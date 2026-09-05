use crate::error::{LsmError, Result};
use crate::utils;

pub mod reader;
pub mod writer;

// On-disk entry layout (multi-byte ints LittleEndian):
// [CRC32 u32][op u8][key_len u16][val_len u32][key][value]
// The CRC covers every byte after the CRC field, i.e. [op .. end of value].
const CRC_LEN: usize = 4;
const OP_LEN: usize = 1;
const KEY_LEN_LEN: usize = 2;
const VAL_LEN_LEN: usize = 4;
pub(crate) const HEADER_LEN: usize = CRC_LEN + OP_LEN + KEY_LEN_LEN + VAL_LEN_LEN;

const OP_BYTE_OFFSET: usize = CRC_LEN;
const KEY_LEN_OFFSET: usize = CRC_LEN + OP_LEN;
const VAL_LEN_OFFSET: usize = CRC_LEN + OP_LEN + KEY_LEN_LEN;

pub const MAX_KEY_LEN: usize = 65535;
pub const MAX_VALUE_LEN: usize = u32::MAX as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalOp {
    Put,
    Delete,
}

impl WalOp {
    pub fn from_byte(b: u8) -> Result<WalOp> {
        match b {
            0x01 => Ok(WalOp::Put),
            0x02 => Ok(WalOp::Delete),
            other => Err(LsmError::InvalidOp(format!(
                "unknown wal op byte: 0x{other:02x}"
            ))),
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            WalOp::Put => 0x01,
            WalOp::Delete => 0x02,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalEntry {
    pub op: WalOp,
    pub key: Vec<u8>,
    // Empty for Delete; a Put with an empty value is valid user data.
    pub value: Vec<u8>,
}

pub fn encode_entry(op: WalOp, key: &[u8], value: &[u8], buf: &mut Vec<u8>) -> Result<()> {
    if key.len() > MAX_KEY_LEN {
        return Err(LsmError::KeyTooLarge(key.len(), MAX_KEY_LEN));
    }
    if value.len() > MAX_VALUE_LEN {
        return Err(LsmError::ValueTooLarge(value.len(), MAX_VALUE_LEN));
    }
    if op == WalOp::Delete && !value.is_empty() {
        return Err(LsmError::InvalidOp(
            "delete entry must not carry a value".to_string(),
        ));
    }

    let start = buf.len();
    buf.reserve(HEADER_LEN + key.len() + value.len());
    buf.extend_from_slice(&[0; CRC_LEN]);
    buf.push(op.to_byte());
    utils::put_u16(buf, key.len() as u16);
    utils::put_u32(buf, value.len() as u32);
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);

    let crc = crc32fast::hash(&buf[start + CRC_LEN..]);
    buf[start..start + CRC_LEN].copy_from_slice(&crc.to_le_bytes());
    Ok(())
}

pub fn decode_entry(bytes: &[u8]) -> Result<WalEntry> {
    if bytes.len() < HEADER_LEN {
        return Err(LsmError::Corruption(format!(
            "wal entry shorter than {HEADER_LEN}-byte header: {} bytes",
            bytes.len()
        )));
    }

    let op = WalOp::from_byte(bytes[OP_BYTE_OFFSET])?;

    let key_len = utils::read_u16(bytes, KEY_LEN_OFFSET)? as usize;
    let val_len = utils::read_u32(bytes, VAL_LEN_OFFSET)? as usize;

    let payload_end = HEADER_LEN + key_len + val_len;
    if payload_end > bytes.len() {
        return Err(LsmError::Corruption(format!(
            "wal entry truncated: header declares {} payload bytes, only {} available",
            key_len + val_len,
            bytes.len() - HEADER_LEN
        )));
    }

    // CRC is scoped to this single entry, not trailing chained entries.
    let stored = crate::utils::read_u32(bytes, 0)?;
    if crc32fast::hash(&bytes[CRC_LEN..payload_end]) != stored {
        return Err(LsmError::Corruption("wal checksum mismatch".to_string()));
    }

    Ok(WalEntry {
        op,
        key: bytes[HEADER_LEN..HEADER_LEN + key_len].to_vec(),
        value: bytes[HEADER_LEN + key_len..payload_end].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_into(op: WalOp, key: &[u8], value: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_entry(op, key, value, &mut buf).expect("encode should succeed");
        buf
    }

    #[test]
    fn roundtrip_put_entry() {
        let bytes = encode_into(WalOp::Put, b"hello", b"world");
        let entry = decode_entry(&bytes).expect("decode should succeed");
        assert_eq!(
            entry,
            WalEntry {
                op: WalOp::Put,
                key: b"hello".to_vec(),
                value: b"world".to_vec(),
            }
        );
    }

    #[test]
    fn roundtrip_delete_entry() {
        let bytes = encode_into(WalOp::Delete, b"gone", b"");
        let entry = decode_entry(&bytes).expect("decode should succeed");
        assert_eq!(
            entry,
            WalEntry {
                op: WalOp::Delete,
                key: b"gone".to_vec(),
                value: Vec::new(),
            }
        );
    }

    #[test]
    fn roundtrip_empty_key_and_value_put() {
        let bytes = encode_into(WalOp::Put, b"", b"");
        let entry = decode_entry(&bytes).expect("decode should succeed");
        assert_eq!(
            entry,
            WalEntry {
                op: WalOp::Put,
                key: Vec::new(),
                value: Vec::new(),
            }
        );
    }

    #[test]
    fn crc_mismatch_detected() {
        let mut bytes = encode_into(WalOp::Put, b"hello", b"world");
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        let err = decode_entry(&bytes).expect_err("corrupted entry must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn invalid_op_byte_rejected() {
        let mut bytes = encode_into(WalOp::Put, b"hello", b"world");
        bytes[4] = 0x7F;
        let err = decode_entry(&bytes).expect_err("invalid op byte must be rejected");
        assert!(matches!(err, LsmError::InvalidOp(_)));
    }

    #[test]
    fn key_too_large_rejected_on_encode() {
        let key = vec![0u8; MAX_KEY_LEN + 1];
        let mut buf = Vec::new();
        let err = encode_entry(WalOp::Put, &key, b"v", &mut buf)
            .expect_err("oversized key must be rejected");
        assert!(matches!(err, LsmError::KeyTooLarge(_, 65535)));
    }

    #[test]
    fn max_length_key_roundtrips() {
        let key = vec![0xABu8; MAX_KEY_LEN];
        let bytes = encode_into(WalOp::Put, &key, b"v");
        let entry = decode_entry(&bytes).expect("max-length key must decode");
        assert_eq!(entry.key, key);
        assert_eq!(entry.value, b"v");
    }

    #[test]
    fn delete_with_value_rejected_on_encode() {
        let mut buf = Vec::new();
        let err = encode_entry(WalOp::Delete, b"k", b"payload", &mut buf)
            .expect_err("delete with a value must be rejected");
        assert!(matches!(err, LsmError::InvalidOp(_)));
    }

    #[test]
    fn decode_truncated_header_rejected() {
        let bytes = encode_into(WalOp::Put, b"k", b"v");
        let err = decode_entry(&bytes[..3]).expect_err("truncated header must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn decode_truncated_payload_rejected() {
        let bytes = encode_into(WalOp::Put, b"k", b"v");
        let err = decode_entry(&bytes[..bytes.len() - 2])
            .expect_err("truncated payload must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn op_to_from_byte_roundtrip() {
        for op in [WalOp::Put, WalOp::Delete] {
            assert_eq!(
                WalOp::from_byte(op.to_byte()).expect("op byte should roundtrip"),
                op
            );
        }
    }

    #[test]
    fn encode_appends_and_can_chain() {
        let mut buf = Vec::new();
        encode_entry(WalOp::Put, b"hello", b"world", &mut buf).expect("encode put");
        encode_entry(WalOp::Delete, b"gone", b"", &mut buf).expect("encode delete");

        let first_len = 11 + 5 + 5;
        let second_len = 11 + 4;
        assert_eq!(buf.len(), first_len + second_len);

        let first = decode_entry(&buf).expect("decode first entry");
        assert_eq!(
            first,
            WalEntry {
                op: WalOp::Put,
                key: b"hello".to_vec(),
                value: b"world".to_vec(),
            }
        );

        let second = decode_entry(&buf[first_len..]).expect("decode second entry");
        assert_eq!(
            second,
            WalEntry {
                op: WalOp::Delete,
                key: b"gone".to_vec(),
                value: Vec::new(),
            }
        );
    }
}
