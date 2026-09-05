//! Codec for the record framing used inside sstable data blocks.

use super::{FLAG_DELETED, MAX_KEY_LEN, MAX_VALUE_LEN};
use crate::error::{LsmError, Result};
use crate::utils;

// On-disk record in a data block:
//   [key_len u16 LE][flags u8][value_len u32 LE][key][value]
//   flags bit0 (FLAG_DELETED) = tombstone. A Put with empty value is flags=0, value_len=0.
pub(crate) const RECORD_HEADER_LEN: usize = 2 + 1 + 4;

pub(crate) fn record_len(key_len: usize, value_len: usize) -> usize {
    RECORD_HEADER_LEN + key_len + value_len
}

/// Appends one record to `buf` after validating length limits. The on-disk
/// layout is `[key_len u16 LE][flags u8][value_len u32 LE][key][value]`.
pub(crate) fn put_record(buf: &mut Vec<u8>, key: &[u8], value: &[u8], deleted: bool) -> Result<()> {
    if key.len() > MAX_KEY_LEN {
        return Err(LsmError::KeyTooLarge(key.len(), MAX_KEY_LEN));
    }
    if value.len() > MAX_VALUE_LEN {
        return Err(LsmError::ValueTooLarge(value.len(), MAX_VALUE_LEN));
    }
    let flags = if deleted { FLAG_DELETED } else { 0 };
    utils::put_u16(buf, key.len() as u16);
    buf.push(flags);
    utils::put_u32(buf, value.len() as u32);
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);
    Ok(())
}

/// Decodes one record at `offset`, returning (key, value, deleted, consumed).
/// Bounds-checked: `Corruption` if fewer than the 7 header bytes remain at
/// `offset`, or if the header-declared payload runs past the end of `bytes`.
pub(crate) fn read_record(bytes: &[u8], offset: usize) -> Result<(Vec<u8>, Vec<u8>, bool, usize)> {
    let remaining = bytes.len().saturating_sub(offset);
    if remaining < RECORD_HEADER_LEN {
        return Err(LsmError::Corruption(format!(
            "sstable record at offset {offset} shorter than header"
        )));
    }
    let key_len = utils::read_u16(bytes, offset)? as usize;
    let flags = bytes[offset + 2];
    let value_len = utils::read_u32(bytes, offset + 3)? as usize;
    let total = record_len(key_len, value_len);
    if total > remaining {
        return Err(LsmError::Corruption(format!(
            "sstable record at offset {offset} truncated: header declares {total} bytes, only {remaining} available"
        )));
    }
    let key_start = offset + RECORD_HEADER_LEN;
    let value_start = key_start + key_len;
    // Only bit0 (FLAG_DELETED) is meaningful; upper flag bits are reserved for
    // forward compatibility and must not make a record invalid.
    let deleted = (flags & FLAG_DELETED) != 0;
    Ok((
        bytes[key_start..value_start].to_vec(),
        bytes[value_start..offset + total].to_vec(),
        deleted,
        total,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_put_roundtrip() {
        let mut buf = Vec::new();
        put_record(&mut buf, b"k", b"v", false).expect("put must succeed");
        assert_eq!(buf.len(), 7 + 1 + 1);
        let mut expected = vec![1, 0, 0, 1, 0, 0, 0];
        expected.extend_from_slice(b"k");
        expected.extend_from_slice(b"v");
        assert_eq!(buf, expected);

        let (key, value, deleted, consumed) = read_record(&buf, 0).expect("read must succeed");
        assert_eq!(key, b"k");
        assert_eq!(value, b"v");
        assert!(!deleted);
        assert_eq!(consumed, 9);
    }

    #[test]
    fn record_tombstone_roundtrip() {
        let mut buf = Vec::new();
        put_record(&mut buf, b"gone", b"", true).expect("put must succeed");

        let (key, value, deleted, consumed) = read_record(&buf, 0).expect("read must succeed");
        assert_eq!(key, b"gone");
        assert!(value.is_empty());
        assert!(deleted);
        assert_eq!(consumed, 7 + 4);
    }

    #[test]
    fn put_empty_value_is_not_tombstone() {
        let mut buf = Vec::new();
        put_record(&mut buf, b"k", b"", false).expect("put must succeed");

        let (_, value, deleted, _) = read_record(&buf, 0).expect("read must succeed");
        assert!(!deleted);
        assert!(value.is_empty());
    }

    #[test]
    fn record_flags_upper_bits_ignored() {
        // flags 0x05 = bit0 (tombstone) + reserved bit2 set.
        let mut buf = vec![1, 0, 0x05, 0, 0, 0, 0];
        buf.push(b'k');
        let (_, _, deleted, consumed) = read_record(&buf, 0).expect("read must succeed");
        assert!(deleted);
        assert_eq!(consumed, 8);

        // flags 0x02 = reserved bit1 only, no tombstone.
        let buf = vec![1, 0, 0x02, 0, 0, 0, 0, b'k'];
        let (_, _, deleted, _) = read_record(&buf, 0).expect("read must succeed");
        assert!(!deleted);
    }

    #[test]
    fn record_len_matches_written() {
        for (key_len, value_len) in [
            (0, 0),
            (1, 1),
            (7, 0),
            (0, 300),
            (100, 100),
            (65535, 0),
            (300, 4096),
        ] {
            let key = vec![0xABu8; key_len];
            let value = vec![0xCDu8; value_len];
            let mut buf = Vec::new();
            put_record(&mut buf, &key, &value, key_len % 2 == 0).expect("put must succeed");

            assert_eq!(buf.len(), record_len(key_len, value_len));
            let (_, _, deleted, consumed) = read_record(&buf, 0).expect("read must succeed");
            assert_eq!(consumed, record_len(key_len, value_len));
            assert_eq!(deleted, key_len % 2 == 0);
        }
    }

    #[test]
    fn read_record_truncated_header_errors() {
        assert!(matches!(
            read_record(&[0u8; 3], 0),
            Err(LsmError::Corruption(_))
        ));

        let mut buf = Vec::new();
        put_record(&mut buf, b"hello", b"world", false).expect("put must succeed");
        // Offset into a tail too short to hold the 7-byte header.
        assert!(matches!(
            read_record(&buf, buf.len() - 2),
            Err(LsmError::Corruption(_))
        ));
        assert!(matches!(
            read_record(&buf, buf.len()),
            Err(LsmError::Corruption(_))
        ));
    }

    #[test]
    fn read_record_truncated_payload_errors() {
        let mut buf = Vec::new();
        put_record(&mut buf, b"hello world", b"some value", false).expect("put must succeed");
        let err = read_record(&buf[..buf.len() - 1], 0).expect_err("truncated payload must error");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn key_too_large_rejected() {
        let key = vec![0u8; 65536];
        let mut buf = Vec::new();
        let err = put_record(&mut buf, &key, b"v", false).expect_err("oversized key must error");
        assert!(matches!(err, LsmError::KeyTooLarge(65536, 65535)));
        assert!(buf.is_empty());
    }

    #[test]
    fn put_record_appends_and_chains() {
        let mut buf = Vec::new();
        put_record(&mut buf, b"first", b"one", false).expect("put first");
        put_record(&mut buf, b"second", b"", true).expect("put second");

        let first_len = 7 + 5 + 3;
        let second_len = 7 + 6;
        assert_eq!(buf.len(), first_len + second_len);

        let (key, value, deleted, c1) = read_record(&buf, 0).expect("read first");
        assert_eq!(key, b"first");
        assert_eq!(value, b"one");
        assert!(!deleted);
        assert_eq!(c1, first_len);

        let (key, value, deleted, c2) = read_record(&buf, c1).expect("read second");
        assert_eq!(key, b"second");
        assert!(value.is_empty());
        assert!(deleted);
        assert_eq!(c2, second_len);
        assert_eq!(c1 + c2, buf.len());
    }
}
