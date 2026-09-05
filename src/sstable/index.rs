//! Sparse index codec for sstable data blocks.
//!
//! The index maps each data block's first key to its (offset, size) pair so a
//! reader can binary-search for the candidate block containing a key.

use super::MAX_KEY_LEN;
use crate::error::{LsmError, Result};
use crate::utils;

// On-disk index entry (self-describing, no count):
//   [key_len u16 LE][first_key][offset u64 LE][size u64 LE]
pub(crate) const INDEX_ENTRY_HEADER_LEN: usize = 2 + 8 + 8;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct IndexEntry {
    pub first_key: Vec<u8>,
    pub offset: u64,
    pub size: u64,
}

/// Appends every entry to `buf` after validating length limits. Each entry is
/// written as `[key_len u16 LE][first_key][offset u64 LE][size u64 LE]`.
pub(crate) fn encode_index(buf: &mut Vec<u8>, entries: &[IndexEntry]) -> Result<()> {
    for e in entries {
        if e.first_key.len() > MAX_KEY_LEN {
            return Err(LsmError::KeyTooLarge(e.first_key.len(), MAX_KEY_LEN));
        }
    }
    for e in entries {
        utils::put_u16(buf, e.first_key.len() as u16);
        buf.extend_from_slice(&e.first_key);
        utils::put_u64(buf, e.offset);
        utils::put_u64(buf, e.size);
    }
    Ok(())
}

/// Decodes a self-describing index: entries are concatenated without a count,
/// so parsing walks until the buffer is consumed exactly. `Corruption` if a
/// trailing fragment is too short to hold an entry or a declared key length
/// runs past the end of `bytes`.
pub(crate) fn parse_index(bytes: &[u8]) -> Result<Vec<IndexEntry>> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        if remaining < INDEX_ENTRY_HEADER_LEN {
            return Err(LsmError::Corruption(format!(
                "sstable index entry at offset {offset} truncated: {remaining} bytes left, need at least {INDEX_ENTRY_HEADER_LEN}"
            )));
        }
        let key_len = utils::read_u16(bytes, offset)? as usize;
        let entry_len = INDEX_ENTRY_HEADER_LEN + key_len;
        if entry_len > remaining {
            return Err(LsmError::Corruption(format!(
                "sstable index entry at offset {offset} truncated: header declares {entry_len} bytes, only {remaining} available"
            )));
        }
        let key_end = offset + 2 + key_len;
        entries.push(IndexEntry {
            first_key: bytes[offset + 2..key_end].to_vec(),
            offset: utils::read_u64(bytes, key_end)?,
            size: utils::read_u64(bytes, key_end + 8)?,
        });
        offset += entry_len;
    }
    Ok(entries)
}

/// Returns the index of the block that could contain `key`: the rightmost
/// entry whose `first_key <= key`. Returns `None` if `key` sorts before the
/// first entry's `first_key`. `entries` must be sorted ascending by
/// `first_key` (the builder guarantees this).
pub(crate) fn locate_block(entries: &[IndexEntry], key: &[u8]) -> Option<usize> {
    // partition_point splits at the first entry whose first_key > key, so the
    // predecessor of that point is the rightmost entry with first_key <= key.
    let split = entries.partition_point(|e| e.first_key.as_slice() <= key);
    if split == 0 { None } else { Some(split - 1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(key: &[u8], offset: u64, size: u64) -> IndexEntry {
        IndexEntry {
            first_key: key.to_vec(),
            offset,
            size,
        }
    }

    #[test]
    fn encode_parse_roundtrip() {
        let entries = vec![
            entry(b"aaa", 0, 100),
            entry(b"bbb", 100, 200),
            entry(b"zzz", 300, 50),
        ];
        let mut buf = Vec::new();
        encode_index(&mut buf, &entries).expect("encode must succeed");
        assert_eq!(buf.len(), (18 + 3) * 3);
        let parsed = parse_index(&buf).expect("parse must succeed");
        assert_eq!(parsed, entries);
    }

    #[test]
    fn encode_parse_empty() {
        let mut buf = Vec::new();
        encode_index(&mut buf, &[]).expect("encode empty must succeed");
        assert!(buf.is_empty());
        assert_eq!(parse_index(&[]).expect("parse empty must succeed"), vec![]);
        assert_eq!(
            parse_index(&buf).expect("parse empty buffer must succeed"),
            vec![]
        );
    }

    #[test]
    fn parse_truncated_mid_entry_errors() {
        let mut buf = Vec::new();
        encode_index(
            &mut buf,
            &[
                entry(b"aaa", 0, 100),
                entry(b"bbb", 100, 200),
                entry(b"zzz", 300, 50),
            ],
        )
        .expect("encode must succeed");
        buf.truncate(buf.len() - 5);
        let err = parse_index(&buf).expect_err("truncated index must error");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn parse_trailing_garbage_errors() {
        let mut buf = Vec::new();
        encode_index(&mut buf, &[entry(b"aaa", 0, 100), entry(b"zzz", 300, 50)])
            .expect("encode must succeed");
        buf.extend_from_slice(&[0xDE, 0xAD, 0xBE]);
        let err = parse_index(&buf).expect_err("trailing garbage must error");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn parse_oversized_key_len_errors() {
        // key_len field 0xFFFF but far fewer than 65535 key bytes available.
        let mut bytes = vec![0xFF, 0xFF];
        bytes.extend_from_slice(&[0u8; 20]);
        let err = parse_index(&bytes).expect_err("oversized declared key_len must error");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn encode_key_too_large_errors() {
        let mut buf = Vec::new();
        let err = encode_index(&mut buf, &[entry(&[0u8; 65536], 0, 0)])
            .expect_err("oversized key must error");
        assert!(matches!(err, LsmError::KeyTooLarge(65536, 65535)));
        assert!(buf.is_empty());
    }

    #[test]
    fn max_length_key_roundtrip() {
        let key = vec![0x5Au8; MAX_KEY_LEN];
        let entries = vec![entry(&key, 42, 7)];
        let mut buf = Vec::new();
        encode_index(&mut buf, &entries).expect("max-length key must encode");
        assert_eq!(buf.len(), INDEX_ENTRY_HEADER_LEN + MAX_KEY_LEN);
        let parsed = parse_index(&buf).expect("max-length key must parse");
        assert_eq!(parsed, entries);
    }

    #[test]
    fn locate_rightmost_le() {
        let entries = vec![
            entry(b"a", 0, 10),
            entry(b"d", 10, 10),
            entry(b"m", 20, 10),
            entry(b"z", 30, 10),
        ];
        assert_eq!(locate_block(&entries, b"a"), Some(0));
        assert_eq!(locate_block(&entries, b"b"), Some(0));
        assert_eq!(locate_block(&entries, b"d"), Some(1));
        assert_eq!(locate_block(&entries, b"m"), Some(2));
        assert_eq!(locate_block(&entries, b"q"), Some(2));
        assert_eq!(locate_block(&entries, b"z"), Some(3));
        assert_eq!(locate_block(&entries, b"zz"), Some(3));
        assert_eq!(locate_block(&entries, b" "), None);
        assert_eq!(locate_block(&entries, b""), None);
    }

    #[test]
    fn locate_empty_entries() {
        assert_eq!(locate_block(&[], b"any"), None);
    }

    #[test]
    fn binary_search_big_list() {
        let entries: Vec<IndexEntry> = (0..10_000u32)
            .map(|i| entry(format!("k{i:05}").as_bytes(), i as u64 * 100, 100))
            .collect();
        assert_eq!(locate_block(&entries, b"k00000"), Some(0));
        assert_eq!(locate_block(&entries, b"k05000"), Some(5000));
        assert_eq!(locate_block(&entries, b"k09999"), Some(9999));
        assert_eq!(locate_block(&entries, b"k00001a"), Some(1));
        assert_eq!(locate_block(&entries, b"k09876a"), Some(9876));
        assert_eq!(locate_block(&entries, b"k10000"), Some(9999));
        assert_eq!(locate_block(&entries, b"z"), Some(9999));
        assert_eq!(locate_block(&entries, b"j"), None);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(locate_block(&entries, &e.first_key), Some(i));
        }
    }
}
