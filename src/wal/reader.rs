use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::{LsmError, Result};
use crate::memtable::MemTable;
use crate::utils;
use crate::wal::{HEADER_LEN, KEY_LEN_OFFSET, VAL_LEN_OFFSET, WalEntry, WalOp, decode_entry};

/// Read-only view over a write-ahead log file.
///
/// Recovery semantics: a log that **ends mid-entry** (a torn crash tail — fewer
/// bytes left than the current entry's full frame) is treated as benign: reading
/// stops cleanly and returns every complete entry before the tear. A fully
/// present frame that fails to decode (CRC mismatch or an invalid op byte) is
/// **genuine mid-file corruption** and is reported as
/// [`LsmError::Corruption`]; it is never silently skipped.
///
/// Limitation: a bit-flip that *inflates* an entry's declared key/value lengths
/// so the frame appears to run past the end of the file is indistinguishable
/// from a torn tail and is therefore reported as one (benign). Only corruption
/// of fully-present frames is detectable without an out-of-band length marker.
///
/// `WalReader` is re-scan-able: every call to [`WalReader::read_all`] or
/// [`WalReader::replay_into`] rewinds to the start of the file and re-reads it.
pub struct WalReader {
    file: File,
    path: PathBuf,
}

impl WalReader {
    /// Opens the WAL at `path` for reading. Errors with `LsmError::Io` if the
    /// file does not exist or cannot be opened.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        Ok(WalReader {
            file,
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Reads every complete entry from the log, in file order.
    ///
    /// Stops cleanly (returning the entries collected so far, without an error)
    /// when the file ends in the middle of an entry — the signature of a crash
    /// interrupting an append. Returns `LsmError::Corruption` if a fully
    /// present frame fails its CRC or carries an invalid op byte.
    pub fn read_all(&mut self) -> Result<Vec<WalEntry>> {
        self.scan()
    }

    /// Replays the log into `memtable` (`Put` → `put`, `Delete` → `delete`)
    /// and returns the number of entries applied.
    ///
    /// Same tail semantics as [`WalReader::read_all`]: a partial trailing entry
    /// is ignored and its predecessors are still applied. Replay is
    /// all-or-nothing with respect to corruption: if the log contains genuine
    /// mid-file corruption nothing is applied and `LsmError::Corruption` is
    /// returned.
    pub fn replay_into(&mut self, memtable: &MemTable) -> Result<u64> {
        let entries = self.scan()?;
        let mut applied: u64 = 0;
        for entry in &entries {
            match entry.op {
                WalOp::Put => memtable.put(&entry.key, &entry.value)?,
                WalOp::Delete => memtable.delete(&entry.key)?,
            }
            applied += 1;
        }
        Ok(applied)
    }

    /// Returns the path this reader was opened at.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Walks the whole file from offset 0, decoding one frame at a time.
    ///
    /// Stops without error when fewer bytes remain than the next frame's header
    /// (`HEADER_LEN`) or when the header declares a frame longer than what is
    /// left (both are crash-torn tails). Errors on any fully present frame that
    /// `decode_entry` rejects.
    fn scan(&mut self) -> Result<Vec<WalEntry>> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        self.file.read_to_end(&mut bytes)?;

        let mut entries = Vec::new();
        let mut offset = 0usize;
        loop {
            let remaining = bytes.len() - offset;
            if remaining < HEADER_LEN {
                break;
            }
            let header = &bytes[offset..offset + HEADER_LEN];
            let key_len = utils::read_u16(header, KEY_LEN_OFFSET)? as usize;
            let val_len = utils::read_u32(header, VAL_LEN_OFFSET)? as usize;

            let frame_len = HEADER_LEN + key_len + val_len;
            if frame_len > remaining {
                break;
            }

            let frame = &bytes[offset..offset + frame_len];
            let entry = decode_entry(frame).map_err(|err| {
                LsmError::Corruption(format!(
                    "wal entry at byte {offset} failed to decode: {err}"
                ))
            })?;
            offset += frame_len;
            entries.push(entry);
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::WalReader;
    use crate::error::LsmError;
    use crate::memtable::MemTable;
    use crate::wal::writer::WalWriter;
    use crate::wal::{HEADER_LEN, WalEntry, WalOp, encode_entry};
    use std::io::Write;
    use std::path::Path;
    use tempfile::tempdir;

    fn write_entries(path: &Path, entries: &[(WalOp, &[u8], &[u8])]) {
        let mut writer = WalWriter::create(path).expect("create wal writer");
        for (op, key, value) in entries {
            writer.append(*op, key, value).expect("append wal entry");
        }
        writer.sync().expect("sync wal");
    }

    fn encode_frame(op: WalOp, key: &[u8], value: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_entry(op, key, value, &mut buf).expect("encode entry");
        buf
    }

    fn append_raw_tail(path: &Path, tail: &[u8]) {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("open wal in append mode");
        file.write_all(tail).expect("append raw tail bytes");
        file.sync_all().expect("sync raw tail");
    }

    #[test]
    fn open_missing_file_errors() {
        let dir = tempdir().expect("tempdir should succeed");
        let result = WalReader::open(dir.path().join("missing.wal"));
        assert!(
            matches!(result, Err(LsmError::Io(_))),
            "opening a nonexistent file must yield an Io error"
        );
    }

    #[test]
    fn read_all_empty_file() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("empty.wal");
        std::fs::File::create(&path).expect("create empty wal");

        let mut reader = WalReader::open(&path).expect("open empty wal");
        assert_eq!(
            reader.read_all().expect("read empty wal"),
            Vec::new(),
            "empty wal must yield no entries"
        );
    }

    #[test]
    fn roundtrip_put_and_delete_entries() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("roundtrip.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"a", b"v1"),
                (WalOp::Delete, b"b", b""),
                (WalOp::Put, b"c", b"v3"),
            ],
        );

        let mut reader = WalReader::open(&path).expect("open wal");
        let entries = reader.read_all().expect("read_all should succeed");
        assert_eq!(
            entries,
            vec![
                WalEntry {
                    op: WalOp::Put,
                    key: b"a".to_vec(),
                    value: b"v1".to_vec(),
                },
                WalEntry {
                    op: WalOp::Delete,
                    key: b"b".to_vec(),
                    value: Vec::new(),
                },
                WalEntry {
                    op: WalOp::Put,
                    key: b"c".to_vec(),
                    value: b"v3".to_vec(),
                },
            ]
        );
    }

    #[test]
    fn read_all_stops_at_partial_trailing_header() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("tail-header.wal");
        write_entries(&path, &[(WalOp::Put, b"a", b"1"), (WalOp::Put, b"b", b"2")]);

        let frame = encode_frame(WalOp::Put, b"k", b"v");
        append_raw_tail(&path, &frame[..HEADER_LEN]);

        let mut reader = WalReader::open(&path).expect("open wal");
        let entries = reader
            .read_all()
            .expect("read_all must tolerate a partial trailing header");
        assert_eq!(entries.len(), 2, "must keep the two complete entries");
        assert_eq!(entries[0].key, b"a");
        assert_eq!(entries[1].key, b"b");
    }

    #[test]
    fn read_all_stops_at_partial_trailing_payload() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("tail-payload.wal");
        write_entries(&path, &[(WalOp::Put, b"a", b"1")]);

        let frame = encode_frame(WalOp::Put, &[0xABu8; 100], b"");
        append_raw_tail(&path, &frame[..HEADER_LEN + 5]);

        let mut reader = WalReader::open(&path).expect("open wal");
        let entries = reader
            .read_all()
            .expect("read_all must tolerate a partial trailing payload");
        assert_eq!(entries.len(), 1, "must keep the single complete entry");
        assert_eq!(entries[0].key, b"a");
    }

    #[test]
    fn read_all_errors_on_mid_file_corruption() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("corrupt.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"key0", b"val0"),
                (WalOp::Put, b"key1", b"val1"),
                (WalOp::Put, b"key2", b"val2"),
            ],
        );

        let mut bytes = std::fs::read(&path).expect("read wal bytes");
        let first_len = HEADER_LEN + b"key0".len() + b"val0".len();
        let flip_at = first_len + HEADER_LEN + 1;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write corrupted wal back");

        let mut reader = WalReader::open(&path).expect("open wal");
        let err = reader
            .read_all()
            .expect_err("corrupted mid-file entry must be reported");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn read_all_errors_on_corrupt_final_entry() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("corrupt-final.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"key0", b"val0"),
                (WalOp::Put, b"key1", b"val1"),
            ],
        );

        let mut bytes = std::fs::read(&path).expect("read wal bytes");
        let first_len = HEADER_LEN + b"key0".len() + b"val0".len();
        let flip_at = first_len + HEADER_LEN + 1;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write corrupted wal back");

        let mut reader = WalReader::open(&path).expect("open wal");
        let err = reader
            .read_all()
            .expect_err("corrupted fully-present final entry must be reported");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn read_all_is_repeatable() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("repeat.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"a", b"1"),
                (WalOp::Delete, b"b", b""),
                (WalOp::Put, b"c", b"3"),
            ],
        );

        let mut reader = WalReader::open(&path).expect("open wal");
        let first = reader.read_all().expect("first read_all");
        let second = reader.read_all().expect("second read_all");
        assert_eq!(
            first, second,
            "repeated read_all must return identical entries"
        );
        assert_eq!(first.len(), 3);
    }

    #[test]
    fn replay_puts_and_deletes_into_memtable() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("replay.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"k1", b"v1"),
                (WalOp::Put, b"k2", b"v2"),
                (WalOp::Delete, b"k1", b""),
            ],
        );

        let mut reader = WalReader::open(&path).expect("open wal");
        let memtable = MemTable::new();
        let count = reader
            .replay_into(&memtable)
            .expect("replay should succeed");
        assert_eq!(count, 3, "all three entries must count as applied");
        assert_eq!(memtable.get(b"k1"), None, "k1 was deleted by replay");
        assert_eq!(memtable.get(b"k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn replay_partial_tail_applies_complete_entries() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("replay-tail.wal");
        write_entries(
            &path,
            &[(WalOp::Put, b"k1", b"v1"), (WalOp::Put, b"k2", b"v2")],
        );

        let frame = encode_frame(WalOp::Put, b"k", b"v");
        append_raw_tail(&path, &frame[..HEADER_LEN]);

        let mut reader = WalReader::open(&path).expect("open wal");
        let memtable = MemTable::new();
        let count = reader
            .replay_into(&memtable)
            .expect("replay must tolerate a partial trailing header");
        assert_eq!(count, 2);
        assert_eq!(memtable.get(b"k1"), Some(b"v1".to_vec()));
        assert_eq!(memtable.get(b"k2"), Some(b"v2".to_vec()));
    }

    #[test]
    fn replay_mid_file_corruption_errors_and_does_not_apply() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join("replay-corrupt.wal");
        write_entries(
            &path,
            &[
                (WalOp::Put, b"k0", b"v0"),
                (WalOp::Put, b"k1", b"v1"),
                (WalOp::Put, b"k2", b"v2"),
            ],
        );

        let mut bytes = std::fs::read(&path).expect("read wal bytes");
        let first_len = HEADER_LEN + b"k0".len() + b"v0".len();
        let flip_at = first_len + HEADER_LEN + 1;
        bytes[flip_at] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write corrupted wal back");

        let mut reader = WalReader::open(&path).expect("open wal");
        let memtable = MemTable::new();
        let err = reader
            .replay_into(&memtable)
            .expect_err("corrupted mid-file entry must abort replay");
        assert!(matches!(err, LsmError::Corruption(_)));
    }
}
