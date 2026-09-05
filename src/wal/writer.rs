use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::wal::{WalOp, encode_entry};

/// Append-only write-ahead log.
///
/// Writes are buffered in memory until [`WalWriter::sync`] flushes them to the
/// OS and fsyncs; **dropping the writer does NOT make data durable**. Entries
/// chain end-to-end using the codec framing.
///
/// Error contract: if [`WalWriter::append`] or [`WalWriter::sync`] returns an
/// error the writer may hold a partially-written (torn) tail and its internal
/// offset is unknown. The caller MUST discard it (and rotate to a fresh WAL);
/// it must not be reused for further appends. Recovery tolerates torn tails by
/// stopping at the first corrupt entry.
pub struct WalWriter {
    file: BufWriter<File>,
    path: PathBuf,
    buf: Vec<u8>,
}

impl WalWriter {
    /// Creates a new WAL at `path`, truncating any existing file.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::create(path.as_ref())?;
        Ok(WalWriter {
            file: BufWriter::new(file),
            path: path.as_ref().to_path_buf(),
            buf: Vec::new(),
        })
    }

    /// Appends one entry (Put or Delete) to the log buffer. Not yet durable.
    ///
    /// On error, the writer must be discarded (see the type-level contract).
    pub fn append(&mut self, op: WalOp, key: &[u8], value: &[u8]) -> Result<()> {
        self.buf.clear();
        encode_entry(op, key, value, &mut self.buf)?;
        self.file.write_all(&self.buf)?;
        Ok(())
    }

    /// Flushes buffered bytes to the OS and fsyncs (durability point).
    pub fn sync(&mut self) -> Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok(())
    }

    /// Returns the path this WAL was created at.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::WalWriter;
    use crate::error::LsmError;
    use crate::wal::{HEADER_LEN, WalEntry, WalOp, decode_entry, encode_entry};
    use tempfile::tempdir;

    fn test_writer_path(dir: &std::path::Path) -> std::path::PathBuf {
        dir.join("test.wal")
    }

    fn encoded_len(key: &[u8], value: &[u8]) -> usize {
        HEADER_LEN + key.len() + value.len()
    }

    fn expect_entries(reader_bytes: &[u8], expected: &[WalEntry]) {
        let mut offset = 0;
        let mut decoded = Vec::new();
        while offset < reader_bytes.len() {
            let entry = decode_entry(&reader_bytes[offset..]).expect("decode chained entry");
            offset += encoded_len(&entry.key, &entry.value);
            decoded.push(entry);
        }
        assert_eq!(
            offset,
            reader_bytes.len(),
            "decode walk must consume all bytes"
        );
        assert_eq!(decoded, expected);
    }

    #[test]
    fn create_makes_file() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let writer = WalWriter::create(&path).expect("create should succeed");
        assert!(path.exists(), "wal file must exist after create");
        assert_eq!(writer.path(), path);
    }

    #[test]
    fn append_put_and_read_back_raw_bytes() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        writer
            .append(WalOp::Put, b"hello", b"world")
            .expect("append should succeed");
        writer.sync().expect("sync should succeed");
        drop(writer);

        let bytes = std::fs::read(&path).expect("read wal file");
        assert_eq!(bytes.len(), encoded_len(b"hello", b"world"));
        assert_eq!(bytes.len(), 11 + 5 + 5);
        let expected = {
            let mut buf = Vec::new();
            encode_entry(WalOp::Put, b"hello", b"world", &mut buf).expect("encode reference");
            buf
        };
        assert_eq!(
            bytes, expected,
            "file bytes must match the codec frame exactly"
        );
    }

    #[test]
    fn append_multiple_entries_frame() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        writer.append(WalOp::Put, b"a", b"b").expect("append put a");
        writer
            .append(WalOp::Delete, b"c", b"")
            .expect("append delete c");
        writer
            .append(WalOp::Put, b"", b"")
            .expect("append empty put");
        writer
            .append(WalOp::Put, b"kk", b"vvv")
            .expect("append put kk");
        writer.sync().expect("sync should succeed");
        drop(writer);

        let bytes = std::fs::read(&path).expect("read wal file");
        assert!(!bytes.is_empty(), "wal must not be empty");
        let expected_len = encoded_len(b"a", b"b")
            + encoded_len(b"c", b"")
            + encoded_len(b"", b"")
            + encoded_len(b"kk", b"vvv");
        assert_eq!(bytes.len(), expected_len);
        expect_entries(
            &bytes,
            &[
                WalEntry {
                    op: WalOp::Put,
                    key: b"a".to_vec(),
                    value: b"b".to_vec(),
                },
                WalEntry {
                    op: WalOp::Delete,
                    key: b"c".to_vec(),
                    value: Vec::new(),
                },
                WalEntry {
                    op: WalOp::Put,
                    key: Vec::new(),
                    value: Vec::new(),
                },
                WalEntry {
                    op: WalOp::Put,
                    key: b"kk".to_vec(),
                    value: b"vvv".to_vec(),
                },
            ],
        );
    }

    #[test]
    fn append_oversized_key_errors_and_file_empty() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        let key = vec![0u8; 65536];
        let err = writer
            .append(WalOp::Put, &key, b"v")
            .expect_err("oversized key must be rejected");
        assert!(matches!(err, LsmError::KeyTooLarge(_, 65535)));
        drop(writer);

        let bytes = std::fs::read(&path).expect("read wal file");
        assert_eq!(
            bytes.len(),
            0,
            "failed append must not leave any bytes on disk"
        );
    }

    #[test]
    fn append_delete_with_value_errors() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        let err = writer
            .append(WalOp::Delete, b"k", b"payload")
            .expect_err("delete with a value must be rejected");
        assert!(matches!(err, LsmError::InvalidOp(_)));
    }

    #[test]
    fn sync_then_read_back() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        let expected: Vec<WalEntry> = (0..5)
            .map(|i| WalEntry {
                op: WalOp::Put,
                key: format!("key{i}").into_bytes(),
                value: format!("value{i}").into_bytes(),
            })
            .collect();
        for entry in &expected {
            writer
                .append(entry.op, &entry.key, &entry.value)
                .expect("append should succeed");
        }
        writer.sync().expect("sync should succeed");
        drop(writer);

        let bytes = std::fs::read(&path).expect("read wal file");
        let total: usize = expected.iter().map(|e| encoded_len(&e.key, &e.value)).sum();
        assert_eq!(bytes.len(), total);
        expect_entries(&bytes, &expected);
    }

    #[test]
    fn append_after_sync_continues() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = test_writer_path(dir.path());
        let mut writer = WalWriter::create(&path).expect("create should succeed");
        writer.append(WalOp::Put, b"a", b"1").expect("append a");
        writer.sync().expect("sync a");
        writer.append(WalOp::Put, b"b", b"2").expect("append b");
        writer.sync().expect("sync b");
        drop(writer);

        let bytes = std::fs::read(&path).expect("read wal file");
        assert_eq!(
            bytes.len(),
            encoded_len(b"a", b"1") + encoded_len(b"b", b"2")
        );
        expect_entries(
            &bytes,
            &[
                WalEntry {
                    op: WalOp::Put,
                    key: b"a".to_vec(),
                    value: b"1".to_vec(),
                },
                WalEntry {
                    op: WalOp::Put,
                    key: b"b".to_vec(),
                    value: b"2".to_vec(),
                },
            ],
        );
    }
}
