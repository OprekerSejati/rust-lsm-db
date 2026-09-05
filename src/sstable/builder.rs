//! SsTableBuilder: accumulates ascending records into ~`TARGET_BLOCK_SIZE`
//! data blocks with a sparse in-memory index, then serializes
//! `[data][index][bloom?][footer]` to an `.sst` file. The bloom block is only
//! emitted when the builder was constructed with [`SsTableBuilder::with_expected_items`];
//! otherwise `bloom_offset` stays 0 and the layout is identical to fase-3.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use super::bloom::{encode_bloom, new_bloom};
use super::index::{IndexEntry, encode_index};
use super::record::{put_record, record_len};
use super::{FOOTER_LEN, MAGIC, MAX_KEY_LEN, MAX_VALUE_LEN, TARGET_BLOCK_SIZE};
use crate::error::{LsmError, Result};
use crate::utils;

/// Accumulates strictly-ascending records and writes them out as an sstable.
///
/// Records land in the active `current` block until it would exceed
/// `TARGET_BLOCK_SIZE`, at which point the block is finalized into `data` and
/// an [`IndexEntry`] records its byte offset. An empty builder writes a valid
/// footer-only file.
///
/// A bloom filter is built lazily: with `expected_items > 0`, the first
/// successful [`SsTableBuilder::add`] creates it (so an empty builder writes
/// no bloom block at all) and every subsequently added key — tombstones
/// included — is inserted.
#[derive(Default)]
pub struct SsTableBuilder {
    data: Vec<u8>,    // finalized blocks, concatenated
    current: Vec<u8>, // active block accumulating records
    current_first_key: Option<Vec<u8>>,
    index: Vec<IndexEntry>,
    first_key: Option<Vec<u8>>, // first key added (min of the whole table)
    last_key: Option<Vec<u8>>,  // most recent key added (max); also enforces strict ordering
    bloom: Option<fastbloom::BloomFilter>, // Some iff expected_items > 0 and a key was added
    expected_items: usize,      // bloom sizing hint; 0 disables the bloom block
}

impl SsTableBuilder {
    /// Returns a builder that writes **no** bloom block (`bloom_offset = 0`),
    /// producing a plain `[data][index][footer]` table. Use
    /// [`SsTableBuilder::with_expected_items`] to enable a bloom filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a builder that writes a bloom filter block sized for
    /// `expected_items` keys. `expected_items` is a sizing hint, not a cap:
    /// the filter still records every added key, and an empty table still
    /// writes no bloom block.
    ///
    /// Estimate trade-off (important for compaction): passing `0` disables the
    /// bloom entirely; over-estimating inflates the on-disk bloom block roughly
    /// in proportion to the hint (~1.2 bytes/key at 1% false-positive rate);
    /// under-estimating only raises the false-positive rate slightly and never
    /// causes a false negative. When the key count is unknown, prefer a modest
    /// over-estimate to disabling the filter.
    pub fn with_expected_items(expected_items: usize) -> Self {
        Self {
            expected_items,
            ..Self::default()
        }
    }

    /// Adds one entry. Keys must be strictly ascending in byte order; a key
    /// equal to or smaller than the previous one yields
    /// `LsmError::InvalidOp`. An empty-string key is valid (and may only be
    /// first). `deleted=true` writes a tombstone record.
    ///
    /// Validation (length caps, then ordering) happens before any state is
    /// mutated, so a rejected `add` leaves the builder untouched. A successful
    /// add also inserts the key into the bloom filter when one is enabled —
    /// including tombstone keys, so a reader can prove a `Deleted` exists
    /// without descending to disk.
    pub fn add(&mut self, key: &[u8], value: &[u8], deleted: bool) -> Result<()> {
        if key.len() > MAX_KEY_LEN {
            return Err(LsmError::KeyTooLarge(key.len(), MAX_KEY_LEN));
        }
        if value.len() > MAX_VALUE_LEN {
            return Err(LsmError::ValueTooLarge(value.len(), MAX_VALUE_LEN));
        }
        self.last_key.as_deref().map_or(Ok(()), |prev| {
            if key <= prev {
                Err(LsmError::InvalidOp(
                    "sstable keys must be strictly ascending".to_string(),
                ))
            } else {
                Ok(())
            }
        })?;

        if !self.current.is_empty()
            && self.current.len() + record_len(key.len(), value.len()) > TARGET_BLOCK_SIZE
        {
            self.finalize_block();
        }
        if self.current.is_empty() {
            self.current_first_key = Some(key.to_vec());
        }
        if self.expected_items > 0 {
            let bloom = self
                .bloom
                .get_or_insert_with(|| new_bloom(self.expected_items));
            bloom.insert(key);
        }
        // Bloom insert above can only over-approximate (a key in bloom but not
        // yet in data), which costs an extra false-positive disk probe at worst
        // — never a false negative — so running it before the infallible
        // put_record below is safe. put_record cannot fail: caps were checked
        // above; the `?` is future-proofing rather than an unchecked append.
        put_record(&mut self.current, key, value, deleted)?;
        if self.last_key.is_none() {
            self.first_key = Some(key.to_vec());
        }
        self.last_key = Some(key.to_vec());
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty() && self.current.is_empty()
    }

    /// Exact size of the data section so far: finalized `data` plus the active
    /// `current` block. The final file is larger (index + footer are appended
    /// at [`SsTableBuilder::build`]); this is a lower bound on total file size.
    pub fn estimated_data_size(&self) -> usize {
        self.data.len() + self.current.len()
    }

    /// Smallest key added so far, if any. Because keys are strictly ascending,
    /// this is the first key inserted.
    pub fn min_key(&self) -> Option<Vec<u8>> {
        self.first_key.clone()
    }

    /// Largest key added so far, if any. Because keys are strictly ascending,
    /// this is the most recently inserted key.
    pub fn max_key(&self) -> Option<Vec<u8>> {
        self.last_key.clone()
    }

    /// Finalizes any pending block and writes `[data][index][bloom?][footer]`
    /// to `path`, truncating an existing file.
    ///
    /// When a bloom filter is enabled (`expected_items > 0`) and at least one
    /// key was added, the bloom block sits between the index and the footer and
    /// `bloom_offset` points at its start; the reader treats the region
    /// `[index_offset..bloom_offset]` as the index. Otherwise `bloom_offset` is
    /// 0 and the layout is byte-identical to fase-3 (`[data][index][footer]`),
    /// which keeps pre-bloom tables readable.
    ///
    /// Durability contract: this is NOT an atomic publish — `File::create`
    /// truncates `path` up front, so an I/O error mid-write destroys a
    /// pre-existing file at that path and leaves a torn partial table (its
    /// missing/invalid footer makes the reader reject it, but the old good
    /// table is gone). Callers that may overwrite a live table must build to a
    /// fresh temp path, fsync, then rename over the target. No fsync is issued
    /// here; flushing is the caller's concern (F5 engine).
    pub fn build(mut self, path: impl AsRef<Path>) -> Result<()> {
        self.finalize_block();
        let index_offset = self.data.len() as u64;
        let mut index_bytes = Vec::new();
        encode_index(&mut index_bytes, &self.index)?;

        let mut bloom_bytes = Vec::new();
        if let Some(bloom) = &self.bloom {
            encode_bloom(&mut bloom_bytes, bloom);
        }
        let bloom_offset = if bloom_bytes.is_empty() {
            0
        } else {
            index_offset + index_bytes.len() as u64
        };

        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&self.data)?;
        writer.write_all(&index_bytes)?;
        writer.write_all(&bloom_bytes)?;

        let mut footer = Vec::with_capacity(FOOTER_LEN);
        utils::put_u64(&mut footer, index_offset);
        utils::put_u64(&mut footer, bloom_offset);
        footer.extend_from_slice(&MAGIC);
        writer.write_all(&footer)?;
        writer.flush()?;
        Ok(())
    }

    /// Moves the active block into `data`, recording its absolute offset as
    /// the `data` length before the append. `data` only ever grows here, so
    /// recorded offsets match the file layout exactly.
    fn finalize_block(&mut self) {
        if let Some(first_key) = self.current_first_key.take() {
            let offset = self.data.len() as u64;
            let size = self.current.len() as u64;
            self.index.push(IndexEntry {
                first_key,
                offset,
                size,
            });
            self.data.append(&mut self.current);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::error::LsmError;
    use crate::sstable::index::parse_index;
    use crate::sstable::record::read_record;
    use crate::sstable::{FOOTER_LEN, MAGIC, TARGET_BLOCK_SIZE};
    use crate::utils;

    use super::SsTableBuilder;

    fn read_footer(bytes: &[u8]) -> (u64, u64) {
        assert!(bytes.len() >= FOOTER_LEN);
        let footer = &bytes[bytes.len() - FOOTER_LEN..];
        (
            utils::read_u64(footer, 0).expect("footer index_offset"),
            utils::read_u64(footer, 8).expect("footer bloom_offset"),
        )
    }

    fn read_all_records(bytes: &[u8]) -> Vec<(Vec<u8>, Vec<u8>, bool)> {
        let (index_offset, _) = read_footer(bytes);
        let mut records = Vec::new();
        let mut offset = 0usize;
        while offset < index_offset as usize {
            let (key, value, deleted, consumed) =
                read_record(bytes, offset).expect("record must parse");
            assert!(consumed > 0);
            records.push((key, value, deleted));
            offset += consumed;
        }
        records
    }

    #[test]
    fn build_writes_valid_file_layout() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("layout.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"k1", b"v1", false).expect("add k1");
        builder.add(b"k2", b"v2", false).expect("add k2");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");

        let data_len = 11usize + 11; // two records: 7 header + 2 key + 2 value
        let index_len = 20usize; // one sparse entry (both records share a block): 2 key + 18 header
        assert_eq!(bytes.len(), data_len + index_len + FOOTER_LEN);

        let (index_offset, bloom_offset) = read_footer(&bytes);
        assert_eq!(index_offset, data_len as u64);
        assert_eq!(bloom_offset, 0);
        assert_eq!(&bytes[bytes.len() - 8..], &MAGIC);

        let (key, value, deleted, consumed) = read_record(&bytes, 0).expect("first record");
        assert_eq!(key, b"k1");
        assert_eq!(value, b"v1");
        assert!(!deleted);
        let (key, value, deleted, consumed2) =
            read_record(&bytes, consumed).expect("second record");
        assert_eq!(key, b"k2");
        assert_eq!(value, b"v2");
        assert!(!deleted);
        assert_eq!(consumed + consumed2, data_len);
    }

    #[test]
    fn single_block_index_offset() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("single.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"apple", b"red", false).expect("add apple");
        builder
            .add(b"banana", b"", true)
            .expect("add banana tombstone");
        builder
            .add(b"cherry", b"dark red", false)
            .expect("add cherry");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, _bloom) = read_footer(&bytes);

        // apple: 7+5+3=15, banana: 7+6=13, cherry: 7+6+8=21.
        let data_len = 15 + 13 + 21;
        assert_eq!(index_offset, data_len as u64);

        let entries =
            parse_index(&bytes[data_len..bytes.len() - FOOTER_LEN]).expect("index must parse");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].first_key, b"apple");
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, data_len as u64);
    }

    #[test]
    fn multi_block_split() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("multi.sst");

        let mut builder = SsTableBuilder::new();
        for i in 0..500u32 {
            let key = format!("k{i:05}");
            let value = vec![0u8; 32];
            builder
                .add(key.as_bytes(), &value, false)
                .expect("add must succeed");
        }
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, _bloom) = read_footer(&bytes);

        let entries = parse_index(&bytes[index_offset as usize..bytes.len() - FOOTER_LEN])
            .expect("index must parse");
        // 500 records x 45 bytes each = 22500 bytes; blocks hold at most ~4096.
        assert!(
            entries.len() >= 5,
            "expected >=5 blocks, got {}",
            entries.len()
        );

        assert_eq!(entries[0].offset, 0);
        for pair in entries.windows(2) {
            assert_eq!(pair[1].offset, pair[0].offset + pair[0].size);
        }
        let last = entries.last().expect("at least one entry");
        assert_eq!(last.offset + last.size, index_offset);

        for entry in &entries {
            let (first_key, _, _, _) =
                read_record(&bytes, entry.offset as usize).expect("block first record");
            assert_eq!(first_key, entry.first_key);
            assert!(
                entry.size <= TARGET_BLOCK_SIZE as u64,
                "finalized block must not exceed target unless it is a lone oversized record"
            );
        }
    }

    #[test]
    fn exact_block_boundary_splits_correctly() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("boundary.sst");

        // Empty record = 7 bytes, leaving 4089 for the rest of block 1.
        // A filler record with key_len 1356 costs 7 + 1356 = 1363; three of them
        // fill 4089 exactly, so block 1 totals precisely TARGET_BLOCK_SIZE.
        let mut builder = SsTableBuilder::new();
        builder.add(b"", b"", false).expect("add empty seed");
        let filler_len = 1356;
        for i in 0..3u32 {
            let key = format!("{i:04}{}", "a".repeat(filler_len - 4));
            assert_eq!(key.len(), filler_len);
            builder.add(key.as_bytes(), b"", false).expect("add filler");
        }
        builder.add(b"z-final", b"x", false).expect("add boundary");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, _bloom) = read_footer(&bytes);
        let entries = parse_index(&bytes[index_offset as usize..bytes.len() - FOOTER_LEN])
            .expect("index must parse");

        assert_eq!(entries.len(), 2, "boundary record must start a new block");
        assert_eq!(entries[0].size, TARGET_BLOCK_SIZE as u64);
        assert_eq!(entries[0].first_key, b"");
        assert_eq!(entries[1].first_key, b"z-final");
    }

    #[test]
    fn build_twice_same_path_truncates() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("truncate.sst");

        let mut first = SsTableBuilder::new();
        for i in 0..500u32 {
            let key = format!("k{i:05}");
            first.add(key.as_bytes(), &[0u8; 32], false).unwrap();
        }
        first.build(&path).expect("first build must succeed");
        let first_len = fs::metadata(&path).expect("metadata").len();

        let mut second = SsTableBuilder::new();
        second.add(b"only", b"one", false).unwrap();
        second.build(&path).expect("second build must succeed");
        let second_bytes = fs::read(&path).expect("read after second build");

        assert!(second_bytes.len() < first_len as usize);
        let (index_offset, _bloom) = read_footer(&second_bytes);
        assert_eq!(
            index_offset as usize,
            second_bytes.len() - FOOTER_LEN - (2 + b"only".len() + 8 + 8)
        );
    }

    #[test]
    fn empty_builder_writes_footer_only() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("empty.sst");

        SsTableBuilder::new()
            .build(&path)
            .expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        assert_eq!(bytes.len(), FOOTER_LEN);

        let (index_offset, bloom_offset) = read_footer(&bytes);
        assert_eq!(index_offset, 0);
        assert_eq!(bloom_offset, 0);
        assert_eq!(&bytes[bytes.len() - 8..], &MAGIC);

        assert!(
            parse_index(&bytes[..0])
                .expect("empty index must parse")
                .is_empty()
        );
    }

    #[test]
    fn oversized_record_is_its_own_block() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("oversized.sst");

        let value = vec![0xABu8; 10_000];
        let mut builder = SsTableBuilder::new();
        builder.add(b"big", &value, false).expect("add oversized");
        builder.add(b"small", b"v", false).expect("add small");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, _bloom) = read_footer(&bytes);

        let entries = parse_index(&bytes[index_offset as usize..bytes.len() - FOOTER_LEN])
            .expect("index must parse");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].first_key, b"big");
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, (7 + 3 + 10_000) as u64);
        assert_eq!(entries[1].offset, entries[0].size);
    }

    #[test]
    fn out_of_order_key_rejected() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("order.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"a", b"1", false).expect("add a");
        let err = builder
            .add(b"a", b"2", false)
            .expect_err("duplicate must error");
        assert!(matches!(err, LsmError::InvalidOp(_)));

        builder
            .add(b"c", b"3", false)
            .expect("add c after rejection");
        let err = builder
            .add(b"b", b"4", false)
            .expect_err("b after c must error");
        assert!(matches!(err, LsmError::InvalidOp(_)));

        builder.add(b"d", b"5", false).expect("add d");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let records = read_all_records(&bytes);
        let got: Vec<(&[u8], &[u8])> = records
            .iter()
            .map(|(k, v, _)| (k.as_slice(), v.as_slice()))
            .collect();
        assert_eq!(
            got,
            vec![(&b"a"[..], &b"1"[..]), (b"c", b"3"), (b"d", b"5")]
        );
    }

    #[test]
    fn empty_key_first_allowed_then_reject() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("empty_key.sst");

        let mut builder = SsTableBuilder::new();
        builder
            .add(b"", b"first", false)
            .expect("empty key is valid");
        let err = builder
            .add(b"", b"dup", false)
            .expect_err("duplicate empty must error");
        assert!(matches!(err, LsmError::InvalidOp(_)));

        builder
            .add(b"\x00", b"after", false)
            .expect("0x00 sorts after empty");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let records = read_all_records(&bytes);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, b"");
        assert_eq!(records[0].1, b"first");
        assert_eq!(records[1].0, b"\x00");
        assert_eq!(records[1].1, b"after");
    }

    #[test]
    fn oversized_key_rejected_cleanly() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("bigkey.sst");

        let key = vec![0u8; 65536];
        let mut builder = SsTableBuilder::new();
        let err = builder
            .add(&key, b"v", false)
            .expect_err("oversized key must error");
        assert!(matches!(err, LsmError::KeyTooLarge(65536, 65535)));

        assert!(builder.is_empty(), "failed add must not mutate builder");
        builder
            .add(b"small", b"value", false)
            .expect("valid add after rejection");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let records = read_all_records(&bytes);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].0, b"small");
        assert_eq!(records[0].1, b"value");
    }

    #[test]
    fn estimated_data_size_tracks_current_and_data() {
        let mut builder = SsTableBuilder::new();
        assert_eq!(builder.estimated_data_size(), 0);
        assert!(builder.is_empty());

        builder.add(b"kk", b"v", false).expect("add kk");
        assert_eq!(builder.estimated_data_size(), 10); // 7 header + 2 + 1
        builder.add(b"ll", b"w", false).expect("add ll");
        builder.add(b"mm", b"x", false).expect("add mm");
        assert_eq!(builder.estimated_data_size(), 30);
        assert!(!builder.is_empty());
    }

    #[test]
    fn is_empty_tracks_adds() {
        let mut builder = SsTableBuilder::new();
        assert!(builder.is_empty());
        builder.add(b"k", b"v", false).expect("add must succeed");
        assert!(!builder.is_empty());
    }

    #[test]
    fn min_max_key_track_extremes() {
        let mut builder = SsTableBuilder::new();
        assert_eq!(builder.min_key(), None);
        assert_eq!(builder.max_key(), None);

        builder.add(b"a", b"1", false).expect("add a");
        builder.add(b"b", b"2", false).expect("add b");
        builder.add(b"c", b"3", false).expect("add c");
        assert_eq!(builder.min_key(), Some(b"a".to_vec()));
        assert_eq!(builder.max_key(), Some(b"c".to_vec()));

        // Tombstone counts as a key for range purposes.
        builder.add(b"d", b"", true).expect("add delete d");
        assert_eq!(builder.max_key(), Some(b"d".to_vec()));
    }

    #[test]
    fn min_max_key_across_block_boundaries() {
        let mut builder = SsTableBuilder::with_expected_items(200);
        builder.add(b"", b"", false).expect("add empty first");
        for i in 0..500u32 {
            let key = format!("k{i:05}");
            builder.add(key.as_bytes(), &[0u8; 32], false).expect("add");
        }
        builder.add(b"zz", b"x", false).expect("add last");
        assert_eq!(builder.min_key(), Some(b"".to_vec()));
        assert_eq!(builder.max_key(), Some(b"zz".to_vec()));
    }

    #[test]
    fn tombstone_record_written() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("tombstone.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"a", b"v1", false).expect("add put a");
        builder.add(b"b", b"", true).expect("add delete b");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");

        let (key, value, deleted, consumed) = read_record(&bytes, 0).expect("first record");
        assert_eq!(key, b"a");
        assert_eq!(value, b"v1");
        assert!(!deleted);

        let (key, value, deleted, _) = read_record(&bytes, consumed).expect("second record");
        assert_eq!(key, b"b");
        assert!(value.is_empty());
        assert!(deleted);
    }

    #[test]
    fn build_to_missing_dir_errors() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("no_such_dir").join("t.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"k", b"v", false).expect("add must succeed");
        let err = builder
            .build(&path)
            .expect_err("build to missing dir must error");
        assert!(matches!(err, LsmError::Io(_)));
    }

    fn bloom_bytes(bytes: &[u8], bloom_offset: u64) -> &[u8] {
        &bytes[bloom_offset as usize..bytes.len() - FOOTER_LEN]
    }

    fn decode_file_bloom(bytes: &[u8]) -> fastbloom::BloomFilter {
        let (_index_offset, bloom_offset) = read_footer(bytes);
        assert!(bloom_offset > 0, "expected a bloom block to be present");
        crate::sstable::bloom::decode_bloom(bloom_bytes(bytes, bloom_offset))
            .expect("bloom block must decode")
    }

    /// Builds `count` ascending keys `k00000..` (each a 32-byte value) into a
    /// fresh `.sst`. `expected_items` > 0 routes through
    /// `with_expected_items`, otherwise `new()` (bloom disabled).
    fn build_sequential(
        dir: &tempfile::TempDir,
        name: &str,
        count: usize,
        expected_items: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut builder = if expected_items > 0 {
            SsTableBuilder::with_expected_items(expected_items)
        } else {
            SsTableBuilder::new()
        };
        let value = vec![0u8; 32];
        for i in 0..count {
            let key = format!("k{i:05}");
            builder.add(key.as_bytes(), &value, false).expect("add");
        }
        builder.build(&path).expect("build");
        path
    }

    #[test]
    fn build_with_bloom_writes_bloom_offset() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("bloom.sst");

        let mut builder = SsTableBuilder::with_expected_items(100);
        for i in 0..5u32 {
            let key = format!("k{i}");
            builder.add(key.as_bytes(), b"v", false).expect("add");
        }
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, bloom_offset) = read_footer(&bytes);
        assert!(
            index_offset < bloom_offset,
            "bloom must start after the index"
        );
        assert!(
            bloom_offset < bytes.len() as u64 - FOOTER_LEN as u64,
            "bloom must end before the footer"
        );

        let index = parse_index(&bytes[index_offset as usize..bloom_offset as usize])
            .expect("index region must parse");
        assert_eq!(index.len(), 1, "all five keys share one block");

        let decoded = decode_file_bloom(&bytes);
        for i in 0..5u32 {
            let key = format!("k{i}");
            assert!(
                decoded.contains(key.as_bytes()),
                "key {key} must be in bloom"
            );
        }
    }

    #[test]
    fn tombstone_key_in_bloom() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("tombstone_bloom.sst");

        let mut builder = SsTableBuilder::with_expected_items(10);
        builder.add(b"a", b"v1", false).expect("add a");
        builder.add(b"b", b"", true).expect("add delete b");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let decoded = decode_file_bloom(&bytes);
        assert!(decoded.contains(b"a"), "live key must be in bloom");
        assert!(
            decoded.contains(b"b"),
            "tombstone key must be in bloom so Deleted is discoverable"
        );
    }

    #[test]
    fn no_bloom_default_writes_zero() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("no_bloom.sst");

        let mut builder = SsTableBuilder::new();
        builder.add(b"k1", b"v1", false).expect("add k1");
        builder.add(b"k2", b"v2", false).expect("add k2");
        builder.build(&path).expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        let (index_offset, bloom_offset) = read_footer(&bytes);
        assert_eq!(bloom_offset, 0, "no bloom block without expected_items");

        // F3 byte math is unchanged: data + index + footer, no bloom padding.
        let data_len = 22usize; // two records of 7 header + 2 key + 2 value
        let index_len = 20usize; // one sparse entry
        assert_eq!(index_offset, data_len as u64);
        assert_eq!(bytes.len(), data_len + index_len + FOOTER_LEN);
        assert_eq!(&bytes[bytes.len() - 8..], &MAGIC);
    }

    #[test]
    fn empty_table_no_bloom_even_when_enabled() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = dir.path().join("empty_bloom.sst");

        SsTableBuilder::with_expected_items(10)
            .build(&path)
            .expect("build must succeed");

        let bytes = fs::read(&path).expect("file must be readable");
        assert_eq!(bytes.len(), FOOTER_LEN, "no keys -> footer-only file");
        let (index_offset, bloom_offset) = read_footer(&bytes);
        assert_eq!(index_offset, 0);
        assert_eq!(
            bloom_offset, 0,
            "lazily-created bloom must be skipped when no key was added"
        );
    }

    #[test]
    fn bloom_file_is_larger() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let no_bloom_path = build_sequential(&dir, "no_bloom.sst", 100, 0);
        let bloom_path = build_sequential(&dir, "with_bloom.sst", 100, 100);

        let no_bloom = fs::read(&no_bloom_path).expect("no-bloom file must be readable");
        let with_bloom = fs::read(&bloom_path).expect("bloom file must be readable");
        assert!(
            with_bloom.len() > no_bloom.len(),
            "bloom file must be larger"
        );

        assert_eq!(&no_bloom[no_bloom.len() - 8..], &MAGIC);
        assert_eq!(&with_bloom[with_bloom.len() - 8..], &MAGIC);

        let (nb_index_offset, nb_bloom_offset) = read_footer(&no_bloom);
        assert_eq!(nb_bloom_offset, 0);
        let nb_entries =
            parse_index(&no_bloom[nb_index_offset as usize..no_bloom.len() - FOOTER_LEN])
                .expect("no-bloom index must parse");
        assert!(!nb_entries.is_empty());

        let (wb_index_offset, wb_bloom_offset) = read_footer(&with_bloom);
        let wb_entries =
            parse_index(&with_bloom[wb_index_offset as usize..wb_bloom_offset as usize])
                .expect("bloom file index must parse");
        assert_eq!(
            wb_entries.len(),
            nb_entries.len(),
            "same data must produce the same index"
        );

        let decoded = decode_file_bloom(&with_bloom);
        for i in 0..100u32 {
            let key = format!("k{i:05}");
            assert!(
                decoded.contains(key.as_bytes()),
                "key {key} must be in bloom"
            );
        }
    }

    #[test]
    fn bloom_contains_all_after_reopen_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir must succeed");
        let path = build_sequential(&dir, "roundtrip.sst", 200, 200);

        let bytes = fs::read(&path).expect("file must be readable");
        let decoded = decode_file_bloom(&bytes);
        for i in 0..200u32 {
            let key = format!("k{i:05}");
            assert!(
                decoded.contains(key.as_bytes()),
                "key {key} lost through file roundtrip"
            );
        }
    }
}
