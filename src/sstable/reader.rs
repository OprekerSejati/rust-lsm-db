//! SsTableReader: opens a table written by `SsTableBuilder`, loads the sparse
//! index (and, when present, the bloom filter block) into memory, and answers
//! point lookups by probing the bloom first — a negative probe returns `None`
//! without touching the index or disk — then binary-searching the index,
//! reading one data block, and scanning it linearly.
//!
//! All reads are positional (`read_at` on Unix; a fresh handle per read
//! elsewhere), so [`SsTableReader::get`] takes `&self` and a reader can be
//! shared across threads behind an `Arc` without any internal lock — matching
//! the engine's lock-free read design.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use super::bloom::decode_bloom;
use super::index::{IndexEntry, locate_block, parse_index};
use super::record::read_record;
use super::{FOOTER_LEN, LookupValue, MAGIC};
use crate::error::{LsmError, Result};
use crate::utils;

/// One decoded record from a table scan: `(key, value, deleted)`.
pub(crate) type Record = (Vec<u8>, Vec<u8>, bool);

/// Read-only handle over one `.sst` table. The sparse index and — on a
/// bloom-bearing table — the decoded bloom filter are loaded into memory at
/// open; each lookup probes the bloom (skipping the disk on a negative) and
/// otherwise reads only the single data block that could hold the key. `get`
/// takes `&self` (positional reads), so a reader is `Send + Sync` and can be
/// shared behind an `Arc` without a lock.
pub struct SsTableReader {
    #[cfg(unix)]
    file: File,
    file_len: u64,
    index_offset: u64, // data occupies [0..index_offset)
    index: Vec<IndexEntry>,
    bloom: Option<fastbloom::BloomFilter>,
    path: PathBuf,
}

impl SsTableReader {
    /// Opens and validates a table: reads the footer from the last
    /// `FOOTER_LEN` bytes, checks the magic, and parses the sparse index from
    /// `[index_offset .. end_of_index]`. `end_of_index` is `file_len -
    /// FOOTER_LEN` when `bloom_offset == 0` (no bloom block yet), else
    /// `bloom_offset`. When a bloom block is present (`bloom_offset != 0`) it
    /// is decoded from `[bloom_offset .. file_len - FOOTER_LEN]` into memory;
    /// a `bloom_offset` that runs into the footer is rejected. Corruption
    /// errors surface for any structural problem.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();

        if file_len < FOOTER_LEN as u64 {
            return Err(LsmError::Corruption(format!(
                "sstable {path:?} is {} bytes, shorter than the {FOOTER_LEN}-byte footer",
                file_len
            )));
        }

        let mut footer = [0u8; FOOTER_LEN];
        file.seek(SeekFrom::End(-(FOOTER_LEN as i64)))?;
        file.read_exact(&mut footer)?;

        if footer[FOOTER_LEN - 8..] != MAGIC {
            return Err(LsmError::Corruption(format!(
                "sstable {path:?} has an invalid magic (not an sstable or truncated footer)"
            )));
        }

        let index_offset = utils::read_u64(&footer, 0)?;
        let bloom_offset = utils::read_u64(&footer, 8)?;

        // The bloom block (when present) occupies [bloom_offset ..
        // file_len - FOOTER_LEN]. Validate this before deriving end_of_index so
        // the index region can never extend into the footer, and so this guard
        // is reachable even when the footer bytes would otherwise parse as
        // valid index entries.
        let bloom_end = file_len - FOOTER_LEN as u64;
        if bloom_offset != 0 && bloom_offset > bloom_end {
            return Err(LsmError::Corruption(format!(
                "sstable {path:?} bloom_offset {bloom_offset} exceeds region end {bloom_end} (overlaps the footer)"
            )));
        }

        // Data occupies [0..index_offset); the index runs from index_offset to
        // end_of_index. Without a bloom block the index extends to the footer.
        let end_of_index = if bloom_offset == 0 {
            file_len - FOOTER_LEN as u64
        } else {
            bloom_offset
        };
        if index_offset > end_of_index || end_of_index > file_len {
            return Err(LsmError::Corruption(format!(
                "sstable {path:?} footer offsets out of range: index_offset={index_offset}, bloom_offset={bloom_offset}, file_len={file_len}"
            )));
        }

        let index_len = (end_of_index - index_offset) as usize;
        let mut index_bytes = vec![0u8; index_len];
        if index_len > 0 {
            file.seek(SeekFrom::Start(index_offset))?;
            file.read_exact(&mut index_bytes)?;
        }
        let index = parse_index(&index_bytes)?;

        // Load the bloom block. A zero-length region ([bloom_offset ..
        // bloom_end] empty) fails decode_bloom's length check and surfaces as
        // Corruption.
        let bloom = if bloom_offset == 0 {
            None
        } else {
            let bloom_len = (bloom_end - bloom_offset) as usize;
            let mut bloom_bytes = vec![0u8; bloom_len];
            if bloom_len > 0 {
                file.seek(SeekFrom::Start(bloom_offset))?;
                file.read_exact(&mut bloom_bytes)?;
            }
            Some(decode_bloom(&bloom_bytes)?)
        };

        Ok(SsTableReader {
            #[cfg(unix)]
            file,
            file_len,
            index_offset,
            index,
            bloom,
            path: path.to_path_buf(),
        })
    }

    pub fn num_blocks(&self) -> usize {
        self.index.len()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Probes the loaded bloom filter: `Some(true/false)` on a table that has
    /// one, `None` on an F3 table without a bloom block. Does no disk I/O.
    /// `get` short-circuits on a `false` probe through this same helper; unit
    /// tests use it to prove that short-circuit, and engine-level code may use
    /// it later to rule out whole tables cheaply.
    pub(crate) fn bloom_probe(&self, key: &[u8]) -> Option<bool> {
        self.bloom.as_ref().map(|b| b.contains(key))
    }

    /// Positional read into `buf` starting at absolute `offset`. Uses
    /// `read_exact_at` where available (shared handle, no seek state); falls
    /// back to a freshly-opened handle elsewhere so no internal lock is needed.
    fn read_block_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            self.file.read_exact_at(buf, offset)
        }
        #[cfg(not(unix))]
        {
            let mut f = File::open(&self.path)?;
            f.seek(SeekFrom::Start(offset))?;
            f.read_exact(buf)
        }
    }

    /// Looks up `key`. Returns `Ok(None)` when absent, `Ok(Some(Present(v)))`
    /// for a live value (including an empty one), and `Ok(Some(Deleted))` for
    /// a tombstone. A corrupt block pointer or undecodable record yields
    /// `LsmError::Corruption`.
    ///
    /// When the table carries a bloom filter, a negative probe returns
    /// `Ok(None)` immediately — before the index is consulted or any block is
    /// read. Tombstone keys are inserted into the bloom by the builder, so a
    /// `Deleted` key always probes positive and falls through to the block
    /// scan, keeping the tri-state result exact.
    pub fn get(&self, key: &[u8]) -> Result<Option<LookupValue>> {
        // A negative bloom probe proves the key was never written (the builder
        // inserts every key, tombstones included), so return None before the
        // index is consulted or any block is read from disk.
        if self.bloom_probe(key) == Some(false) {
            return Ok(None);
        }
        let Some(block_idx) = locate_block(&self.index, key) else {
            return Ok(None);
        };
        let entry = &self.index[block_idx];
        // checked_add guards a hostile file whose index declares offset+size
        // beyond u64::MAX; treat overflow as an out-of-range block pointer.
        let block_end = entry.offset.checked_add(entry.size).ok_or_else(|| {
            LsmError::Corruption(format!(
                "sstable {:?} block {block_idx} offset+size overflows u64",
                self.path
            ))
        })?;
        if block_end > self.file_len {
            return Err(LsmError::Corruption(format!(
                "sstable {:?} block {block_idx} at offset {} size {} exceeds file length {}",
                self.path, entry.offset, entry.size, self.file_len
            )));
        }

        let mut block = vec![0u8; entry.size as usize];
        self.read_block_at(entry.offset, &mut block)?;

        let mut offset = 0usize;
        while offset < block.len() {
            let (record_key, value, deleted, consumed) = read_record(&block, offset)?;
            match record_key.as_slice().cmp(key) {
                std::cmp::Ordering::Equal => {
                    return Ok(Some(if deleted {
                        LookupValue::Deleted
                    } else {
                        LookupValue::Present(value)
                    }));
                }
                std::cmp::Ordering::Greater => return Ok(None), // sorted: key cannot appear later
                std::cmp::Ordering::Less => {}
            }
            offset += consumed;
        }
        Ok(None)
    }

    /// Reads every record of the table in ascending key order as
    /// `(key, value, deleted)`. The whole data region `[0 .. index_offset)` is
    /// read into memory once and walked record by record. Used by compaction
    /// (F6) to feed a k-way merge; also a building block for a future public
    /// scan API. Returns `Corruption` if any record fails to decode.
    pub fn records(&self) -> Result<Vec<Record>> {
        let mut data = vec![0u8; self.index_offset as usize];
        if !data.is_empty() {
            self.read_block_at(0, &mut data)?;
        }
        let mut out = Vec::new();
        let mut offset = 0usize;
        while offset < data.len() {
            let (key, value, deleted, consumed) = read_record(&data, offset)?;
            out.push((key, value, deleted));
            offset += consumed;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sstable::builder::SsTableBuilder;
    use std::path::Path;
    use tempfile::{TempDir, tempdir};

    fn build_sst(dir: &TempDir, name: &str, entries: &[(Vec<u8>, Option<Vec<u8>>)]) -> PathBuf {
        let path = dir.path().join(name);
        let mut builder = SsTableBuilder::new();
        for (key, maybe_value) in entries {
            match maybe_value {
                Some(value) => builder.add(key, value, false).expect("add put"),
                None => builder.add(key, &[], true).expect("add delete"),
            }
        }
        builder.build(&path).expect("build sstable");
        path
    }

    fn build_sst_with_bloom(
        dir: &TempDir,
        name: &str,
        expected_items: usize,
        entries: &[(Vec<u8>, Option<Vec<u8>>)],
    ) -> PathBuf {
        let path = dir.path().join(name);
        let mut builder = SsTableBuilder::with_expected_items(expected_items);
        for (key, maybe_value) in entries {
            match maybe_value {
                Some(value) => builder.add(key, value, false).expect("add put"),
                None => builder.add(key, &[], true).expect("add delete"),
            }
        }
        builder.build(&path).expect("build sstable");
        path
    }

    fn read_bytes(path: &Path) -> Vec<u8> {
        std::fs::read(path).expect("read file")
    }

    #[test]
    fn get_present_value() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "present.sst",
            &[
                (b"k1".to_vec(), Some(b"v1".to_vec())),
                (b"k2".to_vec(), Some(b"v2".to_vec())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open");
        assert_eq!(
            reader.get(b"k1").expect("get k1"),
            Some(LookupValue::Present(b"v1".to_vec()))
        );
        assert_eq!(
            reader.get(b"k2").expect("get k2"),
            Some(LookupValue::Present(b"v2".to_vec()))
        );
    }

    #[test]
    fn get_absent_returns_none() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "absent.sst",
            &[
                (b"aaa".to_vec(), Some(b"1".to_vec())),
                (b"ccc".to_vec(), Some(b"3".to_vec())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open");
        assert_eq!(reader.get(b"nope").expect("get"), None);
        assert_eq!(reader.get(b"bbb").expect("get between"), None);
    }

    #[test]
    fn get_tombstone_and_empty_value() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "tomb.sst",
            &[
                (b"del".to_vec(), None),
                (b"empty".to_vec(), Some(b"".to_vec())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open");
        assert_eq!(
            reader.get(b"del").expect("get deleted"),
            Some(LookupValue::Deleted)
        );
        assert_eq!(
            reader.get(b"empty").expect("get empty"),
            Some(LookupValue::Present(Vec::new()))
        );
    }

    #[test]
    fn get_before_first_and_after_last() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "range.sst",
            &[
                (b"m".to_vec(), Some(b"1".to_vec())),
                (b"z".to_vec(), Some(b"2".to_vec())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open");
        assert_eq!(reader.get(b"a").expect("before first"), None);
        assert_eq!(reader.get(b"zzz").expect("after last"), None);
    }

    #[test]
    fn multi_block_lookup() {
        let dir = tempdir().expect("tempdir");
        let mut entries = Vec::new();
        for i in 0..500u32 {
            let key = format!("k{i:05}").into_bytes();
            if i % 50 == 0 {
                entries.push((key, None));
            } else {
                entries.push((key, Some(vec![0u8; 32])));
            }
        }
        let path = build_sst(&dir, "multi.sst", &entries);
        let reader = SsTableReader::open(&path).expect("open");
        assert!(reader.num_blocks() >= 2, "expected multi-block table");

        for (i, (key, maybe_value)) in entries.iter().enumerate() {
            let result = reader.get(key).expect("get inserted key");
            match maybe_value {
                Some(_) => assert!(
                    matches!(result, Some(LookupValue::Present(_))),
                    "key {} idx {i} should be present",
                    String::from_utf8_lossy(key)
                ),
                None => assert_eq!(
                    result,
                    Some(LookupValue::Deleted),
                    "key {} idx {i} should be deleted",
                    String::from_utf8_lossy(key)
                ),
            }
        }
        // Misses between inserted keys.
        for i in 0..500u32 {
            let miss = format!("k{i:05}x").into_bytes();
            assert_eq!(reader.get(&miss).expect("get miss"), None);
        }
    }

    #[test]
    fn empty_table() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty.sst");
        SsTableBuilder::new().build(&path).expect("build empty");
        let reader = SsTableReader::open(&path).expect("open empty");
        assert_eq!(reader.num_blocks(), 0);
        assert_eq!(reader.get(b"anything").expect("get"), None);
    }

    #[test]
    fn corrupt_magic_errors() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(&dir, "magic.sst", &[(b"k".to_vec(), Some(b"v".to_vec()))]);
        let mut bytes = read_bytes(&path);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("rewrite");
        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("bad magic must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn truncated_file_errors() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(&dir, "trunc.sst", &[(b"k".to_vec(), Some(b"v".to_vec()))]);
        let bytes = read_bytes(&path);
        std::fs::write(&path, &bytes[..10]).expect("rewrite truncated");
        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("short file must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn data_block_pointer_out_of_file_errors() {
        // Inflate a block's recorded size in the index so the block pointer
        // runs past EOF; open() still succeeds (index parses fine) but get()
        // must reject the out-of-range block with Corruption.
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "block-ptr.sst",
            &[(b"k".to_vec(), Some(vec![0u8; 64]))],
        );
        let mut bytes = read_bytes(&path);
        // Index region layout per entry: [key_len u16][key][offset u64][size u64].
        // The final u64 (size) ends exactly at bytes.len() - FOOTER_LEN.
        let size_field = bytes.len() - FOOTER_LEN - 8;
        for b in &mut bytes[size_field..size_field + 8] {
            *b = 0xFF;
        }
        std::fs::write(&path, &bytes).expect("rewrite");

        let reader = SsTableReader::open(&path).expect("open must still succeed");
        let err = match reader.get(b"k") {
            Ok(_) => panic!("out-of-range block must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn block_offset_plus_size_overflow_errors() {
        // Second block sits at a nonzero offset; inflating its size to u64::MAX
        // makes offset+size overflow, which must surface as Corruption (not a
        // panic or a huge allocation).
        let dir = tempdir().expect("tempdir");
        // Block 0: one oversized record (value 10_000 > 4096 target). Block 1: "z".
        let mut builder = SsTableBuilder::new();
        builder
            .add(b"a", &[0u8; 10_000], false)
            .expect("add block0");
        builder.add(b"z", b"last", false).expect("add block1");
        let path = dir.path().join("overflow.sst");
        builder.build(&path).expect("build");

        let mut bytes = read_bytes(&path);
        let index_offset = utils::read_u64(&bytes[bytes.len() - FOOTER_LEN..], 0).expect("io");
        // Walk index: skip entry 0 to reach entry 1's size field (its last 8 bytes).
        let mut pos = index_offset as usize;
        let key0_len = utils::read_u16(&bytes, pos).expect("key0 len") as usize;
        pos += 2 + key0_len + 8 + 8; // skip entry 0 entirely
        let key1_len = utils::read_u16(&bytes, pos).expect("key1 len") as usize;
        let size_field = pos + 2 + key1_len + 8; // skip key1 + offset, land on size
        for b in &mut bytes[size_field..size_field + 8] {
            *b = 0xFF;
        }
        std::fs::write(&path, &bytes).expect("rewrite");

        let reader = SsTableReader::open(&path).expect("open must still succeed");
        let err = match reader.get(b"z") {
            Ok(_) => panic!("overflowing block pointer must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn index_malformed_errors() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(&dir, "idx.sst", &[(b"k".to_vec(), Some(b"v".to_vec()))]);
        let mut bytes = read_bytes(&path);
        let index_offset = utils::read_u64(&bytes[bytes.len() - FOOTER_LEN..], 0).expect("io");
        // Overwrite index region with 0xFF garbage.
        let idx_start = index_offset as usize;
        let idx_end = bytes.len() - FOOTER_LEN;
        for b in &mut bytes[idx_start..idx_end] {
            *b = 0xFF;
        }
        std::fs::write(&path, &bytes).expect("rewrite");
        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("garbage index must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn tri_state_distinct() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "tri.sst",
            &[
                (b"live".to_vec(), Some(b"v".to_vec())),
                (b"tomb".to_vec(), None),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open");
        assert!(matches!(
            reader.get(b"live").expect("get live"),
            Some(LookupValue::Present(_))
        ));
        assert_eq!(
            reader.get(b"tomb").expect("get tomb"),
            Some(LookupValue::Deleted)
        );
        assert_eq!(reader.get(b"gone").expect("get gone"), None);
    }

    #[test]
    fn roundtrip_with_bloom_finds_all() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst_with_bloom(
            &dir,
            "bloom-roundtrip.sst",
            10,
            &[
                (b"a1".to_vec(), Some(b"v1".to_vec())),
                (b"del".to_vec(), None),
                (b"empty".to_vec(), Some(Vec::new())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open with bloom");
        assert_eq!(
            reader.get(b"a1").expect("get live"),
            Some(LookupValue::Present(b"v1".to_vec()))
        );
        assert_eq!(
            reader.get(b"empty").expect("get empty-value put"),
            Some(LookupValue::Present(Vec::new()))
        );
        assert_eq!(
            reader.get(b"del").expect("get tombstone"),
            Some(LookupValue::Deleted)
        );
        assert_eq!(reader.get(b"zzz").expect("get absent"), None);
    }

    #[test]
    fn bloom_short_circuits_absent_key() {
        // Corrupt the first data block's index entry (inflate its recorded size
        // past EOF) so any get() that reaches the block read returns
        // Corruption. A key the bloom rules out must still read Ok(None) —
        // the only way it can, with a poisoned block, is by short-circuiting
        // on the bloom before touching the index or disk.
        let dir = tempdir().expect("tempdir");
        let entries: Vec<(Vec<u8>, Option<Vec<u8>>)> = (0..100u32)
            .map(|i| (format!("k{i:04}").into_bytes(), Some(vec![0u8; 16])))
            .collect();
        let path = build_sst_with_bloom(&dir, "short.sst", 100, &entries);

        let mut bytes = read_bytes(&path);
        let index_offset = utils::read_u64(&bytes[bytes.len() - FOOTER_LEN..], 0).expect("io");
        let key0_len = utils::read_u16(&bytes, index_offset as usize).expect("io") as usize;
        // First index entry layout: [key_len u16][first_key][offset u64][size u64].
        let size_field = index_offset as usize + 2 + key0_len + 8;
        for b in &mut bytes[size_field..size_field + 8] {
            *b = 0xFF;
        }
        std::fs::write(&path, &bytes).expect("rewrite");

        let reader = SsTableReader::open(&path).expect("open must still succeed");
        // A present key must pass the bloom and die on the corrupt block read,
        // proving the block is genuinely unreadable.
        assert_eq!(
            reader.bloom_probe(b"k0000"),
            Some(true),
            "present key must not be a false negative"
        );
        assert!(matches!(reader.get(b"k0000"), Err(LsmError::Corruption(_))));

        // An absent key the bloom rules out must short-circuit before reaching
        // the corrupt block, so it reads Ok(None) rather than Corruption.
        let absent = (100..10_000u32)
            .map(|i| format!("k{i:04}").into_bytes())
            .find(|k| reader.bloom_probe(k) == Some(false))
            .expect("some absent key must be bloom-negative");
        assert_eq!(
            reader.get(&absent).expect("absent must short-circuit"),
            None
        );
    }

    #[test]
    fn f3_file_without_bloom_still_reads() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst(
            &dir,
            "no-bloom.sst",
            &[
                (b"k1".to_vec(), Some(b"v1".to_vec())),
                (b"k2".to_vec(), Some(b"v2".to_vec())),
            ],
        );
        let reader = SsTableReader::open(&path).expect("open F3 file");
        assert_eq!(reader.bloom_probe(b"k1"), None, "no bloom loaded");
        assert_eq!(
            reader.get(b"k1").expect("get present"),
            Some(LookupValue::Present(b"v1".to_vec()))
        );
        assert_eq!(reader.get(b"nope").expect("get absent"), None);
    }

    #[test]
    fn corrupt_bloom_block_errors() {
        let dir = tempdir().expect("tempdir");
        let path = build_sst_with_bloom(
            &dir,
            "corrupt-bloom.sst",
            10,
            &[(b"k1".to_vec(), Some(b"v1".to_vec()))],
        );
        let mut bytes = read_bytes(&path);
        let file_len = bytes.len();
        // Locate the real bloom block start from the footer (offset 8 of footer).
        let bloom_offset = utils::read_u64(&bytes[file_len - FOOTER_LEN..], 8).expect("io");
        assert!(bloom_offset > 0, "test file must carry a bloom block");
        // Overwrite the bloom block's num_hashes field (first 4 bytes) with an
        // implausible value so decode_bloom rejects it — exercising the reader's
        // bloom decode Corruption path with structurally-intact index/footer.
        bytes[bloom_offset as usize..bloom_offset as usize + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        std::fs::write(&path, &bytes).expect("rewrite");

        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("corrupt bloom block must error at open"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn empty_table_with_bloom_requested_reads_as_no_bloom() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty-bloom.sst");
        SsTableBuilder::with_expected_items(10)
            .build(&path)
            .expect("build empty with bloom requested");
        let reader = SsTableReader::open(&path).expect("open empty");
        assert_eq!(reader.num_blocks(), 0);
        assert_eq!(reader.bloom_probe(b"anything"), None);
        assert_eq!(reader.get(b"anything").expect("get"), None);
    }

    #[test]
    fn bloom_offset_overlapping_footer_errors() {
        // Set footer bloom_offset past the end of the bloom region (into the
        // footer itself); the open-time guard must reject before index parse.
        let dir = tempdir().expect("tempdir");
        let path = build_sst_with_bloom(
            &dir,
            "bloom-overlap.sst",
            10,
            &[(b"k1".to_vec(), Some(b"v1".to_vec()))],
        );
        let mut bytes = read_bytes(&path);
        let file_len = bytes.len();
        let bloom_field = file_len - FOOTER_LEN + 8; // bloom_offset in footer
        bytes[bloom_field..bloom_field + 8].copy_from_slice(&(file_len as u64 - 1).to_le_bytes());
        std::fs::write(&path, &bytes).expect("rewrite");

        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("bloom_offset overlapping the footer must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn zero_length_bloom_region_errors() {
        // Point bloom_offset exactly at the bloom region end (== footer start):
        // an empty region must fail decode_bloom, not read as no-bloom.
        let dir = tempdir().expect("tempdir");
        let path = build_sst_with_bloom(
            &dir,
            "bloom-empty-region.sst",
            10,
            &[(b"k1".to_vec(), Some(b"v1".to_vec()))],
        );
        let mut bytes = read_bytes(&path);
        let file_len = bytes.len();
        let footer_start = (file_len - FOOTER_LEN) as u64;
        let bloom_field = file_len - FOOTER_LEN + 8; // bloom_offset in footer
        bytes[bloom_field..bloom_field + 8].copy_from_slice(&footer_start.to_le_bytes());
        std::fs::write(&path, &bytes).expect("rewrite");

        let err = match SsTableReader::open(&path) {
            Ok(_) => panic!("zero-length bloom region must error"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn records_returns_all_in_ascending_order() {
        let dir = tempdir().expect("tempdir");
        let entries = vec![
            (b"a".to_vec(), Some(b"1".to_vec())),
            (b"b".to_vec(), None),               // tombstone
            (b"c".to_vec(), Some(b"".to_vec())), // empty-value put
            (b"d".to_vec(), Some(b"4".to_vec())),
        ];
        let path = build_sst(&dir, "rec.sst", &entries);
        let reader = SsTableReader::open(&path).expect("open");
        let records = reader.records().expect("records");
        assert_eq!(
            records,
            vec![
                (b"a".to_vec(), b"1".to_vec(), false),
                (b"b".to_vec(), Vec::new(), true),
                (b"c".to_vec(), Vec::new(), false),
                (b"d".to_vec(), b"4".to_vec(), false),
            ]
        );
    }

    #[test]
    fn records_multi_block_sorted() {
        let dir = tempdir().expect("tempdir");
        let mut entries = Vec::new();
        for i in 0..500u32 {
            entries.push((format!("k{i:05}").into_bytes(), Some(vec![0u8; 32])));
        }
        // Sprinkle tombstones.
        for i in (0..500usize).step_by(100) {
            entries[i].1 = None;
        }
        let path = build_sst(&dir, "rec-multi.sst", &entries);
        let reader = SsTableReader::open(&path).expect("open");
        let records = reader.records().expect("records");
        assert_eq!(records.len(), 500);
        let mut prev: Option<&[u8]> = None;
        for (i, (key, _val, deleted)) in records.iter().enumerate() {
            if let Some(p) = prev {
                assert!(p < key.as_slice(), "records must be ascending");
            }
            prev = Some(key.as_slice());
            let want_deleted = i % 100 == 0;
            assert_eq!(*deleted, want_deleted, "index {i}");
        }
    }

    #[test]
    fn records_empty_table_is_empty() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("empty-rec.sst");
        SsTableBuilder::new().build(&path).expect("build empty");
        let reader = SsTableReader::open(&path).expect("open");
        assert_eq!(reader.records().expect("records"), Vec::new());
    }
}
