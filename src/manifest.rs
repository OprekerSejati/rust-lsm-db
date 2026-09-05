//! MANIFEST: engine metadata persisted atomically.
//!
//! Records which SST files exist at which level plus each file's key range.
//! The manifest is fully rewritten on every commit (not an append log).
//!
//! On-disk layout (multi-byte ints LittleEndian, via [`crate::utils`]):
//! ```text
//! [MAGIC "LSMMANIFEST01" 13 bytes]
//! records grouped by level ascending, within a level sorted by seq ascending:
//!   [level u32][seq u64][name_len u32][name bytes]
//!   [min_len u32][min bytes][max_len u32][max bytes]
//! ```
//! There is no record count: the decoder walks until exactly `bytes.len()`
//! is consumed. Any trailing bytes short of a full record are corruption.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{LsmError, Result};
use crate::utils;

pub(crate) const MAGIC_MANIFEST: &[u8; 13] = b"LSMMANIFEST01";
pub(crate) const MANIFEST_FILE_NAME: &str = "MANIFEST";
pub(crate) const MANIFEST_TMP_FILE_NAME: &str = "MANIFEST.tmp";

// One record = fixed header (level u32 + seq u64 + name_len u32) then
// name, min_len u32, min, max_len u32, max.
const MAGIC_LEN: usize = MAGIC_MANIFEST.len();
const FIXED_HEADER_LEN: usize = 4 + 8 + 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SstMeta {
    pub level: u32,
    pub seq: u64,
    pub name: String,
    pub min: Vec<u8>,
    pub max: Vec<u8>,
}

/// `levels[l]` = SSTs at level `l`, sorted ascending by seq.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Manifest {
    pub levels: Vec<Vec<SstMeta>>,
}

impl Manifest {
    // The engine keeps its own `state.levels` and rewrites the MANIFEST
    // wholesale on every flush/compact, so these accessors are exercised only
    // by tests today. Kept as part of the manifest's public surface for future
    // in-place metadata edits.
    #[allow(dead_code)]
    pub fn ssts_at(&self, level: u32) -> &[SstMeta] {
        self.levels
            .get(level as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Appends `meta` to its level, keeping per-level seq-ascending order.
    pub fn add(&mut self, meta: SstMeta) {
        let level = meta.level as usize;
        if self.levels.len() <= level {
            self.levels.resize_with(level + 1, Vec::new);
        }
        let pos = self.levels[level].partition_point(|m| m.seq < meta.seq);
        self.levels[level].insert(pos, meta);
    }

    #[allow(dead_code)]
    pub fn remove(&mut self, level: u32, seq: u64) -> bool {
        let Some(list) = self.levels.get_mut(level as usize) else {
            return false;
        };
        match list.iter().position(|m| m.seq == seq) {
            Some(pos) => {
                list.remove(pos);
                true
            }
            None => false,
        }
    }
}

/// Pure: MAGIC followed by every SST's record, level by level ascending,
/// each level's SSTs in seq order (the invariant `add` maintains).
pub(crate) fn encode(manifest: &Manifest) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC_MANIFEST);
    for (level, metas) in manifest.levels.iter().enumerate() {
        for meta in metas {
            utils::put_u32(&mut buf, level as u32);
            utils::put_u64(&mut buf, meta.seq);
            let name_len = u32::try_from(meta.name.len())
                .expect("sst file names are short engine-generated strings");
            utils::put_u32(&mut buf, name_len);
            buf.extend_from_slice(meta.name.as_bytes());
            let min_len =
                u32::try_from(meta.min.len()).expect("keys are bounded by the sstable MAX_KEY_LEN");
            utils::put_u32(&mut buf, min_len);
            buf.extend_from_slice(&meta.min);
            let max_len =
                u32::try_from(meta.max.len()).expect("keys are bounded by the sstable MAX_KEY_LEN");
            utils::put_u32(&mut buf, max_len);
            buf.extend_from_slice(&meta.max);
        }
    }
    buf
}

/// Pure: parses a manifest previously produced by [`encode`]. Returns
/// `Corruption` on a bad magic, a truncated record, or trailing garbage.
pub(crate) fn decode(bytes: &[u8]) -> Result<Manifest> {
    if bytes.len() < MAGIC_LEN || bytes[..MAGIC_LEN] != *MAGIC_MANIFEST {
        return Err(LsmError::Corruption("invalid manifest magic".to_string()));
    }

    let mut manifest = Manifest::default();
    let mut offset = MAGIC_LEN;
    while offset < bytes.len() {
        if bytes.len() - offset < FIXED_HEADER_LEN {
            return Err(LsmError::Corruption(format!(
                "manifest record shorter than {FIXED_HEADER_LEN}-byte header: {} bytes remain",
                bytes.len() - offset
            )));
        }

        let level = utils::read_u32(bytes, offset)?;
        offset += 4;
        let seq = utils::read_u64(bytes, offset)?;
        offset += 8;
        let name_len = utils::read_u32(bytes, offset)? as usize;
        offset += 4;

        let name_bytes = read_payload(bytes, &mut offset, name_len, "sst name")?;
        let name = String::from_utf8(name_bytes.to_vec()).map_err(|_| {
            LsmError::Corruption("manifest sst name is not valid utf-8".to_string())
        })?;

        let min_len = utils::read_u32(bytes, offset)? as usize;
        offset += 4;
        let min = read_payload(bytes, &mut offset, min_len, "min key")?.to_vec();

        let max_len = utils::read_u32(bytes, offset)? as usize;
        offset += 4;
        let max = read_payload(bytes, &mut offset, max_len, "max key")?.to_vec();

        let level_usize = level as usize;
        if manifest.levels.len() <= level_usize {
            let need = level_usize.checked_add(1).ok_or_else(|| {
                LsmError::Corruption("manifest level number overflows usize".to_string())
            })?;
            let grow = need - manifest.levels.len();
            manifest.levels.try_reserve(grow).map_err(|_| {
                LsmError::Corruption(format!("manifest level number {level} too large"))
            })?;
            manifest.levels.resize(need, Vec::new());
        }
        // Enforce the per-level ascending-by-seq invariant the engine relies on
        // (newest-first L0 lookup). The encoder never emits out-of-order seqs,
        // so a violation means the file is corrupt.
        let level_seqs = &mut manifest.levels[level_usize];
        if let Some(prev) = level_seqs.last()
            && prev.seq >= seq
        {
            return Err(LsmError::Corruption(format!(
                "manifest level {level} seq out of order: {seq} after {}",
                prev.seq
            )));
        }
        level_seqs.push(SstMeta {
            level,
            seq,
            name,
            min,
            max,
        });
    }
    Ok(manifest)
}

fn read_payload<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    len: usize,
    what: &str,
) -> Result<&'a [u8]> {
    let end = offset.checked_add(len).ok_or_else(|| {
        LsmError::Corruption(format!(
            "manifest {what} length overflows at offset {offset}"
        ))
    })?;
    if end > bytes.len() {
        return Err(LsmError::Corruption(format!(
            "manifest {what} runs past end of file: declared {len} bytes, {} available",
            bytes.len() - *offset
        )));
    }
    let payload = &bytes[*offset..end];
    *offset = end;
    Ok(payload)
}

/// Atomic write: write `MANIFEST.tmp` → flush + `sync_all` → rename over
/// `MANIFEST`. On success no `MANIFEST.tmp` is left behind; on error the temp
/// file is removed best-effort (the previous `MANIFEST` is untouched because
/// the rename is the last step). Note: the parent directory is not fsynced, so
/// the rename may not survive an OS crash — accepted for v0.1.
pub(crate) fn write_atomic(dir: &Path, manifest: &Manifest) -> Result<()> {
    let tmp_path = dir.join(MANIFEST_TMP_FILE_NAME);
    let final_path = dir.join(MANIFEST_FILE_NAME);

    let result = (|| {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&encode(manifest))?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        std::fs::rename(&tmp_path, &final_path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

/// Loads the MANIFEST from `dir`. A missing file means a fresh engine
/// directory → `Ok(Manifest::default())`. Existing content is decoded;
/// malformed content surfaces as `Corruption`.
pub(crate) fn load(dir: &Path) -> Result<Manifest> {
    let path = dir.join(MANIFEST_FILE_NAME);
    match std::fs::read(&path) {
        Ok(bytes) => decode(&bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Manifest::default()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn meta(level: u32, seq: u64, name: &str, min: &[u8], max: &[u8]) -> SstMeta {
        SstMeta {
            level,
            seq,
            name: name.to_string(),
            min: min.to_vec(),
            max: max.to_vec(),
        }
    }

    fn sample_manifest() -> Manifest {
        let mut manifest = Manifest::default();
        manifest.add(meta(0, 1, "L0_000001.sst", b"a", b"m"));
        manifest.add(meta(0, 2, "L0_000002.sst", b"n", b"z"));
        manifest.add(meta(1, 5, "L1_000001.sst", b"a", b"z"));
        manifest
    }

    #[test]
    fn encode_decode_roundtrip() {
        let manifest = sample_manifest();
        let bytes = encode(&manifest);
        let decoded = decode(&bytes).expect("decode of self-encoded manifest must succeed");
        assert_eq!(decoded, manifest);
    }

    #[test]
    fn empty_manifest_is_magic_only() {
        let bytes = encode(&Manifest::default());
        assert_eq!(bytes, MAGIC_MANIFEST.to_vec());
        let decoded = decode(&bytes).expect("magic-only bytes must decode");
        assert_eq!(decoded, Manifest::default());
    }

    #[test]
    fn decode_bad_magic_errors() {
        let err = decode(b"notamanifest").expect_err("wrong magic must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn decode_truncated_record_errors() {
        let bytes = encode(&sample_manifest());
        let truncated = &bytes[..bytes.len() - 3];
        let err = decode(truncated).expect_err("truncated manifest must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn decode_garbage_tail_errors() {
        let mut bytes = encode(&sample_manifest());
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let err = decode(&bytes).expect_err("garbage tail must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn decode_rejects_out_of_order_seq() {
        // Hand-craft a manifest whose second record repeats a seq (well-formed
        // framing, invalid ordering) — decode must reject it as Corruption.
        let mut manifest = Manifest::default();
        manifest.add(meta(0, 1, "a.sst", b"k", b"k"));
        manifest.add(meta(0, 1, "b.sst", b"k", b"k"));
        let bytes = encode(&manifest);
        let err = decode(&bytes).expect_err("duplicate seq must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn add_keeps_seq_sorted() {
        let mut manifest = Manifest::default();
        manifest.add(meta(0, 3, "c.sst", b"k", b"k"));
        manifest.add(meta(0, 1, "a.sst", b"k", b"k"));
        manifest.add(meta(0, 2, "b.sst", b"k", b"k"));
        let seqs: Vec<u64> = manifest.ssts_at(0).iter().map(|m| m.seq).collect();
        assert_eq!(seqs, [1, 2, 3]);
    }

    #[test]
    fn remove_works() {
        let mut manifest = Manifest::default();
        manifest.add(meta(0, 1, "a.sst", b"k", b"k"));
        manifest.add(meta(0, 2, "b.sst", b"k", b"k"));

        assert!(
            manifest.remove(0, 1),
            "removing existing seq must return true"
        );
        let seqs: Vec<u64> = manifest.ssts_at(0).iter().map(|m| m.seq).collect();
        assert_eq!(seqs, [2]);

        assert!(
            !manifest.remove(0, 99),
            "removing unknown seq must return false"
        );
        assert!(
            !manifest.remove(7, 1),
            "removing from nonexistent level must return false"
        );
    }

    #[test]
    fn ssts_at_absent_level() {
        let manifest = Manifest::default();
        assert!(manifest.ssts_at(5).is_empty());
    }

    #[test]
    fn write_load_roundtrip() {
        let dir = tempdir().expect("tempdir should succeed");
        let manifest = sample_manifest();
        write_atomic(dir.path(), &manifest).expect("atomic write should succeed");

        assert!(
            dir.path().join(MANIFEST_FILE_NAME).exists(),
            "MANIFEST file must exist after write"
        );
        assert!(
            !dir.path().join(MANIFEST_TMP_FILE_NAME).exists(),
            "MANIFEST.tmp must not be left behind after successful write"
        );

        let loaded = load(dir.path()).expect("load should succeed");
        assert_eq!(loaded, manifest);
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let dir = tempdir().expect("tempdir should succeed");
        let loaded = load(dir.path()).expect("missing MANIFEST must load as empty");
        assert_eq!(loaded, Manifest::default());
    }

    #[test]
    fn load_corrupt_file_errors() {
        let dir = tempdir().expect("tempdir should succeed");
        let path = dir.path().join(MANIFEST_FILE_NAME);
        std::fs::write(&path, b"this is not a manifest").expect("write garbage file");
        let err = load(dir.path()).expect_err("corrupt MANIFEST must be rejected");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn add_extends_levels() {
        let mut manifest = Manifest::default();
        manifest.add(meta(3, 9, "deep.sst", b"a", b"z"));
        assert_eq!(manifest.ssts_at(3).len(), 1);
        assert!(manifest.ssts_at(0).is_empty());
        assert!(manifest.ssts_at(1).is_empty());
        assert!(manifest.ssts_at(2).is_empty());
    }
}
