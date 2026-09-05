//! `LsmEngine`: the public orchestrator. A synchronous put/get/delete API over
//! a concurrent in-memory MemTable, an append-only WAL for durability, and
//! sorted SSTables on disk. A MemTable that grows past the flush threshold is
//! frozen and flushed to an L0 SSTable by a background worker; a MANIFEST
//! records which SSTables exist so a reopened engine can recover.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Weak;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::compaction::leveled::{Run, merge_runs};
use crate::error::{LsmError, Result};
use crate::manifest::{Manifest, SstMeta};
use crate::memtable::MemTable;
use crate::sstable::LookupValue;
use crate::sstable::builder::SsTableBuilder;
use crate::sstable::reader::SsTableReader;
use crate::wal::WalOp;
use crate::wal::reader::WalReader;
use crate::wal::writer::WalWriter;

pub const DEFAULT_FLUSH_THRESHOLD_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

/// Tuning knobs for [`LsmEngine`].
#[derive(Clone, Debug)]
pub struct LsmOptions {
    /// An active MemTable is frozen into an immutable once its approximate
    /// resident size reaches this many bytes. 0 disables auto-freeze.
    pub flush_threshold_bytes: usize,
    /// When true, every `put`/`delete` fsyncs the WAL before returning, making
    /// each write durable against an OS crash (slow). When false, durability
    /// is guaranteed at [`LsmEngine::flush`] and on clean drop only.
    pub sync_per_write: bool,
    /// Auto-compaction fires (in the background worker, after a flush) once the
    /// number of live L0 tables reaches this many. 0 disables auto-compaction.
    pub compaction_l0_file_threshold: usize,
}

impl Default for LsmOptions {
    fn default() -> Self {
        Self {
            flush_threshold_bytes: DEFAULT_FLUSH_THRESHOLD_BYTES,
            sync_per_write: false,
            compaction_l0_file_threshold: 4,
        }
    }
}

/// A MemTable that was frozen out of the active write path and is waiting to
/// be flushed to an SSTable. Its WAL generation must be deleted once the flush
/// commits.
#[derive(Clone)]
struct PendingImm {
    mem: Arc<MemTable>,
    wal_seq: u64,
}

/// In-memory handle to one on-disk SSTable. Stored behind an `Arc` in engine
/// state so snapshots share one handle (and thus one lazily-opened reader).
/// The reader is opened once on the first lookup and cached; `OnceLock` keeps
/// the result `Send + Sync`.
struct SstHandle {
    meta: SstMeta,
    path: PathBuf,
    reader: OnceLock<std::result::Result<Arc<SsTableReader>, String>>,
}

impl SstHandle {
    fn open_reader(&self) -> Result<Arc<SsTableReader>> {
        self.reader
            .get_or_init(|| {
                SsTableReader::open(&self.path)
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|e| {
                LsmError::Corruption(format!(
                    "failed to open sstable {}: {e}",
                    self.path.display()
                ))
            })
    }
}

#[derive(Default)]
struct EngineState {
    active: Option<Arc<MemTable>>, // always Some after open; swap is one write lock
    immutables: Vec<PendingImm>,   // freeze order, oldest first
    levels: Vec<Vec<Arc<SstHandle>>>, // levels[l] = level l, ascending by seq
}

struct ActiveWal {
    writer: WalWriter,
    seq: u64,
}

struct LsmEngineInner {
    dir: PathBuf,
    options: LsmOptions,
    state: RwLock<EngineState>,
    wal: Mutex<ActiveWal>,
    /// Serializes freeze / flush-commit critical sections.
    state_lock: Mutex<()>,
    /// Serializes whole flush pipelines (drain + manifest write) so concurrent
    /// flushers — the public `flush()` and the background worker — never race
    /// on `MANIFEST.tmp` or reorder L0 commits.
    flush_lock: Mutex<()>,
    next_seq: AtomicU64,
    /// Signal to the background worker that an immutable is pending. Declared
    /// before `runtime` so it drops first on teardown, closing the channel and
    /// letting the worker exit before the runtime shuts down.
    flush_tx: tokio::sync::mpsc::UnboundedSender<()>,
    runtime: tokio::runtime::Runtime,
}

impl Drop for LsmEngineInner {
    /// Best-effort durability on clean drop: fsync the active WAL so a
    /// reopened engine can recover writes that were never flushed. Errors are
    /// ignored (nothing sensible to do at drop time).
    ///
    /// Note: dropping the last handle does NOT quiesce background flushes — a
    /// worker may finish an in-flight flush after the drop returns, so the
    /// directory can still change briefly. Reopen the same directory only
    /// after a call to [`LsmEngine::flush`] (or when no auto-flush can be in
    /// flight) to avoid racing the old worker's writes.
    fn drop(&mut self) {
        if let Ok(mut wal) = self.wal.lock() {
            let _ = wal.writer.sync();
        }
    }
}

/// Background worker: drains frozen immutables to L0 whenever notified. Holds
/// only a `Weak` handle so it never keeps the engine alive; when every strong
/// handle drops, the sender is dropped, the channel closes, and the loop ends.
async fn flush_worker(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<()>,
    engine: Weak<LsmEngineInner>,
) {
    while rx.recv().await.is_some() {
        let Some(engine) = engine.upgrade() else {
            break;
        };
        // SSTable building and compaction are blocking I/O; run them off the
        // async executor. Compaction runs after a flush once enough L0 tables
        // have piled up.
        let _ = tokio::task::spawn_blocking(move || {
            if engine.flush_all_pending().is_ok() && engine.compact_needed() {
                let _ = engine.compact_all_pending();
            }
        })
        .await;
    }
}

impl LsmEngineInner {
    /// Drains every pending immutable to L0 (in freeze order) and rewrites the
    /// MANIFEST, holding `flush_lock` so concurrent flushers never race on
    /// `MANIFEST.tmp` or reorder L0 commits. Immutables stay visible to readers
    /// until their SST commits.
    ///
    /// A flushed immutable's WAL is only deleted AFTER the MANIFEST is durably
    /// rewritten: if a crash lands between an SST rename and the manifest
    /// write, the WAL still exists and recovery replays it, so no committed
    /// data is lost.
    ///
    /// On error, partially-processed immutables remain queued and are retried
    /// on the next flush / worker notification.
    fn flush_all_pending(&self) -> Result<()> {
        let _guard = self
            .flush_lock
            .lock()
            .map_err(|_| LsmError::InvalidOp("flush lock poisoned".to_string()))?;
        let mut flushed_wal_seqs = Vec::new();
        loop {
            let pending = {
                let state = self.state.read().expect("state lock not poisoned");
                state.immutables.first().cloned()
            };
            let Some(pending) = pending else { break };
            self.flush_pending(&pending)?;
            flushed_wal_seqs.push(pending.wal_seq);
        }
        if flushed_wal_seqs.is_empty() {
            // Nothing was committed; the MANIFEST is already current. Skipping
            // the rewrite avoids needless I/O and shortens the post-drop window
            // in which a waking worker touches the directory.
            return Ok(());
        }
        self.write_manifest()?;
        // WALs of the now-manifested immutables are redundant; delete them.
        for wal_seq in flushed_wal_seqs {
            let wal_path = self.dir.join(format!("wal_{wal_seq:06}.wal"));
            let _ = std::fs::remove_file(wal_path);
        }
        Ok(())
    }

    /// Whether the background worker should compact now: at least
    /// `compaction_l0_file_threshold` L0 tables are live (and auto-compaction
    /// is enabled).
    fn compact_needed(&self) -> bool {
        if self.options.compaction_l0_file_threshold == 0 {
            return false;
        }
        let state = self.state.read().expect("state lock not poisoned");
        state
            .levels
            .first()
            .is_some_and(|l0| l0.len() >= self.options.compaction_l0_file_threshold)
    }

    /// Compacts the whole live set (every L0 and L1 table) into a single new
    /// L1 table via k-way merge, dropping tombstones and outdated keys, then
    /// rewrites the MANIFEST and deletes the source tables. Holding
    /// `flush_lock` keeps this mutually exclusive with flush commits so L0
    /// membership cannot change mid-compaction.
    ///
    /// A reader that grabbed an `Arc` handle to a source table before deletion
    /// may hit an I/O error if it reads that file after compaction removes it —
    /// a small race window accepted in v0.1 (deferred file deletion with
    /// versioning is future work).
    fn compact_all_pending(&self) -> Result<()> {
        let _flush_guard = self
            .flush_lock
            .lock()
            .map_err(|_| LsmError::InvalidOp("flush lock poisoned".to_string()))?;

        // Snapshot the input tables (L0 + L1). Cloning the Arcs keeps their
        // lazily-opened readers alive and shared. Compaction only fires when L0
        // is non-empty (a lone L1 table is already the merged result of a prior
        // compaction and needs no re-merge).
        let input_handles: Vec<Arc<SstHandle>> = {
            let state = self.state.read().expect("state lock not poisoned");
            if state.levels.first().is_none_or(|l0| l0.is_empty()) {
                return Ok(()); // nothing to compact
            }
            let mut handles = Vec::new();
            for level in 0..=1 {
                if let Some(list) = state.levels.get(level) {
                    handles.extend(list.iter().cloned());
                }
            }
            handles
        };

        let mut runs = Vec::with_capacity(input_handles.len());
        for handle in &input_handles {
            let reader = handle.open_reader()?;
            let records = reader.records()?;
            runs.push(Run {
                seq: handle.meta.seq,
                records,
            });
        }
        let merged = merge_runs(&runs)?;

        // Write the merged output to a fresh L1 table (unless nothing survived).
        let mut output_handle: Option<Arc<SstHandle>> = None;
        if !merged.is_empty() {
            let sst_seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
            let name = format!("L1_{sst_seq:06}.sst");
            let tmp_path = self.dir.join(format!("{name}.tmp"));
            let final_path = self.dir.join(&name);

            let mut builder = SsTableBuilder::with_expected_items(merged.len());
            for (key, value, deleted) in &merged {
                builder.add(key, value, *deleted)?;
            }
            let min = builder.min_key().unwrap_or_default();
            let max = builder.max_key().unwrap_or_default();
            let write_result: crate::error::Result<()> = (|| {
                builder.build(&tmp_path)?;
                {
                    let f = File::open(&tmp_path)?;
                    f.sync_all()?;
                }
                std::fs::rename(&tmp_path, &final_path)?;
                Ok(())
            })();
            if write_result.is_err() {
                let _ = std::fs::remove_file(&tmp_path);
            }
            write_result?;

            let meta = SstMeta {
                level: 1,
                seq: sst_seq,
                name,
                min,
                max,
            };
            output_handle = Some(Arc::new(SstHandle {
                meta,
                path: final_path,
                reader: OnceLock::new(),
            }));
        }

        // Commit: L0 and L1 are replaced by the single merged L1 output.
        let _state_guard = self
            .state_lock
            .lock()
            .map_err(|_| LsmError::InvalidOp("state lock poisoned".to_string()))?;
        {
            let mut state = self.state.write().expect("state lock not poisoned");
            if state.levels.is_empty() {
                state.levels.push(Vec::new());
            }
            state.levels[0].clear();
            if let Some(handle) = &output_handle {
                if state.levels.len() <= 1 {
                    state.levels.resize(2, Vec::new());
                }
                state.levels[1] = vec![handle.clone()];
            } else {
                state.levels.truncate(1); // keep level 0 empty; drop level 1
            }
        }
        drop(_state_guard);
        self.write_manifest()?;

        // Source tables are now unreferenced by the manifest; delete them
        // best-effort only after the manifest is durable.
        for handle in &input_handles {
            let _ = std::fs::remove_file(&handle.path);
        }
        Ok(())
    }

    /// Writes one frozen MemTable out as an L0 SSTable, then commits it: the
    /// handle joins the state AND the immutable is removed atomically under
    /// `state_lock`, so a reader never sees a key vanish (the immutable stays
    /// readable until its SST is visible).
    fn flush_pending(&self, pending: &PendingImm) -> Result<()> {
        let sst_seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let name = format!("L0_{sst_seq:06}.sst");
        let tmp_path = self.dir.join(format!("{name}.tmp"));
        let final_path = self.dir.join(&name);

        let mut builder = SsTableBuilder::with_expected_items(pending.mem.len());
        for (key, entry) in pending.mem.iter() {
            builder.add(&key, &entry.value, entry.deleted)?;
        }
        let min = builder.min_key().unwrap_or_default();
        let max = builder.max_key().unwrap_or_default();
        builder.build(&tmp_path)?;
        {
            let f = File::open(&tmp_path)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, &final_path)?;

        let meta = SstMeta {
            level: 0,
            seq: sst_seq,
            name,
            min,
            max,
        };

        let _guard = self
            .state_lock
            .lock()
            .map_err(|_| LsmError::InvalidOp("state lock poisoned".to_string()))?;
        {
            let mut state = self.state.write().expect("state lock not poisoned");
            // Insert the handle keeping level-0 seq-ascending, then drop the
            // immutable atomically so readers see either the immutable or the
            // SST for a key, never neither.
            if state.levels.is_empty() {
                state.levels.push(Vec::new());
            }
            let level0 = &mut state.levels[0];
            let pos = level0.partition_point(|h| h.meta.seq < sst_seq);
            level0.insert(
                pos,
                Arc::new(SstHandle {
                    meta,
                    path: final_path,
                    reader: OnceLock::new(),
                }),
            );
            state.immutables.retain(|p| p.wal_seq != pending.wal_seq);
        }
        Ok(())
    }

    fn write_manifest(&self) -> Result<()> {
        let mut manifest = Manifest::default();
        let state = self.state.read().expect("state lock not poisoned");
        for (level, handles) in state.levels.iter().enumerate() {
            for handle in handles {
                let mut meta = handle.meta.clone();
                meta.level = level as u32;
                manifest.add(meta);
            }
        }
        drop(state);
        crate::manifest::write_atomic(&self.dir, &manifest)
    }
}

/// A synchronous, thread-safe key-value store built on an LSM tree.
#[derive(Clone)]
pub struct LsmEngine {
    inner: Arc<LsmEngineInner>,
}

impl LsmEngine {
    /// Opens (creating if necessary) an engine rooted at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, LsmOptions::default())
    }

    /// Opens an engine with custom [`LsmOptions`].
    pub fn open_with_options(path: impl AsRef<Path>, options: LsmOptions) -> Result<Self> {
        let dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let manifest = crate::manifest::load(&dir)?;
        let mut levels: Vec<Vec<Arc<SstHandle>>> = Vec::new();
        for (level, metas) in manifest.levels.iter().enumerate() {
            let mut handles = Vec::new();
            for meta in metas {
                let path = dir.join(&meta.name);
                if !path.exists() {
                    return Err(LsmError::Corruption(format!(
                        "manifest references missing sstable {}",
                        path.display()
                    )));
                }
                handles.push(Arc::new(SstHandle {
                    meta: meta.clone(),
                    path,
                    reader: OnceLock::new(),
                }));
            }
            if levels.len() <= level {
                levels.resize(level + 1, Vec::new());
            }
            levels[level] = handles;
        }

        // Replay every WAL generation, oldest first, into one fresh MemTable.
        // A later generation's puts/deletes must override earlier ones, so the
        // replay order matters.
        let active = Arc::new(MemTable::new());
        let mut wal_seqs = Vec::new();
        let entries = std::fs::read_dir(&dir)?;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name().into_string().map_err(|_| {
                LsmError::Corruption("non-utf8 file name in engine directory".to_string())
            })?;
            if let Some(seq) = parse_wal_name(&name) {
                wal_seqs.push((seq, entry.path()));
            }
        }
        wal_seqs.sort_by_key(|(seq, _)| *seq);
        // Replay all generations first; a corrupt later WAL must not cost the
        // earlier ones (deletion happens only once every replay succeeded and
        // the recovered state is durably re-backed).
        for (_, path) in &wal_seqs {
            let mut reader = WalReader::open(path)?;
            reader.replay_into(&active)?;
        }

        let mut max_seq = 0u64;
        for metas in &manifest.levels {
            for meta in metas {
                max_seq = max_seq.max(meta.seq);
            }
        }
        for (seq, _) in &wal_seqs {
            max_seq = max_seq.max(*seq);
        }
        let next_seq = AtomicU64::new(max_seq + 1);
        let wal_seq = next_seq.fetch_add(1, Ordering::SeqCst);
        let wal_path = dir.join(format!("wal_{wal_seq:06}.wal"));
        let mut writer = WalWriter::create(&wal_path)?;

        // CRITICAL: re-append the recovered MemTable into the fresh WAL and
        // fsync it BEFORE deleting the old generations. Otherwise the replayed
        // data lives only in RAM: after a clean drop (which fsyncs the new,
        // still-empty WAL) and a reopen, nothing would replay and the data
        // would be lost.
        if !active.is_empty() {
            for (key, entry) in active.iter() {
                let op = if entry.deleted {
                    WalOp::Delete
                } else {
                    WalOp::Put
                };
                let value = if entry.deleted {
                    &[][..]
                } else {
                    entry.value.as_slice()
                };
                writer.append(op, &key, value)?;
            }
            writer.sync()?;
        }
        // Old generations are now redundant; removal is best-effort so a
        // transient delete error cannot abort open and strand the engine.
        for (_, path) in &wal_seqs {
            let _ = std::fs::remove_file(path);
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        let (flush_tx, flush_rx) = tokio::sync::mpsc::unbounded_channel();

        let inner = Arc::new(LsmEngineInner {
            dir,
            options,
            state: RwLock::new(EngineState {
                active: Some(active),
                immutables: Vec::new(),
                levels,
            }),
            wal: Mutex::new(ActiveWal {
                writer,
                seq: wal_seq,
            }),
            state_lock: Mutex::new(()),
            flush_lock: Mutex::new(()),
            next_seq,
            flush_tx,
            runtime,
        });
        inner
            .runtime
            .spawn(flush_worker(flush_rx, Arc::downgrade(&inner)));

        Ok(LsmEngine { inner })
    }

    /// Inserts `key -> value`, overwriting any previous value.
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut wal = self
            .inner
            .wal
            .lock()
            .map_err(|_| LsmError::InvalidOp("wal lock poisoned".to_string()))?;
        wal.writer.append(WalOp::Put, key, value)?;
        if self.inner.options.sync_per_write {
            wal.writer.sync()?;
        }
        self.apply_and_maybe_freeze(&mut wal, key, value, false)
    }

    /// Deletes `key`. The tombstone shadows any older value in memory or on
    /// disk until compaction drops it.
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        let mut wal = self
            .inner
            .wal
            .lock()
            .map_err(|_| LsmError::InvalidOp("wal lock poisoned".to_string()))?;
        wal.writer.append(WalOp::Delete, key, &[])?;
        if self.inner.options.sync_per_write {
            wal.writer.sync()?;
        }
        self.apply_and_maybe_freeze(&mut wal, key, &[], true)
    }

    /// Applies the mutation to the active MemTable, then freezes it if it has
    /// grown past the flush threshold. Runs while holding the WAL lock so the
    /// active MemTable rotation is serialized with appends.
    fn apply_and_maybe_freeze(
        &self,
        wal: &mut ActiveWal,
        key: &[u8],
        value: &[u8],
        deleted: bool,
    ) -> Result<()> {
        let active = self.snapshot_active();
        if deleted {
            active.delete(key)?;
        } else {
            active.put(key, value)?;
        }

        if self.inner.options.flush_threshold_bytes > 0
            && active.approximate_size() >= self.inner.options.flush_threshold_bytes
        {
            let froze = self.freeze_active(wal)?;
            if froze {
                // Ask the background worker to drain the new immutable.
                let _ = self.inner.flush_tx.send(());
            }
        }
        Ok(())
    }

    /// Returns the current value for `key`, or `None` if absent or deleted.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let (active, immutables, levels) = {
            let state = self
                .inner
                .state
                .read()
                .map_err(|_| LsmError::InvalidOp("state lock poisoned".to_string()))?;
            (
                state
                    .active
                    .as_ref()
                    .expect("active is set after open")
                    .clone(),
                state.immutables.clone(),
                state.levels.clone(),
            )
        };

        // 1. active memtable
        if let Some(entry) = active.get_entry(key) {
            return Ok(if entry.deleted {
                None
            } else {
                Some(entry.value)
            });
        }
        // 2. immutables, newest first
        for pending in immutables.iter().rev() {
            if let Some(entry) = pending.mem.get_entry(key) {
                return Ok(if entry.deleted {
                    None
                } else {
                    Some(entry.value)
                });
            }
        }
        // 3. SSTables, level by level, newest (highest seq) first within a level
        for level_handles in &levels {
            for handle in level_handles.iter().rev() {
                if key < handle.meta.min.as_slice() || key > handle.meta.max.as_slice() {
                    continue;
                }
                let reader = handle.open_reader()?;
                match reader.get(key)? {
                    Some(LookupValue::Present(v)) => return Ok(Some(v)),
                    Some(LookupValue::Deleted) => return Ok(None), // tombstone: stop
                    None => {}
                }
            }
        }
        Ok(None)
    }

    /// Flushes the active MemTable (if non-empty) and every frozen immutable to
    /// L0 SSTables, then persists the MANIFEST. This is the durability point.
    pub fn flush(&self) -> Result<()> {
        {
            let mut wal = self
                .inner
                .wal
                .lock()
                .map_err(|_| LsmError::InvalidOp("wal lock poisoned".to_string()))?;
            wal.writer.sync()?;
            self.freeze_active(&mut wal)?;
        }
        self.inner.flush_all_pending()
    }

    /// Compacts every live L0 and L1 table into a single L1 table (k-way merge,
    /// dropping tombstones and outdated keys) and rewrites the MANIFEST. No-op
    /// when there are no L0 tables. Synchronous; the background worker triggers
    /// this automatically after flushes once L0 exceeds
    /// [`LsmOptions::compaction_l0_file_threshold`].
    pub fn compact(&self) -> Result<()> {
        self.inner.compact_all_pending()
    }

    fn snapshot_active(&self) -> Arc<MemTable> {
        let state = self.inner.state.read().expect("state lock not poisoned");
        state
            .active
            .as_ref()
            .expect("active is set after open")
            .clone()
    }

    /// Rotates the active MemTable into the immutable list (if non-empty) and
    /// starts a fresh MemTable + WAL generation. Caller must hold the WAL lock,
    /// which serializes freezes against concurrent puts.
    ///
    /// Returns `true` if an immutable was frozen (a notification to the worker
    /// should be sent).
    fn freeze_active(&self, wal: &mut ActiveWal) -> Result<bool> {
        let is_empty = self
            .inner
            .state
            .read()
            .expect("state lock not poisoned")
            .active
            .as_ref()
            .expect("active is set after open")
            .is_empty();
        if is_empty {
            return Ok(false);
        }
        // Rotate the WAL first so a creation failure leaves engine state
        // untouched (no dangling PendingImm sharing a wal_seq).
        let new_seq = self.inner.next_seq.fetch_add(1, Ordering::SeqCst);
        let path = self.inner.dir.join(format!("wal_{new_seq:06}.wal"));
        let new_writer = WalWriter::create(path)?;

        let _guard = self
            .inner
            .state_lock
            .lock()
            .map_err(|_| LsmError::InvalidOp("state lock poisoned".to_string()))?;
        let mut state = self.inner.state.write().expect("state lock not poisoned");
        let cur = state.active.take().expect("active is set after open");
        state.active = Some(Arc::new(MemTable::new()));
        state.immutables.push(PendingImm {
            mem: cur,
            wal_seq: wal.seq,
        });
        drop(state);
        wal.writer = new_writer;
        wal.seq = new_seq;
        Ok(true)
    }
}

/// Parses `wal_<seq>.wal` returning the seq, or `None` if the name is not a
/// WAL file the engine wrote.
fn parse_wal_name(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("wal_")?;
    let seq = rest.strip_suffix(".wal")?;
    seq.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const LARGE_THRESHOLD: usize = usize::MAX; // effectively disable auto-freeze

    fn open_dir(dir: &TempDir) -> LsmEngine {
        LsmEngine::open(dir.path()).expect("open engine")
    }

    fn open_dir_opts(dir: &TempDir, options: LsmOptions) -> LsmEngine {
        LsmEngine::open_with_options(dir.path(), options).expect("open engine")
    }

    fn sst_count(dir: &TempDir) -> usize {
        std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("L0_"))
            .count()
    }

    fn count_files(dir: &TempDir, prefix: &str) -> usize {
        std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(prefix))
            .count()
    }

    fn l1_filename(dir: &TempDir) -> Option<String> {
        std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .find(|n| n.starts_with("L1_"))
    }

    #[test]
    fn put_get_delete_end_to_end() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"a", b"v1").unwrap();
        assert_eq!(engine.get(b"a").unwrap(), Some(b"v1".to_vec()));

        engine.put(b"b", b"first").unwrap();
        engine.put(b"b", b"second").unwrap();
        assert_eq!(engine.get(b"b").unwrap(), Some(b"second".to_vec()));

        engine.delete(b"a").unwrap();
        assert_eq!(engine.get(b"a").unwrap(), None);
        assert_eq!(engine.get(b"missing").unwrap(), None);
    }

    #[test]
    fn tombstone_shadows_flushed_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();
        engine.flush().unwrap(); // k now in an L0 SSTable
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));

        engine.delete(b"k").unwrap(); // tombstone in the active memtable
        assert_eq!(
            engine.get(b"k").unwrap(),
            None,
            "memtable tombstone must beat SST"
        );

        engine.flush().unwrap(); // tombstone now also on disk
        assert_eq!(engine.get(b"k").unwrap(), None, "must not resurrect");
    }

    #[test]
    fn flush_creates_sst_and_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"apple", b"red").unwrap();
        engine.put(b"cherry", b"dark red").unwrap();
        engine.flush().unwrap();

        assert_eq!(sst_count(&dir), 1);
        let manifest = crate::manifest::load(dir.path()).expect("load manifest");
        assert_eq!(manifest.ssts_at(0).len(), 1);
        let meta = &manifest.ssts_at(0)[0];
        assert_eq!(meta.min, b"apple");
        assert_eq!(meta.max, b"cherry");
    }

    #[test]
    fn get_across_levels_newest_wins() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v1").unwrap();
        engine.flush().unwrap();
        engine.put(b"k", b"v2").unwrap();
        engine.flush().unwrap();
        assert_eq!(sst_count(&dir), 2);
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn recovery_from_wal_after_clean_drop() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"a", b"1").unwrap();
            engine.put(b"b", b"2").unwrap();
            engine.put(b"c", b"3").unwrap();
        } // drop: inner Drop fsyncs the active WAL
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(engine.get(b"c").unwrap(), Some(b"3".to_vec()));
    }

    #[test]
    fn recovery_from_sst_after_flush() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"a", b"v").unwrap();
            engine.flush().unwrap();
            engine.put(b"b", b"w").unwrap(); // stays in WAL
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"a").unwrap(), Some(b"v".to_vec())); // from SST
        assert_eq!(engine.get(b"b").unwrap(), Some(b"w".to_vec())); // from WAL replay
        assert_eq!(engine.get(b"zzz").unwrap(), None);
    }

    #[test]
    fn recovery_tombstone_persisted() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
            engine.delete(b"k").unwrap();
            engine.flush().unwrap();
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"k").unwrap(), None);
    }

    #[test]
    fn open_fresh_dir_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir(&dir);
        assert_eq!(engine.get(b"anything").unwrap(), None);
        engine.put(b"x", b"y").unwrap();
        assert_eq!(engine.get(b"x").unwrap(), Some(b"y".to_vec()));
    }

    #[test]
    fn flush_empty_engine_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir(&dir);
        engine.flush().unwrap();
        assert_eq!(sst_count(&dir), 0);
        assert_eq!(engine.get(b"k").unwrap(), None);
    }

    #[test]
    fn sync_per_write_recovers_without_flush() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    sync_per_write: true,
                    ..Default::default()
                },
            );
            engine.put(b"a", b"1").unwrap();
            engine.put(b"b", b"2").unwrap();
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
    }

    #[test]
    fn manifest_missing_sst_file_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
            engine.flush().unwrap();
        }
        // Delete the SSTable but keep MANIFEST.
        let manifest = crate::manifest::load(dir.path()).expect("load manifest");
        let name = manifest.ssts_at(0)[0].name.clone();
        std::fs::remove_file(dir.path().join(name)).expect("remove sst");
        let err = match LsmEngine::open(dir.path()) {
            Ok(_) => panic!("missing sst must fail open"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn empty_key_value_valid() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"", b"").unwrap();
        assert_eq!(engine.get(b"").unwrap(), Some(b"".to_vec()));
        engine.flush().unwrap();
        assert_eq!(engine.get(b"").unwrap(), Some(b"".to_vec()));
    }

    #[test]
    fn recovery_multiple_wal_generations_applied_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            // Small threshold forces several freeze/wal-rotation cycles without
            // relying on the background worker (we drop quickly), simulating a
            // crash with multiple unwritten generations.
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: 32,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v1").unwrap();
            for i in 0..10u32 {
                let key = format!("a{i:03}");
                engine.put(key.as_bytes(), b"x").unwrap();
            }
            engine.put(b"k", b"v2").unwrap(); // overwrite across a generation boundary
            for i in 0..10u32 {
                let key = format!("b{i:03}");
                engine.put(key.as_bytes(), b"y").unwrap();
            }
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(
            engine.get(b"k").unwrap(),
            Some(b"v2".to_vec()),
            "latest value across generations must win"
        );
        for i in 0..10u32 {
            let key = format!("a{i:03}");
            assert_eq!(engine.get(key.as_bytes()).unwrap(), Some(b"x".to_vec()));
            let key = format!("b{i:03}");
            assert_eq!(engine.get(key.as_bytes()).unwrap(), Some(b"y".to_vec()));
        }
    }

    #[test]
    fn flush_is_idempotent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();
        engine.flush().unwrap();
        let count = sst_count(&dir);
        engine.flush().unwrap();
        assert_eq!(sst_count(&dir), count, "second flush must not add files");
    }

    #[test]
    fn concurrent_flushes_do_not_race() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();

        let engine_a = engine.clone();
        let engine_b = engine.clone();
        let handle_a = std::thread::spawn(move || {
            for _ in 0..25 {
                engine_a.flush().expect("flush must succeed");
            }
        });
        let handle_b = std::thread::spawn(move || {
            for _ in 0..25 {
                engine_b.flush().expect("flush must succeed");
            }
        });
        handle_a.join().unwrap();
        handle_b.join().unwrap();

        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
        drop(engine);
        let reopened = open_dir(&dir);
        assert_eq!(reopened.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn get_during_flush_never_misses_committed_data() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();

        let engine_clone = engine.clone();
        let flusher = std::thread::spawn(move || {
            for _ in 0..200 {
                engine_clone.flush().expect("flush must succeed");
            }
        });
        for _ in 0..20_000 {
            assert_eq!(
                engine.get(b"k").unwrap(),
                Some(b"v".to_vec()),
                "key must be readable during flush"
            );
        }
        flusher.join().unwrap();
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn auto_flush_produces_sst_without_explicit_flush() {
        // A small threshold + background worker must flush to disk on its own.
        // Auto-compaction is disabled so this test asserts the flush path only.
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: 128,
                compaction_l0_file_threshold: 0,
                ..Default::default()
            },
        );
        for i in 0..200u32 {
            let key = format!("key{i:05}");
            engine.put(key.as_bytes(), b"value-data").unwrap();
        }

        // Poll until the worker has flushed at least one L0 file.
        let mut seen = 0;
        for _ in 0..200 {
            seen = sst_count(&dir);
            if seen >= 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(seen >= 1, "background worker must flush to L0");

        // Every key must still be readable (from memtable, immutable, or SST).
        for i in 0..200u32 {
            let key = format!("key{i:05}");
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(b"value-data".to_vec())
            );
        }

        // A final explicit flush must converge the WAL count to exactly one
        // (the active generation); all flushed generations' WALs are deleted.
        engine.flush().unwrap();
        let wals = std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("wal_"))
            .count();
        assert_eq!(wals, 1, "only the active WAL may remain after a flush");
    }

    #[test]
    fn worker_drops_when_engine_dropped() {
        // Dropping the engine must let the worker exit (no leaked thread); the
        // strongest signal we can assert cheaply is that drop + reopen of many
        // engines in a loop neither hangs nor panics.
        for _ in 0..20 {
            let dir = tempfile::tempdir().expect("tempdir");
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: 64,
                    ..Default::default()
                },
            );
            for i in 0..50u32 {
                let key = format!("k{i}");
                engine.put(key.as_bytes(), b"v").unwrap();
            }
            engine.flush().unwrap();
        }
    }

    #[test]
    fn manifest_corrupt_on_existing_data_fails_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"a", b"1").unwrap();
            engine.flush().unwrap(); // MANIFEST now references an SST
            engine.put(b"b", b"2").unwrap(); // survives only in the WAL
        }
        // Corrupt the MANIFEST in place; a perfectly replayable WAL must not
        // mask the corruption (recovery must fail fast, not silently go empty).
        std::fs::write(dir.path().join("MANIFEST"), b"garbage not a manifest")
            .expect("overwrite manifest with garbage");
        let err = match LsmEngine::open(dir.path()) {
            Ok(_) => panic!("open must fail-fast on a corrupt MANIFEST"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn orphan_sst_not_in_manifest_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
            engine.flush().unwrap(); // k is now in an SST referenced by MANIFEST
        }
        // Hand-craft an SST the engine never wrote. Its bytes are deliberately
        // not a valid SST: recovery would fail if it consulted the file, so a
        // successful open + read proves only MANIFEST-referenced SSTs load.
        std::fs::write(dir.path().join("L0_999999.sst"), b"not a real sstable")
            .expect("write orphan sst");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
        assert!(
            dir.path().join("L0_999999.sst").exists(),
            "orphan sst must be left untouched"
        );
    }

    #[test]
    fn recovery_survives_repeated_reopen_without_flush() {
        // Regression: recovered data must be durably re-backed in the fresh WAL
        // so that a second clean drop + reopen still finds it. Previously the
        // replay lived only in RAM and vanished on the next reopen cycle.
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
        } // drop #1: fsyncs the active WAL
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            assert_eq!(
                engine.get(b"k").unwrap(),
                Some(b"v".to_vec()),
                "first reopen replays the WAL"
            );
        } // drop #2: must fsync a WAL that still holds k
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(
            engine.get(b"k").unwrap(),
            Some(b"v".to_vec()),
            "second reopen must still find k after recovery"
        );
    }

    #[test]
    fn recovery_reappends_replayed_tombstone() {
        // The recovered WAL must carry tombstones too, so a delete followed by
        // reopen+reopen keeps the key deleted.
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
            engine.delete(b"k").unwrap();
        }
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            assert_eq!(engine.get(b"k").unwrap(), None);
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(
            engine.get(b"k").unwrap(),
            None,
            "tombstone must survive recovery"
        );
    }

    #[test]
    fn manifest_referenced_sst_corrupt_fails_on_first_read() {
        // A MANIFEST-listed SST whose bytes are garbage must still let open()
        // succeed (readers are lazy) but make the first get() that needs it
        // fail with Corruption — distinct from a missing file, which fails at
        // open.
        let dir = tempfile::tempdir().expect("tempdir");
        let sst_name;
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v").unwrap();
            engine.flush().unwrap();
            let manifest = crate::manifest::load(dir.path()).expect("load manifest");
            sst_name = manifest.ssts_at(0)[0].name.clone();
        }
        // Corrupt the real SST file in place (same path, garbage content).
        std::fs::write(dir.path().join(&sst_name), b"corrupted, not an sstable")
            .expect("corrupt sst");

        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Open is lazy: succeeds even though the SST is unreadable.
        let err = match engine.get(b"k") {
            Ok(v) => panic!("corrupt sst must surface an error, got {v:?}"),
            Err(e) => e,
        };
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn orphan_wal_not_matching_pattern_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Files that are not `wal_<digits>.wal` must never be replayed or
        // parsed as WALs (and must not fail open).
        std::fs::write(dir.path().join("notawal.txt"), b"plain text, not a wal")
            .expect("write stray file");
        std::fs::write(
            dir.path().join("wal_abc.wal"),
            b"non-numeric seq, not a wal",
        )
        .expect("write stray file");

        // Opening succeeds only if neither stray file is treated as a WAL.
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();
        engine.flush().unwrap(); // flush must not mistake the stray files either
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
        drop(engine);

        assert!(dir.path().join("notawal.txt").exists());
        assert!(dir.path().join("wal_abc.wal").exists());
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn recovery_is_idempotent_across_reopens() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"a", b"1").unwrap();
            engine.flush().unwrap();
            engine.put(b"b", b"2").unwrap();
            engine.flush().unwrap();
        }
        // First recovery: data comes back exactly once from the SSTs.
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
        engine.put(b"c", b"3").unwrap();
        engine.flush().unwrap();
        let sst_after_write = sst_count(&dir);
        drop(engine);

        // Second recovery of the same directory: values must not be duplicated
        // and recovery must not add new SST files.
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(engine.get(b"a").unwrap(), Some(b"1".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), Some(b"2".to_vec()));
        assert_eq!(engine.get(b"c").unwrap(), Some(b"3".to_vec()));
        assert_eq!(
            sst_count(&dir),
            sst_after_write,
            "reopening must not add or drop SST files"
        );
    }

    #[test]
    fn tombstone_shadows_across_flush_generations() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            engine.put(b"k", b"v1").unwrap();
            engine.flush().unwrap(); // L0 gen 1: v1
            engine.delete(b"k").unwrap();
            engine.flush().unwrap(); // L0 gen 2: tombstone
            engine.put(b"k", b"v3").unwrap();
            engine.flush().unwrap(); // L0 gen 3: v3
            assert_eq!(
                engine.get(b"k").unwrap(),
                Some(b"v3".to_vec()),
                "a fresh put must beat an older flushed tombstone"
            );

            engine.delete(b"k").unwrap();
            engine.flush().unwrap(); // L0 gen 4: tombstone (newest)
        }
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(
            engine.get(b"k").unwrap(),
            None,
            "newest tombstone must shadow older generations after recovery"
        );
    }

    #[test]
    fn background_worker_recovery_consistency() {
        // Small threshold: puts auto-freeze many WAL generations while the
        // background worker flushes earlier ones to L0 concurrently, so on drop
        // some generations were flushed (SST + MANIFEST, WAL deleted) and others
        // were not. flush() before drop quiesces the worker (its drain is
        // synchronous), making recovery deterministic.
        let dir = tempfile::tempdir().expect("tempdir");
        const N_KEYS: usize = 300;
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: 128,
                    compaction_l0_file_threshold: 0, // isolate flush-worker recovery
                    ..Default::default()
                },
            );
            for i in 0..N_KEYS {
                let key = format!("key{i:05}");
                engine.put(key.as_bytes(), b"value-data").unwrap();
            }
            engine.flush().unwrap(); // converge: drain every frozen generation
        }
        assert!(
            sst_count(&dir) >= 1,
            "auto-flush cycles must have produced L0 files"
        );

        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        for i in 0..N_KEYS {
            let key = format!("key{i:05}");
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(b"value-data".to_vec()),
                "key {i} must survive recovery"
            );
        }
    }

    #[test]
    fn many_small_flushes_recover() {
        // Ten small L0 generations (one explicit flush per round) with
        // interleaved puts and deletes over the same key set. After recovery,
        // every key must reflect exactly its last write, in round order — this
        // catches ordering/lost-update bugs across many small SSTs and lets
        // flushed tombstones prove they still shadow older generations.
        let dir = tempfile::tempdir().expect("tempdir");
        let keys: Vec<String> = (0..20).map(|i| format!("k{i:02}")).collect();
        let mut expected: Vec<Option<Vec<u8>>> = vec![None; keys.len()];
        {
            let engine = open_dir_opts(
                &dir,
                LsmOptions {
                    flush_threshold_bytes: LARGE_THRESHOLD,
                    ..Default::default()
                },
            );
            for round in 0..10u32 {
                for (i, key) in keys.iter().enumerate() {
                    match (round + i as u32) % 3 {
                        0 => {
                            engine.delete(key.as_bytes()).unwrap();
                            expected[i] = None;
                        }
                        _ => {
                            let val = format!("v{round}-{i}");
                            engine.put(key.as_bytes(), val.as_bytes()).unwrap();
                            expected[i] = Some(val.into_bytes());
                        }
                    }
                }
                engine.flush().unwrap();
            }
        }

        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Exactly one L0 SST per drained flush generation. Explicit flush()
        // never notifies the background worker, so auto-compaction (which only
        // fires on worker notifications) cannot coalesce these generations here.
        assert_eq!(
            sst_count(&dir),
            10,
            "every round's flush must leave one L0 generation on disk"
        );
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(
                engine.get(key.as_bytes()).unwrap().as_deref(),
                expected[i].as_deref(),
                "key {i} must reflect its last write across generations"
            );
        }
    }

    #[test]
    fn compact_merges_l0_to_l1() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Three L0 generations with overlapping + deleted keys.
        engine.put(b"a", b"v1").unwrap();
        engine.put(b"b", b"1").unwrap();
        engine.flush().unwrap(); // gen1: a=v1, b=1
        engine.put(b"a", b"v2").unwrap();
        engine.delete(b"b").unwrap();
        engine.put(b"c", b"3").unwrap();
        engine.flush().unwrap(); // gen2: a=v2 (overwrites), b=tombstone, c=3
        engine.put(b"c", b"33").unwrap();
        engine.flush().unwrap(); // gen3: c=33
        assert_eq!(count_files(&dir, "L0_"), 3);

        engine.compact().expect("compact");

        // L0 gone, one L1 file, manifest reflects it.
        assert_eq!(count_files(&dir, "L0_"), 0, "L0 tables must be removed");
        assert_eq!(count_files(&dir, "L1_"), 1, "one merged L1 table");
        assert_eq!(engine.get(b"a").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(engine.get(b"b").unwrap(), None, "tombstone dropped b");
        assert_eq!(engine.get(b"c").unwrap(), Some(b"33".to_vec()));
    }

    #[test]
    fn compact_noop_when_no_l0() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Empty engine: compact is a no-op.
        engine.compact().expect("compact empty");
        assert_eq!(count_files(&dir, "L1_"), 0);

        // One L0 → compacts to L1; compacting again with no L0 must not
        // rewrite L1 (file count stays 1, seq does not advance).
        engine.put(b"k", b"v").unwrap();
        engine.flush().unwrap();
        engine.compact().expect("compact");
        assert_eq!(count_files(&dir, "L1_"), 1);
        let l1_before = l1_filename(&dir).expect("one L1 file");
        engine.compact().expect("compact again");
        assert_eq!(
            l1_filename(&dir).as_ref(),
            Some(&l1_before),
            "no-op must not re-write L1 (seq must not advance)"
        );
        assert_eq!(engine.get(b"k").unwrap(), Some(b"v".to_vec()));
    }

    #[test]
    fn compact_merge_with_existing_l1() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Two keys go to L1 via the first compaction.
        engine.put(b"l1only", b"survivor").unwrap();
        engine.put(b"shared", b"old").unwrap();
        engine.flush().unwrap();
        engine.compact().expect("first compact"); // L1: l1only, shared=old
        assert_eq!(engine.get(b"l1only").unwrap(), Some(b"survivor".to_vec()));

        // New L0: `shared` is overwritten, `l0only` is new; `l1only` exists ONLY
        // in L1 and must survive the merge because L1 is part of the input.
        engine.put(b"shared", b"new").unwrap();
        engine.put(b"l0only", b"x").unwrap();
        engine.flush().unwrap();
        engine.compact().expect("second compact"); // merges L1 + L0

        assert_eq!(
            engine.get(b"l1only").unwrap(),
            Some(b"survivor".to_vec()),
            "L1-only key must survive merge"
        );
        assert_eq!(engine.get(b"shared").unwrap(), Some(b"new".to_vec()));
        assert_eq!(engine.get(b"l0only").unwrap(), Some(b"x".to_vec()));
        assert_eq!(count_files(&dir, "L0_"), 0);
        assert_eq!(count_files(&dir, "L1_"), 1);
    }

    #[test]
    fn compact_drops_final_tombstone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"k", b"v").unwrap();
        engine.flush().unwrap();
        engine.delete(b"k").unwrap();
        engine.flush().unwrap();
        engine.compact().expect("compact");

        assert_eq!(engine.get(b"k").unwrap(), None);
        // Recovery must still see k as absent (tombstone was final).
        drop(engine);
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        assert_eq!(
            engine.get(b"k").unwrap(),
            None,
            "delete must survive compaction+recovery"
        );
    }

    #[test]
    fn compact_empty_result_removes_all_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        engine.put(b"a", b"1").unwrap();
        engine.flush().unwrap();
        engine.put(b"a", b"2").unwrap();
        engine.delete(b"a").unwrap();
        engine.flush().unwrap();
        engine.compact().expect("compact"); // both keys deleted → empty result

        assert_eq!(engine.get(b"a").unwrap(), None);
        assert_eq!(count_files(&dir, "L1_"), 0, "empty merge must write no L1");
        assert_eq!(count_files(&dir, "L0_"), 0);
    }

    #[test]
    fn compact_while_putting_no_crash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: LARGE_THRESHOLD,
                ..Default::default()
            },
        );
        // Seed some L0 data so compact has work to do.
        for i in 0..50u32 {
            let key = format!("seed{i}");
            engine.put(key.as_bytes(), b"s").unwrap();
        }
        engine.flush().unwrap();

        let engine_writer = engine.clone();
        let writer = std::thread::spawn(move || {
            for i in 0..2000u32 {
                let key = format!("w{i}");
                engine_writer.put(key.as_bytes(), b"v").unwrap();
            }
        });
        let engine_compactor = engine.clone();
        let compactor = std::thread::spawn(move || {
            for _ in 0..20 {
                engine_compactor.compact().expect("compact must not error");
            }
        });
        writer.join().unwrap();
        compactor.join().unwrap();

        // All written keys still present (flush to converge, then verify).
        engine.flush().unwrap();
        for i in 0..2000u32 {
            let key = format!("w{i}");
            assert_eq!(engine.get(key.as_bytes()).unwrap(), Some(b"v".to_vec()));
        }
    }

    #[test]
    fn recovery_across_compact_cycles() {
        // Compaction must be crash-safe across a real (non-empty) L1 result:
        // reopen after a compaction that produced L1, continue writing, then
        // compact again and reopen once more. Every key survives.
        let dir = tempfile::tempdir().expect("tempdir");
        let opts = || LsmOptions {
            flush_threshold_bytes: LARGE_THRESHOLD,
            ..Default::default()
        };
        {
            let engine = open_dir_opts(&dir, opts());
            engine.put(b"a", b"v1").unwrap();
            engine.flush().unwrap();
            engine.compact().expect("first compact"); // real L1 with a=v1
        }
        {
            let engine = open_dir_opts(&dir, opts());
            assert_eq!(
                engine.get(b"a").unwrap(),
                Some(b"v1".to_vec()),
                "recovered from L1"
            );
            engine.put(b"b", b"w").unwrap();
            engine.flush().unwrap();
            engine.compact().expect("second compact"); // L1(a) + L0(b) -> L1
        }
        {
            let engine = open_dir_opts(&dir, opts());
            assert_eq!(engine.get(b"a").unwrap(), Some(b"v1".to_vec()));
            assert_eq!(engine.get(b"b").unwrap(), Some(b"w".to_vec()));
            assert_eq!(count_files(&dir, "L0_"), 0);
            assert_eq!(count_files(&dir, "L1_"), 1);
        }
    }

    #[test]
    fn auto_compact_after_many_flushes() {
        // Small flush threshold + low compaction threshold: the background
        // worker should flush to L0 and then compact into L1 on its own, with
        // no explicit flush()/compact() call.
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: 64,
                compaction_l0_file_threshold: 2,
                ..Default::default()
            },
        );
        for i in 0..400u32 {
            let key = format!("key{i:05}");
            engine.put(key.as_bytes(), b"value-data").unwrap();
        }

        // Poll until the worker has produced at least one L1 file (the auto
        // compact path), bounded to keep the test deterministic.
        let mut saw_l1 = false;
        for _ in 0..400 {
            if count_files(&dir, "L1_") >= 1 {
                saw_l1 = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(saw_l1, "background worker must auto-compact L0 into L1");

        // Converge and verify every key.
        engine.flush().unwrap();
        for i in 0..400u32 {
            let key = format!("key{i:05}");
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(b"value-data".to_vec())
            );
        }
    }

    #[test]
    fn compaction_threshold_zero_disables_auto_compact() {
        // compaction_l0_file_threshold = 0 must never auto-compact (no L1 file
        // appears) even after many flushes.
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = open_dir_opts(
            &dir,
            LsmOptions {
                flush_threshold_bytes: 64,
                compaction_l0_file_threshold: 0,
                ..Default::default()
            },
        );
        for i in 0..200u32 {
            let key = format!("key{i:05}");
            engine.put(key.as_bytes(), b"value-data").unwrap();
        }
        // Give the worker time to flush several L0 generations.
        std::thread::sleep(std::time::Duration::from_millis(500));
        engine.flush().unwrap();
        assert_eq!(
            count_files(&dir, "L1_"),
            0,
            "auto-compaction must be disabled when threshold is 0"
        );
        // Data must still be intact in L0.
        for i in 0..200u32 {
            let key = format!("key{i:05}");
            assert_eq!(
                engine.get(key.as_bytes()).unwrap(),
                Some(b"value-data".to_vec())
            );
        }
    }
}
