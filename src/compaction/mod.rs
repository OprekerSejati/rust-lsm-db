//! Compaction: background merging of SSTables to bound read/write amplification
//! and reclaim space from deleted keys.
//!
//! This module houses the pure in-memory k-way merge over sorted runs
//! (`leveled::merge_runs`). The engine drives it: it gathers every live L0 and
//! L1 table, merges them into one fresh L1 table, and updates the MANIFEST.

pub(crate) mod leveled;
