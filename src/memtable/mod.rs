use crossbeam_skiplist::SkipMap;

use crate::error::Result;
use std::sync::atomic::{AtomicUsize, Ordering};

// Estimated fixed per-entry cost (SkipMap node, vec headers, flags). Tuned to
// give a flush trigger that trails real resident memory by a small constant.
const ENTRY_OVERHEAD: usize = 64;

/// Tombstones are represented by an explicit `deleted` flag rather than an
/// empty-value sentinel, because empty values are legitimate user data.
#[derive(Clone, Debug)]
pub struct Entry {
    pub value: Vec<u8>,
    pub deleted: bool,
}

pub struct MemTable {
    map: SkipMap<Vec<u8>, Entry>,
    // Monotonic non-decreasing estimate of resident bytes. Each put/delete adds
    // key.len() + value.len() + ENTRY_OVERHEAD; overwrites never decrement, so
    // this slightly over-estimates under heavy overwrite. That bias is safe:
    // under-reporting would delay an auto-flush, over-reporting only flushes early.
    approximate_size: AtomicUsize,
}

impl MemTable {
    pub fn new() -> Self {
        MemTable {
            map: SkipMap::new(),
            approximate_size: AtomicUsize::new(0),
        }
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.bump_approximate_size(key.len() + value.len());
        self.map.insert(
            key.to_vec(),
            Entry {
                value: value.to_vec(),
                deleted: false,
            },
        );
        Ok(())
    }

    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        let entry = self.map.get(key)?;
        let entry = entry.value();
        if entry.deleted {
            None
        } else {
            Some(entry.value.clone())
        }
    }

    /// Looks up `key` returning the full entry, so a caller can distinguish a
    /// tombstone from an absent key. `None` means absent; a returned entry with
    /// `deleted == true` is a tombstone that shadows older data (the engine's
    /// lookup chain stops there instead of falling through to disk).
    pub fn get_entry(&self, key: &[u8]) -> Option<Entry> {
        let entry = self.map.get(key)?;
        Some(entry.value().clone())
    }

    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.bump_approximate_size(key.len());
        self.map.insert(
            key.to_vec(),
            Entry {
                value: Vec::new(),
                deleted: true,
            },
        );
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn approximate_size(&self) -> usize {
        self.approximate_size.load(Ordering::Relaxed)
    }

    /// Iterates entries in ascending key order, yielding owned clones.
    ///
    /// The underlying `SkipMap` iterator is **weakly consistent** (not a
    /// snapshot): iterating a live table while writers are in flight may miss or
    /// duplicate entries. Callers that need a stable view must first freeze the
    /// table (swap it to immutable under a lock) so no writers remain — this is
    /// the contract flush (F3/F5) relies on.
    pub fn iter(&self) -> impl Iterator<Item = (Vec<u8>, Entry)> + '_ {
        self.map.iter().map(|e| {
            let (key, entry) = (e.key(), e.value());
            (key.clone(), entry.clone())
        })
    }

    fn bump_approximate_size(&self, payload_bytes: usize) {
        self.approximate_size
            .fetch_add(payload_bytes + ENTRY_OVERHEAD, Ordering::Relaxed);
    }
}

impl Default for MemTable {
    fn default() -> Self {
        MemTable::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_returns_value() {
        let table = MemTable::new();
        table.put(b"key1", b"v1").unwrap();
        assert_eq!(table.get(b"key1"), Some(b"v1".to_vec()));
    }

    #[test]
    fn overwrite_same_key_keeps_latest_value() {
        let table = MemTable::new();
        table.put(b"key", b"a").unwrap();
        table.put(b"key", b"b").unwrap();
        assert_eq!(table.get(b"key"), Some(b"b".to_vec()));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn get_missing_key_returns_none() {
        let table = MemTable::new();
        assert_eq!(table.get(b"absent"), None);
    }

    #[test]
    fn delete_existing_key_makes_get_none() {
        let table = MemTable::new();
        table.put(b"key", b"v").unwrap();
        table.delete(b"key").unwrap();
        assert_eq!(table.get(b"key"), None);
    }

    #[test]
    fn delete_missing_key_is_ok_and_get_none() {
        let table = MemTable::new();
        table.delete(b"never-existed").unwrap();
        assert_eq!(table.get(b"never-existed"), None);
    }

    #[test]
    fn put_after_delete_revives_key() {
        let table = MemTable::new();
        table.put(b"key", b"v").unwrap();
        table.delete(b"key").unwrap();
        table.put(b"key", b"v2").unwrap();
        assert_eq!(table.get(b"key"), Some(b"v2".to_vec()));
    }

    #[test]
    fn empty_table_is_empty() {
        let table = MemTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        table.put(b"key", b"v").unwrap();
        assert!(!table.is_empty());
    }

    #[test]
    fn approximate_size_grows_on_put() {
        let table = MemTable::new();
        for i in 0..10 {
            let key = format!("key-{i}");
            let value = format!("val-{i}");
            table.put(key.as_bytes(), value.as_bytes()).unwrap();
        }
        let size = table.approximate_size();
        assert!(size > 0);
        assert!(size >= 10 * (5 + 5), "size {size} too small");
    }

    #[test]
    fn iter_yields_sorted_entries_including_tombstones() {
        let table = MemTable::new();
        table.put(b"b", b"bv").unwrap();
        table.put(b"a", b"av").unwrap();
        table.delete(b"c").unwrap();

        let entries: Vec<(Vec<u8>, Entry)> = table.iter().collect();
        let keys: Vec<&[u8]> = entries.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(keys, [b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]);

        assert_eq!(entries[0].1.value, b"av");
        assert!(!entries[0].1.deleted);
        assert_eq!(entries[1].1.value, b"bv");
        assert!(!entries[1].1.deleted);
        assert!(entries[2].1.deleted, "tombstone must be visible in iter");
    }

    #[test]
    fn iter_skips_nothing() {
        let table = MemTable::new();
        table.put(b"k1", b"v1").unwrap();
        table.put(b"k2", b"v2").unwrap();
        table.delete(b"k3").unwrap();
        let collected: Vec<(Vec<u8>, Entry)> = table.iter().collect();
        assert_eq!(collected.len(), table.len());
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn concurrent_writes_do_not_corrupt() {
        let table = std::sync::Arc::new(MemTable::new());
        let mut handles = Vec::new();
        for thread_id in 0..8 {
            let table = std::sync::Arc::clone(&table);
            handles.push(std::thread::spawn(move || {
                for i in 0..1000 {
                    let key = format!("k-{thread_id}-{i}");
                    let value = format!("v-{i}");
                    table.put(key.as_bytes(), value.as_bytes()).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(table.len(), 8000);
        for thread_id in 0..8 {
            for i in 0..1000 {
                let key = format!("k-{thread_id}-{i}");
                let expected = format!("v-{i}");
                assert_eq!(table.get(key.as_bytes()), Some(expected.into_bytes()));
            }
        }
    }

    #[test]
    fn empty_key_and_value_are_valid() {
        let table = MemTable::new();
        table.put(b"", b"").unwrap();
        assert_eq!(table.get(b""), Some(b"".to_vec()));
    }

    #[test]
    fn get_entry_distinguishes_present_tombstone_absent() {
        let table = MemTable::new();
        table.put(b"live", b"v").unwrap();
        table.put(b"gone", b"old").unwrap();
        table.delete(b"gone").unwrap();

        let live = table
            .get_entry(b"live")
            .expect("present key must yield an entry");
        assert!(!live.deleted);
        assert_eq!(live.value, b"v");

        let gone = table
            .get_entry(b"gone")
            .expect("tombstoned key must yield an entry");
        assert!(gone.deleted, "tombstone must be visible via get_entry");

        assert!(table.get_entry(b"absent").is_none());
    }
}
