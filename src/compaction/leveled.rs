//! Leveled-compaction primitives. The k-way merge here is the heart of every
//! compaction: it folds several sorted runs into one ascending, key-unique,
//! tombstone-free output. All inputs come from `SsTableReader::records()` and
//! the merged output is written to a fresh table by the engine driver.

use crate::error::Result;
use crate::sstable::reader::Record;

/// One sorted run: the records of a single source SSTable. `seq` is the
/// table's file sequence number — a HIGHER seq is NEWER and wins when keys
/// collide across runs.
pub(crate) struct Run {
    pub seq: u64,
    pub records: Vec<Record>,
}

/// k-way merges `runs` into a single ascending, key-unique result.
///
/// Semantics (compaction of the WHOLE live set — see plan D-F6.1/2):
/// - For a key present in multiple runs, the record from the run with the
///   LARGEST `seq` wins (newest data).
/// - If the winning record is a tombstone (`deleted == true`), the key is
///   dropped entirely (final delete: no older data survives outside the merged
///   input, so a tombstone need not be carried forward).
/// - Otherwise the key is emitted once as `(key, value, false)`.
/// - `deleted` is only ever `false` in the output (tombstones are consumed).
///
/// Runs may be empty (`records.is_empty()` → ignored) and `runs` itself may be
/// empty → `Ok(vec![])`. Duplicate `seq` values are a caller error (engine
/// guarantees unique file seqs) — you may `debug_assert` this.
pub(crate) fn merge_runs(runs: &[Run]) -> Result<Vec<Record>> {
    #[cfg(debug_assertions)]
    {
        let mut seqs: Vec<u64> = runs.iter().map(|r| r.seq).collect();
        seqs.sort_unstable();
        debug_assert!(
            seqs.windows(2).all(|w| w[0] != w[1]),
            "merge_runs: duplicate run seqs in input: {seqs:?}"
        );
    }

    let active: Vec<usize> = (0..runs.len())
        .filter(|&i| !runs[i].records.is_empty())
        .collect();
    let mut cursor = vec![0usize; runs.len()];
    let mut out: Vec<Record> = Vec::with_capacity(runs.iter().map(|r| r.records.len()).sum());

    loop {
        // Smallest current key among the runs that still have records.
        let mut min_key: Option<&[u8]> = None;
        for &i in &active {
            if cursor[i] >= runs[i].records.len() {
                continue;
            }
            let key = runs[i].records[cursor[i]].0.as_slice();
            if min_key.is_none_or(|mk| key < mk) {
                min_key = Some(key);
            }
        }
        let Some(min_key) = min_key else { break };

        // Every run whose current key equals `min_key` consumes it now (its
        // records are ascending and key-unique, so it cannot appear again);
        // the record from the highest-seq such run is the winner.
        let mut winner_seq: Option<u64> = None;
        let mut winner: Option<&Record> = None;
        for &i in &active {
            if cursor[i] >= runs[i].records.len() {
                continue;
            }
            let record = &runs[i].records[cursor[i]];
            if record.0.as_slice() != min_key {
                continue;
            }
            cursor[i] += 1;
            let seq = runs[i].seq;
            if winner_seq.is_none_or(|best| seq > best) {
                winner_seq = Some(seq);
                winner = Some(record);
            }
        }

        let winner = winner.expect("min_key must belong to at least one active run");
        if !winner.2 {
            out.push((winner.0.clone(), winner.1.clone(), false));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(seq: u64, pairs: &[(&[u8], Option<&[u8]>)]) -> Run {
        let records = pairs
            .iter()
            .map(|(key, value)| match value {
                Some(v) => (key.to_vec(), v.to_vec(), false),
                None => (key.to_vec(), Vec::new(), true),
            })
            .collect();
        Run { seq, records }
    }

    fn put(key: &[u8], value: &[u8]) -> Record {
        (key.to_vec(), value.to_vec(), false)
    }

    fn key(i: usize) -> Vec<u8> {
        format!("k{i:02}").into_bytes()
    }

    fn key_val(i: usize, seq: u64) -> Vec<u8> {
        format!("v{seq}-{i:02}").into_bytes()
    }

    #[test]
    fn single_run_passthru() {
        let runs = vec![run(1, &[(b"a", Some(b"1")), (b"b", Some(b"2"))])];
        assert_eq!(
            merge_runs(&runs).unwrap(),
            vec![put(b"a", b"1"), put(b"b", b"2")]
        );
    }

    #[test]
    fn single_run_tombstone_dropped() {
        let runs = vec![run(1, &[(b"a", Some(b"1")), (b"b", None)])];
        assert_eq!(merge_runs(&runs).unwrap(), vec![put(b"a", b"1")]);
    }

    #[test]
    fn overlap_newest_wins() {
        let runs = vec![
            run(1, &[(b"a", Some(b"old")), (b"b", Some(b"old"))]),
            run(2, &[(b"a", Some(b"new"))]),
        ];
        assert_eq!(
            merge_runs(&runs).unwrap(),
            vec![put(b"a", b"new"), put(b"b", b"old")]
        );
    }

    #[test]
    fn three_runs_newest_seq_wins() {
        let runs = vec![
            run(1, &[(b"k", Some(b"v1"))]),
            run(5, &[(b"k", Some(b"v5"))]),
            run(3, &[(b"k", Some(b"v3"))]),
        ];
        assert_eq!(merge_runs(&runs).unwrap(), vec![put(b"k", b"v5")]);
    }

    #[test]
    fn old_tombstone_beaten_by_newer_value() {
        let runs = vec![run(1, &[(b"k", None)]), run(2, &[(b"k", Some(b"v2"))])];
        assert_eq!(merge_runs(&runs).unwrap(), vec![put(b"k", b"v2")]);
    }

    #[test]
    fn newest_tombstone_drops_key() {
        let runs = vec![run(1, &[(b"k", Some(b"v1"))]), run(2, &[(b"k", None)])];
        assert!(merge_runs(&runs).unwrap().is_empty());
    }

    #[test]
    fn ascending_and_unique() {
        let runs = vec![
            Run {
                seq: 1,
                records: (0..50).map(|i| (key(i), key_val(i, 1), false)).collect(),
            },
            Run {
                seq: 2,
                records: (25..50).map(|i| (key(i), key_val(i, 2), false)).collect(),
            },
            Run {
                seq: 3,
                records: (0..40).map(|i| (key(i), key_val(i, 3), false)).collect(),
            },
            Run {
                seq: 4,
                records: (10..50).map(|i| (key(i), key_val(i, 4), false)).collect(),
            },
        ];

        let out = merge_runs(&runs).unwrap();

        // Independent oracle: replay the runs in ascending-seq order; each write
        // overwrites the last, so the highest-seq record per key survives.
        let mut expected: std::collections::BTreeMap<Vec<u8>, Vec<u8>> =
            std::collections::BTreeMap::new();
        let mut by_seq: Vec<&Run> = runs.iter().collect();
        by_seq.sort_by_key(|r| r.seq);
        for r in by_seq {
            for (k, v, deleted) in &r.records {
                assert!(!deleted, "no tombstones in this test");
                expected.insert(k.clone(), v.clone());
            }
        }
        let expected: Vec<Record> = expected.into_iter().map(|(k, v)| (k, v, false)).collect();

        assert_eq!(out, expected);
        for w in out.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "keys must be strictly ascending and unique"
            );
        }
    }

    #[test]
    fn empty_value_put_is_not_tombstone() {
        // The empty-value put is NEWER (seq 2) than the tombstone (seq 1) and
        // must win and be emitted as a real (empty) value — never read as a
        // deletion because its payload is empty.
        let runs = vec![run(1, &[(b"k", None)]), run(2, &[(b"k", Some(b""))])];
        assert_eq!(merge_runs(&runs).unwrap(), vec![put(b"k", b"")]);
    }

    #[test]
    fn empty_runs_and_empty_input() {
        assert!(merge_runs(&[]).unwrap().is_empty());

        let e1 = Run {
            seq: 1,
            records: Vec::new(),
        };
        let e2 = Run {
            seq: 3,
            records: Vec::new(),
        };
        assert!(merge_runs(&[e1]).unwrap().is_empty());

        let a = run(2, &[(b"a", Some(b"1"))]);
        let b = run(4, &[(b"b", Some(b"2"))]);
        assert_eq!(
            merge_runs(&[e2, a, b]).unwrap(),
            vec![put(b"a", b"1"), put(b"b", b"2")]
        );
    }

    #[test]
    fn mixed_keys_and_runs() {
        let runs = vec![
            run(
                1,
                &[
                    (b"a", Some(b"1")),
                    (b"c", Some(b"3")),
                    (b"e", Some(b"5")),
                    (b"g", Some(b"7")),
                ],
            ),
            run(
                2,
                &[
                    (b"b", Some(b"2")),
                    (b"c", Some(b"C")),
                    (b"d", Some(b"4")),
                    (b"f", Some(b"6")),
                    (b"g", Some(b"G")),
                ],
            ),
        ];
        assert_eq!(
            merge_runs(&runs).unwrap(),
            vec![
                put(b"a", b"1"),
                put(b"b", b"2"),
                put(b"c", b"C"),
                put(b"d", b"4"),
                put(b"e", b"5"),
                put(b"f", b"6"),
                put(b"g", b"G"),
            ]
        );
    }

    #[test]
    fn empty_key_is_valid_and_orders_first() {
        // An empty key sorts before every other key and is a legitimate value.
        let runs = vec![
            run(1, &[(b"", Some(b"first")), (b"a", Some(b"1"))]),
            run(2, &[(b"", Some(b"newer")), (b"b", Some(b"2"))]),
        ];
        assert_eq!(
            merge_runs(&runs).unwrap(),
            vec![put(b"", b"newer"), put(b"a", b"1"), put(b"b", b"2")]
        );
    }

    #[test]
    fn byte_prefix_keys_merge_correctly() {
        // Keys sharing a prefix across runs must interleave byte-wise, not
        // lexically-by-shared-prefix.
        let runs = vec![
            run(
                1,
                &[(b"a", Some(b"1")), (b"aa", Some(b"2")), (b"ab", Some(b"3"))],
            ),
            run(2, &[(b"a", Some(b"X")), (b"aaa", Some(b"4"))]),
        ];
        assert_eq!(
            merge_runs(&runs).unwrap(),
            vec![
                put(b"a", b"X"), // a < aa < aaa < ab in byte order
                put(b"aa", b"2"),
                put(b"aaa", b"4"),
                put(b"ab", b"3"),
            ]
        );
    }
}
