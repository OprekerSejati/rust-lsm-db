//! End-to-end integration tests for `LsmEngine`, driven through the public API
//! only (this crate cannot see `pub(crate)` internals). The focus is
//! consistency under concurrent multi-threaded workloads: puts racing reads,
//! contended overwrites, deletes that must not resurrect flushed data,
//! flush/compact running while a writer is hot, and crash recovery across
//! drops and reopens.
//!
//! Concurrency is synchronized with thread joins and atomics only — no sleeps,
//! so the suite stays fast and has no fixed timing points. Live readers racing
//! writers are inherently racy by design; the load-bearing assertions run after
//! joins/reopens where the final state is deterministic.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lsm_engine::{LsmEngine, LsmOptions};
use tempfile::TempDir;

/// Small thresholds so flushes and auto-compaction fire *during* the load
/// instead of being deferred to an explicit call.
fn small_opts() -> LsmOptions {
    LsmOptions {
        flush_threshold_bytes: 4 * 1024, // force flushes during load
        compaction_l0_file_threshold: 2, // force auto-compaction during load
        sync_per_write: false,
    }
}

/// Effectively disable auto-freeze so data stays in the active MemTable / WAL
/// (used where a test wants to control every flush explicitly).
fn no_auto_flush_opts() -> LsmOptions {
    LsmOptions {
        flush_threshold_bytes: usize::MAX,
        compaction_l0_file_threshold: 0,
        sync_per_write: false,
    }
}

fn open_engine(dir: &TempDir, opts: LsmOptions) -> LsmEngine {
    LsmEngine::open_with_options(dir.path(), opts).expect("open engine")
}

/// Number of SSTable files (L0 or L1) currently on disk.
fn count_sst(dir: &TempDir) -> usize {
    std::fs::read_dir(dir.path())
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name();
            let n = n.to_string_lossy();
            n.starts_with("L0_") || n.starts_with("L1_")
        })
        .count()
}

/// The public surface exercises key + value sizes that are tiny relative to the
/// engine limits; all keys below are short ASCII so they are trivially ordered.
#[test]
fn concurrent_put_get_consistency() {
    // 4 writers over disjoint ranges while 4 readers poll the whole range.
    // Every key has exactly one writer, so any observed value must be that
    // writer's exact value — never a stale or partial one.
    const WRITERS: usize = 4;
    const PER_WRITER: usize = 5_000;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(open_engine(&dir, small_opts()));

    let stop = Arc::new(AtomicBool::new(false));

    let writer_handles: Vec<_> = (0..WRITERS)
        .map(|t| {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                for i in 0..PER_WRITER {
                    let key = format!("w{t}-{i:05}");
                    let val = format!("v{t}");
                    engine
                        .put(key.as_bytes(), val.as_bytes())
                        .expect("put must not fail");
                }
            })
        })
        .collect();

    let keyspace: Vec<String> = (0..WRITERS)
        .flat_map(|t| (0..PER_WRITER).map(move |i| format!("w{t}-{i:05}")))
        .collect();
    let expected_value = |key: &str| -> Vec<u8> {
        let t = key.as_bytes()[1] - b'0';
        format!("v{t}").into_bytes()
    };
    let keyspace: Arc<Vec<String>> = Arc::new(keyspace);

    let reader_handles: Vec<_> = (0..WRITERS)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let stop = Arc::clone(&stop);
            let keyspace = Arc::clone(&keyspace);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for key in keyspace.iter() {
                        if let Some(got) = engine
                            .get(key.as_bytes())
                            .expect("reader get must not fail")
                        {
                            let expected = expected_value(key);
                            assert_eq!(
                                got, expected,
                                "reader saw a value that no writer ever wrote for {key}"
                            );
                        }
                    }
                }
            })
        })
        .collect();

    for h in writer_handles {
        h.join().expect("writer must not panic");
    }
    stop.store(true, Ordering::Relaxed);
    for h in reader_handles {
        h.join().expect("reader must not panic");
    }

    // Converge everything to SSTs, then verify every key from disk (SST + bloom
    // lookup path). Auto-flush + auto-compaction already wrote many files.
    engine.flush().expect("final flush");
    assert!(count_sst(&dir) > 0, "load must have produced SSTables");
    for key in keyspace.iter() {
        let expected = expected_value(key);
        assert_eq!(
            engine.get(key.as_bytes()).expect("post flush get").unwrap(),
            expected,
            "key {key} must read back its exact writer value"
        );
    }
}

#[test]
fn concurrent_overwrite_last_writer_wins() {
    // 4 threads hammer the SAME keys concurrently; a read (any point in time)
    // must always observe one of the 4 complete values — never a missing,
    // torn, or interleaved value. Final value is any of the 4 (the real last
    // writer is racy by design). An explicit flush/compact loop runs alongside
    // so overwritten generations also move across memtable -> L0 -> L1.
    const THREADS: usize = 4;
    const KEYS: usize = 100;
    const ITERS: usize = 150;
    const FLUSH_EVERY: usize = 50;
    // Writer values are `t.to_string()` for t in 0..THREADS → single digits
    // '0'..'0'+(THREADS-1). Derived so bumping THREADS can't silently weaken
    // the "complete writer value" check below.
    const MAX_WRITER_DIGIT: u8 = b'0' + (THREADS - 1) as u8;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(open_engine(&dir, small_opts()));

    let keys: Vec<String> = (0..KEYS).map(|i| format!("k{i}")).collect();
    let done = Arc::new(AtomicBool::new(false));

    let mut writers = Vec::new();
    for t in 0..THREADS {
        let engine = Arc::clone(&engine);
        let keys = keys.clone();
        let val = t.to_string();
        let done = Arc::clone(&done);
        writers.push(std::thread::spawn(move || {
            for it in 0..ITERS {
                for key in &keys {
                    engine
                        .put(key.as_bytes(), val.as_bytes())
                        .expect("contended put must not fail");
                }
                if (it + 1) % FLUSH_EVERY == 0 {
                    // Move the current generation to disk mid-load.
                    engine.flush().expect("flush during overwrite load");
                    engine.compact().expect("compact during overwrite load");
                }
            }
            done.store(true, Ordering::SeqCst);
        }));
    }

    // A reader racing the overwriters: every observed value must be a complete
    // value from one of the 4 writers.
    let reader_engine = Arc::clone(&engine);
    let reader_keys = keys.clone();
    let reader = std::thread::spawn(move || {
        while !done.load(Ordering::SeqCst) {
            for key in &reader_keys {
                if let Some(v) = reader_engine
                    .get(key.as_bytes())
                    .expect("reader get must not fail")
                {
                    let ok = v.len() == 1 && v[0].is_ascii_digit() && v[0] <= MAX_WRITER_DIGIT;
                    assert!(
                        ok,
                        "reader observed a torn/interleaved value for {key}: {v:?}"
                    );
                }
            }
        }
    });
    for h in writers {
        h.join().expect("writer must not panic");
    }
    reader.join().expect("reader must not panic");

    engine.flush().expect("final flush");
    for key in &keys {
        let v = engine
            .get(key.as_bytes())
            .expect("post flush get")
            .expect("contended key must exist after a completed write");
        assert!(
            v.len() == 1 && v[0].is_ascii_digit() && v[0] <= MAX_WRITER_DIGIT,
            "key {key} must hold one complete writer value, got {v:?}"
        );
    }
}

#[test]
fn delete_under_load_no_resurrect() {
    // Seed values flushed to an SST, then a delete + re-put "load" happens with
    // a reader polling live. The reader may observe None or the fresh value but
    // never the stale seeded value. After converge + drop + reopen, deleted
    // keys must be gone and re-put keys must hold their fresh value: a flushed
    // tombstone must not let the old SST value resurrect.
    let keys: Vec<String> = (0..7).map(|i| format!("a{i}")).collect();
    const FRESH: &[u8] = b"fresh";
    let seed = |e: &LsmEngine| {
        for (i, k) in keys.iter().enumerate() {
            e.put(k.as_bytes(), format!("seed{i}").as_bytes())
                .expect("seed put");
        }
        e.flush().expect("seed flush"); // seed values now in an SST
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let stop = Arc::new(AtomicBool::new(false));
    // Set by the reader if it ever observes a resurrected seed value on a key
    // that must stay deleted; asserted after the scope so a violation surfaces
    // as a clean failure instead of a panic that hangs the thread scope.
    let resurrected = Arc::new(AtomicBool::new(false));

    let engine = open_engine(&dir, no_auto_flush_opts());
    seed(&engine);

    // Phase A (main): delete a0..a4, flush -> tombstones on disk. Re-puts of
    // a2 land later (phase B) so a2 exercises delete-then-reput across two
    // generations, while a0/a1/a3/a4 stay deleted.
    for k in &keys[0..5] {
        engine.delete(k.as_bytes()).expect("delete");
    }
    engine.flush().expect("flush tombstones to SST");

    std::thread::scope(|s| {
        // Phase B (concurrent): deleter drops a3,a4 (already tombstoned in the
        // SST) while the putter re-puts a2,a5,a6 with FRESH. A reader polls the
        // whole key set for the whole phase. The scope's implicit join is the
        // only synchronization needed: the engine lives beyond the scope, so
        // the reader can be stopped by flag and joined here.
        let engine = &engine;
        let keys = &keys;
        let _deleter = s.spawn(move || {
            for k in &keys[3..5] {
                engine.delete(k.as_bytes()).expect("concurrent delete");
            }
        });
        let _putter = s.spawn(move || {
            for k in [&keys[2], &keys[5], &keys[6]] {
                engine.put(k.as_bytes(), FRESH).expect("concurrent put");
            }
        });
        let _reader = {
            let stop = Arc::clone(&stop);
            let resurrected = Arc::clone(&resurrected);
            s.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    for (i, k) in keys.iter().enumerate() {
                        // a0..a4 are tombstoned and must never show a seeded
                        // value again (resurrection). a5/a6 were never deleted
                        // in phase A, so their seed value is legitimate until
                        // the putter overwrites it — leave them unconstrained
                        // during the race.
                        if i < 5
                            && let Some(v) = engine.get(k.as_bytes()).expect("live get")
                            && v != FRESH
                        {
                            resurrected.store(true, Ordering::Relaxed);
                        }
                    }
                }
            })
        };

        stop.store(true, Ordering::Relaxed);
    }); // scope joins all three threads

    assert!(
        !resurrected.load(Ordering::Relaxed),
        "live reads observed a tombstoned key come back with a stale value"
    );

    // Converge + drop + reopen on the same directory.
    engine.flush().expect("final flush");
    drop(engine);
    let reopened = open_engine(&dir, no_auto_flush_opts());
    for (i, k) in keys.iter().enumerate() {
        let got = reopened.get(k.as_bytes()).expect("reopen get");
        match i {
            0 | 1 | 3 | 4 => assert_eq!(got, None, "deleted key {k} resurrected"),
            _ => {
                assert_eq!(got.as_deref(), Some(FRESH), "re-put key {k} lost its value")
            }
        }
    }
}

#[test]
fn flush_and_compact_run_during_load() {
    // A hot writer pours unique keys in while another thread repeatedly runs
    // flush()+compact() over the same live engine. The race the plan cares
    // about is flush/compact committing against concurrent writes; we converge
    // with a final flush and verify every key from disk.
    const N_KEYS: usize = 20_000;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = Arc::new(open_engine(&dir, small_opts()));

    let writer_done = Arc::new(AtomicBool::new(false));

    let writer_writer_done = Arc::clone(&writer_done);
    let writer_engine = Arc::clone(&engine);
    let writer = std::thread::spawn(move || {
        for i in 0..N_KEYS {
            let key = format!("d{i:05}");
            writer_engine
                .put(key.as_bytes(), b"small-value")
                .expect("load put");
        }
        writer_writer_done.store(true, Ordering::SeqCst);
    });

    let flusher_engine = Arc::clone(&engine);
    let flusher_done = Arc::clone(&writer_done);
    let flusher = std::thread::spawn(move || {
        for _ in 0..6 {
            if flusher_done.load(Ordering::SeqCst) {
                break;
            }
            flusher_engine.flush().expect("concurrent flush");
            flusher_engine.compact().expect("concurrent compact");
        }
    });

    writer.join().expect("writer must not panic");
    flusher.join().expect("flusher must not panic");

    engine.flush().expect("final flush");
    for i in 0..N_KEYS {
        let key = format!("d{i:05}");
        assert_eq!(
            engine
                .get(key.as_bytes())
                .expect("post-load get")
                .as_deref(),
            Some(b"small-value".as_slice()),
            "key {key} lost after flush/compact raced live writes"
        );
    }
}

#[test]
fn crash_recovery_mid_load() {
    // Load past many auto-flush generations (bulk recovered from SSTs +
    // MANIFEST), then write a small unflushed tail whose ONLY durable copy is
    // the active WAL (recovered by WAL replay on reopen). Drop fsyncs that WAL
    // (clean drop); reopen must stitch both halves back together.
    const N_KEYS: usize = 10_000;
    const TAIL: usize = 20; // well under the 4KiB auto-freeze threshold

    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(&dir, small_opts());
        for i in 0..N_KEYS {
            let key = format!("r{i:05}");
            engine
                .put(key.as_bytes(), format!("v{i}").as_bytes())
                .expect("load put");
        }
        // Converge the auto-flushed generations to disk, then write TAIL keys
        // that stay in the active WAL (active MemTable is far below the flush
        // threshold, so no background flush can be in flight at drop).
        engine.flush().expect("flush before drop");
        for i in 0..TAIL {
            let key = format!("t{i:05}");
            engine
                .put(key.as_bytes(), format!("tail{i}").as_bytes())
                .expect("tail put");
        }
    } // engine dropped here: active WAL fsynced

    assert!(
        count_sst(&dir) > 0,
        "load must have flushed generations to SST"
    );

    let engine = open_engine(&dir, small_opts());
    for i in 0..N_KEYS {
        let key = format!("r{i:05}");
        let expect = format!("v{i}");
        assert_eq!(
            engine.get(key.as_bytes()).expect("reopen get").as_deref(),
            Some(expect.as_bytes()),
            "key {key} must survive drop + reopen (SST + MANIFEST path)"
        );
    }
    for i in 0..TAIL {
        let key = format!("t{i:05}");
        let expect = format!("tail{i}");
        assert_eq!(
            engine.get(key.as_bytes()).expect("tail get").as_deref(),
            Some(expect.as_bytes()),
            "tail key {key} must survive via WAL replay"
        );
    }
}

#[test]
fn sync_per_write_durability() {
    // With sync_per_write, every put is fsynced to the WAL before returning.
    // Auto-flush is disabled, so after a drop (no flush) the ONLY durable copy
    // is the WAL; reopening must replay all 500 puts.
    const N_KEYS: usize = 500;

    let dir = tempfile::tempdir().expect("tempdir");
    {
        let engine = open_engine(
            &dir,
            LsmOptions {
                flush_threshold_bytes: usize::MAX, // keep everything in the WAL
                compaction_l0_file_threshold: 0,
                sync_per_write: true,
            },
        );
        for i in 0..N_KEYS {
            let key = format!("s{i:05}");
            engine
                .put(key.as_bytes(), format!("sync{i}").as_bytes())
                .expect("synced put");
        }
    } // drop without flush

    assert_eq!(count_sst(&dir), 0, "nothing may have flushed to SST");

    let engine = open_engine(&dir, no_auto_flush_opts());
    for i in 0..N_KEYS {
        let key = format!("s{i:05}");
        let expect = format!("sync{i}");
        assert_eq!(
            engine.get(key.as_bytes()).expect("reopen get").as_deref(),
            Some(expect.as_bytes()),
            "key {key} must be durable via per-write WAL fsync"
        );
    }
}
