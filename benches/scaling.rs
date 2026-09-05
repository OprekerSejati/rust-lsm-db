//! Multi-threaded scaling + value-size throughput harness.
//!
//! Criterion measures one closure serially, which cannot express concurrency,
//! so this is a plain `harness = false` binary: it runs N worker threads
//! against a shared engine, measures wall-clock time, and prints a CSV of
//! ops/sec to stdout for the chart script to consume.
//!
//! Run: `cargo bench --bench scaling` (release build).
//!
//! Output rows are `metric,param,ops_per_sec`:
//!   put_threads,<threads>,...   puts/sec with that many writer threads
//!   get_threads,<threads>,...   gets/sec with that many reader threads (L1 disk data)
//!   value_size,<bytes>,...      puts/sec single-threaded at that value size
//!
//! Methodology notes (honesty): the engine serializes writers on a single WAL
//! mutex, and all writers share one SkipMap, so put throughput does NOT scale
//! with threads — contention on that lock/map can even make aggregate put
//! throughput decline as writers pile up. The read path holds no global lock
//! (shared Arc readers + bloom short-circuit), so get throughput should grow
//! with reader threads. Keys and values are pre-generated outside the timed
//! loops so only engine work is measured. Numbers are machine- and
//! page-cache-dependent and only meaningful relative to each other.

use std::hint::black_box;
use std::time::Instant;

use lsm_engine::{LsmEngine, LsmOptions};

const THREADS: [usize; 4] = [1, 2, 4, 8];
const BURSTS: usize = 5;
const KEY_VALUE_LEN: usize = 128;
const PUT_OPS_PER_THREAD: usize = 10_000;
const GET_OPS_PER_THREAD: usize = 50_000;
const DISK_KEYS: usize = 100_000;
const VALUE_SIZES: [usize; 4] = [16, 128, 1024, 4096];
const VALUE_PUT_OPS: usize = 5_000;

/// Keep everything in the active memtable (no auto-flush) so put scaling
/// measures the pure WAL + memtable path and value-size puts never flush
/// mid-measurement.
fn no_flush_opts() -> LsmOptions {
    LsmOptions {
        flush_threshold_bytes: usize::MAX,
        compaction_l0_file_threshold: 0,
        sync_per_write: false,
    }
}

fn key(i: u64) -> Vec<u8> {
    format!("k{i}").into_bytes()
}

/// Runs `threads` workers, each performing `ops_per_thread` calls of `op`
/// against `engine`; returns the best ops/sec across `BURSTS` timed windows.
/// `keys` is a pre-generated pool; worker `t` owns the disjoint slice
/// `[t*ops_per_thread .. (t+1)*ops_per_thread)`. `op` receives the engine and
/// one key buffer, so no key formatting happens inside the timed region.
fn run_bursts(
    engine: &LsmEngine,
    threads: usize,
    ops_per_thread: usize,
    keys: &[Vec<u8>],
    op: impl Fn(&LsmEngine, &[u8]) + Sync + Send,
) -> f64 {
    let mut best = f64::MAX;
    let op = &op;
    for _ in 0..BURSTS {
        let start = Instant::now();
        std::thread::scope(|s| {
            for t in 0..threads {
                let engine = &*engine;
                let keys = &keys[t * ops_per_thread..(t + 1) * ops_per_thread];
                s.spawn(move || {
                    for k in keys {
                        op(engine, k);
                    }
                });
            }
        });
        let elapsed = start.elapsed().as_secs_f64();
        let total_ops = (threads * ops_per_thread) as f64;
        best = best.min(total_ops / elapsed);
    }
    best
}

/// Opens a fresh engine (its tempdir is returned and kept alive by the caller).
fn open_engine() -> (tempfile::TempDir, LsmEngine) {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = LsmEngine::open_with_options(dir.path(), no_flush_opts()).expect("open");
    (dir, engine)
}

fn main() {
    println!("metric,param,ops_per_sec");

    // --- put throughput vs writer threads ---
    // Writers use disjoint keyspaces (worker t owns keys t*N..(t+1)*N) so we
    // measure insert throughput, not overwrite/contention on the same keys.
    let (_dir, engine) = open_engine();
    let max_threads = *THREADS.iter().max().unwrap();
    let put_keys: Vec<Vec<u8>> = (0..(max_threads * PUT_OPS_PER_THREAD) as u64)
        .map(key)
        .collect();
    for &t in &THREADS {
        let ops = run_bursts(&engine, t, PUT_OPS_PER_THREAD, &put_keys, |engine, k| {
            engine.put(k, &[b'x'; KEY_VALUE_LEN]).expect("put");
        });
        println!("put_threads,{t},{ops:.0}");
        eprintln!("put_threads {t}: {ops:.0} ops/s");
    }
    drop(engine);
    drop(_dir);

    // --- get throughput vs reader threads (data compacted to L1) ---
    let (_dir, engine) = open_engine();
    let disk_keys: Vec<Vec<u8>> = (0..DISK_KEYS as u64).map(key).collect();
    let v = vec![b'x'; KEY_VALUE_LEN];
    for k in &disk_keys {
        engine.put(k, &v).expect("seed put");
    }
    engine.flush().expect("flush seeds to L0");
    engine.compact().expect("compact seeds to L1"); // single table, all on disk
    // Warm the reader cache for a key the workers will hit.
    assert!(engine.get(&disk_keys[0]).expect("get").is_some());

    for &t in &THREADS {
        // Readers need threads*ops distinct key slots for run_bursts' disjoint
        // windows; cycle the on-disk pool (reads are value-insensitive, so
        // repeats across slots are fine).
        let get_pool: Vec<Vec<u8>> = (0..(t * GET_OPS_PER_THREAD) as u64)
            .map(|i| disk_keys[(i % DISK_KEYS as u64) as usize].clone())
            .collect();
        let ops = run_bursts(&engine, t, GET_OPS_PER_THREAD, &get_pool, |engine, k| {
            black_box(engine.get(k).expect("get"));
        });
        println!("get_threads,{t},{ops:.0}");
        eprintln!("get_threads {t}: {ops:.0} ops/s");
    }
    drop(engine);
    drop(_dir);

    // --- put throughput vs value size (single thread, memtable) ---
    let value_keys: Vec<Vec<u8>> = (0..VALUE_PUT_OPS as u64).map(key).collect();
    for &vlen in &VALUE_SIZES {
        let (_dir, engine) = open_engine();
        let v = vec![b'x'; vlen];
        let ops = run_bursts(&engine, 1, VALUE_PUT_OPS, &value_keys, |engine, k| {
            engine.put(k, &v).expect("put");
        });
        println!("value_size,{vlen},{ops:.0}");
        eprintln!("value_size {vlen}: {ops:.0} ops/s");
        drop(engine);
        drop(_dir);
    }
}
