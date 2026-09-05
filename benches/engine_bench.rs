//! Criterion benchmarks for the public LSM engine API (plan Fase-7 D-F7.3).
//!
//! Four scenarios measure engine-level throughput/latency for put and point
//! get, each against its own private `tempfile`-backed engine with
//! `sync_per_write: false` (the pure LSM write path, not per-op fsync).
//! Keys are 16 bytes ("k" + 15 zero-padded digits), values 128 bytes.
//!
//! Setup/population work happens outside the timed region. The put benchmark
//! uses `iter_batched` so every measured sample starts from a fresh, empty
//! engine: writes mutate memtable state and trigger flushes, so a reused
//! engine would not represent a steady per-write cost.
//!
//! Caveat on the put bench: `put()` freezes the memtable synchronously but the
//! flush + compaction I/O runs on the engine's background worker, so exactly
//! how much of that pipeline lands inside a given sample is scheduling-
//! dependent. The in-window work dominates the measured batch, so the number
//! is meaningful, but sample-to-sample variance (~±10%) can mask small
//! regressions in fine-grained comparisons.

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
};
use lsm_engine::{LsmEngine, LsmOptions};

const VALUE_LEN: usize = 128;
/// Small flush threshold for the put bench: at 16B key + 128B value the
/// memtable crosses it roughly every ~450 puts, so a 2000-put sample triggers
/// ~4 freezes; with the L0 threshold at 4 the background worker also
/// auto-compacts during the batch (see the caveat above on scheduling).
const PUT_FLUSH_THRESHOLD_BYTES: usize = 64 * 1024;
const PUTS_PER_SAMPLE: usize = 2_000;
const HOT_KEYS: usize = 10_000;
const DISK_KEYS: usize = 50_000;
const MISS_KEYS: usize = 10_000;

/// 16-byte key ("k" + 15 zero-padded digits). Zero padding keeps byte order
/// equal to numeric order and matches the plan's 16B key budget.
fn key(i: u64) -> Vec<u8> {
    format!("k{i:015}").into_bytes()
}

fn value() -> Vec<u8> {
    vec![b'x'; VALUE_LEN]
}

/// Common tuning for the read-only get groups: fewer, longer samples keep the
/// run short while stays stable enough for relative comparison.
fn tune_get_group(group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(1));
}

/// Options for the put bench: a small threshold so flush + auto-compaction
/// fire during the measured batch.
fn put_opts() -> LsmOptions {
    LsmOptions {
        flush_threshold_bytes: PUT_FLUSH_THRESHOLD_BYTES,
        compaction_l0_file_threshold: 4,
        sync_per_write: false,
    }
}

/// Options for the get benches: a threshold far above any populate size, so
/// data stays where the scenario puts it (active memtable, or on disk after an
/// explicit `flush()` + `compact()`).
fn get_opts() -> LsmOptions {
    LsmOptions {
        flush_threshold_bytes: 1 << 30,
        compaction_l0_file_threshold: 4,
        sync_per_write: false,
    }
}

/// Deterministic xorshift64 shuffle, so disk lookups see a pseudo-random
/// access pattern instead of a page-cache-friendly sequential scan.
fn shuffled_indices(len: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..len).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..len).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
    indices
}

/// Sequential puts, 2000 per sample, that accumulate past the flush threshold
/// so the background worker flushes memtables to L0 and auto-compacts to L1
/// mid-bench. Each sample starts from a fresh empty engine in its own
/// tempdir; the engine+dir are returned from the timed closure so their drop
/// (and tempdir cleanup) happens after criterion stops the timer.
fn bench_put_sequential(c: &mut Criterion) {
    let batch: Vec<(Vec<u8>, Vec<u8>)> = (0..PUTS_PER_SAMPLE as u64)
        .map(|i| (key(i), value()))
        .collect();

    let mut group = c.benchmark_group("put");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(PUTS_PER_SAMPLE as u64));
    group.bench_function("sequential_flush", |b| {
        b.iter_batched(
            || {
                let dir = tempfile::tempdir().expect("tempdir for put bench");
                let engine =
                    LsmEngine::open_with_options(dir.path(), put_opts()).expect("open engine");
                (engine, dir)
            },
            |(engine, dir)| {
                for (k, v) in &batch {
                    engine.put(k, v).expect("put");
                }
                (engine, dir)
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Random point gets served by the active memtable (10_000 keys populated, no
/// flush). The engine is read-only after setup, so one instance reused across
/// samples measures steady-state memtable lookups.
fn bench_get_hot_memtable(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..HOT_KEYS as u64).map(key).collect();
    let dir = tempfile::tempdir().expect("tempdir for hot get bench");
    let engine = LsmEngine::open_with_options(dir.path(), get_opts()).expect("open engine");
    let v = value();
    for k in &keys {
        engine.put(k, &v).expect("seed put");
    }
    assert!(engine.get(&keys[0]).expect("seed get").is_some());

    let order = shuffled_indices(keys.len());
    let mut pos = 0usize;
    let mut group = c.benchmark_group("get");
    tune_get_group(&mut group);
    group.bench_function("hot_memtable", |b| {
        b.iter(|| {
            let k = &keys[order[pos % order.len()]];
            pos = pos.wrapping_add(1);
            black_box(engine.get(k).expect("hot get"))
        });
    });
    group.finish();
}

/// Random point gets served from L1 SSTables: 50_000 keys populated, then an
/// explicit `flush()` + `compact()` pushes everything to a single L1 table, so
/// lookups exercise the disk-read path (bloom probe + index + one block).
/// NOTE: after warm-up the ~7MB table is resident in the OS page cache, so
/// this measures the L1 lookup path from cache — not cold-disk seek latency.
fn bench_get_disk_l1(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..DISK_KEYS as u64).map(key).collect();
    let dir = tempfile::tempdir().expect("tempdir for disk get bench");
    let engine = LsmEngine::open_with_options(dir.path(), get_opts()).expect("open engine");
    let v = value();
    for k in &keys {
        engine.put(k, &v).expect("seed put");
    }
    engine.flush().expect("flush seed data to L0");
    engine.compact().expect("compact seed data to L1");
    assert!(engine.get(&keys[0]).expect("seed get").is_some());

    let order = shuffled_indices(keys.len());
    let mut pos = 0usize;
    let mut group = c.benchmark_group("get");
    tune_get_group(&mut group);
    group.bench_function("disk_l1", |b| {
        b.iter(|| {
            let k = &keys[order[pos % order.len()]];
            pos = pos.wrapping_add(1);
            black_box(engine.get(k).expect("disk get"))
        });
    });
    group.finish();
}

/// Point gets for keys that do NOT exist: 10_000 even keys populated and
/// compacted to L1, probed with the (in-range, absent) odd keys so the bloom
/// filter / min-max skip path short-circuits before any data block read.
fn bench_get_miss(c: &mut Criterion) {
    let keys: Vec<Vec<u8>> = (0..MISS_KEYS as u64).map(|i| key(2 * i)).collect();
    let probes: Vec<Vec<u8>> = (0..MISS_KEYS as u64).map(|i| key(2 * i + 1)).collect();
    let dir = tempfile::tempdir().expect("tempdir for miss get bench");
    let engine = LsmEngine::open_with_options(dir.path(), get_opts()).expect("open engine");
    let v = value();
    for k in &keys {
        engine.put(k, &v).expect("seed put");
    }
    engine.flush().expect("flush seed data to L0");
    engine.compact().expect("compact seed data to L1");
    assert!(engine.get(&keys[0]).expect("seed get").is_some());
    // An in-range absent key: the min/max range-skip must NOT bail here (the
    // probe sits inside the table's range), so the bloom filter is what
    // short-circuits.
    assert!(engine.get(&probes[0]).expect("probe get").is_none());

    let order = shuffled_indices(probes.len());
    let mut pos = 0usize;
    let mut group = c.benchmark_group("get");
    tune_get_group(&mut group);
    group.bench_function("miss", |b| {
        b.iter(|| {
            let k = &probes[order[pos % order.len()]];
            pos = pos.wrapping_add(1);
            black_box(engine.get(k).expect("miss get"))
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_put_sequential,
    bench_get_hot_memtable,
    bench_get_disk_l1,
    bench_get_miss,
);
criterion_main!(benches);
