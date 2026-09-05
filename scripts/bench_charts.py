#!/usr/bin/env python3
"""Generate performance/architecture charts for the LSM engine.

Reads `charts/data/scaling.csv` (produced by `cargo bench --bench scaling`)
and writes:
  charts/thread_scaling.png   ops/s vs worker threads (put & get)
  charts/value_size.png       put ops/s vs value size
  charts/architecture.png     LSM pipeline diagram

Usage:
  python3 scripts/bench_charts.py            # use existing charts/data/scaling.csv
  python3 scripts/bench_charts.py --rerun    # re-run `cargo bench --bench scaling` first
"""

import argparse
import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO, "charts", "data", "scaling.csv")
OUT = os.path.join(REPO, "charts")


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def rerun_bench():
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    out = subprocess.run(
        ["cargo", "bench", "--bench", "scaling"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    # Keep only the CSV rows (skip criterion/compiler chatter).
    rows = [ln for ln in out.splitlines() if ln.startswith(("put_threads,", "get_threads,", "value_size,"))]
    with open(CSV_PATH, "w") as f:
        f.write("metric,param,ops_per_sec\n")
        f.write("\n".join(rows) + "\n")
    print(f"wrote {CSV_PATH}")


def styled():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 11,
        }
    )


def chart_thread_scaling(rows):
    put = {int(r["param"]): float(r["ops_per_sec"]) for r in rows if r["metric"] == "put_threads"}
    get = {int(r["param"]): float(r["ops_per_sec"]) for r in rows if r["metric"] == "get_threads"}
    xs = sorted(put)  # same thread set as get
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, [get[x] for x in xs], "-o", color="#1f77b4", label="get (disk L1, concurrent readers)", lw=2)
    ax.plot(xs, [put[x] for x in xs], "-s", color="#d62728", label="put (WAL-serialized writers)", lw=2)
    for x in xs:
        ax.annotate(f"{get[x]/1e6:.2f}M", (x, get[x]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#1f77b4")
        ax.annotate(f"{put[x]/1e6:.2f}M", (x, put[x]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#d62728")
    ax.set_xticks(xs)
    ax.set_xlabel("worker threads")
    ax.set_ylabel("throughput (ops/sec)")
    ax.set_title("Read path scales with threads; writes serialize on the WAL")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "thread_scaling.png"), dpi=150)
    plt.close(fig)


def chart_value_size(rows):
    sizes = {int(r["param"]): float(r["ops_per_sec"]) for r in rows if r["metric"] == "value_size"}
    xs = sorted(sizes)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.semilogx(xs, [sizes[x] for x in xs], "-o", color="#2ca02c", lw=2)
    for x in xs:
        ax.annotate(f"{sizes[x]/1e6:.2f}M", (x, sizes[x]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("value size (bytes)")
    ax.set_ylabel("put throughput (ops/sec)")
    ax.set_title("Put throughput vs value size (single writer, no fsync)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "value_size.png"), dpi=150)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, fontsize=10, text_kw=None):
    from matplotlib.patches import FancyBboxPatch

    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012",
        linewidth=1.3, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(b)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        **(text_kw or {}),
    )


def arrow(ax, x1, y1, x2, y2, style="-|>", color="black", lw=1.4, dashed=False):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        linestyle="--" if dashed else "-"),
    )


def chart_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    api = "#e3f2fd"
    wal = "#fff3e0"
    mem = "#e8f5e9"
    sst = "#fce4ec"
    bg = "#f1f3f5"
    fs = 9

    # ---- WRITE PATH (top) ----
    ax.text(1, 98, "WRITE PATH", fontsize=11, fontweight="bold", va="top")
    box(ax, 1, 84, 12, 9, "put/delete", api, "#1976d2", fontsize=fs)
    box(ax, 16, 84, 19, 9, "WAL\nappend/CRC/fsync-opt", wal, "#f57c00", fontsize=fs)
    box(ax, 38, 84, 18, 9, "Active MemTable\nSkipMap", mem, "#388e3c", fontsize=fs)
    box(ax, 59, 84, 18, 9, "Immutable\nMemTable", mem, "#2e7d32", fontsize=fs)
    box(ax, 80, 84, 18, 9, "L0 SSTable", sst, "#c2185b", fontsize=fs)
    arrow(ax, 13, 88.5, 16, 88.5)
    arrow(ax, 35, 88.5, 38, 88.5)
    arrow(ax, 56, 88.5, 59, 88.5)
    arrow(ax, 77, 88.5, 80, 88.5)

    # auto-compaction down to L1
    box(ax, 80, 66, 18, 9, "L1 SSTable\nmerged", sst, "#880e4f", fontsize=fs)
    arrow(ax, 89, 84, 89, 76, style="-|>", dashed=True)
    ax.text(93, 80, "auto-compact", fontsize=8, color="#880e4f", va="center")

    # ---- READ PATH (middle) ----
    ax.text(1, 56, "READ PATH", fontsize=11, fontweight="bold", va="top")
    box(ax, 1, 42, 12, 9, "get(k)", api, "#1976d2", fontsize=fs)
    box(ax, 17, 42, 16, 9, "Active", mem, "#388e3c", fontsize=fs)
    box(ax, 37, 42, 16, 9, "Immutable", mem, "#2e7d32", fontsize=fs)
    box(ax, 57, 42, 20, 9, "L0 newest-first\nbloom+min/max", sst, "#c2185b", fontsize=fs)
    box(ax, 80, 42, 18, 9, "L1 bloom", sst, "#880e4f", fontsize=fs)
    arrow(ax, 13, 46.5, 17, 46.5)
    arrow(ax, 33, 46.5, 37, 46.5)
    arrow(ax, 53, 46.5, 57, 46.5)
    arrow(ax, 77, 46.5, 80, 46.5)
    ax.text(1, 37.5, "first hit wins; tombstone returns None and stops the search",
            fontsize=8, style="italic", color="#555")

    # ---- DURABILITY & RECOVERY (bottom) ----
    ax.text(1, 30, "DURABILITY & RECOVERY", fontsize=11, fontweight="bold", va="top")
    box(ax, 1, 12, 20, 9, "MANIFEST\nSST+level+range", bg, "#37474f", fontsize=fs)
    box(ax, 25, 12, 20, 9, "WAL files", bg, "#455a64", fontsize=fs)
    box(ax, 49, 12, 20, 9, "open(): manifest+\nreplay WAL", bg, "#546e7a", fontsize=fs)
    arrow(ax, 59, 21, 59, 30, style="-|>", color="#546e7a", dashed=True)
    ax.text(61, 26, "recovery feeds Active MemTable", fontsize=8, color="#546e7a")
    arrow(ax, 35, 33, 22, 33, style="-|>", color="#37474f", dashed=True)
    ax.text(28.5, 34.5, "flush/compact rewrite MANIFEST", fontsize=8, color="#37474f", ha="center")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true", help="re-run cargo bench --bench scaling first")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.rerun:
        rerun_bench()
    if not os.path.exists(CSV_PATH):
        print(f"missing {CSV_PATH} — run with --rerun", file=sys.stderr)
        sys.exit(1)

    styled()
    rows = load_rows()
    chart_thread_scaling(rows)
    chart_value_size(rows)
    chart_architecture()
    print("wrote:")
    for name in ("thread_scaling.png", "value_size.png", "architecture.png"):
        p = os.path.join(OUT, name)
        print(f"  {p} ({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    main()
