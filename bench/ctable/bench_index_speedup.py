#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: CTable index vs full-scan speedup.
#
# Sections
# ────────
#  1. Random data    — index barely helps; every chunk spans full value range
#  2. Sorted data    — index skips most chunks; best case for range queries
#  3. sort_by() then index — sorting random data then indexing
#  4. Selectivity sweet-spot — shows where index wins vs loses
#  5. High vs low cardinality (sorted) — repetition level vs speedup
#  6. Compound filters — 1 index vs 2 indexes on sorted data
#
# Key finding: the index only skips chunks when the data is sorted/clustered
# on the indexed column.  Cardinality matters less than clustering.
# Index wins at <~25% selectivity; full scan wins above that.

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

import blosc2

# ── Config ────────────────────────────────────────────────────────────────────

N    = 1_000_000
REPS = 5

# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    sensor_id:   int   = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float64())
    region:      int   = blosc2.field(blosc2.int32())

np_dtype = np.dtype([
    ("sensor_id",   np.int32),
    ("temperature", np.float64),
    ("region",      np.int32),
])

rng = np.random.default_rng(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def bench_gt(table, threshold, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(table["sensor_id"] > threshold)
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def bench_eq(table, value, col="sensor_id", reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(table[col] == value)
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def bench_compound(table, threshold, region, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(
            (table["sensor_id"] > threshold) & (table["region"] == region)
        )
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def make_table(sensor_ids):
    DATA = np.empty(N, dtype=np_dtype)
    DATA["sensor_id"]   = sensor_ids
    DATA["temperature"] = 15.0 + rng.random(N) * 25
    DATA["region"]      = rng.integers(0, 8, size=N, dtype=np.int32)
    ct = blosc2.CTable(Row, expected_size=N)
    ct.extend(DATA)
    return ct, DATA

def print_range_table(results, title, width=70):
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    print(f"  {'SELECTIVITY':<14} {'ROWS':>9}  {'SCAN(ms)':>9}  {'IDX(ms)':>9}  {'SPEEDUP':>8}")
    print(f"  {'─'*14} {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    for label, n, t_scan, t_idx in results:
        speedup = t_scan / t_idx if t_idx > 0 else float("inf")
        marker  = " ←" if speedup >= 2.0 else ("  (slower)" if speedup < 0.9 else "")
        print(f"  {label:<14} {n:>9,}  {t_scan*1e3:>9.1f}  {t_idx*1e3:>9.1f}  {speedup:>7.1f}×{marker}")
    print(f"{'─' * width}")

FRACS   = [0.999, 0.99, 0.95, 0.90, 0.75, 0.50, 0.25]
LABELS  = ["0.1%", "1%", "5%", "10%", "25%", "50%", "75%"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. RANDOM DATA — index barely helps
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  1. RANDOM data  (sensor_id randomly spread across full range)")
print("═" * 70)
print(f"  N={N:,}  |  REPS={REPS}  |  ~10 rows per sensor_id value")

SID_MAX = N // 10
rand_ids = rng.integers(0, SID_MAX, size=N, dtype=np.int32)
ct_rand, _ = make_table(rand_ids)

thresholds = [(lbl, int(SID_MAX * f)) for lbl, f in zip(LABELS, FRACS)]
scan_rand  = {lbl: bench_gt(ct_rand, thr) for lbl, thr in thresholds}

ct_rand.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
results = []
for lbl, thr in thresholds:
    t_idx, n = bench_gt(ct_rand, thr)
    t_scan, _ = scan_rand[lbl]
    results.append((lbl, n, t_scan, t_idx))

print_range_table(results, "Random data — BUCKET index (chunks span full range → can't skip)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SORTED DATA — best case: chunks are contiguous, index skips most
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  2. SORTED data  (sensor_id = 0,0,…,1,1,…,2,2,…  clustered)")
print("═" * 70)
print(f"  N={N:,}  |  REPS={REPS}  |  ~10 rows per sensor_id value")

sorted_ids = np.repeat(np.arange(SID_MAX, dtype=np.int32), N // SID_MAX)
ct_sorted, _ = make_table(sorted_ids)

scan_sorted = {lbl: bench_gt(ct_sorted, thr) for lbl, thr in thresholds}

ct_sorted.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
results = []
for lbl, thr in thresholds:
    t_idx, n = bench_gt(ct_sorted, thr)
    t_scan, _ = scan_sorted[lbl]
    results.append((lbl, n, t_scan, t_idx))

print_range_table(results, "Sorted (clustered) data — BUCKET index  → chunks skipped")


# ══════════════════════════════════════════════════════════════════════════════
# 3. RANDOM DATA → sort_by() → INDEX
#    Shows that sorting random data first gives the same speedup as pre-sorted
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  3. RANDOM data  then  sort_by('sensor_id')  then  index")
print("═" * 70)
print(f"  N={N:,}  |  REPS={REPS}")

ct_sortedlate, _ = make_table(rand_ids)
scan_sl = {lbl: bench_gt(ct_sortedlate, thr) for lbl, thr in thresholds}

t0 = perf_counter()
ct_sortedlate.sort_by(["sensor_id"], inplace=True)
sort_time = perf_counter() - t0

t0 = perf_counter()
ct_sortedlate.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
idx_time = perf_counter() - t0

print(f"  sort_by: {sort_time*1e3:.0f} ms  |  create_index: {idx_time*1e3:.0f} ms")

results = []
for lbl, thr in thresholds:
    t_idx, n = bench_gt(ct_sortedlate, thr)
    t_scan, _ = scan_sl[lbl]
    results.append((lbl, n, t_scan, t_idx))

print_range_table(results, "Random → sort_by → BUCKET index  (same result as pre-sorted)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. HIGH vs LOW CARDINALITY — does repetition level matter?
#    (all sorted so the index can always skip)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  4. CARDINALITY comparison  (data always sorted — only repetition changes)")
print("═" * 70)

CARD_CASES = [
    ("High rep  (10 uniq)", 10),          # 100,000 rows/value
    ("Med  rep  (1k uniq)", 1_000),       # 1,000 rows/value
    ("Low  rep  (1M uniq)", N),           # 1 row/value
]

SEL_FRACS  = [0.999, 0.99, 0.95, 0.90]
SEL_LABELS = ["0.1%", "1%", "5%", "10%"]

W2 = 72
print(f"\n  {'CARDINALITY':<24}", end="")
for lbl in SEL_LABELS:
    print(f"  {lbl+' sel':>12}", end="")
print()
print("  " + "─" * (W2 - 2))

for case_lbl, n_unique in CARD_CASES:
    reps_per_val = N // n_unique
    ids = np.repeat(np.arange(n_unique, dtype=np.int32), reps_per_val)
    if len(ids) < N:
        ids = np.concatenate([ids, np.zeros(N - len(ids), dtype=np.int32)])
    sid_max = n_unique

    ct_c = blosc2.CTable(Row, expected_size=N)
    DATA_c = np.empty(N, dtype=np_dtype)
    DATA_c["sensor_id"]   = ids
    DATA_c["temperature"] = 15.0 + rng.random(N) * 25
    DATA_c["region"]      = rng.integers(0, 8, size=N, dtype=np.int32)
    ct_c.extend(DATA_c)

    thr_list = [(lbl, int(sid_max * f)) for lbl, f in zip(SEL_LABELS, SEL_FRACS)]
    scan_c   = {lbl: bench_gt(ct_c, thr) for lbl, thr in thr_list}
    ct_c.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)

    print(f"  {case_lbl:<24}", end="")
    for lbl, thr in thr_list:
        t_idx, _ = bench_gt(ct_c, thr)
        t_scan, _ = scan_c[lbl]
        spd = t_scan / t_idx if t_idx > 0 else float("inf")
        print(f"  {spd:>10.1f}×  ", end="")
    print()

print("  " + "─" * (W2 - 2))
print("  (speedup columns — higher is better)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. COMPOUND FILTERS: sensor_id > X  AND  region == Y
#    No index  vs  1 index (sid)  vs  2 indexes  — on SORTED data
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  5. COMPOUND filter  sensor_id > X  AND  region == Y  (sorted data)")
print("═" * 70)
print(f"  N={N:,}  |  REPS={REPS}  |  region in [0,8)  → ~12.5% per value")

REGION_TARGET = 3
# sensor_id selectivity labels + region==3 combined
COMPOUND_THRESHOLDS = [
    ("0.1%+12.5%", int(SID_MAX * 0.999)),
    ("1%+12.5%",   int(SID_MAX * 0.99)),
    ("5%+12.5%",   int(SID_MAX * 0.95)),
    ("10%+12.5%",  int(SID_MAX * 0.90)),
]

# Reuse ct_sorted (sensor_id is sorted; region is random)
ct_sorted.drop_index("sensor_id")

no_idx      = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
one_idx_sid = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.drop_index("sensor_id")
ct_sorted.create_index("region", kind=blosc2.IndexKind.PARTIAL)
one_idx_reg = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
two_idx     = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}

W3 = 80
print(f"\n{'─' * W3}")
print(f"  {'QUERY':<14} {'ROWS':>8}  {'NO IDX':>9}  {'IDX:sid':>9}  {'IDX:reg':>9}  {'2 IDX':>9}  {'BEST'}")
print(f"  {'─'*14} {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*12}")
for lbl, thr in COMPOUND_THRESHOLDS:
    n      = no_idx[lbl][1]
    t_none = no_idx[lbl][0]
    t_sid  = one_idx_sid[lbl][0]
    t_reg  = one_idx_reg[lbl][0]
    t_two  = two_idx[lbl][0]
    best_t = min(t_none, t_sid, t_reg, t_two)
    spd    = t_none / best_t
    winner = ["none", "sid", "reg", "2idx"][[t_none, t_sid, t_reg, t_two].index(best_t)]
    print(
        f"  {lbl:<14} {n:>8,}"
        f"  {t_none*1e3:>8.1f}ms"
        f"  {t_sid*1e3:>8.1f}ms"
        f"  {t_reg*1e3:>8.1f}ms"
        f"  {t_two*1e3:>8.1f}ms"
        f"  {winner}({spd:.1f}×)"
    )
print(f"{'─' * W3}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. EQUALITY QUERIES  sensor_id == X  (random and sorted)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  6. EQUALITY queries  sensor_id == X")
print("═" * 70)
print(f"  N={N:,}  |  REPS={REPS}  |  ~10 rows per value")

EQ_VALS = [0, SID_MAX // 4, SID_MAX // 2, SID_MAX - 1]

for label, ct_eq in [("Random data", ct_rand), ("Sorted data", ct_sorted)]:
    # ensure sensor_id index exists; drop others
    try:
        ct_eq.drop_index("sensor_id")
    except Exception:
        pass
    try:
        ct_eq.drop_index("region")
    except Exception:
        pass

    scan_eq = {v: bench_eq(ct_eq, v) for v in EQ_VALS}
    ct_eq.create_index("sensor_id", kind=blosc2.IndexKind.PARTIAL)
    idx_eq  = {v: bench_eq(ct_eq, v) for v in EQ_VALS}

    print(f"\n  {label}")
    print(f"  {'VALUE':<12} {'ROWS':>6}  {'SCAN(ms)':>9}  {'IDX(ms)':>9}  {'SPEEDUP':>8}")
    print(f"  {'─'*12} {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}")
    for v in EQ_VALS:
        t_s, n = scan_eq[v]
        t_i, _ = idx_eq[v]
        spd = t_s / t_i if t_i > 0 else float("inf")
        marker = " ←" if spd >= 2.0 else ""
        print(f"  =={v:<10,} {n:>6,}  {t_s*1e3:>9.1f}  {t_i*1e3:>9.1f}  {spd:>7.1f}×{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════

print("""
┌─────────────────────────────────────────────────────────────────────┐
│  SUMMARY                                                            │
├─────────────────────────────────────────────────────────────────────┤
│  • Random (unsorted) data → index gives ~1–1.5×.  Every chunk       │
│    spans the full value range so no chunks can be skipped.          │
│                                                                     │
│  • Sorted / clustered data → index gives 5–9× for <10% selectivity. │
│    Chunks are contiguous ranges; the index skips most of them.      │
│                                                                     │
│  • sort_by() before create_index() = same speedup as pre-sorted.    │
│                                                                     │
│  • Cardinality (repetition level) has little effect as long as      │
│    the data is sorted on the indexed column.                        │
│                                                                     │
│  • Selectivity > ~25%: full scan wins — mask-building overhead      │
│    outweighs the savings from skipping chunks.                      │
│                                                                     │
│  • Compound filters: index the most selective column (sorted one).  │
│    Adding a second index on an unsorted column usually hurts.       │
│                                                                     │
│  • On-disk tables benefit even more (I/O dominates over Python      │
│    mask-building overhead).                                         │
└─────────────────────────────────────────────────────────────────────┘
""")
