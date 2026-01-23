#!/usr/bin/env python3
"""
Micro-benchmark: Isolate individual components.
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sochdb.vector import VectorIndex

# Config
DIM = 384
N_VECTORS = 10000
N_ITERATIONS = 10000

np.random.seed(42)

print("=" * 70)
print("COMPONENT MICRO-BENCHMARKS")
print("=" * 70)

# Generate data
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
query = vectors[0].copy()

# ============================================================================
# Test 1: Pure NumPy distance (baseline)
# ============================================================================
print("\n1. PURE NUMPY DISTANCE (10k vectors)")
times = []
for _ in range(100):
    start = time.perf_counter_ns()
    distances = 1.0 - (query @ vectors.T)
    elapsed = time.perf_counter_ns() - start
    times.append(elapsed / 1000)

p50 = np.percentile(times, 50)
print(f"   p50: {p50:.1f}µs ({p50/N_VECTORS*1000:.1f}ns per distance)")

# ============================================================================
# Test 2: SochDB single search at various ef_search
# ============================================================================
print("\n2. SOCHDB SEARCH LATENCY vs ef_search")

index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
ids = np.arange(N_VECTORS, dtype=np.uint64)
index.insert_batch_fast(ids, vectors)

for ef in [16, 50, 100, 200, 500]:
    index.ef_search = ef
    
    times = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        results = index.search(query, k=10)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1000)
    
    p50 = np.percentile(times, 50)
    cost_per_ef = p50 / ef
    print(f"   ef={ef:3d}: {p50:7.1f}µs ({cost_per_ef:.1f}µs/candidate)")

# ============================================================================
# Test 3: Overhead breakdown estimate
# ============================================================================
print("\n3. OVERHEAD BREAKDOWN (estimated)")

# Measure fixed overhead (ef=1 case)
index.ef_search = 1
times_ef1 = []
for _ in range(1000):
    start = time.perf_counter_ns()
    results = index.search(query, k=1)
    elapsed = time.perf_counter_ns() - start
    times_ef1.append(elapsed / 1000)

p50_ef1 = np.percentile(times_ef1, 50)

# Measure ef=500 
index.ef_search = 500
times_ef500 = []
for _ in range(1000):
    start = time.perf_counter_ns()
    results = index.search(query, k=10)
    elapsed = time.perf_counter_ns() - start
    times_ef500.append(elapsed / 1000)

p50_ef500 = np.percentile(times_ef500, 50)

# Calculate marginal cost
marginal_cost = (p50_ef500 - p50_ef1) / 499
print(f"   Fixed overhead (FFI + setup): {p50_ef1:.1f}µs")
print(f"   Per-candidate cost: {marginal_cost:.2f}µs")
print(f"   At ef=500, variable cost: {499 * marginal_cost:.1f}µs")

# ============================================================================
# Test 4: Python overhead test
# ============================================================================
print("\n4. PYTHON FFI CALL OVERHEAD")

# Just measure len() calls (minimal FFI)
times_len = []
for _ in range(N_ITERATIONS):
    start = time.perf_counter_ns()
    _ = len(index)
    elapsed = time.perf_counter_ns() - start
    times_len.append(elapsed / 1000)

p50_len = np.percentile(times_len, 50)
print(f"   len() call: {p50_len:.2f}µs")

# Measure search with k=1, ef=1 (minimal work)
index.ef_search = 1
times_min = []
for _ in range(1000):
    start = time.perf_counter_ns()
    results = index.search(query, k=1)
    elapsed = time.perf_counter_ns() - start
    times_min.append(elapsed / 1000)

p50_min = np.percentile(times_min, 50)
print(f"   Minimal search (k=1, ef=1): {p50_min:.1f}µs")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n   NumPy 10k distances:  {np.percentile([t for t in times], 50):.1f}µs")
print(f"   SochDB ef=500 search: {p50_ef500:.1f}µs")
print(f"   Fixed overhead:       {p50_ef1:.1f}µs")
print(f"   Per-candidate cost:   {marginal_cost:.2f}µs")
print(f"\n   Theoretical SIMD @ 384D: ~0.1µs per distance")
print(f"   Actual per-candidate:    {marginal_cost:.2f}µs")
print(f"   Gap ratio:               {marginal_cost * 1000 / 100:.1f}x")
