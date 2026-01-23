#!/usr/bin/env python3
"""
Compare search() vs search_fast() performance.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sochdb.vector import VectorIndex

# Configuration
DIM = 384
N_VECTORS = 10000
N_QUERIES = 1000
K = 10

np.random.seed(42)

print("=" * 70)
print("SEARCH vs SEARCH_FAST COMPARISON")
print("=" * 70)
print(f"Config: {N_VECTORS} vectors, {DIM}D, {N_QUERIES} queries, k={K}")

# Generate data
vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

queries = vectors[:N_QUERIES].copy() + np.random.randn(N_QUERIES, DIM).astype(np.float32) * 0.01
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

# Ground truth
similarities = queries @ vectors.T
ground_truth = np.argsort(-similarities, axis=1)[:, :K]

# Create index
print("\nCreating index...")
index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
ids = np.arange(N_VECTORS, dtype=np.uint64)
index.insert_batch_fast(ids, vectors)
index.ef_search = 500
print(f"Index created with {len(index)} vectors")

# Warmup
print("\nWarming up...")
for i in range(100):
    index.search(queries[i % N_QUERIES], k=K)
    index.search_fast(queries[i % N_QUERIES], k=K)

# Benchmark search()
print("\nBenchmarking search()...")
times_search = []
recalls_search = []
for i in range(N_QUERIES):
    start = time.perf_counter_ns()
    results = index.search(queries[i], k=K)
    elapsed = time.perf_counter_ns() - start
    times_search.append(elapsed / 1000)  # µs
    
    pred = [r[0] for r in results]
    recall = len(set(pred) & set(ground_truth[i])) / K
    recalls_search.append(recall)

p50_search = np.percentile(times_search, 50)
p99_search = np.percentile(times_search, 99)
recall_search = np.mean(recalls_search)

print(f"  search():     p50={p50_search:.1f}µs ({p50_search/1000:.2f}ms), p99={p99_search:.1f}µs, recall={recall_search:.3f}")

# Benchmark search_fast()
print("\nBenchmarking search_fast()...")
times_fast = []
recalls_fast = []
for i in range(N_QUERIES):
    start = time.perf_counter_ns()
    results = index.search_fast(queries[i], k=K)
    elapsed = time.perf_counter_ns() - start
    times_fast.append(elapsed / 1000)  # µs
    
    pred = [r[0] for r in results]
    recall = len(set(pred) & set(ground_truth[i])) / K
    recalls_fast.append(recall)

p50_fast = np.percentile(times_fast, 50)
p99_fast = np.percentile(times_fast, 99)
recall_fast = np.mean(recalls_fast)

print(f"  search_fast(): p50={p50_fast:.1f}µs ({p50_fast/1000:.2f}ms), p99={p99_fast:.1f}µs, recall={recall_fast:.3f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
speedup = p50_search / p50_fast
print(f"  search():      {p50_search:.1f}µs ({p50_search/1000:.2f}ms)")
print(f"  search_fast(): {p50_fast:.1f}µs ({p50_fast/1000:.2f}ms)")
print(f"  Speedup:       {speedup:.2f}x faster")
print(f"  Recall diff:   {recall_search:.3f} vs {recall_fast:.3f}")

if speedup > 1.2:
    print(f"\n  ✅ search_fast() is {speedup:.1f}x faster!")
elif speedup < 0.8:
    print(f"\n  ⚠️ search_fast() is slower - investigating needed")
else:
    print(f"\n  ≈ Performance is similar")
