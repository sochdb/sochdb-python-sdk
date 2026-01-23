#!/usr/bin/env python3
"""
Deep Profiling: HNSW Hot Path Analysis
=======================================

Principle: Before optimizing, MEASURE. Identify the actual bottleneck.

For robotics/edge use case, we need:
- Sub-millisecond latency (<1ms for real-time control loops)
- Consistent P99 (no GC pauses, no lock contention spikes)
- Low memory footprint

This script profiles each component of the search pipeline:
1. Distance calculation (should be SIMD-bound)
2. Memory access patterns (cache hits/misses)
3. Python FFI overhead
4. Heap allocations in hot path
"""

import sys
import time
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sochdb.vector import VectorIndex, _FFI

# =============================================================================
# Configuration
# =============================================================================
DIM = 384
N_VECTORS = 10000
N_WARMUP = 100
N_ITERATIONS = 1000

np.random.seed(42)

# =============================================================================
# Micro-benchmarks
# =============================================================================

def profile_pure_distance_calculation():
    """
    Profile JUST the distance calculation, no HNSW overhead.
    
    This tells us the theoretical minimum time for distance ops.
    FAISS achieves ~0.3ms for 10k vectors at 384D = 30ns per distance.
    
    For k=10 search with ef_search=500, we compute ~500-2000 distances.
    At 30ns each, that's 15-60µs. If we're at 1.5ms, something is 25x slower.
    """
    print("\n" + "="*70)
    print("1. PURE DISTANCE CALCULATION PROFILE")
    print("="*70)
    
    # Generate test data
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = np.random.randn(DIM).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # NumPy baseline (uses BLAS/MKL)
    times_numpy = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter_ns()
        # Cosine distance = 1 - dot product (for normalized vectors)
        distances = 1.0 - (query @ vectors.T)
        elapsed = time.perf_counter_ns() - start
        times_numpy.append(elapsed)
    
    p50_numpy = np.percentile(times_numpy, 50) / 1000  # µs
    p99_numpy = np.percentile(times_numpy, 99) / 1000
    
    print(f"\n  NumPy (BLAS) - {N_VECTORS} distances @ {DIM}D:")
    print(f"    p50: {p50_numpy:.1f}µs ({p50_numpy/N_VECTORS*1000:.1f}ns per distance)")
    print(f"    p99: {p99_numpy:.1f}µs")
    
    # Pure Python baseline (to show overhead)
    times_python = []
    for _ in range(min(10, N_ITERATIONS)):  # Only 10 iterations, it's slow
        start = time.perf_counter_ns()
        dists = []
        for i in range(100):  # Only 100 vectors
            d = 1.0 - sum(query[j] * vectors[i, j] for j in range(DIM))
            dists.append(d)
        elapsed = time.perf_counter_ns() - start
        times_python.append(elapsed)
    
    p50_python = np.percentile(times_python, 50) / 1000
    print(f"\n  Pure Python - 100 distances @ {DIM}D:")
    print(f"    p50: {p50_python:.1f}µs ({p50_python/100*1000:.1f}ns per distance)")
    print(f"    → Python is {p50_python/100*N_VECTORS/p50_numpy:.0f}x slower than NumPy")
    
    return p50_numpy, p99_numpy


def profile_sochdb_ffi_overhead():
    """
    Profile the Python → Rust FFI boundary overhead.
    
    FFI calls have overhead:
    - Python → C: ~100-500ns per call
    - Data marshaling: depends on size
    - Return value unpacking
    
    If we make many small FFI calls, overhead dominates.
    """
    print("\n" + "="*70)
    print("2. PYTHON → RUST FFI OVERHEAD")
    print("="*70)
    
    # Create index
    index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    
    # Insert vectors
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.insert_batch_fast(ids, vectors)
    index.ef_search = 500
    
    # Profile single search (includes all overhead)
    query = vectors[0] + np.random.randn(DIM).astype(np.float32) * 0.01
    query = query / np.linalg.norm(query)
    
    times_search = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter_ns()
        results = index.search(query, k=10)
        elapsed = time.perf_counter_ns() - start
        times_search.append(elapsed)
    
    p50_search = np.percentile(times_search, 50) / 1000  # µs
    p99_search = np.percentile(times_search, 99) / 1000
    
    print(f"\n  Full search (k=10, ef_search=500):")
    print(f"    p50: {p50_search:.1f}µs ({p50_search/1000:.2f}ms)")
    print(f"    p99: {p99_search:.1f}µs ({p99_search/1000:.2f}ms)")
    
    # Profile just the FFI call overhead (empty operation)
    times_len = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter_ns()
        _ = len(index)  # Simple FFI call
        elapsed = time.perf_counter_ns() - start
        times_len.append(elapsed)
    
    p50_len = np.percentile(times_len, 50) / 1000
    print(f"\n  Simple FFI call (len()):")
    print(f"    p50: {p50_len:.2f}µs")
    print(f"    → FFI overhead is ~{p50_len:.0f}µs per call")
    
    return p50_search, p99_search


def profile_memory_allocation():
    """
    Profile heap allocations during search.
    
    Every malloc/free in the hot path kills performance:
    - malloc: 50-500ns (varies with fragmentation)
    - GC pressure: unpredictable latency spikes
    
    The HNSW search should use pre-allocated scratch buffers.
    """
    print("\n" + "="*70)
    print("3. MEMORY ALLOCATION ANALYSIS")
    print("="*70)
    
    import tracemalloc
    
    # Create index
    index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.insert_batch_fast(ids, vectors)
    index.ef_search = 500
    
    query = vectors[0] + np.random.randn(DIM).astype(np.float32) * 0.01
    query = query / np.linalg.norm(query)
    
    # Warmup
    for _ in range(N_WARMUP):
        index.search(query, k=10)
    
    # Measure allocations
    tracemalloc.start()
    for _ in range(100):
        results = index.search(query, k=10)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n  Memory during 100 searches:")
    print(f"    Current: {current / 1024:.1f} KB")
    print(f"    Peak: {peak / 1024:.1f} KB")
    print(f"    Per search: ~{(peak - current) / 100:.1f} bytes")


def profile_batch_vs_single():
    """
    Profile batch search vs single search.
    
    Batch operations amortize:
    - FFI call overhead
    - Memory allocation
    - CPU cache warmup
    
    If batch is much faster per-query, FFI overhead is the bottleneck.
    """
    print("\n" + "="*70)
    print("4. BATCH vs SINGLE SEARCH")
    print("="*70)
    
    # Create index
    index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.insert_batch_fast(ids, vectors)
    index.ef_search = 500
    
    # Generate queries
    queries = vectors[:100] + np.random.randn(100, DIM).astype(np.float32) * 0.01
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Single search
    times_single = []
    for q in queries:
        start = time.perf_counter_ns()
        results = index.search(q, k=10)
        elapsed = time.perf_counter_ns() - start
        times_single.append(elapsed)
    
    p50_single = np.percentile(times_single, 50) / 1000
    total_single = sum(times_single) / 1e6  # ms
    
    print(f"\n  Single search x100:")
    print(f"    Total time: {total_single:.1f}ms")
    print(f"    Per query (p50): {p50_single:.1f}µs")
    
    # Check if batch search exists
    if hasattr(index, 'search_batch'):
        start = time.perf_counter_ns()
        all_results = index.search_batch(queries, k=10)
        elapsed = time.perf_counter_ns() - start
        total_batch = elapsed / 1e6
        
        print(f"\n  Batch search (100 queries):")
        print(f"    Total time: {total_batch:.1f}ms")
        print(f"    Per query: {total_batch/100*1000:.1f}µs")
        print(f"    → Batch is {total_single/total_batch:.1f}x faster")
    else:
        print(f"\n  ⚠️ No batch search API available")
        print(f"    → Potential optimization: add search_batch to FFI")


def profile_cache_locality():
    """
    Profile cache behavior during graph traversal.
    
    HNSW graph traversal is memory-bound:
    - L1 cache: 4 cycles (~1ns)
    - L2 cache: 10-12 cycles (~3ns)
    - L3 cache: 30-40 cycles (~10ns)
    - RAM: 100+ cycles (~60ns)
    
    If vectors are scattered in memory, we get L3/RAM hits.
    Sequential access patterns get L1/L2 hits.
    """
    print("\n" + "="*70)
    print("5. CACHE LOCALITY ANALYSIS")
    print("="*70)
    
    # Create index
    index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.insert_batch_fast(ids, vectors)
    index.ef_search = 500
    
    # Sequential queries (cache-friendly)
    queries_seq = vectors[:100].copy()
    queries_seq = queries_seq / np.linalg.norm(queries_seq, axis=1, keepdims=True)
    
    times_seq = []
    for q in queries_seq:
        start = time.perf_counter_ns()
        results = index.search(q, k=10)
        elapsed = time.perf_counter_ns() - start
        times_seq.append(elapsed)
    
    # Random queries (cache-unfriendly)
    random_indices = np.random.permutation(N_VECTORS)[:100]
    queries_rand = vectors[random_indices].copy()
    queries_rand = queries_rand / np.linalg.norm(queries_rand, axis=1, keepdims=True)
    
    times_rand = []
    for q in queries_rand:
        start = time.perf_counter_ns()
        results = index.search(q, k=10)
        elapsed = time.perf_counter_ns() - start
        times_rand.append(elapsed)
    
    p50_seq = np.percentile(times_seq, 50) / 1000
    p50_rand = np.percentile(times_rand, 50) / 1000
    
    print(f"\n  Sequential query patterns:")
    print(f"    p50: {p50_seq:.1f}µs")
    
    print(f"\n  Random query patterns:")
    print(f"    p50: {p50_rand:.1f}µs")
    
    print(f"\n  → Random is {p50_rand/p50_seq:.2f}x slower (cache misses)")


def profile_ef_search_scaling():
    """
    Profile how latency scales with ef_search.
    
    ef_search controls the search beam width:
    - Higher = more distance calculations = better recall
    - Lower = fewer calculations = faster but worse recall
    
    Latency should scale ~linearly with ef_search.
    """
    print("\n" + "="*70)
    print("6. ef_search SCALING ANALYSIS")
    print("="*70)
    
    # Create index
    index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    index.insert_batch_fast(ids, vectors)
    
    query = vectors[0] + np.random.randn(DIM).astype(np.float32) * 0.01
    query = query / np.linalg.norm(query)
    
    # Ground truth
    similarities = query @ vectors.T
    gt = np.argsort(-similarities)[:10]
    
    print(f"\n  ef_search | Latency (p50) | Recall@10")
    print(f"  {'-'*45}")
    
    ef_values = [16, 32, 64, 100, 200, 400, 800]
    results_scaling = []
    
    for ef in ef_values:
        index.ef_search = ef
        
        times = []
        recalls = []
        for _ in range(100):
            start = time.perf_counter_ns()
            results = index.search(query, k=10)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)
            
            pred = [r[0] for r in results]
            recall = len(set(pred) & set(gt)) / 10
            recalls.append(recall)
        
        p50 = np.percentile(times, 50) / 1000
        avg_recall = np.mean(recalls)
        results_scaling.append((ef, p50, avg_recall))
        
        print(f"  {ef:9} | {p50:12.1f}µs | {avg_recall:.3f}")
    
    # Calculate scaling factor
    t1, t2 = results_scaling[0][1], results_scaling[-1][1]
    ef1, ef2 = results_scaling[0][0], results_scaling[-1][0]
    scaling = (t2 - t1) / (ef2 - ef1)
    
    print(f"\n  → Latency scales at ~{scaling:.1f}µs per ef_search increment")
    print(f"  → Cost per distance calc: ~{t1/ef1:.2f}µs")


def compare_with_faiss():
    """
    Direct comparison with FAISS using identical parameters.
    
    This identifies WHERE the gap comes from.
    """
    print("\n" + "="*70)
    print("7. DIRECT COMPARISON WITH FAISS")
    print("="*70)
    
    try:
        import faiss
    except ImportError:
        print("  FAISS not installed, skipping comparison")
        return
    
    # Generate data
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0] + np.random.randn(DIM).astype(np.float32) * 0.01
    query = query / np.linalg.norm(query)
    query = np.ascontiguousarray(query.reshape(1, -1))
    vectors = np.ascontiguousarray(vectors)
    
    # FAISS HNSW with SAME parameters
    faiss_index = faiss.IndexHNSWFlat(DIM, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 500
    faiss_index.add(vectors)
    
    # SochDB with SAME parameters
    sochdb_index = VectorIndex(dimension=DIM, max_connections=32, ef_construction=200)
    ids = np.arange(N_VECTORS, dtype=np.uint64)
    sochdb_index.insert_batch_fast(ids, vectors)
    sochdb_index.ef_search = 500
    
    # Warmup
    for _ in range(N_WARMUP):
        faiss_index.search(query, 10)
        sochdb_index.search(query[0], k=10)
    
    # Profile FAISS
    times_faiss = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter_ns()
        D, I = faiss_index.search(query, 10)
        elapsed = time.perf_counter_ns() - start
        times_faiss.append(elapsed)
    
    # Profile SochDB
    times_sochdb = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter_ns()
        results = sochdb_index.search(query[0], k=10)
        elapsed = time.perf_counter_ns() - start
        times_sochdb.append(elapsed)
    
    p50_faiss = np.percentile(times_faiss, 50) / 1000
    p99_faiss = np.percentile(times_faiss, 99) / 1000
    p50_sochdb = np.percentile(times_sochdb, 50) / 1000
    p99_sochdb = np.percentile(times_sochdb, 99) / 1000
    
    print(f"\n  Same parameters: M=32, ef_construction=200, ef_search=500")
    print(f"\n  FAISS:")
    print(f"    p50: {p50_faiss:.1f}µs ({p50_faiss/1000:.2f}ms)")
    print(f"    p99: {p99_faiss:.1f}µs")
    
    print(f"\n  SochDB:")
    print(f"    p50: {p50_sochdb:.1f}µs ({p50_sochdb/1000:.2f}ms)")
    print(f"    p99: {p99_sochdb:.1f}µs")
    
    gap = p50_sochdb / p50_faiss
    print(f"\n  → SochDB is {gap:.1f}x slower than FAISS")
    
    # Break down the gap
    print(f"\n  GAP ANALYSIS:")
    
    # Time per ef_search unit
    time_per_ef_faiss = p50_faiss / 500
    time_per_ef_sochdb = p50_sochdb / 500
    print(f"    FAISS: {time_per_ef_faiss:.2f}µs per ef_search candidate")
    print(f"    SochDB: {time_per_ef_sochdb:.2f}µs per ef_search candidate")
    
    # Theoretical distance calculation time
    # 384 floats * 2 ops (mul+add) / 8 floats per SIMD = 96 SIMD ops
    # At 4GHz with 2 SIMD units: 96 / 2 / 4 = 12 cycles = 3ns
    print(f"\n    Theoretical minimum (SIMD): ~3ns per distance")
    print(f"    FAISS achieves: {time_per_ef_faiss*1000:.0f}ns per candidate")
    print(f"    SochDB achieves: {time_per_ef_sochdb*1000:.0f}ns per candidate")
    
    # The gap suggests:
    if gap > 3:
        print(f"\n  ⚠️ DIAGNOSIS: {gap:.1f}x gap suggests:")
        if gap > 5:
            print(f"    - Memory allocation in hot path")
            print(f"    - Lock contention")
            print(f"    - Poor cache locality")
        else:
            print(f"    - SIMD not fully utilized")
            print(f"    - FFI overhead")
            print(f"    - Graph traversal overhead")


def main():
    print("="*70)
    print("🔬 SOCHDB DEEP PROFILING")
    print("="*70)
    print(f"\nConfiguration: {N_VECTORS} vectors, {DIM}D, {N_ITERATIONS} iterations")
    
    p50_numpy, _ = profile_pure_distance_calculation()
    p50_search, p99_search = profile_sochdb_ffi_overhead()
    profile_memory_allocation()
    profile_batch_vs_single()
    profile_cache_locality()
    profile_ef_search_scaling()
    compare_with_faiss()
    
    # Summary
    print("\n" + "="*70)
    print("📊 PROFILING SUMMARY")
    print("="*70)
    
    print(f"\n  NumPy distance calc (10k @ 384D): {p50_numpy:.1f}µs")
    print(f"  SochDB full search (ef=500, k=10): {p50_search:.1f}µs")
    print(f"\n  Overhead ratio: {p50_search/p50_numpy:.1f}x")
    print(f"  (Should be ~1-2x if SIMD is efficient)")
    
    print("\n  Key bottleneck indicators:")
    if p50_search > 2000:  # > 2ms
        print("  ⚠️ HIGH: Lock contention or memory allocation likely")
    elif p50_search > 1000:  # > 1ms
        print("  ⚠️ MEDIUM: SIMD underutilization or FFI overhead")
    else:
        print("  ✅ LOW: Performance is competitive")


if __name__ == "__main__":
    main()
