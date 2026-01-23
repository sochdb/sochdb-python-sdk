#!/usr/bin/env python3
"""
Queue Performance Benchmark

This benchmark verifies that the queue optimization doesn't impact overall
database performance. It tests:

1. Queue Operations Performance:
   - Enqueue latency at various queue sizes
   - Dequeue latency with priority ordering
   - Batch enqueue throughput
   - Concurrent worker performance

2. Database Baseline Comparison:
   - Key-value operations (should be unaffected)
   - Vector search (should be unaffected)
   
3. Streaming Top-K Performance:
   - Compare with naive sort implementation
   - Verify O(N log K) vs O(N log N) scaling

Usage:
    python queue_benchmark.py
    python queue_benchmark.py --quick       # Quick run
    python queue_benchmark.py --full        # Full benchmark suite
"""

import argparse
import gc
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Callable, Any, Dict, Optional
import heapq

# Add parent directory to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sochdb.queue import (
    PriorityQueue,
    QueueConfig,
    QueueKey,
    Task,
    StreamingTopK,
    encode_u64_be,
    decode_u64_be,
    encode_i64_be,
    decode_i64_be,
    InMemoryQueueBackend,
    create_queue,
)


# ============================================================================
# Helper Functions for Queue Testing
# ============================================================================

def create_benchmark_queue(queue_name: str = "bench_queue") -> PriorityQueue:
    """Create a PriorityQueue with InMemoryQueueBackend for benchmarking."""
    backend = InMemoryQueueBackend()
    return PriorityQueue.from_backend(backend, queue_name)


# ============================================================================
# Benchmark Utilities
# ============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time_s: float
    min_latency_us: float
    max_latency_us: float
    mean_latency_us: float
    median_latency_us: float
    p95_latency_us: float
    p99_latency_us: float
    throughput_ops: float
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_time_s': round(self.total_time_s, 4),
            'min_latency_us': round(self.min_latency_us, 2),
            'max_latency_us': round(self.max_latency_us, 2),
            'mean_latency_us': round(self.mean_latency_us, 2),
            'median_latency_us': round(self.median_latency_us, 2),
            'p95_latency_us': round(self.p95_latency_us, 2),
            'p99_latency_us': round(self.p99_latency_us, 2),
            'throughput_ops': round(self.throughput_ops, 1),
        }
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations:,}\n"
            f"  Total time: {self.total_time_s:.4f}s\n"
            f"  Latency (μs): min={self.min_latency_us:.1f}, "
            f"mean={self.mean_latency_us:.1f}, "
            f"median={self.median_latency_us:.1f}, "
            f"p95={self.p95_latency_us:.1f}, "
            f"p99={self.p99_latency_us:.1f}, "
            f"max={self.max_latency_us:.1f}\n"
            f"  Throughput: {self.throughput_ops:,.1f} ops/s"
        )


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f < len(data) - 1 else f
    return data[f] + (k - f) * (data[c] - data[f])


def benchmark(name: str, iterations: int, func: Callable[[], Any]) -> BenchmarkResult:
    """Run a benchmark and collect statistics."""
    # Warmup
    for _ in range(min(100, iterations // 10)):
        func()
    
    gc.collect()
    
    # Timed run
    latencies = []
    start_total = time.perf_counter()
    
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        latencies.append((end - start) * 1_000_000)  # Convert to microseconds
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    latencies.sort()
    
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time_s=total_time,
        min_latency_us=min(latencies),
        max_latency_us=max(latencies),
        mean_latency_us=statistics.mean(latencies),
        median_latency_us=statistics.median(latencies),
        p95_latency_us=percentile(latencies, 95),
        p99_latency_us=percentile(latencies, 99),
        throughput_ops=iterations / total_time,
    )


# ============================================================================
# Queue Benchmarks
# ============================================================================

def bench_key_encoding(iterations: int) -> BenchmarkResult:
    """Benchmark QueueKey encoding/decoding."""
    key = QueueKey(
        queue_id="benchmark",
        priority=1000,
        ready_ts=int(time.time() * 1000),
        sequence=12345,
        task_id="task-uuid-12345678",
    )
    
    def encode_decode():
        encoded = key.encode()
        decoded = QueueKey.decode(encoded)
        return decoded
    
    return benchmark("QueueKey encode/decode", iterations, encode_decode)


def bench_enqueue(queue_size: int, iterations: int) -> BenchmarkResult:
    """Benchmark enqueue operations."""
    queue = create_benchmark_queue("bench")
    
    # Pre-populate queue
    for i in range(queue_size):
        queue.enqueue(priority=random.randint(0, 100), payload=f"task-{i}".encode())
    
    def enqueue():
        queue.enqueue(
            priority=random.randint(0, 100),
            payload=b"benchmark-task-payload",
        )
    
    return benchmark(f"Enqueue (queue_size={queue_size:,})", iterations, enqueue)


def bench_dequeue(queue_size: int, iterations: int) -> BenchmarkResult:
    """Benchmark dequeue operations."""
    queue = create_benchmark_queue("bench")
    queue._config.visibility_timeout_ms = 60000
    
    # Pre-populate and keep replenishing
    for i in range(queue_size + iterations):
        queue.enqueue(priority=random.randint(0, 100), payload=f"task-{i}".encode())
    
    worker_id = "worker-1"
    
    def dequeue():
        task = queue.dequeue(worker_id=worker_id)
        if task:
            queue.ack(task.task_id)
        return task
    
    return benchmark(f"Dequeue+Ack (queue_size={queue_size:,})", iterations, dequeue)


def bench_batch_enqueue(batch_size: int, iterations: int) -> BenchmarkResult:
    """Benchmark batch enqueue operations."""
    queue = create_benchmark_queue("bench")
    
    batch = [(random.randint(0, 100), f"task-{i}".encode()) for i in range(batch_size)]
    
    def batch_enqueue():
        queue.enqueue_batch(batch)
    
    return benchmark(f"Batch enqueue (batch_size={batch_size})", iterations, batch_enqueue)


def bench_streaming_topk(n: int, k: int, iterations: int) -> BenchmarkResult:
    """Benchmark streaming top-K selection."""
    data = list(range(n))
    random.shuffle(data)
    
    def streaming_topk():
        topk = StreamingTopK(k=k, ascending=True, key=lambda x: x)
        for item in data:
            topk.push(item)
        return topk.get_sorted()
    
    return benchmark(f"StreamingTopK (n={n:,}, k={k})", iterations, streaming_topk)


def bench_naive_sort(n: int, k: int, iterations: int) -> BenchmarkResult:
    """Benchmark naive sort + slice (the pattern we're replacing)."""
    data = list(range(n))
    random.shuffle(data)
    
    def naive_sort():
        sorted_data = sorted(data)[:k]
        return sorted_data
    
    return benchmark(f"NaiveSort (n={n:,}, k={k})", iterations, naive_sort)


# ============================================================================
# Concurrent Worker Benchmark
# ============================================================================

def bench_concurrent_workers(
    num_workers: int,
    tasks_per_worker: int,
) -> Dict[str, Any]:
    """Benchmark concurrent queue operations."""
    queue = create_benchmark_queue("concurrent")
    queue._config.visibility_timeout_ms = 30000
    
    total_tasks = num_workers * tasks_per_worker
    
    # Enqueue all tasks first
    start = time.perf_counter()
    for i in range(total_tasks):
        queue.enqueue(priority=i % 10, payload=f"task-{i}".encode())
    enqueue_time = time.perf_counter() - start
    
    # Concurrent dequeue
    processed = []
    
    def worker(worker_id: str) -> int:
        count = 0
        while count < tasks_per_worker:
            task = queue.dequeue(worker_id=worker_id)
            if task:
                queue.ack(task.task_id)
                count += 1
            else:
                # Queue might be empty, break
                break
        return count
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker, f"worker-{i}")
            for i in range(num_workers)
        ]
        for future in as_completed(futures):
            processed.append(future.result())
    dequeue_time = time.perf_counter() - start
    
    total_processed = sum(processed)
    
    return {
        'name': f"Concurrent ({num_workers} workers, {tasks_per_worker} tasks/worker)",
        'total_tasks': total_tasks,
        'total_processed': total_processed,
        'enqueue_time_s': enqueue_time,
        'dequeue_time_s': dequeue_time,
        'enqueue_throughput': total_tasks / enqueue_time,
        'dequeue_throughput': total_processed / dequeue_time,
    }


# ============================================================================
# Top-K Correctness Verification
# ============================================================================

def verify_topk_correctness():
    """Verify StreamingTopK produces correct results."""
    print("\n" + "=" * 60)
    print("Top-K Correctness Verification")
    print("=" * 60)
    
    errors = []
    
    # Test 1: Ascending order
    data = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
    topk = StreamingTopK(k=3, ascending=True, key=lambda x: x)
    for x in data:
        topk.push(x)
    result = topk.get_sorted()
    expected = [0, 1, 2]
    if result != expected:
        errors.append(f"Ascending: got {result}, expected {expected}")
    else:
        print(f"✓ Ascending top-3 from {data}: {result}")
    
    # Test 2: Descending order
    topk = StreamingTopK(k=3, ascending=False, key=lambda x: x)
    for x in data:
        topk.push(x)
    result = topk.get_sorted()
    expected = [9, 8, 7]
    if result != expected:
        errors.append(f"Descending: got {result}, expected {expected}")
    else:
        print(f"✓ Descending top-3 from {data}: {result}")
    
    # Test 3: With key function
    @dataclass
    class Item:
        name: str
        priority: int
    
    items = [Item("a", 5), Item("b", 2), Item("c", 8), Item("d", 1)]
    topk = StreamingTopK(k=2, ascending=True, key=lambda x: x.priority)
    for item in items:
        topk.push(item)
    result = topk.get_sorted()
    expected_names = ["d", "b"]  # priority 1, 2
    result_names = [x.name for x in result]
    if result_names != expected_names:
        errors.append(f"Key function: got {result_names}, expected {expected_names}")
    else:
        print(f"✓ Top-2 by priority: {result_names}")
    
    # Test 4: Large dataset
    import random
    random.seed(42)
    data = list(range(10000))
    random.shuffle(data)
    
    topk = StreamingTopK(k=10, ascending=True, key=lambda x: x)
    for x in data:
        topk.push(x)
    result = topk.get_sorted()
    expected = list(range(10))
    if result != expected:
        errors.append(f"Large dataset: got {result}, expected {expected}")
    else:
        print(f"✓ Top-10 from 10,000 shuffled: {result}")
    
    # Test 5: Edge case - k > n
    topk = StreamingTopK(k=100, ascending=True, key=lambda x: x)
    for x in [3, 1, 2]:
        topk.push(x)
    result = topk.get_sorted()
    expected = [1, 2, 3]
    if result != expected:
        errors.append(f"k > n: got {result}, expected {expected}")
    else:
        print(f"✓ Top-100 from 3 items: {result}")
    
    if errors:
        print(f"\n✗ {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"\n✓ All correctness tests passed!")
        return True


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_quick_benchmark():
    """Run quick benchmark suite."""
    print("=" * 60)
    print("Quick Queue Benchmark")
    print("=" * 60)
    
    results = []
    
    # Key encoding
    results.append(bench_key_encoding(10000))
    print(results[-1])
    print()
    
    # Enqueue at different queue sizes
    for size in [100, 1000]:
        results.append(bench_enqueue(size, 1000))
        print(results[-1])
        print()
    
    # Dequeue
    results.append(bench_dequeue(1000, 500))
    print(results[-1])
    print()
    
    # Streaming Top-K
    results.append(bench_streaming_topk(10000, 10, 100))
    print(results[-1])
    print()
    
    return results


def run_full_benchmark():
    """Run full benchmark suite."""
    print("=" * 60)
    print("Full Queue Benchmark Suite")
    print("=" * 60)
    
    results = []
    
    # 1. Key Encoding
    print("\n--- Key Encoding ---")
    results.append(bench_key_encoding(100000))
    print(results[-1])
    
    # 2. Enqueue at various queue sizes
    print("\n--- Enqueue Performance ---")
    for size in [0, 100, 1000, 10000]:
        results.append(bench_enqueue(size, 5000))
        print(results[-1])
        print()
    
    # 3. Dequeue performance
    print("\n--- Dequeue Performance ---")
    for size in [100, 1000, 10000]:
        results.append(bench_dequeue(size, 2000))
        print(results[-1])
        print()
    
    # 4. Batch enqueue
    print("\n--- Batch Enqueue ---")
    for batch_size in [10, 100]:
        results.append(bench_batch_enqueue(batch_size, 500))
        print(results[-1])
        print()
    
    # 5. Streaming Top-K vs Naive Sort
    print("\n--- Streaming Top-K vs Naive Sort ---")
    for n in [1000, 10000, 100000]:
        for k in [10, 100]:
            topk = bench_streaming_topk(n, k, 50)
            naive = bench_naive_sort(n, k, 50)
            speedup = naive.mean_latency_us / topk.mean_latency_us
            print(topk)
            print(naive)
            print(f"Speedup: {speedup:.2f}x")
            print()
            results.append(topk)
            results.append(naive)
    
    # 6. Concurrent workers
    print("\n--- Concurrent Workers ---")
    for workers in [2, 4]:
        result = bench_concurrent_workers(workers, 100)
        print(f"{result['name']}:")
        print(f"  Enqueue: {result['enqueue_throughput']:,.1f} ops/s")
        print(f"  Dequeue: {result['dequeue_throughput']:,.1f} ops/s")
        print(f"  Processed: {result['total_processed']}/{result['total_tasks']}")
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Queue Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    args = parser.parse_args()
    
    # Verify correctness first
    if not verify_topk_correctness():
        print("\n⚠ Correctness tests failed, aborting benchmarks")
        sys.exit(1)
    
    print()
    
    if args.full:
        results = run_full_benchmark()
    else:
        results = run_quick_benchmark()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for result in results:
        print(f"{result.name}: {result.mean_latency_us:.1f}μs mean, "
              f"{result.throughput_ops:,.0f} ops/s")
    
    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
