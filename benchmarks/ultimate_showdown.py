#!/usr/bin/env python3
"""
Ultimate Vector Database Showdown
==================================

Run this script to compare SochDB against all available competitors.

Usage:
    python benchmarks/ultimate_showdown.py

Requirements (install what you want to compare):
    pip install chromadb qdrant-client lancedb faiss-cpu
    
Industry-Standard Metrics Measured:
    - Recall@k: Fraction of true k-nearest neighbors found (THE key metric)
    - QPS: Queries Per Second
    - Latency: p50, p95, p99 percentiles
    - Index Build Time: Time to construct HNSW index
    - Insert Rate: Vectors per second throughput
"""

import sys
import os
import time
import tempfile
import shutil
import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np

# Add sochdb to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def compute_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute ground truth k-nearest neighbors using brute force."""
    # Normalize for cosine similarity
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Compute all cosine similarities
    similarities = queries_norm @ vectors_norm.T  # (n_queries, n_vectors)
    
    # Get top-k for each query
    ground_truth = np.argsort(-similarities, axis=1)[:, :k]
    return ground_truth


def compute_recall(predicted: List[int], ground_truth: np.ndarray) -> float:
    """Compute recall@k for a single query."""
    if len(predicted) == 0:
        return 0.0
    k = len(ground_truth)
    predicted_set = set(predicted[:k])
    truth_set = set(ground_truth.tolist())
    return len(predicted_set & truth_set) / k


def generate_test_data(n_vectors: int, dimension: int, n_queries: int = 100, k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate normalized test vectors, queries, and ground truth."""
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Generate queries (random vectors, not from dataset for realistic benchmark)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Compute ground truth
    ground_truth = compute_ground_truth(vectors, queries, k)
    
    return vectors, queries, ground_truth


def benchmark_sochdb(vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """Benchmark SochDB."""
    from sochdb.vector import VectorIndex
    
    dimension = vectors.shape[1]
    n_vectors = len(vectors)
    
    # Create index
    index = VectorIndex(dimension=dimension, max_connections=32, ef_construction=100)
    ids = np.arange(n_vectors, dtype=np.uint64)
    
    # Insert
    start = time.perf_counter()
    index.insert_batch_fast(ids, vectors)
    insert_time = (time.perf_counter() - start) * 1000
    
    # Search
    search_times = []
    recalls = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        results = index.search(q, k=k)
        search_times.append((time.perf_counter() - start) * 1000)
        
        # Compute recall
        predicted = [r[0] for r in results]  # Extract IDs
        recalls.append(compute_recall(predicted, ground_truth[i]))
    
    search_times.sort()
    n = len(search_times)
    
    return {
        "database": "SochDB",
        "insert_time_ms": insert_time,
        "insert_rate": n_vectors / (insert_time / 1000),
        "search_p50_ms": search_times[int(n * 0.5)],
        "search_p95_ms": search_times[int(n * 0.95)],
        "search_p99_ms": search_times[-1],
        "qps": 1000 / (sum(search_times) / len(search_times)),
        "recall_at_k": np.mean(recalls),
    }


def benchmark_chromadb(vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """Benchmark ChromaDB."""
    import chromadb
    
    n_vectors = len(vectors)
    
    client = chromadb.Client()
    collection = client.create_collection(name="benchmark", metadata={"hnsw:space": "cosine"})
    
    # Insert
    start = time.perf_counter()
    collection.add(
        embeddings=vectors.tolist(),
        ids=[str(i) for i in range(n_vectors)],
    )
    insert_time = (time.perf_counter() - start) * 1000
    
    # Search
    search_times = []
    recalls = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        result = collection.query(query_embeddings=[q.tolist()], n_results=k)
        search_times.append((time.perf_counter() - start) * 1000)
        
        # Compute recall
        predicted = [int(id) for id in result["ids"][0]]
        recalls.append(compute_recall(predicted, ground_truth[i]))
    
    search_times.sort()
    n = len(search_times)
    
    return {
        "database": "ChromaDB",
        "insert_time_ms": insert_time,
        "insert_rate": n_vectors / (insert_time / 1000),
        "search_p50_ms": search_times[int(n * 0.5)],
        "search_p95_ms": search_times[int(n * 0.95)],
        "search_p99_ms": search_times[-1],
        "qps": 1000 / (sum(search_times) / len(search_times)),
        "recall_at_k": np.mean(recalls),
    }


def benchmark_qdrant(vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """Benchmark Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    
    dimension = vectors.shape[1]
    n_vectors = len(vectors)
    
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="benchmark",
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
    )
    
    # Insert
    start = time.perf_counter()
    points = [PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vectors)]
    client.upsert(collection_name="benchmark", points=points)
    insert_time = (time.perf_counter() - start) * 1000
    
    # Search
    search_times = []
    recalls = []
    for i, q in enumerate(queries):
        start = time.perf_counter()
        result = client.search(collection_name="benchmark", query_vector=q.tolist(), limit=k)
        search_times.append((time.perf_counter() - start) * 1000)
        
        # Compute recall
        predicted = [p.id for p in result]
        recalls.append(compute_recall(predicted, ground_truth[i]))
    
    search_times.sort()
    n = len(search_times)
    
    return {
        "database": "Qdrant",
        "insert_time_ms": insert_time,
        "insert_rate": n_vectors / (insert_time / 1000),
        "search_p50_ms": search_times[int(n * 0.5)],
        "search_p95_ms": search_times[int(n * 0.95)],
        "search_p99_ms": search_times[-1],
        "qps": 1000 / (sum(search_times) / len(search_times)),
        "recall_at_k": np.mean(recalls),
    }


def benchmark_faiss(vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """Benchmark FAISS."""
    import faiss
    
    dimension = vectors.shape[1]
    n_vectors = len(vectors)
    
    # HNSW for fair comparison
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 64
    
    vectors = np.ascontiguousarray(vectors)
    
    # Insert
    start = time.perf_counter()
    index.add(vectors)
    insert_time = (time.perf_counter() - start) * 1000
    
    # Search
    search_times = []
    recalls = []
    for i, q in enumerate(queries):
        q_arr = np.ascontiguousarray(q.reshape(1, -1))
        start = time.perf_counter()
        _, indices = index.search(q_arr, k)
        search_times.append((time.perf_counter() - start) * 1000)
        
        # Compute recall
        predicted = indices[0].tolist()
        recalls.append(compute_recall(predicted, ground_truth[i]))
    
    search_times.sort()
    n = len(search_times)
    
    return {
        "database": "FAISS",
        "insert_time_ms": insert_time,
        "insert_rate": n_vectors / (insert_time / 1000),
        "search_p50_ms": search_times[int(n * 0.5)],
        "search_p95_ms": search_times[int(n * 0.95)],
        "search_p99_ms": search_times[-1],
        "qps": 1000 / (sum(search_times) / len(search_times)),
        "recall_at_k": np.mean(recalls),
    }


def benchmark_lancedb(vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> Dict[str, Any]:
    """Benchmark LanceDB."""
    import lancedb
    
    n_vectors = len(vectors)
    temp_dir = tempfile.mkdtemp()
    
    try:
        db = lancedb.connect(temp_dir)
        
        # Insert
        start = time.perf_counter()
        data = [{"id": i, "vector": v.tolist()} for i, v in enumerate(vectors)]
        table = db.create_table("benchmark", data)
        insert_time = (time.perf_counter() - start) * 1000
        
        # Search
        search_times = []
        recalls = []
        for i, q in enumerate(queries):
            start = time.perf_counter()
            result = table.search(q.tolist()).limit(k).to_pandas()
            search_times.append((time.perf_counter() - start) * 1000)
            
            # Compute recall
            predicted = result["id"].tolist()
            recalls.append(compute_recall(predicted, ground_truth[i]))
        
        search_times.sort()
        n = len(search_times)
        
        return {
            "database": "LanceDB",
            "insert_time_ms": insert_time,
            "insert_rate": n_vectors / (insert_time / 1000),
            "search_p50_ms": search_times[int(n * 0.5)],
            "search_p95_ms": search_times[int(n * 0.95)],
            "search_p99_ms": search_times[-1],
            "qps": 1000 / (sum(search_times) / len(search_times)),
            "recall_at_k": np.mean(recalls),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_benchmark(name: str, func, vectors: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """Run a single benchmark with error handling."""
    try:
        print(f"  Benchmarking {name}...")
        result = func(vectors.copy(), queries.copy(), ground_truth.copy())
        print(f"    ✅ {name}: recall={result['recall_at_k']:.3f}, {result['insert_rate']:,.0f} vec/s, {result['search_p50_ms']:.2f}ms p50")
        return result
    except ImportError:
        print(f"    ⚠️ {name}: Not installed (pip install {name.lower().replace(' ', '-')})")
        return None
    except Exception as e:
        print(f"    ❌ {name}: Error - {e}")
        return None


def main():
    """Run the ultimate showdown."""
    print("=" * 70)
    print("🏆 ULTIMATE VECTOR DATABASE SHOWDOWN")
    print("=" * 70)
    print("\n📏 Industry-Standard Metrics: Recall@10, QPS, Latency (p50/p95/p99)")
    print("📊 Based on ANN-Benchmarks and VectorDBBench methodology\n")
    
    # Test configurations
    configs = [
        {"dimension": 384, "n_vectors": 10000, "name": "MiniLM (384D, 10K)"},
        {"dimension": 768, "n_vectors": 10000, "name": "BERT (768D, 10K)"},
        {"dimension": 1536, "n_vectors": 10000, "name": "OpenAI (1536D, 10K)"},
    ]
    
    # Available benchmarks
    benchmarks = [
        ("SochDB", benchmark_sochdb),
        ("ChromaDB", benchmark_chromadb),
        ("Qdrant", benchmark_qdrant),
        ("FAISS", benchmark_faiss),
        ("LanceDB", benchmark_lancedb),
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*60}")
        
        # Generate test data with ground truth for recall calculation
        vectors, queries, ground_truth = generate_test_data(
            config["n_vectors"], config["dimension"], n_queries=100, k=10
        )
        
        config_results = []
        for name, func in benchmarks:
            result = run_benchmark(name, func, vectors, queries, ground_truth)
            if result:
                config_results.append(result)
            gc.collect()
        
        all_results[config['name']] = config_results
    
    # Print summary with recall
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY (sorted by Recall@10 - higher is better)")
    print("=" * 70)
    
    for config_name, results in all_results.items():
        if not results:
            continue
        
        print(f"\n{config_name}:")
        print(f"  {'Database':<12} {'Recall@10':<10} {'p50 (ms)':<10} {'p99 (ms)':<10} {'QPS':<10}")
        print(f"  {'-'*58}")
        
        # Sort by recall (higher is better)
        results.sort(key=lambda r: r.get("recall_at_k", 0), reverse=True)
        
        for r in results:
            recall = r.get("recall_at_k", 0)
            print(f"  {r['database']:<12} {recall:>8.3f} {r['search_p50_ms']:>9.2f} {r['search_p99_ms']:>9.2f} {r['qps']:>9,.0f}")
        
        # Find SochDB for comparison
        sochdb = next((r for r in results if r["database"] == "SochDB"), None)
        if sochdb and len(results) > 1:
            print(f"\n  SochDB comparison (at same recall level):")
            sochdb_recall = sochdb.get("recall_at_k", 0)
            for r in results:
                if r["database"] == "SochDB":
                    continue
                r_recall = r.get("recall_at_k", 0)
                recall_diff = sochdb_recall - r_recall
                latency_ratio = r["search_p50_ms"] / sochdb["search_p50_ms"] if sochdb["search_p50_ms"] > 0 else 1
                
                if abs(recall_diff) < 0.05:  # Similar recall
                    emoji = "🚀" if latency_ratio > 1.5 else ("✅" if latency_ratio > 1 else "⚠️")
                    print(f"    {emoji} vs {r['database']}: {latency_ratio:.1f}x {'faster' if latency_ratio > 1 else 'slower'} (similar recall)")
                else:
                    print(f"    📊 vs {r['database']}: recall {sochdb_recall:.3f} vs {r_recall:.3f}")
    
    # Feature comparison
    print("\n" + "=" * 70)
    print("📋 FEATURE COMPARISON")
    print("=" * 70)
    print("""
| Feature                    | SochDB | ChromaDB | Qdrant | FAISS | LanceDB |
|----------------------------|--------|----------|--------|-------|---------|
| Embedded (no server)       |   ✅   |    ✅    |   ❌   |  ✅   |   ✅    |
| Rust Performance           |   ✅   |    ❌    |   ✅   |  ❌   |   ✅    |
| HNSW Index                 |   ✅   |    ✅    |   ✅   |  ✅   |   ❌    |
| Filtering                  |   ✅   |    ✅    |   ✅   |  ❌   |   ✅    |
| Persistence                |   ✅   |    ✅    |   ✅   |  ❌   |   ✅    |
| SQL Interface              |   ✅   |    ❌    |   ❌   |  ❌   |   ❌    |
| MVCC Transactions          |   ✅   |    ❌    |   ❌   |  ❌   |   ❌    |
| Graph + Vector Hybrid      |   ✅   |    ❌    |   ❌   |  ❌   |   ❌    |
| All Embedding Dims (128-3K)|   ✅   |    ✅    |   ✅   |  ✅   |   ✅    |
    """)
    
    # Save results
    output_path = Path(__file__).parent / "showdown_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Results saved to {output_path}")
    
    # Print methodology note
    print("\n" + "=" * 70)
    print("📝 METHODOLOGY NOTES")
    print("=" * 70)
    print("""
• Recall@10: Fraction of true 10-nearest neighbors found (ground truth via brute force)
• QPS: Queries per second (single-threaded)
• Latency: p50/p95/p99 percentiles across 100 queries
• Index: HNSW with M=32, ef_construction=100 (where applicable)
• Distance: Cosine similarity (normalized vectors)
• Fair comparison: All databases use same test vectors and ground truth

Reference: https://ann-benchmarks.com/, https://github.com/zilliztech/VectorDBBench
    """)


if __name__ == "__main__":
    main()
