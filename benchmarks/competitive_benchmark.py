#!/usr/bin/env python3
"""
Comprehensive Vector Database Competitive Benchmark
====================================================

Tests SochDB against major vector database competitors using REAL embeddings
from Azure OpenAI (text-embedding-3-large @ 3072 dimensions).

Competitors tested:
1. ChromaDB - Simple, embedded, Python-focused
2. Qdrant - Rust-based, HNSW, good filtering  
3. LanceDB - Columnar, embedded
4. FAISS - Facebook's foundational library

Note: Cloud-only services (Pinecone, Weaviate Cloud, MongoDB Atlas) are excluded
as they require network calls and don't provide fair local comparisons.

Usage:
    python benchmarks/competitive_benchmark.py

Requirements:
    pip install chromadb qdrant-client lancedb faiss-cpu openai python-dotenv numpy
"""

import os
import sys
import time
import json
import gc
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics

import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent.parent / '.env')

# Add sochdb to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Embedding model
    embedding_model: str = "text-embedding-3-small"  # 1536 dimensions
    embedding_dimension: int = 1536
    
    # Test sizes (progressive)
    test_sizes: List[int] = field(default_factory=lambda: [100, 1_000, 10_000])
    
    # Queries per test
    n_queries: int = 100
    
    # Top-k for search
    top_k: int = 10
    
    # Number of runs for statistical significance
    n_runs: int = 3
    
    # Sample texts for embeddings
    sample_texts: List[str] = field(default_factory=lambda: [
        "Machine learning enables computers to learn from data without being explicitly programmed.",
        "Neural networks are computing systems inspired by biological neural networks in the brain.",
        "Deep learning uses multiple layers of neural networks to progressively extract features.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Reinforcement learning trains agents to make sequences of decisions.",
        "Transfer learning leverages knowledge from one domain to solve problems in another.",
        "Generative AI creates new content including text, images, and code.",
        "Vector databases store and search high-dimensional embedding vectors efficiently.",
        "Approximate nearest neighbor search trades perfect accuracy for massive speedups.",
        "HNSW graphs enable logarithmic search complexity in vector databases.",
        "Product quantization compresses vectors while maintaining search quality.",
        "Semantic search understands meaning rather than just matching keywords.",
        "RAG combines retrieval with generation for accurate AI responses.",
        "Embeddings capture semantic meaning in dense vector representations.",
        "Transformers revolutionized NLP with attention mechanisms.",
        "Large language models can generate human-like text at scale.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Prompt engineering optimizes inputs for better AI outputs.",
        "AI agents can autonomously complete complex multi-step tasks.",
    ])


# =============================================================================
# Embedding Generator
# =============================================================================

class EmbeddingGenerator:
    """Generate real embeddings using Azure OpenAI."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._client = None
        self._cache: Dict[str, np.ndarray] = {}
        
    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            
            self._client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        return self._client
    
    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Embed a batch of texts."""
        # Check cache first
        uncached_texts = [t for t in texts if t not in self._cache]
        
        if uncached_texts:
            if show_progress:
                print(f"    Generating {len(uncached_texts)} embeddings via Azure OpenAI...")
            
            # Batch in chunks of 100 (API limit)
            batch_size = 100
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.config.embedding_model,
                )
                
                for j, item in enumerate(response.data):
                    self._cache[batch[j]] = np.array(item.embedding, dtype=np.float32)
        
        # Return embeddings in order
        return np.array([self._cache[t] for t in texts], dtype=np.float32)
    
    def generate_dataset(self, n_vectors: int) -> Tuple[np.ndarray, List[str]]:
        """Generate a dataset of embeddings by cycling through sample texts."""
        texts = []
        for i in range(n_vectors):
            base_text = self.config.sample_texts[i % len(self.config.sample_texts)]
            # Add variation to make embeddings unique
            text = f"{base_text} (variant {i})"
            texts.append(text)
        
        embeddings = self.embed_batch(texts, show_progress=True)
        return embeddings, texts


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    database: str
    n_vectors: int
    dimension: int
    
    # Insert metrics
    insert_time_ms: float
    insert_rate: float  # vectors/sec
    
    # Search metrics  
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float
    search_qps: float  # queries/sec
    
    # Recall (if computed)
    recall_at_k: Optional[float] = None
    
    # Memory (if available)
    memory_mb: Optional[float] = None
    
    # Additional info
    notes: str = ""


@dataclass 
class CompetitiveResults:
    """Aggregated competitive benchmark results."""
    timestamp: str
    config: Dict[str, Any]
    results: List[BenchmarkResult]
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [
                {
                    "database": r.database,
                    "n_vectors": r.n_vectors,
                    "dimension": r.dimension,
                    "insert_time_ms": r.insert_time_ms,
                    "insert_rate": r.insert_rate,
                    "search_p50_ms": r.search_p50_ms,
                    "search_p95_ms": r.search_p95_ms,
                    "search_p99_ms": r.search_p99_ms,
                    "search_qps": r.search_qps,
                    "recall_at_k": r.recall_at_k,
                    "memory_mb": r.memory_mb,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }


# =============================================================================
# Database Benchmarks
# =============================================================================

class BaseBenchmark:
    """Base class for database benchmarks."""
    
    name: str = "Base"
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.temp_dir = None
        
    def setup(self, dimension: int):
        """Initialize the database."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown(self):
        """Cleanup resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        gc.collect()
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        """Insert vectors, return time in ms."""
        raise NotImplementedError
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors."""
        raise NotImplementedError
    
    def run_benchmark(self, vectors: np.ndarray, queries: np.ndarray) -> BenchmarkResult:
        """Run complete benchmark."""
        n_vectors = len(vectors)
        dimension = vectors.shape[1]
        ids = list(range(n_vectors))
        
        # Setup
        self.setup(dimension)
        
        try:
            # Insert benchmark
            insert_time = self.insert(vectors, ids)
            insert_rate = n_vectors / (insert_time / 1000)
            
            # Search benchmark
            search_times = []
            for query in queries:
                start = time.perf_counter()
                _ = self.search(query, self.config.top_k)
                search_times.append((time.perf_counter() - start) * 1000)
            
            search_times.sort()
            n = len(search_times)
            
            return BenchmarkResult(
                database=self.name,
                n_vectors=n_vectors,
                dimension=dimension,
                insert_time_ms=insert_time,
                insert_rate=insert_rate,
                search_p50_ms=search_times[int(n * 0.5)],
                search_p95_ms=search_times[int(n * 0.95)],
                search_p99_ms=search_times[int(n * 0.99)] if n >= 100 else search_times[-1],
                search_qps=1000 / statistics.mean(search_times),
            )
        finally:
            self.teardown()


class SochDBBenchmark(BaseBenchmark):
    """SochDB benchmark."""
    
    name = "SochDB"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        from sochdb.vector import VectorIndex
        
        # Optimal HNSW config for high dimensions
        self.index = VectorIndex(
            dimension=dimension,
            max_connections=32,  # Higher for 3072-dim
            ef_construction=100,
        )
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        ids_arr = np.array(ids, dtype=np.uint64)
        start = time.perf_counter()
        self.index.insert_batch_fast(ids_arr, vectors)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        return self.index.search(query, k=k)


class ChromaDBBenchmark(BaseBenchmark):
    """ChromaDB benchmark."""
    
    name = "ChromaDB"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.temp_dir,
                anonymized_telemetry=False,
            ))
            self.collection = self.client.create_collection(
                name="benchmark",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError("ChromaDB not installed: pip install chromadb")
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        start = time.perf_counter()
        # ChromaDB requires string IDs
        self.collection.add(
            embeddings=vectors.tolist(),
            ids=[str(i) for i in ids],
        )
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        results = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
        )
        # Return (id, distance) pairs
        return [(int(id), d) for id, d in zip(results["ids"][0], results["distances"][0])]


class QdrantBenchmark(BaseBenchmark):
    """Qdrant benchmark (in-memory mode)."""
    
    name = "Qdrant"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            self.client = QdrantClient(":memory:")
            self.client.create_collection(
                collection_name="benchmark",
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            self.PointStruct = PointStruct
        except ImportError:
            raise ImportError("Qdrant not installed: pip install qdrant-client")
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        start = time.perf_counter()
        points = [
            self.PointStruct(id=i, vector=v.tolist())
            for i, v in zip(ids, vectors)
        ]
        self.client.upsert(
            collection_name="benchmark",
            points=points,
        )
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        results = self.client.search(
            collection_name="benchmark",
            query_vector=query.tolist(),
            limit=k,
        )
        return [(r.id, r.score) for r in results]


class LanceDBBenchmark(BaseBenchmark):
    """LanceDB benchmark."""
    
    name = "LanceDB"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            import lancedb
            
            self.db = lancedb.connect(self.temp_dir)
            self.dimension = dimension
            self.table = None
        except ImportError:
            raise ImportError("LanceDB not installed: pip install lancedb")
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        import pyarrow as pa
        
        start = time.perf_counter()
        data = [
            {"id": i, "vector": v.tolist()}
            for i, v in zip(ids, vectors)
        ]
        self.table = self.db.create_table("benchmark", data)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        results = self.table.search(query.tolist()).limit(k).to_pandas()
        return [(r["id"], r["_distance"]) for _, r in results.iterrows()]


class FAISSBenchmark(BaseBenchmark):
    """FAISS benchmark (IVF-Flat with HNSW)."""
    
    name = "FAISS"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            import faiss
            
            self.dimension = dimension
            # Use HNSW for fair comparison
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
            self.index.hnsw.efConstruction = 100
            self.index.hnsw.efSearch = 64
        except ImportError:
            raise ImportError("FAISS not installed: pip install faiss-cpu")
    
    def insert(self, vectors: np.ndarray, ids: List[int]) -> float:
        # FAISS requires contiguous C-array
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = time.perf_counter()
        self.index.add(vectors)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query, k)
        return [(int(i), float(d)) for i, d in zip(indices[0], distances[0])]


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Run competitive benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.embedder = EmbeddingGenerator(config)
        self.results: List[BenchmarkResult] = []
        
    def run_all(self) -> CompetitiveResults:
        """Run all benchmarks."""
        print("=" * 80)
        print("COMPETITIVE VECTOR DATABASE BENCHMARK")
        print("=" * 80)
        print(f"Embedding model: {self.config.embedding_model}")
        print(f"Dimension: {self.config.embedding_dimension}")
        print(f"Test sizes: {self.config.test_sizes}")
        print(f"Queries per test: {self.config.n_queries}")
        print("=" * 80)
        
        # Initialize benchmarks
        benchmarks = [
            SochDBBenchmark(self.config),
            ChromaDBBenchmark(self.config),
            QdrantBenchmark(self.config),
            LanceDBBenchmark(self.config),
            FAISSBenchmark(self.config),
        ]
        
        for n_vectors in self.config.test_sizes:
            print(f"\n{'='*60}")
            print(f"DATASET: {n_vectors:,} vectors @ {self.config.embedding_dimension} dimensions")
            print(f"{'='*60}")
            
            # Generate dataset
            print(f"\nGenerating {n_vectors} embeddings...")
            vectors, _ = self.embedder.generate_dataset(n_vectors)
            
            # Generate query vectors (subset of data)
            query_indices = np.random.choice(n_vectors, min(self.config.n_queries, n_vectors), replace=False)
            queries = vectors[query_indices]
            
            # Run each benchmark
            for benchmark in benchmarks:
                print(f"\n  {benchmark.name}:")
                try:
                    result = benchmark.run_benchmark(vectors.copy(), queries.copy())
                    self.results.append(result)
                    
                    print(f"    Insert: {result.insert_time_ms:.1f}ms ({result.insert_rate:,.0f} vec/s)")
                    print(f"    Search: p50={result.search_p50_ms:.2f}ms, p99={result.search_p99_ms:.2f}ms")
                    print(f"    QPS: {result.search_qps:,.0f}")
                except ImportError as e:
                    print(f"    ⚠️ Skipped: {e}")
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                
                gc.collect()
        
        return CompetitiveResults(
            timestamp=datetime.now().isoformat(),
            config={
                "embedding_model": self.config.embedding_model,
                "dimension": self.config.embedding_dimension,
                "test_sizes": self.config.test_sizes,
                "n_queries": self.config.n_queries,
            },
            results=self.results,
        )
    
    def print_summary(self, results: CompetitiveResults):
        """Print summary comparison."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Group by test size
        for n_vectors in self.config.test_sizes:
            size_results = [r for r in results.results if r.n_vectors == n_vectors]
            if not size_results:
                continue
            
            print(f"\n{n_vectors:,} vectors @ {self.config.embedding_dimension}D:")
            print("-" * 70)
            print(f"{'Database':<12} {'Insert (vec/s)':<16} {'Search p50':<12} {'Search p99':<12} {'QPS':<10}")
            print("-" * 70)
            
            # Sort by search p50 (fastest first)
            size_results.sort(key=lambda r: r.search_p50_ms)
            
            for r in size_results:
                print(f"{r.database:<12} {r.insert_rate:>14,.0f} {r.search_p50_ms:>10.2f}ms {r.search_p99_ms:>10.2f}ms {r.search_qps:>9,.0f}")
        
        # Overall winner
        print("\n" + "=" * 80)
        print("OVERALL ANALYSIS")
        print("=" * 80)
        
        # Find SochDB results for largest test
        largest_size = max(self.config.test_sizes)
        sochdb_result = next((r for r in results.results if r.database == "SochDB" and r.n_vectors == largest_size), None)
        
        if sochdb_result:
            competitors = [r for r in results.results if r.database != "SochDB" and r.n_vectors == largest_size]
            
            print(f"\nSochDB vs Competitors ({largest_size:,} vectors):")
            for comp in competitors:
                insert_ratio = sochdb_result.insert_rate / comp.insert_rate if comp.insert_rate > 0 else float('inf')
                search_ratio = comp.search_p50_ms / sochdb_result.search_p50_ms if sochdb_result.search_p50_ms > 0 else float('inf')
                
                insert_status = "🚀" if insert_ratio > 1.5 else ("✅" if insert_ratio > 0.8 else "⚠️")
                search_status = "🚀" if search_ratio > 1.5 else ("✅" if search_ratio > 0.8 else "⚠️")
                
                print(f"  vs {comp.database}:")
                print(f"    Insert: {insert_status} {insert_ratio:.1f}x {'faster' if insert_ratio > 1 else 'slower'}")
                print(f"    Search: {search_status} {search_ratio:.1f}x {'faster' if search_ratio > 1 else 'slower'}")


# =============================================================================
# Feature Comparison Matrix
# =============================================================================

def print_feature_comparison():
    """Print feature comparison matrix."""
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON MATRIX")
    print("=" * 80)
    
    features = [
        ("Embedded (no server)", ["SochDB", "ChromaDB", "LanceDB", "FAISS"]),
        ("HNSW Index", ["SochDB", "ChromaDB", "Qdrant", "FAISS"]),
        ("Filtering Support", ["SochDB", "ChromaDB", "Qdrant", "LanceDB"]),
        ("Product Quantization", ["SochDB", "Qdrant", "FAISS"]),
        ("Rust-based (fast)", ["SochDB", "Qdrant", "LanceDB"]),
        ("SQL Interface", ["SochDB"]),
        ("MVCC Transactions", ["SochDB"]),
        ("Graph + Vector Hybrid", ["SochDB"]),
        ("Python SDK", ["SochDB", "ChromaDB", "Qdrant", "LanceDB", "FAISS"]),
        ("Persistence", ["SochDB", "ChromaDB", "Qdrant", "LanceDB"]),
        ("Distributed/Sharding", ["Qdrant"]),
        ("Multi-modal", ["LanceDB"]),
    ]
    
    databases = ["SochDB", "ChromaDB", "Qdrant", "LanceDB", "FAISS"]
    
    print(f"\n{'Feature':<30}", end="")
    for db in databases:
        print(f"{db:<12}", end="")
    print()
    print("-" * 90)
    
    for feature, supported in features:
        print(f"{feature:<30}", end="")
        for db in databases:
            status = "✅" if db in supported else "❌"
            print(f"{status:<12}", end="")
        print()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the competitive benchmark."""
    # Check for API key
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ Error: AZURE_OPENAI_API_KEY not set in .env file")
        print("   Please configure your Azure OpenAI credentials")
        sys.exit(1)
    
    # Configure benchmark
    config = BenchmarkConfig(
        test_sizes=[100, 1_000, 10_000],  # Adjust based on API limits/cost
        n_queries=100,
    )
    
    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all()
    
    # Print summary
    runner.print_summary(results)
    
    # Print feature comparison
    print_feature_comparison()
    
    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\n📊 Results saved to {output_path}")


if __name__ == "__main__":
    main()
