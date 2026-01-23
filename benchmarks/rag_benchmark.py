#!/usr/bin/env python3
"""
Real-World RAG Benchmark
========================

Tests vector databases in realistic RAG (Retrieval-Augmented Generation) scenarios
using Azure OpenAI embeddings. This benchmark simulates actual production workloads:

1. Document Ingestion - Chunked document embedding and storage
2. Semantic Search - Finding relevant context for LLM queries
3. Hybrid Queries - Filtering + vector search (where supported)
4. Batch Operations - Multi-query processing for concurrent users
5. Memory Efficiency - Measuring memory footprint at scale

This test uses REAL embeddings from Azure OpenAI, not synthetic random vectors.
"""

import os
import sys
import time
import json
import gc
import psutil
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import statistics
import traceback

import numpy as np
from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Add sochdb to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Simulated Document Corpus
# =============================================================================

DOCUMENT_CORPUS = [
    # Technical documentation
    "SochDB is a high-performance vector database written in Rust, designed for AI applications.",
    "The HNSW algorithm provides logarithmic search complexity for approximate nearest neighbor queries.",
    "Product quantization reduces memory usage by 8-32x while maintaining search quality above 95%.",
    "Vector embeddings capture semantic meaning in dense numerical representations.",
    "The database supports cosine similarity, Euclidean distance, and dot product metrics.",
    
    # AI/ML concepts
    "Transformer models use self-attention mechanisms to process sequences in parallel.",
    "Large language models like GPT-4 can generate human-quality text across many domains.",
    "Fine-tuning adapts pre-trained models to specific tasks with smaller datasets.",
    "RAG combines retrieval with generation to ground LLM responses in factual information.",
    "Semantic search understands query intent rather than just matching keywords.",
    
    # Software engineering
    "Rust provides memory safety without garbage collection through its ownership system.",
    "FFI (Foreign Function Interface) allows Python to call native Rust code efficiently.",
    "SIMD instructions process multiple data elements in parallel for faster computations.",
    "MVCC (Multi-Version Concurrency Control) enables concurrent reads without locks.",
    "Persistent data structures maintain immutability while allowing efficient updates.",
    
    # Database concepts
    "B+ trees provide O(log n) lookups and efficient range scans for ordered data.",
    "Write-ahead logging ensures durability by persisting changes before committing.",
    "Indexes trade storage space and write speed for faster query performance.",
    "Sharding distributes data across multiple nodes for horizontal scalability.",
    "Replication provides high availability and read scalability in distributed systems.",
    
    # Production concerns
    "Monitoring and observability are crucial for maintaining production systems.",
    "Load balancing distributes traffic across multiple servers to prevent overload.",
    "Rate limiting protects APIs from abuse and ensures fair resource allocation.",
    "Circuit breakers prevent cascading failures in microservices architectures.",
    "Caching reduces latency and backend load for frequently accessed data.",
    
    # Security
    "Authentication verifies user identity through credentials or tokens.",
    "Authorization controls what authenticated users can access or modify.",
    "Encryption protects data both in transit (TLS) and at rest (AES).",
    "API keys should be rotated regularly and never committed to version control.",
    "Input validation prevents injection attacks and data corruption.",
    
    # Performance optimization
    "Batch processing amortizes overhead by handling multiple items together.",
    "Connection pooling reduces the cost of establishing database connections.",
    "Prefetching anticipates data needs to hide memory latency.",
    "Cache locality groups related data to minimize cache misses.",
    "Lock-free algorithms avoid contention in concurrent workloads.",
]

QUERY_TEMPLATES = [
    "How does {} work?",
    "What is the best approach for {}?",
    "Explain the concept of {}",
    "How to implement {}",
    "What are the benefits of {}",
    "Describe the architecture of {}",
    "How to optimize {}",
    "What is the difference between {} and alternatives?",
]

QUERY_TOPICS = [
    "vector search", "HNSW indexing", "product quantization",
    "semantic similarity", "embedding generation", "RAG systems",
    "Rust performance", "memory safety", "SIMD optimization",
    "database transactions", "concurrent access", "data persistence",
    "API security", "rate limiting", "load balancing",
    "cache optimization", "batch processing", "horizontal scaling",
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGBenchmarkConfig:
    """RAG benchmark configuration."""
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Corpus settings
    n_documents: int = 1000  # Number of document chunks
    chunk_size: int = 512    # Characters per chunk (simulated)
    
    # Query settings
    n_queries: int = 50
    top_k: int = 5  # Typical for RAG context
    
    # Batch query settings (simulating concurrent users)
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 50])
    
    # Memory test sizes
    memory_test_sizes: List[int] = field(default_factory=lambda: [1000, 5000, 10000])


# =============================================================================
# Embedding Service
# =============================================================================

class AzureEmbeddingService:
    """Azure OpenAI embedding service with caching."""
    
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model
        self._client = None
        self._cache: Dict[str, np.ndarray] = {}
        self._api_calls = 0
        self._cached_hits = 0
    
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
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching."""
        results = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text in self._cache:
                self._cached_hits += 1
                results.append((i, self._cache[text]))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        if texts_to_embed:
            self._api_calls += 1
            response = self.client.embeddings.create(
                input=texts_to_embed,
                model=self.model,
            )
            
            for j, item in enumerate(response.data):
                vec = np.array(item.embedding, dtype=np.float32)
                self._cache[texts_to_embed[j]] = vec
                results.append((text_indices[j], vec))
        
        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return np.array([v for _, v in results], dtype=np.float32)
    
    def stats(self) -> Dict[str, int]:
        return {
            "api_calls": self._api_calls,
            "cache_hits": self._cached_hits,
            "cache_size": len(self._cache),
        }


# =============================================================================
# Database Wrappers
# =============================================================================

class DatabaseWrapper:
    """Base class for database wrappers."""
    
    name: str = "Base"
    
    def __init__(self):
        self.temp_dir = None
    
    def setup(self, dimension: int):
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown(self):
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        gc.collect()
    
    def insert_documents(self, vectors: np.ndarray, metadata: List[Dict]) -> float:
        """Insert documents with metadata, return time in ms."""
        raise NotImplementedError
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        """Search for k nearest neighbors."""
        raise NotImplementedError
    
    def search_batch(self, query_vectors: np.ndarray, k: int) -> List[List[Dict]]:
        """Batch search."""
        return [self.search(q, k) for q in query_vectors]
    
    def search_with_filter(self, query_vector: np.ndarray, k: int, filter_dict: Dict) -> List[Dict]:
        """Filtered search (not all databases support this)."""
        raise NotImplementedError
    
    def memory_usage_mb(self) -> float:
        """Estimate memory usage."""
        return psutil.Process().memory_info().rss / 1024 / 1024


class SochDBWrapper(DatabaseWrapper):
    """SochDB wrapper."""
    
    name = "SochDB"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        from sochdb.vector import VectorIndex
        
        self.index = VectorIndex(
            dimension=dimension,
            max_connections=32,
            ef_construction=100,
        )
        self.metadata_store = {}
    
    def insert_documents(self, vectors: np.ndarray, metadata: List[Dict]) -> float:
        ids = np.arange(len(vectors), dtype=np.uint64)
        
        start = time.perf_counter()
        self.index.insert_batch_fast(ids, vectors)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Store metadata separately
        for i, meta in enumerate(metadata):
            self.metadata_store[i] = meta
        
        return elapsed
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        results = self.index.search(query_vector, k=k)
        return [
            {"id": int(id), "score": float(score), "metadata": self.metadata_store.get(int(id), {})}
            for id, score in results
        ]


class ChromaDBWrapper(DatabaseWrapper):
    """ChromaDB wrapper."""
    
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
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError("ChromaDB not installed")
    
    def insert_documents(self, vectors: np.ndarray, metadata: List[Dict]) -> float:
        start = time.perf_counter()
        self.collection.add(
            embeddings=vectors.tolist(),
            ids=[str(i) for i in range(len(vectors))],
            metadatas=metadata,
        )
        return (time.perf_counter() - start) * 1000
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            include=["metadatas", "distances"],
        )
        return [
            {"id": int(id), "score": float(d), "metadata": m}
            for id, d, m in zip(results["ids"][0], results["distances"][0], results["metadatas"][0])
        ]
    
    def search_with_filter(self, query_vector: np.ndarray, k: int, filter_dict: Dict) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=filter_dict,
            include=["metadatas", "distances"],
        )
        if not results["ids"][0]:
            return []
        return [
            {"id": int(id), "score": float(d), "metadata": m}
            for id, d, m in zip(results["ids"][0], results["distances"][0], results["metadatas"][0])
        ]


class QdrantWrapper(DatabaseWrapper):
    """Qdrant wrapper."""
    
    name = "Qdrant"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            self.client = QdrantClient(":memory:")
            self.client.create_collection(
                collection_name="documents",
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            self.PointStruct = PointStruct
        except ImportError:
            raise ImportError("Qdrant not installed")
    
    def insert_documents(self, vectors: np.ndarray, metadata: List[Dict]) -> float:
        start = time.perf_counter()
        points = [
            self.PointStruct(id=i, vector=v.tolist(), payload=m)
            for i, (v, m) in enumerate(zip(vectors, metadata))
        ]
        self.client.upsert(collection_name="documents", points=points)
        return (time.perf_counter() - start) * 1000
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        results = self.client.search(
            collection_name="documents",
            query_vector=query_vector.tolist(),
            limit=k,
        )
        return [
            {"id": r.id, "score": r.score, "metadata": r.payload}
            for r in results
        ]
    
    def search_with_filter(self, query_vector: np.ndarray, k: int, filter_dict: Dict) -> List[Dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Convert filter dict to Qdrant filter
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter_dict.items()
        ]
        
        results = self.client.search(
            collection_name="documents",
            query_vector=query_vector.tolist(),
            query_filter=Filter(must=conditions),
            limit=k,
        )
        return [
            {"id": r.id, "score": r.score, "metadata": r.payload}
            for r in results
        ]


class FAISSWrapper(DatabaseWrapper):
    """FAISS wrapper (no filtering support)."""
    
    name = "FAISS"
    
    def setup(self, dimension: int):
        super().setup(dimension)
        try:
            import faiss
            
            self.index = faiss.IndexHNSWFlat(dimension, 32)
            self.index.hnsw.efConstruction = 100
            self.index.hnsw.efSearch = 64
            self.metadata_store = {}
        except ImportError:
            raise ImportError("FAISS not installed")
    
    def insert_documents(self, vectors: np.ndarray, metadata: List[Dict]) -> float:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        start = time.perf_counter()
        self.index.add(vectors)
        elapsed = (time.perf_counter() - start) * 1000
        
        for i, m in enumerate(metadata):
            self.metadata_store[i] = m
        
        return elapsed
    
    def search(self, query_vector: np.ndarray, k: int) -> List[Dict]:
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query, k)
        return [
            {"id": int(i), "score": float(d), "metadata": self.metadata_store.get(int(i), {})}
            for i, d in zip(indices[0], distances[0]) if i >= 0
        ]


# =============================================================================
# RAG Benchmark
# =============================================================================

class RAGBenchmark:
    """Run RAG-focused benchmarks."""
    
    def __init__(self, config: RAGBenchmarkConfig):
        self.config = config
        self.embedder = AzureEmbeddingService(config.embedding_model)
        
    def generate_corpus(self) -> Tuple[List[str], List[Dict]]:
        """Generate document corpus with metadata."""
        documents = []
        metadata = []
        categories = ["technical", "ai", "engineering", "database", "operations", "security", "performance"]
        
        for i in range(self.config.n_documents):
            # Cycle through base documents with variations
            base_doc = DOCUMENT_CORPUS[i % len(DOCUMENT_CORPUS)]
            doc = f"{base_doc} (Document {i}, chunk {i % 10})"
            documents.append(doc)
            
            metadata.append({
                "doc_id": i,
                "category": categories[i % len(categories)],
                "chunk_index": i % 10,
                "source": f"doc_{i // 10}.md",
            })
        
        return documents, metadata
    
    def generate_queries(self) -> List[str]:
        """Generate realistic queries."""
        queries = []
        for i in range(self.config.n_queries):
            template = QUERY_TEMPLATES[i % len(QUERY_TEMPLATES)]
            topic = QUERY_TOPICS[i % len(QUERY_TOPICS)]
            queries.append(template.format(topic))
        return queries
    
    def run_benchmark(self, database: DatabaseWrapper) -> Dict[str, Any]:
        """Run full RAG benchmark on a database."""
        results = {
            "database": database.name,
            "dimension": self.config.embedding_dimension,
            "n_documents": self.config.n_documents,
        }
        
        try:
            # Setup
            database.setup(self.config.embedding_dimension)
            
            # Generate and embed corpus
            print(f"    Generating corpus ({self.config.n_documents} documents)...")
            documents, metadata = self.generate_corpus()
            
            print(f"    Embedding documents...")
            doc_vectors = self.embedder.embed(documents)
            
            # Test 1: Document Ingestion
            print(f"    Testing document ingestion...")
            insert_time = database.insert_documents(doc_vectors, metadata)
            results["insert_time_ms"] = insert_time
            results["insert_rate"] = self.config.n_documents / (insert_time / 1000)
            
            # Generate and embed queries
            print(f"    Embedding queries...")
            queries = self.generate_queries()
            query_vectors = self.embedder.embed(queries)
            
            # Test 2: Single Query Search
            print(f"    Testing single query search...")
            search_times = []
            for qv in query_vectors:
                start = time.perf_counter()
                _ = database.search(qv, self.config.top_k)
                search_times.append((time.perf_counter() - start) * 1000)
            
            search_times.sort()
            n = len(search_times)
            results["search_p50_ms"] = search_times[int(n * 0.5)]
            results["search_p95_ms"] = search_times[int(n * 0.95)]
            results["search_p99_ms"] = search_times[-1]
            results["search_qps"] = 1000 / statistics.mean(search_times)
            
            # Test 3: Batch Query Search
            print(f"    Testing batch query search...")
            batch_results = {}
            for batch_size in self.config.batch_sizes:
                batch_queries = query_vectors[:batch_size]
                start = time.perf_counter()
                _ = database.search_batch(batch_queries, self.config.top_k)
                batch_time = (time.perf_counter() - start) * 1000
                batch_results[batch_size] = {
                    "total_ms": batch_time,
                    "per_query_ms": batch_time / batch_size,
                    "qps": batch_size / (batch_time / 1000),
                }
            results["batch_search"] = batch_results
            
            # Test 4: Filtered Search (if supported)
            print(f"    Testing filtered search...")
            try:
                filter_times = []
                for qv in query_vectors[:10]:
                    start = time.perf_counter()
                    _ = database.search_with_filter(qv, self.config.top_k, {"category": "technical"})
                    filter_times.append((time.perf_counter() - start) * 1000)
                
                results["filtered_search_p50_ms"] = statistics.median(filter_times)
                results["filtered_search_supported"] = True
            except (NotImplementedError, Exception) as e:
                results["filtered_search_supported"] = False
                results["filtered_search_note"] = str(e)
            
            # Test 5: Memory Usage
            results["memory_mb"] = database.memory_usage_mb()
            
            results["status"] = "success"
            
        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            traceback.print_exc()
        finally:
            database.teardown()
        
        return results
    
    def run_all(self) -> Dict[str, Any]:
        """Run benchmarks on all databases."""
        print("=" * 80)
        print("RAG-REALISTIC VECTOR DATABASE BENCHMARK")
        print("=" * 80)
        print(f"Embedding Model: {self.config.embedding_model}")
        print(f"Dimension: {self.config.embedding_dimension}")
        print(f"Documents: {self.config.n_documents}")
        print(f"Queries: {self.config.n_queries}")
        print("=" * 80)
        
        databases = [
            SochDBWrapper(),
            ChromaDBWrapper(),
            QdrantWrapper(),
            FAISSWrapper(),
        ]
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "embedding_model": self.config.embedding_model,
                "dimension": self.config.embedding_dimension,
                "n_documents": self.config.n_documents,
                "n_queries": self.config.n_queries,
            },
            "embedding_stats": None,
            "results": [],
        }
        
        for db in databases:
            print(f"\n{'='*60}")
            print(f"Benchmarking: {db.name}")
            print(f"{'='*60}")
            
            try:
                result = self.run_benchmark(db)
                all_results["results"].append(result)
                
                if result["status"] == "success":
                    print(f"\n    Results:")
                    print(f"      Insert: {result['insert_time_ms']:.1f}ms ({result['insert_rate']:,.0f} vec/s)")
                    print(f"      Search: p50={result['search_p50_ms']:.2f}ms, p99={result['search_p99_ms']:.2f}ms")
                    print(f"      QPS: {result['search_qps']:,.0f}")
                    print(f"      Filtering: {'✅' if result.get('filtered_search_supported') else '❌'}")
                    print(f"      Memory: {result['memory_mb']:.1f}MB")
            except ImportError as e:
                print(f"    ⚠️ Skipped: {e}")
                all_results["results"].append({
                    "database": db.name,
                    "status": "skipped",
                    "reason": str(e),
                })
            
            gc.collect()
        
        all_results["embedding_stats"] = self.embedder.stats()
        
        return all_results
    
    def print_comparison(self, results: Dict[str, Any]):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        
        successful = [r for r in results["results"] if r.get("status") == "success"]
        
        if not successful:
            print("No successful benchmarks to compare")
            return
        
        print(f"\n{'Database':<12} {'Insert (vec/s)':<16} {'Search p50':<12} {'Search p99':<12} {'QPS':<10} {'Filter':<8}")
        print("-" * 80)
        
        # Sort by search latency
        successful.sort(key=lambda r: r.get("search_p50_ms", float('inf')))
        
        for r in successful:
            filter_status = "✅" if r.get("filtered_search_supported") else "❌"
            print(f"{r['database']:<12} {r['insert_rate']:>14,.0f} {r['search_p50_ms']:>10.2f}ms {r['search_p99_ms']:>10.2f}ms {r['search_qps']:>9,.0f} {filter_status:<8}")
        
        # SochDB analysis
        sochdb = next((r for r in successful if r["database"] == "SochDB"), None)
        if sochdb:
            print("\n" + "-" * 80)
            print("SochDB vs Competitors:")
            for r in successful:
                if r["database"] == "SochDB":
                    continue
                
                insert_ratio = sochdb["insert_rate"] / r["insert_rate"] if r["insert_rate"] > 0 else 0
                search_ratio = r["search_p50_ms"] / sochdb["search_p50_ms"] if sochdb["search_p50_ms"] > 0 else 0
                
                insert_emoji = "🚀" if insert_ratio > 1.5 else ("✅" if insert_ratio > 0.8 else "⚠️")
                search_emoji = "🚀" if search_ratio > 1.5 else ("✅" if search_ratio > 0.8 else "⚠️")
                
                print(f"  vs {r['database']}: Insert {insert_emoji} {insert_ratio:.1f}x, Search {search_emoji} {search_ratio:.1f}x faster")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the RAG benchmark."""
    # Check credentials
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("❌ AZURE_OPENAI_API_KEY not set")
        print("Please configure .env file with Azure OpenAI credentials")
        sys.exit(1)
    
    config = RAGBenchmarkConfig(
        n_documents=1000,
        n_queries=50,
    )
    
    benchmark = RAGBenchmark(config)
    results = benchmark.run_all()
    
    # Print comparison
    benchmark.print_comparison(results)
    
    # Save results
    output_path = Path(__file__).parent / "rag_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📊 Results saved to {output_path}")
    
    # Print embedding stats
    print(f"\n📡 Embedding API Stats:")
    print(f"   API calls: {results['embedding_stats']['api_calls']}")
    print(f"   Cache hits: {results['embedding_stats']['cache_hits']}")
    print(f"   Cache size: {results['embedding_stats']['cache_size']}")


if __name__ == "__main__":
    main()
