#!/usr/bin/env python3
"""
Real Embedding Semantic Search Demo
====================================

Demonstrates SochDB's vector search using REAL Azure OpenAI embeddings.
This shows the actual end-to-end experience of building a semantic search system.

Usage:
    python benchmarks/real_search_demo.py
"""

import sys
import os
import time
from pathlib import Path

# Add sochdb and load env
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / '.env')

import numpy as np
from openai import AzureOpenAI
from sochdb.vector import VectorIndex


# =============================================================================
# Sample Knowledge Base
# =============================================================================

KNOWLEDGE_BASE = [
    # Vector Database Concepts
    "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search with logarithmic complexity.",
    "Product quantization compresses high-dimensional vectors into smaller codes, reducing memory usage by 8-32x while maintaining search quality.",
    "Cosine similarity measures the angle between two vectors, ideal for comparing semantic embeddings regardless of magnitude.",
    "Vector embeddings are dense numerical representations that capture semantic meaning of text, images, or other data.",
    
    # SochDB Features  
    "SochDB is a high-performance vector database written in Rust, designed for AI and machine learning applications.",
    "SochDB supports SIMD-accelerated distance calculations using NEON on ARM and AVX2 on x86 processors.",
    "SochDB provides an SQL interface for querying vectors, similar to how SQLite works for traditional data.",
    "SochDB implements MVCC (Multi-Version Concurrency Control) for safe concurrent read and write operations.",
    
    # Machine Learning
    "Transformer models use self-attention mechanisms to process sequences in parallel, enabling training on massive datasets.",
    "Large language models like GPT-4 generate human-like text by predicting the next token based on context.",
    "RAG (Retrieval-Augmented Generation) combines vector search with LLMs to ground responses in factual information.",
    "Fine-tuning adapts pre-trained models to specific tasks using smaller, domain-specific datasets.",
    
    # Software Engineering
    "Rust provides memory safety without garbage collection through its ownership and borrowing system.",
    "FFI (Foreign Function Interface) enables Python to call native Rust code with near-zero overhead.",
    "Lock-free data structures avoid mutex contention in concurrent workloads using atomic operations.",
    "Write-ahead logging ensures database durability by persisting changes before committing transactions.",
]


def main():
    """Run the real embedding demo."""
    print("=" * 70)
    print("🔍 REAL SEMANTIC SEARCH WITH AZURE OPENAI + SOCHDB")
    print("=" * 70)
    
    # Check credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("❌ AZURE_OPENAI_API_KEY not set in .env")
        return
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    
    print(f"\n📚 Knowledge Base: {len(KNOWLEDGE_BASE)} documents")
    
    # Generate embeddings for knowledge base
    print("\n⏳ Generating embeddings...")
    start = time.perf_counter()
    response = client.embeddings.create(
        input=KNOWLEDGE_BASE,
        model="text-embedding-3-small",
    )
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    embed_time = time.perf_counter() - start
    
    print(f"   Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
    print(f"   Dimension: {embeddings.shape[1]}")
    
    # Build SochDB index
    print("\n🔧 Building SochDB index...")
    start = time.perf_counter()
    index = VectorIndex(
        dimension=embeddings.shape[1],
        max_connections=16,
        ef_construction=100,
    )
    ids = np.arange(len(embeddings), dtype=np.uint64)
    index.insert_batch_fast(ids, embeddings)
    build_time = (time.perf_counter() - start) * 1000
    print(f"   Built in {build_time:.1f}ms")
    
    # Interactive search
    queries = [
        "How does vector search work?",
        "What makes SochDB fast?",
        "How do I use embeddings in my app?",
        "What is RAG and how does it help LLMs?",
        "How does Rust ensure memory safety?",
    ]
    
    print("\n" + "=" * 70)
    print("🔎 SEMANTIC SEARCH RESULTS")
    print("=" * 70)
    
    for query in queries:
        # Embed query
        query_response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
        )
        query_embedding = np.array(query_response.data[0].embedding, dtype=np.float32)
        
        # Search
        start = time.perf_counter()
        results = index.search(query_embedding, k=3)
        search_time = (time.perf_counter() - start) * 1000
        
        print(f"\n❓ Query: \"{query}\"")
        print(f"   ⏱️  Search time: {search_time:.2f}ms")
        print(f"   📄 Top results:")
        
        for rank, (doc_id, score) in enumerate(results, 1):
            doc = KNOWLEDGE_BASE[doc_id]
            # Truncate for display
            display_doc = doc[:80] + "..." if len(doc) > 80 else doc
            similarity = 1 - score  # Convert distance to similarity
            print(f"      {rank}. [{similarity:.3f}] {display_doc}")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"   Embedding generation: {embed_time:.2f}s for {len(KNOWLEDGE_BASE)} docs")
    print(f"   Index build: {build_time:.1f}ms")
    print(f"   Average search: ~{search_time:.2f}ms")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    print("\n✅ Demo complete!")
    print("\n💡 This demonstrates real semantic search using:")
    print("   - Azure OpenAI text-embedding-3-small (1536 dimensions)")
    print("   - SochDB HNSW index with SIMD acceleration")
    print("   - Sub-millisecond search latency")


if __name__ == "__main__":
    main()
