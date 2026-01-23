# SochDB Competitive Benchmarks

This directory contains comprehensive benchmarks comparing SochDB against major vector database competitors.

## Quick Start

```bash
# Install dependencies
pip install chromadb qdrant-client lancedb faiss-cpu python-dotenv openai

# Run the ultimate showdown
python benchmarks/ultimate_showdown.py

# Run real embedding demo (requires Azure OpenAI)
python benchmarks/real_search_demo.py
```

## Benchmark Scripts

### 1. `ultimate_showdown.py` - Comprehensive Comparison
Tests SochDB against all available competitors:
- **ChromaDB** - Python-based, simple embedded database
- **Qdrant** - Rust-based with excellent filtering
- **FAISS** - Facebook's C++ library (no persistence)
- **LanceDB** - Columnar embedded database

Dimensions tested: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)

### 2. `real_search_demo.py` - Real Embedding Demo
Demonstrates semantic search using actual Azure OpenAI embeddings. Requires `.env` with:
```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

### 3. `competitive_benchmark.py` - Full Competitive Suite
Extensive benchmark with real embeddings across multiple test sizes.

### 4. `rag_benchmark.py` - RAG-Realistic Workloads
Simulates actual RAG (Retrieval-Augmented Generation) workloads:
- Document ingestion
- Semantic search
- Batch queries (concurrent users)
- Filtered search
- Memory usage

### 5. `feature_benchmark.py` - Feature Differentiators
Tests SochDB's unique features:
- All commercial embedding dimensions (128-3072)
- Concurrent read/write access
- Batch operation efficiency
- Real embedding performance

## Expected Results

Based on testing, SochDB provides:

| Metric | SochDB | ChromaDB | Qdrant | FAISS | LanceDB |
|--------|--------|----------|--------|-------|---------|
| Insert (vec/s) | 2,000-10,000 | 3,000-5,000 | 5,000-10,000 | 50,000+ | 15,000+ |
| Search p50 | 0.3-0.5ms | 1-2ms | 0.5-1ms | 0.1-0.2ms | 5-10ms |
| Filtering | ✅ | ✅ | ✅ | ❌ | ✅ |
| Embedded | ✅ | ✅ | ❌ | ✅ | ✅ |
| SQL Interface | ✅ | ❌ | ❌ | ❌ | ❌ |

## SochDB Advantages

1. **🚀 Rust-Native Performance** - SIMD-accelerated distance calculations (NEON/AVX2)
2. **📦 Truly Embedded** - No server required, like SQLite for vectors
3. **🔢 All Dimensions** - Supports 128-3072 (MiniLM to OpenAI text-embedding-3-large)
4. **💾 SQL Interface** - Query vectors with familiar SQL syntax
5. **🔒 MVCC Transactions** - Safe concurrent reads and writes
6. **🕸️ Graph + Vector** - Hybrid knowledge graph + semantic search
7. **🐍 Python Simplicity** - Native Python bindings via FFI

## Competitors Overview

| Database | Type | Best For | Limitations |
|----------|------|----------|-------------|
| **Pinecone** | Cloud | Managed simplicity | Cloud-only, cost |
| **Weaviate** | Server | Hybrid search | Requires server |
| **Milvus** | Distributed | Large scale | Complexity |
| **Qdrant** | Server | Filtering | Requires server |
| **ChromaDB** | Embedded | Simple Python | Slower performance |
| **FAISS** | Library | Raw speed | No persistence |
| **LanceDB** | Embedded | Analytics | Slower search |
| **pgvector** | Extension | PostgreSQL users | Limited scale |
| **SochDB** | Embedded | AI/ML apps | Feature-rich |

## Running Benchmarks

```bash
# Full competitive analysis
cd sochdb-python-sdk
python benchmarks/ultimate_showdown.py

# Real embeddings (requires Azure OpenAI)
python benchmarks/real_search_demo.py

# RAG-realistic workloads
python benchmarks/rag_benchmark.py

# Feature tests
python benchmarks/feature_benchmark.py
```

## Environment Setup

For real embedding benchmarks, create `.env` in the project root:

```env
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## Results

Results are saved to:
- `showdown_results.json` - Ultimate showdown results
- `benchmark_results.json` - Competitive benchmark results
- `rag_benchmark_results.json` - RAG benchmark results
- `feature_benchmark_results.json` - Feature benchmark results

---

## 📊 Industry-Standard Performance Metrics

Based on **ANN-Benchmarks** (ann-benchmarks.com), **VectorDBBench** (Zilliz), and **Qdrant Benchmarks**:

### Primary Metrics (Required for Credible Benchmarks)

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **Recall@k** | Fraction of true k-nearest neighbors found | Measures search accuracy - the most critical metric |
| **QPS (Queries Per Second)** | Number of queries processed per second | Raw throughput for parallel workloads |
| **Latency p50/p95/p99** | Response time percentiles | User-perceived performance |
| **Index Build Time** | Time to construct the HNSW index | Critical for data ingestion pipelines |
| **Index Size (Memory)** | RAM required for the index | Cost and scalability factor |

### Recall vs QPS Tradeoff (The Gold Standard)

> **"The speed of vector databases should only be compared if they achieve the same precision."** 
> — Qdrant Benchmarks

ANN search is fundamentally about trading **precision for speed**. Any benchmark comparing two systems must use the **same recall threshold** (typically 0.95 or 0.99).

```
Recall@10 = (# of true neighbors in results) / 10
```

### Standard Benchmark Datasets (ANN-Benchmarks)

| Dataset | Vectors | Dimensions | Distance | Use Case |
|---------|---------|------------|----------|----------|
| **SIFT-1M** | 1,000,000 | 128 | Euclidean | Classic image descriptors |
| **GloVe-100** | 1,200,000 | 100 | Cosine | Word embeddings |
| **Fashion-MNIST** | 60,000 | 784 | Euclidean | Image classification |
| **GIST-960** | 1,000,000 | 960 | Euclidean | Scene recognition |
| **DBpedia-OpenAI-1M** | 1,000,000 | 1536 | Cosine | Real OpenAI embeddings |
| **Deep-Image-96** | 10,000,000 | 96 | Cosine | Large-scale images |

### VectorDBBench Scenarios

VectorDBBench (github.com/zilliztech/VectorDBBench) tests:

| Case Type | Vectors | Dimensions | Purpose |
|-----------|---------|------------|---------|
| Performance768D1M | 1M | 768 | BERT-class embeddings |
| Performance768D10M | 10M | 768 | Scale test |
| Performance1536D500K | 500K | 1536 | OpenAI embeddings |
| Performance1536D5M | 5M | 1536 | Large OpenAI scale |
| CapacityDim128 | Max | 128 | Stress test (SIFT) |
| CapacityDim960 | Max | 960 | Stress test (GIST) |

### Latency Percentiles Explained

| Percentile | Meaning | Target |
|------------|---------|--------|
| **p50 (median)** | Half of requests faster than this | < 1ms |
| **p95** | 95% of requests faster than this | < 5ms |
| **p99** | 99% of requests faster than this | < 10ms |
| **p999** | 99.9% (tail latency) | < 50ms |

High p99/p999 indicates **tail latency issues** that affect user experience.

### HNSW Index Parameters

| Parameter | Effect on Recall | Effect on Speed | Effect on Memory |
|-----------|------------------|-----------------|------------------|
| **M** (connections) | ↑ M = ↑ Recall | ↑ M = ↓ Speed | ↑ M = ↑ Memory |
| **ef_construction** | ↑ ef = ↑ Recall | ↑ ef = ↓ Build | No effect |
| **ef_search** | ↑ ef = ↑ Recall | ↑ ef = ↓ QPS | No effect |

Typical configurations:
- **High Recall (0.99+)**: M=32, ef_construction=256, ef_search=256
- **Balanced (0.95-0.98)**: M=16, ef_construction=128, ef_search=100
- **High Speed (0.90-0.95)**: M=8, ef_construction=64, ef_search=50

### Benchmark Methodology (Best Practices)

1. **Same Hardware**: All systems must run on identical hardware
2. **Same Dataset**: Use standard datasets (SIFT, GloVe, DBpedia)
3. **Same Recall**: Only compare at equivalent precision thresholds
4. **Warm Cache**: Run warmup queries before measurement
5. **Multiple Runs**: Report median of 5+ runs
6. **Separate Client/Server**: Use different machines for client and server (if applicable)

### Reference Hardware (VectorDBBench Standard)

```
Client: 8 vCPUs, 16 GB RAM (Azure Standard D8ls v5)
Server: 8 vCPUs, 32 GB RAM (Azure Standard D8s v3)
CPU: Intel Xeon Platinum 8375C @ 2.90GHz
Memory Limit: 25 GB (to ensure fairness)
```

### How to Interpret Results

#### Good Benchmark Report Shows:
✅ Recall@k vs QPS curves (the gold standard chart)  
✅ Multiple precision thresholds (0.90, 0.95, 0.99)  
✅ Latency percentiles (p50, p95, p99)  
✅ Index build time and memory usage  
✅ Dataset and hardware specifications  

#### Red Flags in Benchmarks:
❌ No recall measurement (speed without accuracy is meaningless)  
❌ Single data point (no precision/speed tradeoff shown)  
❌ Unknown or unreproducible hardware  
❌ Proprietary datasets  

---

## 🏆 SochDB Performance Targets

Based on industry benchmarks, SochDB targets:

| Metric | Target | Compared To |
|--------|--------|-------------|
| Recall@10 | ≥ 0.95 | Standard ANN threshold |
| QPS (single-thread) | ≥ 1,000 | ChromaDB baseline |
| Latency p50 | < 1ms | Qdrant/Milvus class |
| Latency p99 | < 10ms | Production-ready |
| Index Build | < 60s/1M vectors | Competitive |
| Memory | < 2x raw vector size | Efficient |

### Distance Metrics Supported

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Cosine** | 1 - (a·b / \|a\|\|b\|) | Text embeddings (default) |
| **Euclidean (L2)** | √Σ(aᵢ-bᵢ)² | Image features |
| **Dot Product** | -a·b | Pre-normalized vectors |

---

## 📚 References

- **ANN-Benchmarks**: https://ann-benchmarks.com/
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench
- **Qdrant Benchmarks**: https://qdrant.tech/benchmarks/
- **Zilliz Leaderboard**: https://zilliz.com/benchmark
- **Erik Bernhardsson's ANN Benchmarks**: https://github.com/erikbern/ann-benchmarks
