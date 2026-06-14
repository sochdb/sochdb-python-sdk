#!/usr/bin/env python3
"""
SochDB Python SDK - Benchmark: "grep + HNSW" vs "BM25 + HNSW"

Both modes fuse a keyword leg with the HNSW vector leg using Reciprocal Rank
Fusion (RRF). The only difference is the keyword leg:

    * BM25 + HNSW : relevance-ranked keyword search (IDF/TF weighting)
    * grep + HNSW : case-insensitive substring AND-match (every term must appear)

This script builds a reproducible, topic-labelled synthetic corpus and reports,
for each mode:

    * latency   - average query time in milliseconds (lower = faster)
    * precision - fraction of top-k results in the query's ground-truth topic
                  (higher = more accurate)

It then prints a verdict on which mode is faster and which is more accurate.

Run:
    python examples/31_hybrid_grep_vs_bm25.py

No server and no embedding model required - embeddings are derived
deterministically from each document's topic so the keyword and vector signals
agree on relevance.
"""

import math
import random
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from sochdb import Database
from sochdb.namespace import Namespace, CollectionConfig


# --- Reproducible corpus configuration -------------------------------------

SEED = 1234
DIM = 64
NUM_DOCS = 3000
TOP_K = 10
QUERIES_PER_TOPIC = 5
DB_PATH = "./bench_grep_vs_bm25_db"

# Each topic has a unique keyword and a random centroid in embedding space.
# Documents of a topic cluster around its centroid; a query for that topic
# uses the topic keyword and a near-centroid vector. Ground truth for a query
# is therefore "all documents of the same topic".
TOPICS = {
    "databases": ["index", "btree", "transaction", "query", "storage"],
    "networking": ["packet", "router", "latency", "protocol", "bandwidth"],
    "cooking": ["recipe", "saute", "simmer", "garlic", "oven"],
    "astronomy": ["galaxy", "nebula", "orbit", "telescope", "comet"],
    "finance": ["portfolio", "dividend", "equity", "hedge", "liquidity"],
    "biology": ["enzyme", "genome", "protein", "mitosis", "cell"],
}


def build_embeddings(rng: np.random.Generator):
    """Create one random unit centroid per topic."""
    centroids = {}
    for topic in TOPICS:
        v = rng.standard_normal(DIM).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        centroids[topic] = v
    return centroids


def make_doc_vector(centroid: np.ndarray, rng: np.random.Generator) -> list:
    """Centroid + small noise, re-normalised to the unit sphere."""
    noise = rng.standard_normal(DIM).astype(np.float32) * 0.25
    v = centroid + noise
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def build_corpus(rng: np.random.Generator, centroids):
    """Return a list of (doc_id, topic, content, vector)."""
    topics = list(TOPICS.keys())
    docs = []
    for i in range(NUM_DOCS):
        topic = rng.choice(topics)
        keywords = TOPICS[topic]
        # Content = a few topic keywords plus some generic filler words.
        chosen = list(rng.choice(keywords, size=3, replace=False))
        filler = list(rng.choice(
            ["system", "method", "approach", "value", "design", "process"],
            size=4, replace=True,
        ))
        words = chosen + filler
        rng.shuffle(words)
        content = " ".join(words)
        vector = make_doc_vector(centroids[topic], rng)
        docs.append((f"doc_{i}", topic, content, vector))
    return docs


def build_queries(rng: np.random.Generator, centroids):
    """Return a list of (topic, text_query, query_vector)."""
    queries = []
    for topic, keywords in TOPICS.items():
        for _ in range(QUERIES_PER_TOPIC):
            # Query text: two real topic keywords (present in some docs).
            terms = list(rng.choice(keywords, size=2, replace=False))
            text = " ".join(terms)
            qv = make_doc_vector(centroids[topic], rng)
            queries.append((topic, text, qv))
    return queries


def precision_at_k(results, topic_of, query_topic) -> float:
    """Fraction of returned results that belong to the query's topic."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if topic_of.get(str(r.id)) == query_topic)
    return hits / len(results)


def run_mode(collection, queries, topic_of, keyword_mode: str):
    """Run every query in the given hybrid mode; return (avg_ms, avg_precision)."""
    latencies = []
    precisions = []
    for query_topic, text, qv in queries:
        start = time.perf_counter()
        res = collection.hybrid_search(
            vector=qv,
            text_query=text,
            k=TOP_K,
            alpha=0.5,
            keyword_mode=keyword_mode,
        )
        latencies.append((time.perf_counter() - start) * 1000.0)
        precisions.append(precision_at_k(res.results, topic_of, query_topic))
    return statistics.mean(latencies), statistics.mean(precisions)


def main():
    print("=" * 64)
    print("SochDB - grep + HNSW  vs  BM25 + HNSW")
    print("=" * 64)

    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Clean slate for a deterministic run.
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)

    centroids = build_embeddings(rng)
    docs = build_corpus(rng, centroids)
    queries = build_queries(rng, centroids)
    topic_of = {doc_id: topic for doc_id, topic, _c, _v in docs}

    print(f"\nCorpus : {NUM_DOCS} docs across {len(TOPICS)} topics, dim={DIM}")
    print(f"Queries: {len(queries)} ({QUERIES_PER_TOPIC} per topic), k={TOP_K}\n")

    db = Database.open(DB_PATH)
    ns = Namespace(db, "bench")
    config = CollectionConfig(
        name="hybrid",
        dimension=DIM,
        enable_hybrid_search=True,
        content_field="content",
    )
    collection = ns.create_collection(config)

    print("Indexing documents ...")
    t0 = time.perf_counter()
    for doc_id, _topic, content, vector in docs:
        collection.insert(doc_id, vector, metadata={"content": content}, content=content)
    print(f"  indexed {len(docs)} docs in {time.perf_counter() - t0:.2f}s\n")

    # Warm up (loads / rebuilds the in-memory HNSW from the snapshot).
    collection.hybrid_search(queries[0][2], queries[0][1], k=TOP_K, keyword_mode="bm25")
    collection.hybrid_search(queries[0][2], queries[0][1], k=TOP_K, keyword_mode="grep")

    bm25_ms, bm25_prec = run_mode(collection, queries, topic_of, "bm25")
    grep_ms, grep_prec = run_mode(collection, queries, topic_of, "grep")

    print("Results")
    print("-" * 64)
    print(f"{'mode':<16}{'avg latency (ms)':>20}{'precision@%d' % TOP_K:>18}")
    print("-" * 64)
    print(f"{'BM25 + HNSW':<16}{bm25_ms:>20.3f}{bm25_prec:>18.3f}")
    print(f"{'grep + HNSW':<16}{grep_ms:>20.3f}{grep_prec:>18.3f}")
    print("-" * 64)

    faster = "grep + HNSW" if grep_ms < bm25_ms else "BM25 + HNSW"
    speedup = max(bm25_ms, grep_ms) / max(min(bm25_ms, grep_ms), 1e-9)
    if abs(grep_prec - bm25_prec) < 1e-6:
        accurate = "tie"
    else:
        accurate = "grep + HNSW" if grep_prec > bm25_prec else "BM25 + HNSW"

    print(f"\nFaster   : {faster}  ({speedup:.2f}x)")
    print(f"Accurate : {accurate}"
          f"  (BM25={bm25_prec:.3f}  grep={grep_prec:.3f})")
    print(
        "\nNote: grep is an exact substring AND-match (high precision, no ranking "
        "model);\nBM25 adds IDF/TF relevance ranking. Which wins depends on your "
        "data and\nwhether queries use literal terms that appear verbatim in "
        "documents."
    )

    db.close()


if __name__ == "__main__":
    main()
