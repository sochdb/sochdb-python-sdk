#!/usr/bin/env python3
"""
SochDB - 360 degree REAL-WORLD analysis: grep + HNSW  vs  BM25 + HNSW

This is not a synthetic micro-benchmark. It uses:

  * REAL embeddings   - Ollama `nomic-embed-text` (768-d) for documents+queries
  * REAL relevance    - an LLM judge (vLLM `nvidia/Qwen3.6-35B-A3B-NVFP4`,
                        reasoning disabled, temperature 0) grades every returned
                        (query, document) pair YES/NO. No hand-labels, no leakage.
  * A natural corpus  - hand-written passages across 6 domains, with deliberately
                        overlapping vocabulary and paraphrase-style queries so the
                        keyword/semantic trade-off actually surfaces.

It evaluates FIVE retrieval modes:

    1. vector            - pure HNSW ANN
    2. bm25              - pure BM25 keyword
    3. grep              - pure substring AND-match
    4. bm25 + hnsw       - hybrid (RRF fusion)
    5. grep + hnsw       - hybrid (RRF fusion)

and reports, per mode, averaged over all queries:

    * latency_ms   - search time (lower = faster)
    * precision@k  - LLM-judged fraction of returned docs that are relevant
    * recall@k     - relevant found / all relevant discovered (pooled over modes)
    * ndcg@k       - rank-quality of the relevant hits
    * coverage     - fraction of queries that returned at least one result

Finally it prints a 360 degree conclusion comparing grep+HNSW vs BM25+HNSW and
writes a JSON report next to this file.

Endpoints (override via env):
    SOCHDB_EMBED_URL   default http://localhost:11434/api/embeddings
    SOCHDB_EMBED_MODEL default nomic-embed-text
    SOCHDB_LLM_URL     default http://192.168.1.198:8000/v1/chat/completions
    SOCHDB_LLM_MODEL   default nvidia/Qwen3.6-35B-A3B-NVFP4

Run:
    SOCHDB_LIB_PATH=/path/to/dylib/dir PYTHONPATH=src \
        python examples/32_real_grep_vs_bm25_360.py
"""

import hashlib
import json
import math
import os
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from sochdb import Database
from sochdb.namespace import Namespace, CollectionConfig, SearchRequest


# --- Endpoints --------------------------------------------------------------

EMBED_URL = os.environ.get("SOCHDB_EMBED_URL", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.environ.get("SOCHDB_EMBED_MODEL", "nomic-embed-text")
LLM_URL = os.environ.get("SOCHDB_LLM_URL", "http://192.168.1.198:8000/v1/chat/completions")
LLM_MODEL = os.environ.get("SOCHDB_LLM_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

TOP_K = 8
DB_PATH = "./bench_real_360_db"
CACHE_DIR = Path(__file__).with_name(".cache_360")
REPORT_PATH = Path(__file__).with_name("real_360_report.json")
JUDGE_WORKERS = 6


# --- Real corpus ------------------------------------------------------------
# (domain, id, text). Vocabulary deliberately overlaps across domains
# ("index", "memory", "cell", "protocol") so lexical and semantic signals
# genuinely diverge.
CORPUS = [
    # databases
    ("db", "db1", "A B-tree index keeps keys sorted so range scans and lookups stay logarithmic even as the table grows."),
    ("db", "db2", "Write-ahead logging records every change before it touches the data pages, which is how the engine recovers after a crash."),
    ("db", "db3", "Serializable snapshot isolation lets transactions run concurrently while still preventing write skew anomalies."),
    ("db", "db4", "A hash index gives constant-time point lookups but, unlike a B-tree, cannot answer range queries."),
    ("db", "db5", "Vacuuming reclaims space from dead row versions left behind by multi-version concurrency control."),
    ("db", "db6", "Query planners use table statistics to estimate cardinality and pick between a sequential scan and an index scan."),
    ("db", "db7", "A covering index stores all the columns a query needs, so the engine never has to visit the heap."),
    ("db", "db8", "Sharding spreads rows across nodes by a partition key to scale writes horizontally."),
    # networking
    ("net", "net1", "TCP guarantees ordered, reliable delivery by retransmitting any packet that is not acknowledged in time."),
    ("net", "net2", "A router forwards packets between networks by consulting its routing table for the longest prefix match."),
    ("net", "net3", "DNS translates a human-friendly hostname into the IP address a client actually connects to."),
    ("net", "net4", "TLS performs a handshake to agree on keys, then encrypts the session so eavesdroppers learn nothing."),
    ("net", "net5", "Latency is the time a packet takes to reach its destination, while bandwidth is how much data fits per second."),
    ("net", "net6", "A load balancer distributes incoming connections across several backend servers to avoid overloading any one."),
    ("net", "net7", "UDP sends datagrams without acknowledgements, trading reliability for the low latency that live video needs."),
    ("net", "net8", "NAT lets many private devices share one public IP address by rewriting ports on outgoing connections."),
    # cooking
    ("cook", "cook1", "Searing meat over high heat triggers the Maillard reaction, building the browned crust that carries flavor."),
    ("cook", "cook2", "Letting a roast rest after the oven lets the juices redistribute so they do not spill out when sliced."),
    ("cook", "cook3", "Blooming spices in hot oil dissolves their aromatic compounds and deepens the taste of the dish."),
    ("cook", "cook4", "A gentle simmer keeps a stock clear, whereas a rolling boil makes it cloudy and greasy."),
    ("cook", "cook5", "Salting pasta water seasons the noodles from the inside as they absorb the liquid while cooking."),
    ("cook", "cook6", "Resting bread dough lets gluten relax so the loaf stretches instead of tearing when shaped."),
    ("cook", "cook7", "Deglazing a hot pan with wine lifts the caramelized bits into a quick, savory sauce."),
    ("cook", "cook8", "Caramelizing onions slowly converts their starches to sugar, turning them sweet and jammy."),
    # astronomy
    ("astro", "astro1", "A galaxy is a gravitationally bound system of stars, gas, dust, and dark matter spanning thousands of light-years."),
    ("astro", "astro2", "A comet grows a glowing tail as the Sun heats its icy nucleus and blows the released gas outward."),
    ("astro", "astro3", "A planet stays in orbit because the Sun's gravity continuously bends its otherwise straight-line motion."),
    ("astro", "astro4", "A nebula is a vast cloud of gas and dust where gravity can slowly collapse pockets into new stars."),
    ("astro", "astro5", "A telescope gathers faint light over a large aperture so we can see objects far too dim for the eye."),
    ("astro", "astro6", "A black hole forms when a massive star collapses so completely that not even light can escape it."),
    ("astro", "astro7", "Redshift stretches the light of distant galaxies, which is how we know the universe is expanding."),
    ("astro", "astro8", "A supernova is the violent explosion of a dying star that briefly outshines its entire galaxy."),
    # finance
    ("fin", "fin1", "Diversifying a portfolio across uncorrelated assets reduces risk without giving up much expected return."),
    ("fin", "fin2", "A dividend is a share of company profits paid out to shareholders, usually each quarter."),
    ("fin", "fin3", "Compound interest grows savings faster over time because each period earns interest on prior interest."),
    ("fin", "fin4", "A bond pays fixed coupons and returns its principal at maturity, making it less volatile than stocks."),
    ("fin", "fin5", "Liquidity measures how quickly an asset can be sold for cash without moving its price."),
    ("fin", "fin6", "Hedging offsets a position with an opposite one so a loss on one side is cushioned by the other."),
    ("fin", "fin7", "Inflation erodes purchasing power, so money sitting idle slowly buys less each year."),
    ("fin", "fin8", "An index fund tracks a whole market basket cheaply instead of betting on individual stocks."),
    # biology
    ("bio", "bio1", "An enzyme speeds up a biochemical reaction by lowering its activation energy without being consumed."),
    ("bio", "bio2", "DNA stores genetic instructions as a sequence of four bases coiled into a double helix."),
    ("bio", "bio3", "Mitochondria are the cell's power plants, turning nutrients into the ATP that fuels its work."),
    ("bio", "bio4", "During mitosis a cell duplicates its chromosomes and splits into two identical daughter cells."),
    ("bio", "bio5", "Proteins fold into precise three-dimensional shapes that determine the job each one performs."),
    ("bio", "bio6", "Photosynthesis lets plants capture sunlight and convert carbon dioxide and water into sugar."),
    ("bio", "bio7", "Natural selection favors traits that improve survival, gradually shifting a population over generations."),
    ("bio", "bio8", "A neuron transmits signals by firing electrical impulses that release chemicals across a synapse."),
]

# Queries: a mix of (a) literal-keyword queries whose terms appear verbatim in
# some docs (favours grep/BM25) and (b) paraphrase queries whose intent matches
# docs that share NO keyword (favours vector). This is what exposes the trade-off.
QUERIES = [
    # literal keyword present in corpus
    ("q1",  "B-tree index range scan"),
    ("q2",  "write-ahead logging crash recovery"),
    ("q3",  "TLS handshake encrypt session"),
    ("q4",  "comet tail heated by the Sun"),
    ("q5",  "dividend paid to shareholders"),
    ("q6",  "enzyme lowers activation energy"),
    # paraphrase / semantic (keywords largely absent from the target doc)
    ("q7",  "how does a database survive an unexpected power loss"),
    ("q8",  "why do faraway galaxies look redder"),
    ("q9",  "spreading investments to lower risk"),
    ("q10", "the part of a cell that makes energy"),
    ("q11", "keeping a soup broth clear while it cooks"),
    ("q12", "sharing one public address among many devices"),
]


# --- HTTP helpers with on-disk caching --------------------------------------

def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / kind / f"{h}.json"


def embed(text: str, session: requests.Session) -> list:
    """Embed text via Ollama, cached on disk."""
    cp = _cache_path("embed", f"{EMBED_MODEL}|{text}")
    if cp.exists():
        return json.loads(cp.read_text())["embedding"]
    resp = session.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=60)
    resp.raise_for_status()
    emb = resp.json()["embedding"]
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"embedding": emb}))
    return emb


def llm_judge(query: str, doc_text: str, session: requests.Session) -> bool:
    """Ask the LLM whether doc_text is relevant to query. Returns True/False. Cached."""
    cp = _cache_path("judge", f"{LLM_MODEL}|{query}|||{doc_text}")
    if cp.exists():
        return json.loads(cp.read_text())["relevant"]

    prompt = (
        "You are a strict search-relevance judge. Decide if the DOCUMENT directly "
        "answers or is clearly on-topic for the QUERY.\n"
        f"QUERY: {query}\n"
        f"DOCUMENT: {doc_text}\n"
        "Answer with exactly one word: YES or NO."
    )
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = session.post(LLM_URL, json=body, timeout=120)
    resp.raise_for_status()
    content = (resp.json()["choices"][0]["message"].get("content") or "").strip().upper()
    relevant = content.startswith("Y")
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"relevant": relevant, "raw": content}))
    return relevant


# --- Retrieval mode runners -------------------------------------------------

def run_query(collection, qvec, qtext, mode):
    """Return (latency_ms, [doc_id, ...]) for one query in one mode."""
    start = time.perf_counter()
    if mode == "vector":
        res = collection.vector_search(qvec, k=TOP_K)
    elif mode == "bm25":
        res = collection.search(SearchRequest(text_query=qtext, k=TOP_K, keyword_mode="bm25"))
    elif mode == "grep":
        res = collection.search(SearchRequest(text_query=qtext, k=TOP_K, keyword_mode="grep"))
    elif mode == "bm25+hnsw":
        res = collection.hybrid_search(qvec, qtext, k=TOP_K, alpha=0.5, keyword_mode="bm25")
    elif mode == "grep+hnsw":
        res = collection.hybrid_search(qvec, qtext, k=TOP_K, alpha=0.5, keyword_mode="grep")
    else:
        raise ValueError(mode)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return latency_ms, [str(r.id) for r in res.results]


def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_rels, total_relevant):
    if total_relevant == 0:
        return 0.0
    ideal = sorted(ranked_rels, reverse=True)
    # ideal list = as many 1s as min(k, total_relevant)
    ideal = [1.0] * min(len(ranked_rels), total_relevant)
    idcg = dcg(ideal)
    return (dcg(ranked_rels) / idcg) if idcg > 0 else 0.0


def main():
    print("=" * 70)
    print("SochDB - REAL 360 analysis:  grep + HNSW  vs  BM25 + HNSW")
    print("=" * 70)
    print(f"Embeddings : {EMBED_MODEL} @ {EMBED_URL}")
    print(f"LLM judge  : {LLM_MODEL} @ {LLM_URL}")
    print(f"Corpus     : {len(CORPUS)} docs | Queries: {len(QUERIES)} | k={TOP_K}\n")

    session = requests.Session()

    # 1. Connectivity check.
    try:
        _ = embed("connectivity check", session)
    except Exception as e:
        print(f"FATAL: embedding endpoint unreachable: {e}")
        return
    try:
        _ = llm_judge("ping", "pong", session)
    except Exception as e:
        print(f"FATAL: LLM judge endpoint unreachable: {e}")
        return

    # 2. Embed corpus + queries (cached).
    print("Embedding corpus + queries (Ollama, cached) ...")
    doc_text = {doc_id: text for _dom, doc_id, text in CORPUS}
    doc_vecs = {doc_id: embed(text, session) for _dom, doc_id, text in CORPUS}
    dim = len(next(iter(doc_vecs.values())))
    query_vecs = {qid: embed(text, session) for qid, text in QUERIES}
    print(f"  embedded {len(doc_vecs)} docs + {len(query_vecs)} queries (dim={dim})\n")

    # 3. Build the collection.
    if Path(DB_PATH).exists():
        shutil.rmtree(DB_PATH)
    db = Database.open(DB_PATH)
    ns = Namespace(db, "real360")
    collection = ns.create_collection(CollectionConfig(
        name="docs", dimension=dim, enable_hybrid_search=True, content_field="content",
    ))
    for _dom, doc_id, text in CORPUS:
        collection.insert(doc_id, doc_vecs[doc_id], metadata={"content": text}, content=text)

    # warm up (loads in-memory HNSW + metadata store).
    qid0, qtext0 = QUERIES[0]
    for m in ("vector", "bm25", "grep", "bm25+hnsw", "grep+hnsw"):
        run_query(collection, query_vecs[qid0], qtext0, m)

    modes = ["vector", "bm25", "grep", "bm25+hnsw", "grep+hnsw"]

    # 4. Run every query in every mode; collect results + latency.
    print("Running retrieval for all modes ...")
    results = {m: {} for m in modes}          # mode -> qid -> [doc_id]
    latencies = {m: [] for m in modes}        # mode -> [ms]
    pairs_to_judge = set()                     # (qid, doc_id)
    for qid, qtext in QUERIES:
        qvec = query_vecs[qid]
        for m in modes:
            lat, ids = run_query(collection, qvec, qtext, m)
            results[m][qid] = ids
            latencies[m].append(lat)
            for d in ids:
                pairs_to_judge.add((qid, d))
    print(f"  collected {len(pairs_to_judge)} unique (query, doc) pairs to judge\n")

    # 5. LLM-judge the pooled pairs (concurrent, cached).
    qtext_by_id = dict(QUERIES)
    print(f"LLM judging relevance ({JUDGE_WORKERS} workers) ...")
    pairs = sorted(pairs_to_judge)

    def judge_pair(pair):
        qid, doc_id = pair
        rel = llm_judge(qtext_by_id[qid], doc_text[doc_id], session)
        return pair, rel

    judged = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        for pair, rel in ex.map(judge_pair, pairs):
            judged[pair] = rel
    print(f"  judged {len(judged)} pairs in {time.perf_counter() - t0:.1f}s\n")

    # Pooled relevant set per query = union of judged-relevant docs across modes.
    relevant_by_query = {}
    for (qid, doc_id), rel in judged.items():
        if rel:
            relevant_by_query.setdefault(qid, set()).add(doc_id)

    # 6. Score each mode.
    report = {"config": {
        "embed_model": EMBED_MODEL, "llm_model": LLM_MODEL,
        "num_docs": len(CORPUS), "num_queries": len(QUERIES), "top_k": TOP_K,
    }, "modes": {}}

    for m in modes:
        precisions, recalls, ndcgs = [], [], []
        covered = 0
        for qid, _qtext in QUERIES:
            ids = results[m][qid]
            if ids:
                covered += 1
            rels = [1.0 if judged.get((qid, d)) else 0.0 for d in ids]
            total_rel = len(relevant_by_query.get(qid, set()))
            p = (sum(rels) / len(ids)) if ids else 0.0
            r = (sum(rels) / total_rel) if total_rel else 0.0
            precisions.append(p)
            recalls.append(r)
            ndcgs.append(ndcg_at_k(rels, total_rel))
        report["modes"][m] = {
            "latency_ms": round(statistics.mean(latencies[m]), 4),
            "p50_latency_ms": round(statistics.median(latencies[m]), 4),
            "precision_at_k": round(statistics.mean(precisions), 4),
            "recall_at_k": round(statistics.mean(recalls), 4),
            "ndcg_at_k": round(statistics.mean(ndcgs), 4),
            "coverage": round(covered / len(QUERIES), 4),
        }

    # 7. Print the 360 table.
    print("Results (averaged over all queries, LLM-judged)")
    print("-" * 92)
    print(f"{'mode':<14}{'lat ms':>9}{'p50 ms':>9}{'precision@k':>14}"
          f"{'recall@k':>12}{'ndcg@k':>10}{'coverage':>11}")
    print("-" * 92)
    for m in modes:
        s = report["modes"][m]
        print(f"{m:<14}{s['latency_ms']:>9.3f}{s['p50_latency_ms']:>9.3f}"
              f"{s['precision_at_k']:>14.3f}{s['recall_at_k']:>12.3f}"
              f"{s['ndcg_at_k']:>10.3f}{s['coverage']:>11.3f}")
    print("-" * 92)

    # 8. 360 conclusion: grep+hnsw vs bm25+hnsw.
    g = report["modes"]["grep+hnsw"]
    b = report["modes"]["bm25+hnsw"]
    faster = "grep+hnsw" if g["latency_ms"] < b["latency_ms"] else "bm25+hnsw"
    speedup = max(g["latency_ms"], b["latency_ms"]) / max(min(g["latency_ms"], b["latency_ms"]), 1e-9)

    def better(metric):
        if abs(g[metric] - b[metric]) < 1e-6:
            return "tie"
        return "grep+hnsw" if g[metric] > b[metric] else "bm25+hnsw"

    print("\n360 CONCLUSION  (grep + HNSW  vs  BM25 + HNSW)")
    print("-" * 70)
    print(f"  Speed      : {faster} faster ({speedup:.2f}x)  "
          f"[grep {g['latency_ms']:.3f}ms vs bm25 {b['latency_ms']:.3f}ms]")
    print(f"  Precision  : {better('precision_at_k')}  "
          f"[grep {g['precision_at_k']:.3f} vs bm25 {b['precision_at_k']:.3f}]")
    print(f"  Recall     : {better('recall_at_k')}  "
          f"[grep {g['recall_at_k']:.3f} vs bm25 {b['recall_at_k']:.3f}]")
    print(f"  nDCG       : {better('ndcg_at_k')}  "
          f"[grep {g['ndcg_at_k']:.3f} vs bm25 {b['ndcg_at_k']:.3f}]")
    print("-" * 70)
    print(
        "  Interpretation:\n"
        "   * grep + HNSW is a substring AND-match fused with semantic ANN: it is\n"
        "     cheapest and very precise WHEN query terms appear verbatim, but its\n"
        "     keyword leg contributes nothing on paraphrase queries (recall rests\n"
        "     entirely on the HNSW leg).\n"
        "   * BM25 + HNSW ranks partial/!inflected lexical overlap, so its keyword\n"
        "     leg still helps on near-miss wording, usually lifting recall/nDCG at a\n"
        "     small latency cost.\n"
        "  Rule of thumb: pick grep+HNSW for exact-term / code / id lookups and\n"
        "  latency-critical paths; pick BM25+HNSW for natural-language search where\n"
        "  wording varies."
    )

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nJSON report written: {REPORT_PATH}")
    db.close()


if __name__ == "__main__":
    main()
