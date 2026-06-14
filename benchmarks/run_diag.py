#!/usr/bin/env python3
"""Decomposed retrieval diagnostic for SochDB on LoCoMo.

Separates the two things the earlier RAG run conflated:
  (1) RETRIEVAL quality  — evidence recall@k, using LoCoMo's gold `dia_id`
      evidence labels. Independent of any LLM.
  (2) ANSWER quality     — QA accuracy given retrieved context (optional, --answer).

Retrievers compared on identical per-conversation corpora (turn-level units):
  - first_n        : first k turns, no retrieval        (naive floor)
  - bm25           : rank_bm25 over turn text           (lexical baseline)
  - sochdb_vector  : server-side HNSW via gRPC          (the number that matters)
  - sochdb_keyword : embedded Collection BM25           (if native libs present)
  - sochdb_hybrid  : embedded Collection hybrid         (if native libs present)

Embeddings: fastembed BAAI/bge-small-en-v1.5 (CPU ONNX, fast) — replaces the
CPU-crippled ollama path so the vector arm is actually testable at scale.

Example:
  python run_diag.py --locomo-data /root/locomo/data/locomo10.json \
      --address 127.0.0.1:50061 --k 5 10 20 --answer --answer-k 10 \
      --answer-retrievers sochdb_vector bm25
"""
from __future__ import annotations
import argparse, json, re, time, math, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import urllib.request, urllib.error

from sochdb import SochDBClient
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# ---- optional embedded Collection (sochdb keyword/hybrid) ----
try:
    from sochdb import open_collection  # needs libsochdb_storage.so
    _HAS_EMBEDDED = True
except Exception:
    _HAS_EMBEDDED = False


def _tok(s: str) -> List[str]:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


# -------------------- LLM (for optional --answer) --------------------
def _post(url, payload, timeout=120, retries=6):
    body = json.dumps(payload).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e; time.sleep(min(15.0, 2.0 ** a))
    raise RuntimeError(f"POST {url} failed: {last}")


def chat(base, model, system, user, max_tokens=256):
    d = _post(base.rstrip("/") + "/chat/completions", {
        "model": model, "max_tokens": max_tokens, "temperature": 0.0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=180)
    m = d["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "").strip()


def judge(base, model, q, gold, pred):
    try:
        v = chat(base, model,
                 "Grade strictly. Reply only YES or NO. YES iff the candidate answer matches any reference in meaning.",
                 f"Question: {q}\nReference(s): {' | '.join(gold)}\nCandidate: {pred}\nCorrect? (YES/NO):", 4)
        return v.strip().upper().startswith("Y")
    except Exception:
        return False


def sub_hit(pred, gold):
    p = _norm(pred)
    return any(_norm(g) and _norm(g) in p for g in gold)


# -------------------- corpus + evidence --------------------
def session_keys(conv):
    ks = [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")]
    ks.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
    return ks


def build_corpus(conv) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Return (dia_ids, texts, dia2text) for every turn, in order."""
    dia_ids, texts, dia2text = [], [], {}
    for sk in session_keys(conv):
        date = conv.get(f"{sk}_date_time", "") or ""
        for t in (conv.get(sk) or []):
            d = t.get("dia_id")
            txt = t.get("text", t.get("clean_text", ""))
            if not d or not txt:
                continue
            full = f"[{date}] {t.get('speaker','')}: {txt}"
            dia_ids.append(d); texts.append(full); dia2text[d] = full
    return dia_ids, texts, dia2text


def doc_ids(res) -> List[int]:
    """Extract integer ids from a gRPC Document list or embedded SearchResults."""
    out = []
    seq = res if isinstance(res, list) else getattr(res, "results", res)
    try:
        for it in seq:
            i = getattr(it, "id", None)
            if i is None and isinstance(it, dict):
                i = it.get("id")
            if i is not None:
                out.append(int(i))
    except TypeError:
        pass
    return out


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo-data", default="/root/locomo/data/locomo10.json")
    ap.add_argument("--address", default="127.0.0.1:50061")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--samples", type=int, default=0, help="cap conversations; 0=all")
    ap.add_argument("--limit", type=int, default=0, help="cap QAs total; 0=all")
    ap.add_argument("--qa-per-conv", type=int, default=0, help="cap QAs per conversation (balanced subset); 0=all")
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--answer-k", type=int, default=10)
    ap.add_argument("--answer-retrievers", nargs="+", default=["sochdb_vector", "bm25"])
    ap.add_argument("--llm-base", default="http://100.127.255.44:8000/v1")
    ap.add_argument("--llm-model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    ap.add_argument("--max-answer-tokens", type=int, default=256)
    ap.add_argument("--out", default="/root/results_diag/locomo_diag.json")
    args = ap.parse_args()

    raw = json.loads(Path(args.locomo_data).read_text())
    if args.samples:
        raw = raw[: args.samples]
    emb = TextEmbedding(args.embed_model)
    client = SochDBClient(args.address)
    kmax = max(args.k)

    retrievers = ["first_n", "bm25", "sochdb_vector"]
    if _HAS_EMBEDDED:
        retrievers += ["sochdb_keyword", "sochdb_hybrid"]
    print(f"Embedded sochdb keyword/hybrid: {'ON' if _HAS_EMBEDDED else 'OFF (no native lib)'}", flush=True)
    print(f"Retrievers: {retrievers}", flush=True)

    # recall accumulators: retriever -> k -> sum_recall ; plus count
    rec = {r: {k: 0.0 for k in args.k} for r in retrievers}
    # per-category: cat -> {n, retriever -> sum_recall@maxk(strict: all evidence in topk)}
    from collections import defaultdict
    cat_n = defaultdict(int)
    cat_rec = defaultdict(lambda: defaultdict(float))   # cat -> retriever -> sum recall@10
    cat_full = defaultdict(lambda: defaultdict(int))     # cat -> retriever -> #(all evidence retrieved)@10
    RK = 10 if 10 in args.k else max(args.k)
    n_q = 0
    # answer accumulators
    ans = {r: {"n": 0, "sub": 0, "judge": 0} for r in args.answer_retrievers}
    t_start = time.time()

    for si, item in enumerate(raw):
        conv = item["conversation"]
        dia_ids, texts, dia2text = build_corpus(conv)
        if not texts:
            continue
        # embed corpus
        vecs = [v.tolist() for v in emb.embed(texts)]
        dim = len(vecs[0])
        id2dia = {i: dia_ids[i] for i in range(len(dia_ids))}
        # sochdb gRPC HNSW index (create_index + insert_vectors + search; ids are ints)
        coll = f"diag_{item['sample_id']}_{si}"
        try:
            client.create_index(coll, dimension=dim)
            client.insert_vectors(coll, ids=list(range(len(vecs))), vectors=vecs)
            grpc_ok = True
        except Exception as e:
            print(f"  [grpc index err {coll}] {e}", flush=True); grpc_ok = False
        # bm25
        bm25 = BM25Okapi([_tok(t) for t in texts])
        # embedded sochdb collection (keyword/hybrid)
        ecol = None
        if _HAS_EMBEDDED:
            try:
                ecol = open_collection(f"e_{coll}", path=":memory:", dimension=dim)
                ecol.insert_batch(documents=[(i, vecs[i], {"dia": dia_ids[i]}, texts[i]) for i in range(len(vecs))])
            except Exception as e:
                print(f"  [embedded col err] {e}", flush=True); ecol = None

        q_in_conv = 0
        for qa in item.get("qa", []):
            if args.limit and n_q >= args.limit:
                break
            if args.qa_per_conv and q_in_conv >= args.qa_per_conv:
                break
            q = qa.get("question"); a = qa.get("answer"); ev = qa.get("evidence")
            if q is None or a is None or not ev:
                continue
            q_in_conv += 1
            gold = [str(x) for x in (a if isinstance(a, list) else [a])]
            ev_set = set(ev)
            n_q += 1
            qv = list(emb.embed([q]))[0].tolist()

            # candidate top-kmax dia lists per retriever
            cand: Dict[str, List[str]] = {}
            cand["first_n"] = dia_ids[:kmax]
            bm_idx = sorted(range(len(texts)), key=lambda i: bm25.get_scores(_tok(q))[i], reverse=True)[:kmax]
            cand["bm25"] = [dia_ids[i] for i in bm_idx]
            if grpc_ok:
                try:
                    res = client.search(coll, qv, k=kmax)
                    cand["sochdb_vector"] = [id2dia.get(i, "") for i in doc_ids(res)]
                except Exception as e:
                    cand["sochdb_vector"] = []
            else:
                cand["sochdb_vector"] = []
            if ecol is not None:
                try:
                    kr = ecol.keyword_search(q, k=kmax); cand["sochdb_keyword"] = [id2dia.get(i, "") for i in doc_ids(kr)]
                except Exception:
                    cand["sochdb_keyword"] = []
                try:
                    hr = ecol.hybrid_search(qv, q, k=kmax); cand["sochdb_hybrid"] = [id2dia.get(i, "") for i in doc_ids(hr)]
                except Exception:
                    cand["sochdb_hybrid"] = []

            # recall@k
            cat = qa.get("category")
            cat_n[cat] += 1
            for r in retrievers:
                ids = cand.get(r, [])
                for k in args.k:
                    topk = set(ids[:k])
                    rec[r][k] += len(ev_set & topk) / len(ev_set)
                topRK = set(ids[:RK])
                cat_rec[cat][r] += len(ev_set & topRK) / len(ev_set)
                cat_full[cat][r] += int(ev_set <= topRK)  # ALL evidence retrieved

            # optional answering
            if args.answer:
                for r in args.answer_retrievers:
                    ids = cand.get(r, [])[: args.answer_k]
                    ctx = "\n\n".join(dia2text.get(d, "") for d in ids if d)
                    pred = ""
                    try:
                        pred = chat(args.llm_base, args.llm_model,
                                    "Answer the question using ONLY the context. Be concise. If not in context, say 'I don't know'.",
                                    f"Context:\n{ctx[:24000]}\n\nQuestion: {q}\nAnswer:", args.max_answer_tokens)
                    except Exception:
                        pass
                    ans[r]["n"] += 1
                    ans[r]["sub"] += int(sub_hit(pred, gold))
                    ans[r]["judge"] += int(judge(args.llm_base, args.llm_model, q, gold, pred))
            if n_q % 25 == 0:
                print(f"  {n_q} QAs ...", flush=True)
        # cleanup gRPC collection best-effort
        try:
            client.delete_collection(coll)  # may not exist
        except Exception:
            pass
        if args.limit and n_q >= args.limit:
            break

    summary = {
        "dataset": "locomo",
        "metric": "evidence_recall@k (gold dia_id labels) + optional QA accuracy",
        "embed_model": args.embed_model,
        "num_questions": n_q,
        "retrievers": retrievers,
        "recall_at_k": {
            r: {str(k): round(100 * rec[r][k] / n_q, 2) if n_q else 0.0 for k in args.k}
            for r in retrievers
        },
        "wall_time_s": round(time.time() - t_start, 1),
    }
    cat_names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    summary[f"recall@{RK}_by_category"] = {
        f"{c}:{cat_names.get(c, c)}": {
            "n": cat_n[c],
            **{r: round(100 * cat_rec[c][r] / cat_n[c], 1) if cat_n[c] else 0.0 for r in retrievers},
            **{f"{r}_ALLretrieved": round(100 * cat_full[c][r] / cat_n[c], 1) if cat_n[c] else 0.0
               for r in ("sochdb_vector", "bm25")},
        }
        for c in sorted(cat_n)
    }
    if args.answer:
        summary["qa_accuracy"] = {
            r: {
                "n": ans[r]["n"],
                "substring_pct": round(100 * ans[r]["sub"] / ans[r]["n"], 2) if ans[r]["n"] else 0.0,
                "judge_pct": round(100 * ans[r]["judge"] / ans[r]["n"], 2) if ans[r]["n"] else 0.0,
            }
            for r in args.answer_retrievers
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("=== DIAG SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
