#!/usr/bin/env python3
"""End-to-end RAG benchmark over the SochDB Python SDK (gRPC).

Unlike run_agent_memory_sdk.py (retrieval-recall only, no LLM), this harness
exercises the *full* loop with real models:

  1. Ingest a memory benchmark's conversations into SochDB via the SDK
     (`create_agent_memory(...).write_episode(...)`), server-side over gRPC.
  2. Retrieve compiled context per question via the SDK (`memory.search`).
  3. Optionally re-rank the retrieved chunks with a real embedding model
     (ollama nomic-embed-text, OpenAI-compatible /v1/embeddings).
  4. Answer each question with a real chat LLM (vLLM Qwen, OpenAI-compatible
     /v1/chat/completions).
  5. Judge correctness with both a cheap substring check and an LLM judge.

Datasets: LoCoMo (locomo10.json) and MemoryAgentBench Accurate-Retrieval.

Example:
  python run_rag_e2e.py --suite all \
    --address 127.0.0.1:50061 \
    --llm-base http://100.127.255.44:8000/v1 --llm-model nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --embed-base http://localhost:11434/v1 --embed-model nomic-embed-text --rerank
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sochdb import QueryLanes, SochDBClient, create_agent_memory

# --------------------------------------------------------------------------
# HTTP helpers (stdlib only) for the OpenAI-compatible LLM + embedding servers
# --------------------------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: int = 120, retries: int = 8) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            last = e
            # Exponential backoff capped at 20s — rides out brief endpoint
            # outages (vLLM restarts) without killing a multi-hour run.
            time.sleep(min(20.0, 2.0 ** attempt))
    raise RuntimeError(f"POST {url} failed after {retries} tries: {last}")


def embed_texts(texts: List[str], base: str, model: str, batch: int = 64) -> List[List[float]]:
    url = base.rstrip("/") + "/embeddings"
    out: List[List[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        data = _post_json(url, {"model": model, "input": chunk}, timeout=120)
        out.extend(row["embedding"] for row in data["data"])
    return out


def chat(base: str, model: str, system: str, user: str, max_tokens: int = 256) -> Tuple[str, dict]:
    url = base.rstrip("/") + "/chat/completions"
    data = _post_json(
        url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # Qwen3 is a thinking model: without this it spends max_tokens in a
            # reasoning field and returns content=None. Disable for direct answers.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    msg = data["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    usage = data.get("usage", {}) or {}
    return text, usage


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def substring_hit(prediction: str, answers: List[str]) -> bool:
    p = _norm(prediction)
    for a in answers:
        a = _norm(str(a))
        if a and a in p:
            return True
    return False


def llm_judge(base: str, model: str, question: str, gold: List[str], pred: str) -> bool:
    sys_p = (
        "You are a strict grader. Given a question, the reference answer(s), and a "
        "candidate answer, reply with exactly 'YES' if the candidate is correct "
        "(same meaning as any reference), otherwise 'NO'. Reply only YES or NO."
    )
    usr = (
        f"Question: {question}\n"
        f"Reference answer(s): {' | '.join(str(g) for g in gold)}\n"
        f"Candidate answer: {pred}\n\nCorrect? (YES/NO):"
    )
    try:
        verdict, _ = chat(base, model, sys_p, usr, max_tokens=4)
        return verdict.strip().upper().startswith("Y")
    except Exception:
        return False


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def chunk_context(context: str, size: int = 600) -> List[str]:
    """Split compiled context into windows for embedding re-rank."""
    parts = [p.strip() for p in re.split(r"\n{2,}|\n#+ ", context) if p.strip()]
    out: List[str] = []
    for p in parts:
        if len(p) <= size:
            out.append(p)
        else:
            for i in range(0, len(p), size):
                out.append(p[i : i + size])
    return out[:64]


# --------------------------------------------------------------------------
# Answer one question end-to-end
# --------------------------------------------------------------------------

def answer_question(
    memory,
    question: str,
    gold: List[str],
    args,
    agg: dict,
) -> dict:
    t0 = time.perf_counter()
    res = memory.search(question, lanes=args.lanes, token_limit=args.token_limit)
    retrieve_ms = (time.perf_counter() - t0) * 1000.0
    context = res.context or ""
    recall_hit = substring_hit(context, gold)

    rerank_used = False
    if args.rerank and context:
        try:
            chunks = chunk_context(context)
            if chunks:
                vecs = embed_texts([question] + chunks, args.embed_base, args.embed_model)
                qv, cvs = vecs[0], vecs[1:]
                scored = sorted(zip(chunks, cvs), key=lambda c: cosine(qv, c[1]), reverse=True)
                context = "\n\n".join(c for c, _ in scored[: args.rerank_top])
                rerank_used = True
                agg["embed_calls"] += 1
        except Exception as e:
            agg.setdefault("rerank_errors", 0)
            agg["rerank_errors"] += 1

    sys_p = (
        "Answer the question using ONLY the provided context from a conversation "
        "memory. Be concise — a few words or one sentence. If the answer is not in "
        "the context, say 'I don't know'."
    )
    usr = f"Context:\n{context[: args.answer_ctx_chars]}\n\nQuestion: {question}\nAnswer:"
    t1 = time.perf_counter()
    try:
        pred, usage = chat(args.llm_base, args.llm_model, sys_p, usr, max_tokens=args.max_answer_tokens)
    except Exception:
        # Non-fatal: a persistent endpoint outage yields an empty answer (scored
        # as a miss) but the run continues to completion rather than aborting.
        pred, usage = "", {}
        agg["llm_errors"] = agg.get("llm_errors", 0) + 1
    answer_ms = (time.perf_counter() - t1) * 1000.0
    agg["llm_calls"] += 1
    agg["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    agg["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    # vLLM/Anthropic-style cached prefill, when present
    agg["cached_tokens"] += int(
        (usage.get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0) or 0
    )

    sub_hit = substring_hit(pred, gold)
    judge_hit = False
    if args.judge:
        judge_hit = llm_judge(args.llm_base, args.llm_model, question, gold, pred)
        agg["llm_calls"] += 1

    return {
        "question": question,
        "gold": gold,
        "prediction": pred,
        "retrieval_recall_hit": recall_hit,
        "answer_substring_hit": sub_hit,
        "answer_judge_hit": judge_hit,
        "rerank_used": rerank_used,
        "retrieve_ms": round(retrieve_ms, 1),
        "answer_ms": round(answer_ms, 1),
        "retrieved_tokens": res.total_tokens,
    }


# --------------------------------------------------------------------------
# LoCoMo
# --------------------------------------------------------------------------

def _session_keys(conv: Dict[str, Any]) -> List[str]:
    keys = [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")]
    keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
    return keys


def session_text(turns: List[dict]) -> str:
    lines = []
    for t in turns:
        spk = t.get("speaker", "")
        txt = t.get("text", t.get("clean_text", ""))
        lines.append(f"{spk}: {txt}")
    return "\n".join(lines)


def run_locomo(client: SochDBClient, path: Path, args) -> Tuple[dict, list]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    agg = {"llm_calls": 0, "embed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    rows: List[dict] = []
    n_correct_sub = n_correct_judge = n_recall = n = 0
    wall = time.time()

    for item in raw:
        sample_id = item["sample_id"]
        conv = item["conversation"]
        ns = f"locomo-{sample_id}"
        memory = create_agent_memory(client, namespace=ns, token_limit=args.token_limit)
        # Turn-level ingestion (representative LoCoMo RAG granularity). Prepend the
        # session date so temporal questions ("When did ...") are answerable.
        for sk in _session_keys(conv):
            turns = conv.get(sk) or []
            date = conv.get(f"{sk}_date_time", "") or ""
            for turn in turns:
                spk = turn.get("speaker", "")
                txt = turn.get("text", turn.get("clean_text", ""))
                if not txt:
                    continue
                ep = f"[{date}] {spk}: {txt}" if date else f"{spk}: {txt}"
                memory.write_episode(
                    ep,
                    metadata={"sample_id": sample_id, "session": sk, "dia_id": turn.get("dia_id")},
                )

        q_in_conv = 0
        for qa in item.get("qa", []):
            if args.limit and n >= args.limit:
                break
            if args.qa_per_conv and q_in_conv >= args.qa_per_conv:
                break
            q = qa.get("question")
            ans = qa.get("answer")
            if q is None or ans is None:
                continue  # skip adversarial/no-answer for accuracy clarity
            q_in_conv += 1
            gold = ans if isinstance(ans, list) else [ans]
            r = answer_question(memory, q, [str(g) for g in gold], args, agg)
            r["sample_id"] = sample_id
            r["category"] = qa.get("category")
            rows.append(r)
            n += 1
            n_recall += int(r["retrieval_recall_hit"])
            n_correct_sub += int(r["answer_substring_hit"])
            n_correct_judge += int(r["answer_judge_hit"])
            if n % 10 == 0:
                print(f"  [locomo] {n} QAs  sub={n_correct_sub} judge={n_correct_judge}", flush=True)
        if args.limit and n >= args.limit:
            break

    summary = {
        "dataset": "locomo",
        "retriever": f"sochdb-sdk-grpc/{args.lanes}" + (" + ollama-rerank" if args.rerank else ""),
        "llm_model": args.llm_model,
        "embed_model": args.embed_model if args.rerank else None,
        "num_questions": n,
        "retrieval_recall_pct": round(100 * n_recall / n, 2) if n else 0,
        "answer_substring_acc_pct": round(100 * n_correct_sub / n, 2) if n else 0,
        "answer_judge_acc_pct": round(100 * n_correct_judge / n, 2) if n else 0,
        "avg_retrieve_ms": round(sum(r["retrieve_ms"] for r in rows) / n, 1) if n else 0,
        "avg_answer_ms": round(sum(r["answer_ms"] for r in rows) / n, 1) if n else 0,
        "tokens": agg,
        "wall_time_s": round(time.time() - wall, 1),
    }
    return summary, rows


# --------------------------------------------------------------------------
# MemoryAgentBench (Accurate Retrieval) — uses the benchmark's own loaders
# --------------------------------------------------------------------------

MAB_TASKS = [
    {"dataset": "Accurate_Retrieval", "sub_dataset": "ruler_qa1_197K", "chunk_size": 4096, "max_length": 200000},
    {"dataset": "Accurate_Retrieval", "sub_dataset": "ruler_qa2_421K", "chunk_size": 4096, "max_length": 440000},
    {"dataset": "Accurate_Retrieval", "sub_dataset": "longmemeval_s*", "chunk_size": 1024, "max_length": 800000},
]


def run_mab(client: SochDBClient, args) -> List[Tuple[dict, list]]:
    sys.path.insert(0, args.mab_repo)
    from conversation_creator import ConversationCreator  # type: ignore

    results = []
    for task in MAB_TASKS:
        if args.mab_tasks and task["sub_dataset"] not in args.mab_tasks:
            continue
        print(f"=== MAB {task['sub_dataset']} ===", flush=True)
        agent_config = {"agent_name": "Simple_rag_bm25", "model": "gpt-4o-mini"}
        dataset_config = {
            "dataset": task["dataset"],
            "sub_dataset": task["sub_dataset"],
            "chunk_size": task["chunk_size"],
            "context_max_length": task["max_length"],
            "max_test_samples": args.limit or 9999,
            "seed": 42,
        }
        try:
            creator = ConversationCreator(agent_config, dataset_config)
            all_chunks = creator.get_chunks()
            all_qas = creator.get_query_and_answers()
        except Exception as e:
            print(f"  MAB load failed for {task['sub_dataset']}: {e}", flush=True)
            results.append(({"dataset": "memoryagentbench", "sub_dataset": task["sub_dataset"], "error": str(e)}, []))
            continue

        agg = {"llm_calls": 0, "embed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        rows: List[dict] = []
        n = n_recall = n_sub = n_judge = 0
        wall = time.time()
        # One context = one chunk-list + its QA-list (zipped, same order).
        for ctx_id, (chunks, qas) in enumerate(zip(all_chunks, all_qas)):
            ns = f"mab-{task['sub_dataset']}-{ctx_id}"
            memory = create_agent_memory(client, namespace=ns, token_limit=args.token_limit)
            for ch in chunks:
                memory.write_episode(ch)
            for query, answer, _qid in qas:
                if args.limit and n >= args.limit:
                    break
                g = [str(x) for x in (answer if isinstance(answer, list) else [answer])]
                if not g or not g[0]:
                    continue
                r = answer_question(memory, query, g, args, agg)
                rows.append(r)
                n += 1
                n_recall += int(r["retrieval_recall_hit"])
                n_sub += int(r["answer_substring_hit"])
                n_judge += int(r["answer_judge_hit"])
                if n % 10 == 0:
                    print(f"  [{task['sub_dataset']}] {n} QAs sub={n_sub} judge={n_judge}", flush=True)
            if args.limit and n >= args.limit:
                break

        summary = {
            "dataset": "memoryagentbench",
            "sub_dataset": task["sub_dataset"],
            "competency": "Accurate Retrieval",
            "retriever": f"sochdb-sdk-grpc/{args.lanes}" + (" + ollama-rerank" if args.rerank else ""),
            "llm_model": args.llm_model,
            "num_questions": n,
            "retrieval_recall_pct": round(100 * n_recall / n, 2) if n else 0,
            "answer_substring_acc_pct": round(100 * n_sub / n, 2) if n else 0,
            "answer_judge_acc_pct": round(100 * n_judge / n, 2) if n else 0,
            "avg_retrieve_ms": round(sum(x["retrieve_ms"] for x in rows) / n, 1) if n else 0,
            "avg_answer_ms": round(sum(x["answer_ms"] for x in rows) / n, 1) if n else 0,
            "tokens": agg,
            "wall_time_s": round(time.time() - wall, 1),
        }
        results.append((summary, rows))
    return results


def embedder_handshake(args):
    """T4 preflight: scrape the server's sochdb_embedder_info gauge and refuse to
    run a semantic benchmark against the non-semantic hash fallback."""
    url = getattr(args, "metrics_url", None)
    if not url:
        print(
            "[handshake] no --metrics-url given; cannot verify the server's embedder. "
            "Run the server with --metrics-port and pass --metrics-url to enforce.",
            flush=True,
        )
        return
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            body = r.read().decode()
    except Exception as e:
        print(f"[handshake] WARNING: could not scrape {url}: {e}", flush=True)
        return
    m = re.search(r"sochdb_embedder_info\{([^}]*)\}\s+1", body)
    if not m:
        print("[handshake] WARNING: sochdb_embedder_info not exposed by the server.", flush=True)
        return
    labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    fam, model, sem = labels.get("family"), labels.get("model"), labels.get("semantic")
    print(f"[handshake] server embedder: family={fam} model={model} semantic={sem}", flush=True)
    expect = args.expect_embedder
    if expect in ("any", "hash"):
        return  # explicitly not requiring semantics
    if fam == "hash" or sem != "true":
        sys.exit(
            f"[handshake] ABORT: semantic benchmark but the server embedder is non-semantic "
            f"(family={fam}, semantic={sem}). Re-run the server with SOCHDB_EMBEDDER=fastembed:<model>, "
            f"or pass --expect-embedder hash to deliberately benchmark the fallback."
        )
    if expect not in ("semantic", fam, model):
        sys.exit(f"[handshake] ABORT: expected '{expect}' but server reports family={fam} model={model}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["locomo", "mab", "all"], default="all")
    ap.add_argument("--address", default="127.0.0.1:50061")
    ap.add_argument("--metrics-url", default=None, help="server /metrics URL for the embedder handshake")
    ap.add_argument("--expect-embedder", default="semantic",
                    help="semantic|hash|any|<family>|<model>; abort if the server's embedder disagrees")
    ap.add_argument("--llm-base", default="http://100.127.255.44:8000/v1")
    ap.add_argument("--llm-model", default="nvidia/Qwen3.6-35B-A3B-NVFP4")
    ap.add_argument("--embed-base", default="http://localhost:11434/v1")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--rerank", action="store_true", help="re-rank retrieved chunks with ollama embeddings")
    ap.add_argument("--rerank-top", type=int, default=8)
    ap.add_argument("--judge", action="store_true", default=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--lanes", default=QueryLanes.LEXICAL)
    ap.add_argument("--token-limit", type=int, default=8192)
    ap.add_argument("--answer-ctx-chars", type=int, default=24000)
    ap.add_argument("--max-answer-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="cap questions per (sub)dataset; 0 = all")
    ap.add_argument("--qa-per-conv", type=int, default=0, help="cap QAs per conversation (balanced subset); 0 = all")
    ap.add_argument("--locomo-data", default="/root/locomo/data/locomo10.json")
    ap.add_argument("--mab-repo", default="/root/MemoryAgentBench")
    ap.add_argument("--mab-tasks", nargs="*", default=None)
    ap.add_argument("--out-dir", default="results_e2e")
    args = ap.parse_args()
    if args.limit == 0:
        args.limit = None

    # Embedder handshake (T4): confirm the system-under-test's embedder rather
    # than assume it. A semantic benchmark against the non-semantic hash
    # fallback is a void measurement — abort unless explicitly benchmarking it.
    embedder_handshake(args)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = SochDBClient(args.address)
    print(f"Connected SDK -> {args.address}; LLM={args.llm_model} embed={args.embed_model if args.rerank else 'off'}", flush=True)

    all_summaries = []
    if args.suite in ("locomo", "all"):
        print("=== LoCoMo end-to-end ===", flush=True)
        s, rows = run_locomo(client, Path(args.locomo_data), args)
        all_summaries.append(s)
        (out / "locomo_e2e_details.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(s, indent=2), flush=True)
    if args.suite in ("mab", "all"):
        for s, rows in run_mab(client, args):
            all_summaries.append(s)
            tag = s.get("sub_dataset", "mab").replace("*", "star")
            (out / f"mab_{tag}_e2e_details.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(s, indent=2), flush=True)

    (out / "summary_e2e.json").write_text(json.dumps(all_summaries, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(all_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
