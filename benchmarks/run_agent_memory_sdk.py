#!/usr/bin/env python3
"""Memory benchmarks via sochdb-python-sdk (gRPC AgentMemory / ContextService).

Supports:
  - memoryagentbench: Accurate Retrieval (lexical lanes, evidence substring recall)
  - locomo: LoComo10 session retrieval (lexical lanes, evidence substring recall)

Requires sochdb-grpc-server running (default localhost:50051).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sochdb import QueryLanes, SochDBClient, create_agent_memory

MAB_ROOT = os.environ.get(
    "MEMORY_AGENT_BENCH_ROOT", "/Users/sushanth/git-clone/MemoryAgentBench"
)
DEFAULT_LOCOMO = os.environ.get(
    "LOCOMO_PLUS_DATA", "/Users/sushanth/git-clone/Locomo-Plus/data/locomo10.json"
)

MAB_TASKS = [
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "ruler_qa1_197K",
        "chunk_size": 4096,
        "context_max_length": 220000,
        "generation_max_length": 50,
    },
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "ruler_qa2_421K",
        "chunk_size": 4096,
        "context_max_length": 524288,
        "generation_max_length": 50,
    },
    {
        "dataset": "Accurate_Retrieval",
        "sub_dataset": "longmemeval_s*",
        "chunk_size": 4096,
        "context_max_length": 400000,
        "generation_max_length": 50,
    },
]

SKIP_LOCOMO_CATEGORIES = {5}
LOCOMO_CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "common-sense",
    4: "single-hop",
    5: "adversarial",
}


def _import_mab_utils():
    sys.path.insert(0, MAB_ROOT)
    from conversation_creator import ConversationCreator  # noqa: E402
    from utils.eval_other_utils import (  # noqa: E402
        drqa_metric_max_over_ground_truths,
        substring_exact_match_score,
    )

    return ConversationCreator, drqa_metric_max_over_ground_truths, substring_exact_match_score


def evidence_hit(prediction: str, answers, scorer, drqa_metric) -> bool:
    return bool(drqa_metric(scorer, prediction, answers))


def session_to_text(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('speaker', '?')}: {turn.get('text', '')}" for turn in turns
    )


def session_in_context(session_text: str, ctx: str, *, min_len: int = 32) -> bool:
    """Check if ingested dialogue text appears in compiled context."""
    ctx_lower = ctx.lower()
    for line in session_text.split("\n"):
        line = line.strip()
        if len(line) >= min_len and line.lower() in ctx_lower:
            return True
    snippet = session_text[:120].strip().lower()
    return bool(snippet and snippet in ctx_lower)


# MAB chunks are ~4096 chars; truncate compiled context to simulate top-k.
CHARS_PER_CHUNK = 4096


def run_mab_task(
    client: SochDBClient,
    task: Dict[str, Any],
    *,
    max_test_samples: int,
    k_values: List[int],
    lanes: str,
    address: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ConversationCreator, drqa_metric, scorer = _import_mab_utils()
    agent_config = {"agent_name": "Simple_rag_bm25", "model": "gpt-4o-mini"}
    dataset_config = {
        "dataset": task["dataset"],
        "sub_dataset": task["sub_dataset"],
        "chunk_size": task["chunk_size"],
        "context_max_length": task["context_max_length"],
        "generation_max_length": task["generation_max_length"],
        "max_test_samples": max_test_samples,
        "seed": 42,
    }
    creator = ConversationCreator(agent_config, dataset_config)
    all_chunks = creator.get_chunks()
    all_qas = creator.get_query_and_answers()

    contexts_payload = []
    gold: Dict[Tuple[int, int], Any] = {}
    global_qid = 0
    for ctx_id, (chunks, qas) in enumerate(zip(all_chunks, all_qas)):
        queries = []
        for query, answer, _qa_id in qas:
            queries.append({"query_id": global_qid, "query": query})
            gold[(ctx_id, global_qid)] = answer
            global_qid += 1
        contexts_payload.append(
            {"context_id": ctx_id, "chunks": chunks, "queries": queries}
        )

    top_k = max(k_values)
    token_limit = min(task["context_max_length"], 524288)
    per_query: List[Dict[str, Any]] = []
    recall_hits = {k: 0 for k in k_values}
    build_times: List[float] = []
    query_times: List[float] = []

    wall_start = time.time()
    for ctx in contexts_payload:
        ns = f"ctx-{ctx['context_id']}"
        memory = create_agent_memory(client, namespace=ns, token_limit=token_limit)

        build_start = time.perf_counter()
        for chunk in ctx["chunks"]:
            memory.write_episode(chunk)
        build_ms = (time.perf_counter() - build_start) * 1000.0
        build_times.append(build_ms)

        for q in ctx["queries"]:
            answers = gold[(ctx["context_id"], q["query_id"])]
            q_start = time.perf_counter()
            final = memory.search(q["query"], lanes=lanes, token_limit=token_limit)
            query_ms = (time.perf_counter() - q_start) * 1000.0
            query_times.append(query_ms)

            hits_at_k = {}
            for k in k_values:
                truncated = final.context[: k * CHARS_PER_CHUNK]
                hit = evidence_hit(truncated, answers, scorer, drqa_metric)
                hits_at_k[k] = int(hit)
                recall_hits[k] += hits_at_k[k]

            per_query.append(
                {
                    "context_id": ctx["context_id"],
                    "query_id": q["query_id"],
                    "answers": answers,
                    "retrieved_context_chars": len(final.context),
                    "total_tokens": final.total_tokens,
                    "evidence_substring_match_at_k": hits_at_k,
                    "build_ms": build_ms,
                    "query_ms": query_ms,
                    "error": final.error,
                }
            )

    wall = time.time() - wall_start
    n = len(per_query)
    recall_at_k = {
        str(k): round((recall_hits[k] / n) * 100, 2) if n else 0.0
        for k in k_values
    }
    summary = {
        "sub_dataset": task["sub_dataset"],
        "competency": "Accurate Retrieval",
        "retriever": f"sochdb-python-sdk-grpc/{lanes}",
        "grpc_address": address,
        "num_contexts": len(contexts_payload),
        "num_queries": n,
        "evidence_recall_at_k": recall_at_k,
        "avg_index_build_ms": round(sum(build_times) / len(build_times), 4)
        if build_times
        else 0.0,
        "avg_query_ms": round(sum(query_times) / len(query_times), 4)
        if query_times
        else 0.0,
        "wall_time_s": round(wall, 2),
    }
    return summary, per_query


def _session_keys(conv: Dict[str, Any]) -> List[str]:
    keys = [
        k
        for k in conv.keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    ]
    keys.sort(key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
    return keys


def run_locomo(
    client: SochDBClient,
    locomo_path: Path,
    *,
    lanes: str,
    address: str,
) -> Dict[str, Any]:
    raw = json.loads(locomo_path.read_text(encoding="utf-8"))
    namespace = "locomo-sdk"
    memory = create_agent_memory(client, namespace=namespace, token_limit=131072)

    doc_map: Dict[str, int] = {}
    doc_text: Dict[int, str] = {}
    ingest_start = time.perf_counter()
    for item in raw:
        sample_id = item["sample_id"]
        conv = item["conversation"]
        for sk in _session_keys(conv):
            turns = conv.get(sk) or []
            text = session_to_text(turns) if turns else ""
            wr = memory.write_episode(
                text, metadata={"sample_id": sample_id, "session": sk}
            )
            doc_map[f"{sample_id}_{sk}"] = wr.episode_id
            doc_text[wr.episode_id] = text
    ingest_ms = (time.perf_counter() - ingest_start) * 1000.0

    per_question = []
    recall5 = recall10 = mrr_sum = 0.0
    latencies: List[float] = []
    n = 0

    for item in raw:
        sample_id = item["sample_id"]
        for qi, qa in enumerate(item["qa"]):
            cat = qa.get("category", 0)
            if cat in SKIP_LOCOMO_CATEGORIES:
                continue
            gold_ids = []
            for ev in qa.get("evidence", []):
                for part in ev.split(";"):
                    part = part.strip()
                    session_token = part.split(":")[0].replace("D", "") if part else ""
                    if not session_token.isdigit():
                        continue
                    sk = f"session_{session_token}"
                    key = f"{sample_id}_{sk}"
                    if key in doc_map:
                        gold_ids.append(doc_map[key])

            q_start = time.perf_counter()
            result = memory.search(qa["question"], lanes=lanes)
            query_ms = (time.perf_counter() - q_start) * 1000.0
            latencies.append(query_ms)

            ctx = result.context
            hit = any(
                session_in_context(doc_text.get(gid, ""), ctx) for gid in gold_ids
            )
            r5 = r10 = 1.0 if hit else 0.0
            rr = 0.0
            for rank, gid in enumerate(gold_ids, start=1):
                if session_in_context(doc_text.get(gid, ""), ctx):
                    rr = 1.0 / rank
                    break

            recall5 += r5
            recall10 += r10
            mrr_sum += rr
            n += 1
            per_question.append(
                {
                    "id": f"{sample_id}_q{qi}",
                    "category": LOCOMO_CATEGORY_NAMES.get(cat, f"category_{cat}"),
                    "recall_at_5": r5,
                    "recall_at_10": r10,
                    "mrr": rr,
                    "query_ms": query_ms,
                    "total_tokens": result.total_tokens,
                }
            )

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    report = {
        "dataset": "locomo",
        "retriever": f"sochdb-python-sdk-grpc/{lanes}",
        "grpc_address": address,
        "questions": n,
        "recall_at_5": round((recall5 / n) * 100, 2) if n else 0.0,
        "recall_at_10": round((recall10 / n) * 100, 2) if n else 0.0,
        "mrr": round((mrr_sum / n) * 100, 2) if n else 0.0,
        "ingest_ms": round(ingest_ms, 2),
        "p50_query_ms": round(p50, 4),
        "per_question": per_question,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        default=os.environ.get("SOCHDB_ADDRESS", "localhost:50051"),
    )
    parser.add_argument(
        "--suite",
        choices=["mab", "locomo", "all"],
        default="all",
    )
    parser.add_argument("--max-test-samples", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--lanes", default=QueryLanes.LEXICAL)
    parser.add_argument(
        "--locomo-data",
        default=DEFAULT_LOCOMO,
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {args.address} ...")
    with SochDBClient(args.address) as client:
        report: Dict[str, Any] = {
            "benchmark": "sochdb-python-sdk agent memory",
            "version": __import__("sochdb").__version__,
            "grpc_address": args.address,
            "lanes": args.lanes,
        }

        if args.suite in ("mab", "all"):
            tasks = MAB_TASKS
            if args.tasks:
                tasks = [t for t in MAB_TASKS if t["sub_dataset"] in args.tasks]
            mab_summaries = []
            mab_details = {}
            for task in tasks:
                print(
                    f"\n=== MAB {task['sub_dataset']} "
                    f"(samples={args.max_test_samples}, lanes={args.lanes}) ==="
                )
                summary, details = run_mab_task(
                    client,
                    task,
                    max_test_samples=args.max_test_samples,
                    k_values=args.k_values,
                    lanes=args.lanes,
                    address=args.address,
                )
                mab_summaries.append(summary)
                mab_details[task["sub_dataset"]] = details
                topk = " ".join(
                    f"@{k}={summary['evidence_recall_at_k'][str(k)]}%"
                    for k in args.k_values
                )
                print(
                    f"  queries={summary['num_queries']} evidence_recall {topk} "
                    f"avg_query={summary['avg_query_ms']}ms"
                )
            report["memoryagentbench"] = mab_summaries
            mab_path = out_dir / "sdk_membench_results.json"
            with open(mab_path, "w") as f:
                json.dump(report, f, indent=2)
            with open(out_dir / "sdk_membench_details.json", "w") as f:
                json.dump(mab_details, f, indent=2)
            print(f"\nSaved {mab_path}")

        if args.suite in ("locomo", "all"):
            locomo_path = Path(args.locomo_data)
            if not locomo_path.exists():
                raise FileNotFoundError(f"LoComo data not found: {locomo_path}")
            print(f"\n=== LoComo ({locomo_path.name}, lanes={args.lanes}) ===")
            locomo_report = run_locomo(
                client,
                locomo_path,
                lanes=args.lanes,
                address=args.address,
            )
            report["locomo"] = locomo_report
            print(
                f"  questions={locomo_report['questions']} "
                f"recall@5={locomo_report['recall_at_5']}% "
                f"recall@10={locomo_report['recall_at_10']}% "
                f"mrr={locomo_report['mrr']}% "
                f"p50_query={locomo_report['p50_query_ms']}ms"
            )
            locomo_path_out = out_dir / "sdk_locomo_results.json"
            with open(locomo_path_out, "w") as f:
                json.dump(locomo_report, f, indent=2)
            print(f"Saved {locomo_path_out}")

        combined = out_dir / "sdk_agent_memory_benchmark.json"
        with open(combined, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nCombined report: {combined}")


if __name__ == "__main__":
    main()