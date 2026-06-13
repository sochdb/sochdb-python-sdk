#!/usr/bin/env python3
"""
CrewAI + SochDB hosted remote tool example.

This example shows the same CrewAI integration surface as the embedded example,
but points it at a remote SochDB collection over gRPC.

Environment variables:
    SOCHDB_GRPC_ADDRESS   default: studio.agentslab.host:50053
    SOCHDB_NAMESPACE      default: default
    CREWAI_MODEL          default: gpt-4o-mini
    OPENAI_API_KEY        required by CrewAI for the default LLM provider
    SOCHDB_CREWAI_SKIP_KICKOFF=1 to only validate remote storage/search setup

Install:
    pip install -e ".[crewai]"
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path
from typing import Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sochdb import SochDBClient, SochDBKnowledgeStore, create_crewai_tools


DEFAULT_GRPC_ADDRESS = "studio.agentslab.host:50053"
DEFAULT_NAMESPACE = "default"


def deterministic_embed(texts: Sequence[str], dim: int = 32) -> list[list[float]]:
    """Small local embedder so the demo does not require a second model service."""

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        vectors.append([v / norm for v in values])
    return vectors


def build_remote_store(client: SochDBClient, namespace: str) -> tuple[str, SochDBKnowledgeStore]:
    run_id = f"crewai-remote-{int(time.time())}"
    collection_name = f"sdk_crewai_remote_{run_id}"
    client.create_collection(collection_name, dimension=32, namespace=namespace, metric="cosine")

    store = SochDBKnowledgeStore.from_client(
        client,
        collection_name=collection_name,
        namespace=namespace,
        embedder=deterministic_embed,
    )
    store.add_texts(
        [
            "The hosted SochDB demo endpoint listens on studio.agentslab.host:50053.",
            "The corrected 10GB benchmark showed about 506 QPS after one-time index load.",
            "BAAI/bge-base-en-v1.5 is the best published SciFact quality result so far.",
        ],
        metadatas=[
            {"topic": "deployment"},
            {"topic": "benchmark"},
            {"topic": "quality"},
        ],
        ids=[f"{run_id}-deploy", f"{run_id}-bench", f"{run_id}-quality"],
    )
    return collection_name, store


def main() -> None:
    grpc_address = os.environ.get("SOCHDB_GRPC_ADDRESS", DEFAULT_GRPC_ADDRESS)
    namespace = os.environ.get("SOCHDB_NAMESPACE", DEFAULT_NAMESPACE)
    model = os.environ.get("CREWAI_MODEL", "gpt-4o-mini")
    skip_kickoff = os.environ.get("SOCHDB_CREWAI_SKIP_KICKOFF", "").lower() in {
        "1",
        "true",
        "yes",
    }

    client = SochDBClient(grpc_address)
    collection_name, store = build_remote_store(client, namespace)
    print(f"Using remote collection: {collection_name} in namespace={namespace}")
    try:
        if skip_kickoff:
            hits = store.search("What is the 10GB benchmark takeaway?", top_k=2)
            print("\n=== Remote Store Smoke ===\n")
            print(store.format_hits(hits))
            return

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY is required for the CrewAI kickoff. "
                "Set SOCHDB_CREWAI_SKIP_KICKOFF=1 to validate the remote SochDB path without an LLM."
            )

        from crewai import Agent, Crew, Task

        search_tool, remember_tool = create_crewai_tools(store, top_k=3)

        researcher = Agent(
            role="SochDB Remote Researcher",
            goal="Answer questions using the hosted SochDB knowledge base.",
            backstory="You always search the remote collection before making a claim.",
            llm=model,
            tools=[search_tool, remember_tool],
            verbose=True,
        )

        task = Task(
            description=(
                "Find the current 10GB benchmark takeaway and summarize it in 2-3 sentences. "
                "Use the SochDB tools and mention that the knowledge came from the remote store."
            ),
            expected_output="A short grounded summary of the latest 10GB benchmark result.",
            agent=researcher,
        )

        crew = Crew(agents=[researcher], tasks=[task], verbose=True)
        result = crew.kickoff()
        print("\n=== Crew Result ===\n")
        print(result)
    finally:
        client.close()


if __name__ == "__main__":
    main()
