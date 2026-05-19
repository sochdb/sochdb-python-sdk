#!/usr/bin/env python3
"""
CrewAI + SochDB knowledge tool example.

This example shows the supported integration shape in the Python SDK:

- SochDB stores searchable project knowledge
- a user-supplied embedder converts text into vectors
- CrewAI agents use SochDB-backed search and memory tools

Install:
    pip install -e ".[crewai]"

Optional environment:
    OPENAI_API_KEY=<your key>
    CREWAI_MODEL=gpt-4o-mini
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from typing import Sequence

from sochdb import Database, Namespace, SochDBKnowledgeStore, create_crewai_tools


def deterministic_embed(texts: Sequence[str], dim: int = 32) -> list[list[float]]:
    """
    Tiny local embedder for demos and tests.

    This is not semantically strong like OpenAI or sentence-transformers, but it
    keeps the example runnable without another service dependency.
    """

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dim)]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        vectors.append([v / norm for v in values])
    return vectors


def build_knowledge_store() -> SochDBKnowledgeStore:
    tempdir = tempfile.mkdtemp(prefix="sochdb-crewai-")
    db = Database.open(tempdir)
    namespace = Namespace(db, "crewai_demo")
    collection = namespace.create_collection("knowledge", dimension=32)

    store = SochDBKnowledgeStore.from_collection(collection, embedder=deterministic_embed)
    store.add_texts(
        [
            "SochDB supports both embedded and gRPC deployment modes.",
            "The hosted SochDB demo endpoint listens on studio.agentslab.host:50053.",
            "The corrected 10GB benchmark showed about 506 QPS after one-time index load.",
        ],
        metadatas=[
            {"topic": "architecture"},
            {"topic": "deployment"},
            {"topic": "benchmark"},
        ],
        ids=["arch-1", "deploy-1", "bench-1"],
    )
    return store


def main() -> None:
    from crewai import Agent, Crew, Task

    store = build_knowledge_store()
    search_tool, remember_tool = create_crewai_tools(store, top_k=3)

    model = os.environ.get("CREWAI_MODEL", "gpt-4o-mini")

    researcher = Agent(
        role="SochDB Researcher",
        goal="Answer questions using the SochDB knowledge base",
        backstory="You ground answers in the project knowledge store before responding.",
        llm=model,
        tools=[search_tool, remember_tool],
        verbose=True,
    )

    task = Task(
        description=(
            "Find the current 10GB benchmark takeaway and summarize it in 2-3 sentences. "
            "Use the SochDB tools instead of guessing."
        ),
        expected_output="A short grounded summary of the latest 10GB benchmark result.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], verbose=True)
    result = crew.kickoff()

    print("\n=== Crew Result ===\n")
    print(result)


if __name__ == "__main__":
    main()
