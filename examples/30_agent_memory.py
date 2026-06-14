#!/usr/bin/env python3
"""
Example 28: Agent Memory — ContextService + sochdb-memory backend (gRPC)

Requires a running sochdb-grpc server with MemoryBackend enabled.

  cargo run -p sochdb-grpc --release

Usage:
  SOCHDB_ADDRESS=localhost:50051 python examples/30_agent_memory.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sochdb import SochDBClient, QueryLanes, create_agent_memory


def main():
    address = os.environ.get("SOCHDB_ADDRESS", "localhost:50051")
    print("=" * 60)
    print("SochDB — Example 28: Agent Memory")
    print(f"Server: {address}")
    print("=" * 60)

    with SochDBClient(address) as client:
        memory = create_agent_memory(client, namespace="example-agent", token_limit=4096)

        # Write episodes (lexical index immediate; vector enrichment async)
        w1 = memory.write_episode(
            "Caroline went to the LGBTQ support group on May 7, 2023.",
            metadata={"speaker": "Caroline"},
        )
        w2 = memory.write_episode(
            "Melanie painted a sunrise at the beach with her kids.",
            metadata={"speaker": "Melanie"},
        )
        print(f"Wrote episodes: {w1.episode_id}, {w2.episode_id}")
        print(f"  lexical_indexed={w1.lexical_indexed}, enrichment_queued={w1.enrichment_queued}")

        # Lexical recall (fast, no embedding required)
        lexical = memory.search("LGBTQ support group", lanes=QueryLanes.LEXICAL)
        print(f"\nLexical search ({lexical.total_tokens} tokens):")
        print(lexical.context[:500] or "(empty)")

        # Three-lane retrieval (lexical + vector + graph when enriched)
        three_lane = memory.search("beach painting", lanes=QueryLanes.THREE_LANE)
        print(f"\nThree-lane search ({three_lane.total_tokens} tokens):")
        print(three_lane.context[:500] or "(empty)")

        # Token budget utilities
        sample = "You are a helpful assistant with access to conversation memory."
        print(f"\nEstimate tokens: {memory.estimate_tokens(sample)}")
        print(f"Format markdown: {memory.format_context(sample, format='markdown')[:80]}...")


if __name__ == "__main__":
    main()