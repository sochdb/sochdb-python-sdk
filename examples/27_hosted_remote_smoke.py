#!/usr/bin/env python3
"""
Minimal hosted SochDB remote smoke test.

This example:
1. creates a fresh remote collection
2. inserts a couple of documents
3. runs a search and prints a compact summary

Environment variables:
    SOCHDB_GRPC_ADDRESS   default: studio.agentslab.host:50053
    SOCHDB_NAMESPACE      default: default
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sochdb import SochDBClient


DEFAULT_GRPC_ADDRESS = "studio.agentslab.host:50053"
DEFAULT_NAMESPACE = "default"


def main() -> None:
    grpc_address = os.environ.get("SOCHDB_GRPC_ADDRESS", DEFAULT_GRPC_ADDRESS)
    namespace = os.environ.get("SOCHDB_NAMESPACE", DEFAULT_NAMESPACE)

    run_id = f"python-live-{int(time.time())}"
    collection = f"sdk_live_python_{run_id}"
    client = SochDBClient(grpc_address)

    print(f"Connecting to remote SochDB at {grpc_address}")
    client.create_collection(collection, dimension=4, namespace=namespace, metric="cosine")

    inserted_ids = client.add_documents(
        collection,
        [
            {
                "id": f"{run_id}-doc-1",
                "content": "python live parity doc",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "metadata": {"sdk": "python", "run_id": run_id},
            },
            {
                "id": f"{run_id}-doc-2",
                "content": "python live parity second doc",
                "embedding": [0.0, 1.0, 0.0, 0.0],
                "metadata": {"sdk": "python", "run_id": run_id},
            },
        ],
    )

    results = client.search_collection(
        collection,
        [1.0, 0.0, 0.0, 0.0],
        2,
        namespace=namespace,
    )

    top_id = results[0].id if results else None
    print(
        {
            "sdk": "python",
            "collection": collection,
            "inserted_count": len(inserted_ids),
            "result_count": len(results),
            "top_id": top_id,
        }
    )


if __name__ == "__main__":
    main()
