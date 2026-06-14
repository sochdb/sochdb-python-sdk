# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Integration test: as_of point-in-time search over gRPC (requires server)."""

import os
import socket

import pytest

from sochdb import QueryLanes, SochDBClient, create_agent_memory

ADDRESS = os.environ.get("SOCHDB_ADDRESS", "localhost:50051")


def _server_up(host: str = "127.0.0.1", port: int = 50051) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _server_up(), reason="sochdb-grpc-server not running")
def test_as_of_excludes_future_episodes():
    """Demo path: wrong fact at t=10000 hidden when querying as_of=5000."""
    with SochDBClient(ADDRESS) as client:
        memory = create_agent_memory(client, namespace="test-asof-dispatch")

        memory.write_episode(
            "Driver Mike on route 42, quoted $1200.",
            t_valid_from=1_000,
        )
        memory.write_episode(
            "Driver Sarah on route 42, quoted $1400.",
            t_valid_from=10_000,
        )

        past = memory.search("route 42 driver rate", as_of=5_000, lanes=QueryLanes.LEXICAL)
        assert past.error == ""
        ctx = past.context.lower()
        assert "mike" in ctx or "$1200" in ctx or "1200" in ctx
        assert "sarah" not in ctx
        assert "$1400" not in ctx and "1400" not in ctx

        present = memory.search("route 42 driver rate", lanes=QueryLanes.LEXICAL)
        assert present.error == ""
        # Without as_of both episodes may appear.
        assert len(present.context) > 0