# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Structural tests for AgentMemory and ContextService client surface."""

import dataclasses
import inspect

import pytest

from sochdb import (
    AgentMemory,
    ContextQueryResult,
    ContextSectionResult,
    EpisodeWriteResult,
    QueryLanes,
    SochDBClient,
    build_ingest_section,
    build_search_section,
    create_agent_memory,
)
from sochdb.memory.agent_memory import ContextSectionType


def _dc_fields(cls):
    return {f.name for f in dataclasses.fields(cls)}


def test_context_result_dataclasses():
    assert _dc_fields(ContextSectionResult) == {"name", "tokens_used", "truncated", "content"}
    assert _dc_fields(ContextQueryResult) == {"context", "total_tokens", "section_results", "error"}
    assert _dc_fields(EpisodeWriteResult) == {
        "episode_id",
        "t_created",
        "lexical_indexed",
        "ingestion_lag_us",
        "enrichment_queued",
        "error",
    }


def test_query_lanes_constants():
    assert QueryLanes.LEXICAL == "lexical"
    assert QueryLanes.THREE_LANE == "three_lane"
    assert QueryLanes.HYBRID == "hybrid"


def test_build_search_section():
    section = build_search_section(
        "who attended the support group?",
        lanes=QueryLanes.HYBRID,
        namespace="agent-42",
        priority=1,
        as_of=1_700_000_000_000,
    )
    assert section["type"] == ContextSectionType.SEARCH
    assert section["query"] == "who attended the support group?"
    assert section["options"] == {
        "lanes": QueryLanes.HYBRID,
        "namespace": "agent-42",
        "as_of": "1700000000000",
    }
    assert section["priority"] == 1


def test_build_ingest_section_with_metadata():
    section = build_ingest_section(
        "Episode text",
        namespace="agent-42",
        metadata={"speaker": "Caroline"},
    )
    assert section["type"] == ContextSectionType.GET
    assert section["options"]["episode_text"] == "Episode text"
    assert section["options"]["namespace"] == "agent-42"
    assert '"speaker"' in section["options"]["metadata_json"]


def test_agent_memory_factory_and_signatures():
    client = SochDBClient("localhost:50051")
    memory = create_agent_memory(client, namespace="bench-ns", token_limit=8192)
    assert isinstance(memory, AgentMemory)
    assert memory.namespace == "bench-ns"
    assert memory.token_limit == 8192

    search_sig = inspect.signature(memory.search)
    assert "lanes" in search_sig.parameters
    assert "as_of" in search_sig.parameters
    assert search_sig.parameters["lanes"].default == QueryLanes.LEXICAL

    write_sig = inspect.signature(memory.write_episode)
    assert set(write_sig.parameters.keys()) >= {"text", "t_valid_from", "metadata", "namespace"}