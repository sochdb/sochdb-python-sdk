# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Agent memory client — thin wrapper over ContextService + sochdb-memory backend.

Maps to the server-side pipeline:
  write_episode → lexical index (immediate) → async vector enrichment
  query_context → three-lane retrieval + ContextCompiler (exact BPE budget)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..grpc_client import SochDBClient, ContextQueryResult, EpisodeWriteResult


class ContextSectionType:
    """ContextService section types (proto enum values)."""

    GET = 0
    LAST = 1
    SEARCH = 2
    SELECT = 3


class QueryLanes:
    """Retrieval lane presets passed via section options (`lanes=`)."""

    LEXICAL = "lexical"
    THREE_LANE = "three_lane"
    HYBRID = "hybrid"
    BM25 = "bm25"
    TRIGRAM = "trigram"


@dataclass
class AgentMemoryConfig:
    """Defaults for an agent memory namespace."""

    namespace: str = "default"
    token_limit: int = 4096
    output_format: str = "markdown"
    search_lanes: str = QueryLanes.LEXICAL


def build_search_section(
    query: str,
    *,
    name: str = "memory",
    priority: int = 0,
    lanes: str = QueryLanes.LEXICAL,
    namespace: Optional[str] = None,
    as_of: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a SEARCH section dict for `query_context`.

    Options:
      lanes: lexical | three_lane | hybrid
      namespace: memory namespace override
      as_of: unix timestamp (ms) for bi-temporal point-in-time recall
    """
    options: Dict[str, str] = {"lanes": lanes}
    if namespace:
        options["namespace"] = namespace
    if as_of is not None:
        options["as_of"] = str(int(as_of))
    return {
        "name": name,
        "priority": priority,
        "type": ContextSectionType.SEARCH,
        "query": query,
        "options": options,
    }


def build_ingest_section(
    text: str,
    *,
    name: str = "ingest",
    priority: int = 0,
    namespace: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a GET section that ingests episode text (legacy path)."""
    options: Dict[str, str] = {"episode_text": text}
    if namespace:
        options["namespace"] = namespace
    if metadata:
        options["metadata_json"] = json.dumps(metadata)
    return {
        "name": name,
        "priority": priority,
        "type": ContextSectionType.GET,
        "query": "",
        "options": options,
    }


class AgentMemory:
    """
    High-level agent memory API (gRPC / sochdb-memory backend).

    Example::

        client = SochDBClient("localhost:50051")
        memory = AgentMemory(client, namespace="agent-42")

        memory.write_episode("Caroline went to the LGBTQ support group.")
        ctx = memory.search("LGBTQ support group", lanes=QueryLanes.THREE_LANE)
        print(ctx.context, ctx.total_tokens)
    """

    def __init__(
        self,
        client: "SochDBClient",
        namespace: str = "default",
        *,
        session_id: Optional[str] = None,
        token_limit: int = 4096,
        output_format: str = "markdown",
    ):
        self._client = client
        self.namespace = namespace
        self.session_id = session_id or namespace
        self.token_limit = token_limit
        self.output_format = output_format

    def write_episode(
        self,
        text: str,
        *,
        t_valid_from: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> "EpisodeWriteResult":
        """Ingest an episode via ContextService.WriteEpisode."""
        return self._client.write_episode(
            namespace=namespace or self.namespace,
            text=text,
            t_valid_from=t_valid_from,
            metadata=metadata,
        )

    def get_episode(self, doc_id: int, *, namespace: Optional[str] = None) -> str:
        """Fetch episode text by doc id."""
        section = {
            "name": "episode",
            "priority": 0,
            "type": ContextSectionType.GET,
            "query": "",
            "options": {
                "doc_id": str(doc_id),
                "namespace": namespace or self.namespace,
            },
        }
        result = self._client.query_context(
            session_id=self.session_id,
            sections=[section],
            token_limit=8192,
            format=self.output_format,
        )
        return result.context

    def search(
        self,
        query: str,
        *,
        token_limit: Optional[int] = None,
        lanes: str = QueryLanes.LEXICAL,
        format: Optional[str] = None,
        namespace: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> "ContextQueryResult":
        """Retrieve + compile memory into an exact-BPE context string.

        Pass ``as_of`` (unix ms) for point-in-time recall — only episodes whose
        ``t_valid_from`` is at or before ``as_of`` are visible.
        """
        section = build_search_section(
            query,
            lanes=lanes,
            namespace=namespace or self.namespace,
            as_of=as_of,
        )
        return self._client.query_context(
            session_id=self.session_id,
            sections=[section],
            token_limit=token_limit or self.token_limit,
            format=format or self.output_format,
        )

    def compile_context(
        self,
        sections: List[Dict[str, Any]],
        *,
        token_limit: Optional[int] = None,
        format: Optional[str] = None,
    ) -> "ContextQueryResult":
        """Assemble multi-section context under a token budget."""
        return self._client.query_context(
            session_id=self.session_id,
            sections=sections,
            token_limit=token_limit or self.token_limit,
            format=format or self.output_format,
        )

    def estimate_tokens(self, content: str, *, model: str = "") -> int:
        """Exact BPE token count via ContextService.EstimateTokens."""
        return self._client.estimate_tokens(content, model=model)

    def format_context(self, content: str, *, format: Optional[str] = None) -> str:
        """Format raw content (toon/json/markdown/text)."""
        return self._client.format_context(content, format=format or self.output_format)


def create_agent_memory(
    client: "SochDBClient",
    namespace: str = "default",
    **kwargs: Any,
) -> AgentMemory:
    """Factory for AgentMemory."""
    return AgentMemory(client, namespace, **kwargs)