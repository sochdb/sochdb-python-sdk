from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sochdb.grpc_client import Document
from sochdb.integrations.crewai import SochDBKnowledgeHit, SochDBKnowledgeStore


class FakeBackend:
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        self.documents.extend(documents)
        return [doc["id"] for doc in documents]

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SochDBKnowledgeHit]:
        hits: List[SochDBKnowledgeHit] = []
        for doc in self.documents:
            metadata = doc.get("metadata", {})
            if metadata_filter:
                allowed = all(metadata.get(key) == value for key, value in metadata_filter.items())
                if not allowed:
                    continue
            hits.append(
                SochDBKnowledgeHit(
                    id=doc["id"],
                    content=doc.get("content", ""),
                    metadata=metadata,
                    score=0.9,
                )
            )
        return hits[:k]


class FakeGrpcClient:
    def __init__(self) -> None:
        self.add_calls: List[Dict[str, Any]] = []
        self.search_calls: List[Dict[str, Any]] = []

    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        namespace: str = "default",
    ) -> List[str]:
        self.add_calls.append(
            {
                "collection_name": collection_name,
                "documents": documents,
                "namespace": namespace,
            }
        )
        return [doc["id"] for doc in documents]

    def search_collection(
        self,
        collection_name: str,
        query: List[float],
        k: int = 10,
        namespace: str = "default",
        filter: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "k": k,
                "namespace": namespace,
                "filter": filter,
            }
        )
        return [
            Document(
                id="grpc-doc-1",
                content="remote benchmark note",
                embedding=[1.0, 0.0],
                metadata={"topic": "benchmark"},
            )
        ]


def fake_embedder(texts: Sequence[str]) -> List[List[float]]:
    return [[float(len(text)), 1.0] for text in texts]


def test_add_texts_normalizes_metadata() -> None:
    backend = FakeBackend()
    store = SochDBKnowledgeStore(backend=backend, embedder=fake_embedder)

    ids = store.add_texts(
        ["hello world"],
        metadatas=[{"topic": "demo", "count": 3, "nested": {"a": 1}}],
        ids=["doc-1"],
    )

    assert ids == ["doc-1"]
    assert backend.documents[0]["metadata"]["topic"] == "demo"
    assert backend.documents[0]["metadata"]["count"] == "3"
    assert backend.documents[0]["metadata"]["nested"] == '{"a": 1}'


def test_search_returns_filtered_hits() -> None:
    backend = FakeBackend()
    store = SochDBKnowledgeStore(backend=backend, embedder=fake_embedder)
    store.add_texts(
        ["architecture note", "benchmark note"],
        metadatas=[{"topic": "architecture"}, {"topic": "benchmark"}],
        ids=["doc-a", "doc-b"],
    )

    hits = store.search("benchmark", top_k=5, metadata_filter={"topic": "benchmark"})

    assert len(hits) == 1
    assert hits[0].id == "doc-b"
    assert hits[0].metadata["topic"] == "benchmark"


def test_remember_generates_unique_ids() -> None:
    backend = FakeBackend()
    store = SochDBKnowledgeStore(backend=backend, embedder=fake_embedder)

    first = store.remember("fact one")
    second = store.remember("fact two")

    assert first != second
    assert len(backend.documents) == 2


def test_format_hits_is_human_readable() -> None:
    text = SochDBKnowledgeStore.format_hits(
        [
            SochDBKnowledgeHit(
                id="doc-1",
                content="SochDB supports embedded mode.",
                metadata={"topic": "architecture"},
                score=0.75,
            )
        ]
    )

    assert "doc-1" in text
    assert "architecture" in text
    assert "0.7500" in text


def test_from_client_uses_grpc_backend() -> None:
    client = FakeGrpcClient()
    store = SochDBKnowledgeStore.from_client(
        client,
        collection_name="knowledge",
        namespace="crew",
        embedder=fake_embedder,
    )

    inserted_ids = store.add_texts(
        ["remote deployment note"],
        metadatas=[{"topic": "deployment"}],
        ids=["grpc-1"],
    )
    hits = store.search("deployment", top_k=3, metadata_filter={"topic": "benchmark"})

    assert inserted_ids == ["grpc-1"]
    assert client.add_calls[0]["collection_name"] == "knowledge"
    assert client.add_calls[0]["namespace"] == "crew"
    assert client.search_calls[0]["filter"] == {"topic": "benchmark"}
    assert hits[0].id == "grpc-doc-1"
