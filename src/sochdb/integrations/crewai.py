"""
CrewAI integration helpers for SochDB.

This module provides:

- a backend-agnostic SochDB knowledge store
- optional CrewAI tools for search and memory writes

The integration supports both embedded collections and remote gRPC collections.
CrewAI itself remains an optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from ..grpc_client import Document, SochDBClient
from ..namespace import Collection


EmbeddingFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass
class SochDBKnowledgeHit:
    """Search result returned by :class:`SochDBKnowledgeStore`."""

    id: str
    content: str
    metadata: Dict[str, str]
    score: Optional[float] = None


class _KnowledgeBackend(Protocol):
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        ...

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SochDBKnowledgeHit]:
        ...


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not metadata:
        return {}

    normalized: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = str(value)
        else:
            normalized[str(key)] = json.dumps(value, sort_keys=True)
    return normalized


class _EmbeddedCollectionBackend:
    def __init__(self, collection: Collection):
        self._collection = collection

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        ids = [doc["id"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        contents = [doc.get("content", "") for doc in documents]
        self._collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
            documents=contents,
        )
        return ids

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SochDBKnowledgeHit]:
        results = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            where=metadata_filter,
            include=["metadatas", "documents"],
        )
        hits: List[SochDBKnowledgeHit] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        for doc_id, distance, metadata, content in zip(ids, distances, metadatas, documents):
            hits.append(
                SochDBKnowledgeHit(
                    id=str(doc_id),
                    content=str(content or ""),
                    metadata=_normalize_metadata(metadata),
                    score=1.0 - float(distance),
                )
            )
        return hits


class _GrpcCollectionBackend:
    def __init__(
        self,
        client: SochDBClient,
        collection_name: str,
        namespace: str = "default",
    ):
        self._client = client
        self._collection_name = collection_name
        self._namespace = namespace

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        grpc_docs: List[Dict[str, Any]] = []
        for doc in documents:
            grpc_docs.append(
                {
                    "id": doc["id"],
                    "content": doc.get("content", ""),
                    "embedding": list(doc["embedding"]),
                    "metadata": _normalize_metadata(doc.get("metadata")),
                }
            )
        return self._client.add_documents(
            self._collection_name,
            grpc_docs,
            namespace=self._namespace,
        )

    def search(
        self,
        embedding: Sequence[float],
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[SochDBKnowledgeHit]:
        docs: List[Document] = self._client.search_collection(
            self._collection_name,
            list(embedding),
            k=k,
            namespace=self._namespace,
            filter=metadata_filter,
        )
        return [
            SochDBKnowledgeHit(
                id=doc.id,
                content=doc.content,
                metadata=_normalize_metadata(doc.metadata),
            )
            for doc in docs
        ]


class SochDBKnowledgeStore:
    """
    Small adapter that turns SochDB collections into a CrewAI-friendly
    searchable knowledge store.

    The embedder callback is intentionally user-supplied so teams can plug in
    OpenAI, Azure OpenAI, local sentence-transformers, FastEmbed, or any other
    embedding provider they already use.
    """

    def __init__(
        self,
        *,
        backend: _KnowledgeBackend,
        embedder: EmbeddingFn,
    ):
        self._backend = backend
        self._embedder = embedder

    @classmethod
    def from_collection(
        cls,
        collection: Collection,
        *,
        embedder: EmbeddingFn,
    ) -> "SochDBKnowledgeStore":
        return cls(backend=_EmbeddedCollectionBackend(collection), embedder=embedder)

    @classmethod
    def from_client(
        cls,
        client: SochDBClient,
        *,
        collection_name: str,
        embedder: EmbeddingFn,
        namespace: str = "default",
    ) -> "SochDBKnowledgeStore":
        return cls(
            backend=_GrpcCollectionBackend(client, collection_name, namespace=namespace),
            embedder=embedder,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if not texts:
            return []

        embeddings = self._embedder(texts)
        embeddings = [list(map(float, embedding)) for embedding in embeddings]
        if len(embeddings) != len(texts):
            raise ValueError("Embedder returned a different number of embeddings than texts")

        if metadatas is None:
            metadatas = [None] * len(texts)
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        if ids is None:
            ids = [f"sochdb-crewai-{idx}" for idx in range(len(texts))]
        if len(ids) != len(texts):
            raise ValueError("ids length must match texts length")

        documents: List[Dict[str, Any]] = []
        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            documents.append(
                {
                    "id": str(doc_id),
                    "content": text,
                    "embedding": embedding,
                    "metadata": _normalize_metadata(metadata),
                }
            )
        return self._backend.add_documents(documents)

    def remember(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        generated_id = doc_id or f"sochdb-crewai-memory-{uuid.uuid4()}"
        return self.add_texts([text], metadatas=[metadata], ids=[generated_id])[0]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SochDBKnowledgeHit]:
        embedding = self._embedder([query])
        if len(embedding) != 1:
            raise ValueError("Embedder must return exactly one embedding for a single query")
        normalized_filter = _normalize_metadata(metadata_filter)
        return self._backend.search(embedding[0], top_k, normalized_filter or None)

    @staticmethod
    def format_hits(hits: Iterable[SochDBKnowledgeHit]) -> str:
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            score = "" if hit.score is None else f" (score={hit.score:.4f})"
            metadata = ""
            if hit.metadata:
                metadata = f"\nmetadata: {json.dumps(hit.metadata, sort_keys=True)}"
            lines.append(f"{idx}. [{hit.id}]{score}\n{hit.content}{metadata}")
        return "\n\n".join(lines) if lines else "No matching knowledge found."


def crewai_available() -> bool:
    try:
        import crewai  # noqa: F401
    except Exception:
        return False
    return True


def _require_crewai() -> None:
    if not crewai_available():
        raise ImportError(
            "CrewAI integration requires the optional 'crewai' dependency. "
            "Install with: pip install 'sochdb[crewai]' or pip install crewai"
        )


try:
    from pydantic import BaseModel, Field
    try:
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - pydantic v1 compatibility
        ConfigDict = None  # type: ignore[assignment]
    from crewai.tools import BaseTool
    _HAS_CREWAI = True
except Exception:  # pragma: no cover - handled by lazy import checks
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    BaseTool = object  # type: ignore[assignment]
    _HAS_CREWAI = False


if _HAS_CREWAI:
    class SochDBSearchInput(BaseModel):
        query: str = Field(..., description="Natural-language question to search in SochDB.")
        metadata_filter_json: Optional[str] = Field(
            default=None,
            description="Optional JSON object used as a metadata filter.",
        )


    class SochDBSearchTool(BaseTool):
        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)
        else:  # pragma: no cover - pydantic v1 compatibility
            class Config:
                arbitrary_types_allowed = True

        name: str = "sochdb_search"
        description: str = (
            "Searches a SochDB-backed knowledge base for relevant context. "
            "Use this when you need grounded project or domain facts."
        )
        args_schema = SochDBSearchInput

        knowledge_store: SochDBKnowledgeStore
        top_k: int = 5

        def _run(self, query: str, metadata_filter_json: Optional[str] = None) -> str:
            metadata_filter = None
            if metadata_filter_json:
                parsed = json.loads(metadata_filter_json)
                if not isinstance(parsed, dict):
                    raise ValueError("metadata_filter_json must decode to a JSON object")
                metadata_filter = parsed
            hits = self.knowledge_store.search(
                query,
                top_k=self.top_k,
                metadata_filter=metadata_filter,
            )
            return self.knowledge_store.format_hits(hits)


    class SochDBRememberInput(BaseModel):
        content: str = Field(..., description="Text content to store in SochDB.")
        metadata_json: Optional[str] = Field(
            default=None,
            description="Optional JSON object with metadata to store alongside the content.",
        )
        doc_id: Optional[str] = Field(
            default=None,
            description="Optional stable identifier for the stored memory record.",
        )


    class SochDBRememberTool(BaseTool):
        if ConfigDict is not None:
            model_config = ConfigDict(arbitrary_types_allowed=True)
        else:  # pragma: no cover - pydantic v1 compatibility
            class Config:
                arbitrary_types_allowed = True

        name: str = "sochdb_remember"
        description: str = (
            "Stores a new memory or knowledge snippet in SochDB so future tasks "
            "can retrieve it."
        )
        args_schema = SochDBRememberInput

        knowledge_store: SochDBKnowledgeStore

        def _run(
            self,
            content: str,
            metadata_json: Optional[str] = None,
            doc_id: Optional[str] = None,
        ) -> str:
            metadata = None
            if metadata_json:
                parsed = json.loads(metadata_json)
                if not isinstance(parsed, dict):
                    raise ValueError("metadata_json must decode to a JSON object")
                metadata = parsed
            stored_id = self.knowledge_store.remember(content, metadata=metadata, doc_id=doc_id)
            return f"Stored memory in SochDB with id={stored_id}"

else:
    class SochDBSearchTool:  # pragma: no cover - lazy dependency guard
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_crewai()


    class SochDBRememberTool:  # pragma: no cover - lazy dependency guard
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_crewai()


def create_crewai_tools(
    knowledge_store: SochDBKnowledgeStore,
    *,
    top_k: int = 5,
) -> List[Any]:
    """
    Create the default CrewAI tool set backed by SochDB.

    Returns:
        `[SochDBSearchTool(...), SochDBRememberTool(...)]`
    """

    _require_crewai()
    return [
        SochDBSearchTool(knowledge_store=knowledge_store, top_k=top_k),
        SochDBRememberTool(knowledge_store=knowledge_store),
    ]


__all__ = [
    "SochDBKnowledgeHit",
    "SochDBKnowledgeStore",
    "SochDBSearchTool",
    "SochDBRememberTool",
    "create_crewai_tools",
    "crewai_available",
]
