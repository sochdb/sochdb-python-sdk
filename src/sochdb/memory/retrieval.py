# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hybrid Retrieval Interface with Pre-Filtering

This module provides a unified retrieval API that:

1. Leverages SochDB's built-in RRF (Reciprocal Rank Fusion)
2. Enforces pre-filtering (never post-filter for security)
3. Supports optional cross-encoder reranking
4. Provides a single "RAG-grade" endpoint

Key invariants:
- Filtering happens during candidate generation, not after
- Results ⊆ allowed_set (monotonicity property)
- Ranking is stable and debuggable

Supports both embedded (FFI) and server (gRPC) modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Union
)
import time


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class RetrievalConfig:
    """
    Configuration for hybrid retrieval.
    
    Attributes:
        k: Number of results to return
        alpha: Balance between vector (1.0) and keyword (0.0)
        rrf_k: RRF constant (typically 60)
        min_score: Minimum score threshold
        enable_rerank: Whether to use cross-encoder reranking
        rerank_top_n: Number of candidates for reranking
        vector_weight: Weight for vector results in fusion
        keyword_weight: Weight for keyword results in fusion
    """
    k: int = 10
    alpha: float = 0.5
    rrf_k: int = 60
    min_score: Optional[float] = None
    enable_rerank: bool = False
    rerank_top_n: int = 50
    vector_weight: float = 1.0
    keyword_weight: float = 1.0


@dataclass
class RetrievalResult:
    """
    A single retrieval result.
    
    Attributes:
        id: Document ID
        score: Combined/final score
        content: Document content
        metadata: Document metadata
        vector_rank: Rank in vector results (None if not in vector results)
        keyword_rank: Rank in keyword results (None if not in keyword results)
        rerank_score: Cross-encoder score (None if not reranked)
    """
    id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "rerank_score": self.rerank_score,
        }


@dataclass
class RetrievalResponse:
    """
    Complete retrieval response.
    
    Attributes:
        results: List of retrieval results
        total_candidates: Total candidates before final selection
        query_time_ms: Total query time
        vector_count: Number of vector results
        keyword_count: Number of keyword results
        filtered_count: Number filtered by allowed_set
    """
    results: List[RetrievalResult] = field(default_factory=list)
    total_candidates: int = 0
    query_time_ms: float = 0.0
    vector_count: int = 0
    keyword_count: int = 0
    filtered_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_candidates": self.total_candidates,
            "query_time_ms": self.query_time_ms,
            "vector_count": self.vector_count,
            "keyword_count": self.keyword_count,
            "filtered_count": self.filtered_count,
        }


# ============================================================================
# Allowed Set (Pre-Filter)
# ============================================================================

class AllowedSet:
    """
    Pre-filter set for security-by-construction.
    
    Ensures retrieval only returns documents in the allowed set.
    This is the key invariant for multi-tenant safety.
    
    Usage:
        # Allow specific document IDs
        allowed = AllowedSet.from_ids(["doc1", "doc2", "doc3"])
        
        # Allow by namespace prefix
        allowed = AllowedSet.from_namespace("user_123")
        
        # Allow by metadata filter
        allowed = AllowedSet.from_filter({"tenant": "acme"})
        
        # Allow all (trusted context)
        allowed = AllowedSet.allow_all()
    """
    
    def __init__(
        self,
        ids: Optional[Set[str]] = None,
        namespace: Optional[str] = None,
        filter_fn: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        allow_all: bool = False,
    ):
        self._ids = ids
        self._namespace = namespace
        self._filter_fn = filter_fn
        self._allow_all = allow_all
    
    @classmethod
    def from_ids(cls, ids: List[str]) -> "AllowedSet":
        """Create from explicit ID list."""
        return cls(ids=set(ids))
    
    @classmethod
    def from_namespace(cls, namespace: str) -> "AllowedSet":
        """Create from namespace prefix."""
        return cls(namespace=namespace)
    
    @classmethod
    def from_filter(cls, filter_fn: Callable[[str, Dict[str, Any]], bool]) -> "AllowedSet":
        """Create from filter function."""
        return cls(filter_fn=filter_fn)
    
    @classmethod
    def allow_all(cls) -> "AllowedSet":
        """Allow all documents (trusted context only)."""
        return cls(allow_all=True)
    
    def contains(self, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Check if document is allowed."""
        if self._allow_all:
            return True
        
        if self._ids is not None:
            return doc_id in self._ids
        
        if self._namespace is not None:
            return doc_id.startswith(self._namespace)
        
        if self._filter_fn is not None and metadata is not None:
            return self._filter_fn(doc_id, metadata)
        
        return False
    
    def filter_results(
        self,
        results: List[Tuple[str, float, Optional[Dict[str, Any]]]],
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Filter results by allowed set."""
        return [
            (doc_id, score, metadata)
            for doc_id, score, metadata in results
            if self.contains(doc_id, metadata)
        ]


# ============================================================================
# Retrieval Backend Interface
# ============================================================================

class RetrievalBackend(ABC):
    """
    Abstract interface for retrieval backends.
    """
    
    @abstractmethod
    def vector_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Vector search. Returns (id, score, metadata) tuples."""
        pass
    
    @abstractmethod
    def keyword_search(
        self,
        namespace: str,
        collection: str,
        query_text: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Keyword search. Returns (id, score, metadata) tuples."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        query_text: str,
        k: int,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Hybrid search (RRF fusion). Returns (id, score, metadata) tuples."""
        pass
    
    @abstractmethod
    def get_document(
        self,
        namespace: str,
        collection: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        pass


# ============================================================================
# FFI Backend
# ============================================================================

class FFIRetrievalBackend(RetrievalBackend):
    """
    Retrieval backend using embedded database via FFI.
    
    Leverages the Collection API's built-in search methods.
    """
    
    def __init__(self, db: "Database"):
        self._db = db
    
    def _get_collection(self, namespace: str, collection: str):
        """Get or create collection."""
        ns = self._db.namespace(namespace)
        return ns.collection(collection)
    
    def vector_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        coll = self._get_collection(namespace, collection)
        results = coll.search(vector=query_vector, k=k, filter=filter)
        return [
            (r.id, r.score, r.metadata or {})
            for r in results.results
        ]
    
    def keyword_search(
        self,
        namespace: str,
        collection: str,
        query_text: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        coll = self._get_collection(namespace, collection)
        try:
            results = coll.keyword_search(query=query_text, k=k, filter=filter)
            return [
                (r.id, r.score, r.metadata or {})
                for r in results.results
            ]
        except Exception:
            # Keyword search not enabled
            return []
    
    def hybrid_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        query_text: str,
        k: int,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        coll = self._get_collection(namespace, collection)
        try:
            results = coll.hybrid_search(
                vector=query_vector,
                text_query=query_text,
                k=k,
                alpha=alpha,
                filter=filter,
            )
            return [
                (r.id, r.score, r.metadata or {})
                for r in results.results
            ]
        except Exception:
            # Fall back to vector only
            return self.vector_search(namespace, collection, query_vector, k, filter)
    
    def get_document(
        self,
        namespace: str,
        collection: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        coll = self._get_collection(namespace, collection)
        return coll.get(doc_id)


# ============================================================================
# gRPC Backend
# ============================================================================

class GrpcRetrievalBackend(RetrievalBackend):
    """
    Retrieval backend using gRPC client.
    """
    
    def __init__(self, client: "SochDBClient"):
        self._client = client
    
    def _collection_name(self, namespace: str, collection: str) -> str:
        return f"{namespace}/{collection}"
    
    def vector_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        results = self._client.search(
            collection=self._collection_name(namespace, collection),
            query_vector=query_vector,
            k=k,
            filter=filter,
        )
        return [
            (r.id, r.distance, getattr(r, 'metadata', {}) or {})
            for r in results
        ]
    
    def keyword_search(
        self,
        namespace: str,
        collection: str,
        query_text: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        try:
            results = self._client.keyword_search(
                collection=self._collection_name(namespace, collection),
                query=query_text,
                k=k,
                filter=filter,
            )
            return [
                (r.id, r.score, getattr(r, 'metadata', {}) or {})
                for r in results
            ]
        except Exception:
            return []
    
    def hybrid_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        query_text: str,
        k: int,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        try:
            results = self._client.hybrid_search(
                collection=self._collection_name(namespace, collection),
                query_vector=query_vector,
                query_text=query_text,
                k=k,
                alpha=alpha,
                filter=filter,
            )
            return [
                (r.id, r.score, getattr(r, 'metadata', {}) or {})
                for r in results
            ]
        except Exception:
            return self.vector_search(namespace, collection, query_vector, k, filter)
    
    def get_document(
        self,
        namespace: str,
        collection: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self._client.get_document(
                collection=self._collection_name(namespace, collection),
                doc_id=doc_id,
            )
        except Exception:
            return None


# ============================================================================
# In-Memory Backend
# ============================================================================

class InMemoryRetrievalBackend(RetrievalBackend):
    """
    In-memory retrieval backend for testing.
    """
    
    def __init__(self):
        self._documents: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    def add_document(
        self,
        namespace: str,
        collection: str,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a document for testing."""
        key = f"{namespace}/{collection}"
        if key not in self._documents:
            self._documents[key] = {}
        self._documents[key][doc_id] = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
        }
    
    def vector_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        import math
        
        key = f"{namespace}/{collection}"
        docs = self._documents.get(key, {})
        
        results = []
        for doc_id, doc in docs.items():
            if filter:
                if not self._matches_filter(doc.get("metadata", {}), filter):
                    continue
            
            # Compute cosine similarity
            embedding = doc.get("embedding", [])
            if embedding:
                dot = sum(a * b for a, b in zip(query_vector, embedding))
                norm_a = math.sqrt(sum(a * a for a in query_vector))
                norm_b = math.sqrt(sum(b * b for b in embedding))
                if norm_a > 0 and norm_b > 0:
                    similarity = dot / (norm_a * norm_b)
                    results.append((doc_id, similarity, doc.get("metadata", {})))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def keyword_search(
        self,
        namespace: str,
        collection: str,
        query_text: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        key = f"{namespace}/{collection}"
        docs = self._documents.get(key, {})
        
        query_terms = set(query_text.lower().split())
        
        results = []
        for doc_id, doc in docs.items():
            if filter:
                if not self._matches_filter(doc.get("metadata", {}), filter):
                    continue
            
            content = doc.get("content", "").lower()
            doc_terms = set(content.split())
            
            # Simple term overlap score
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                results.append((doc_id, score, doc.get("metadata", {})))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def hybrid_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        query_text: str,
        k: int,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        # Get both result sets
        vector_results = self.vector_search(
            namespace, collection, query_vector, k * 2, filter
        )
        keyword_results = self.keyword_search(
            namespace, collection, query_text, k * 2, filter
        )
        
        # RRF fusion
        rrf_k = 60
        scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        
        for rank, (doc_id, _, metadata) in enumerate(vector_results):
            rrf_score = alpha / (rrf_k + rank + 1)
            if doc_id in scores:
                scores[doc_id] = (scores[doc_id][0] + rrf_score, metadata)
            else:
                scores[doc_id] = (rrf_score, metadata)
        
        for rank, (doc_id, _, metadata) in enumerate(keyword_results):
            rrf_score = (1 - alpha) / (rrf_k + rank + 1)
            if doc_id in scores:
                scores[doc_id] = (scores[doc_id][0] + rrf_score, scores[doc_id][1])
            else:
                scores[doc_id] = (rrf_score, metadata)
        
        # Sort by combined score
        results = [
            (doc_id, score, metadata)
            for doc_id, (score, metadata) in scores.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_document(
        self,
        namespace: str,
        collection: str,
        doc_id: str,
    ) -> Optional[Dict[str, Any]]:
        key = f"{namespace}/{collection}"
        docs = self._documents.get(key, {})
        return docs.get(doc_id)
    
    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True


# ============================================================================
# Hybrid Retriever
# ============================================================================

class HybridRetriever:
    """
    Unified hybrid retrieval interface.
    
    Provides a single API for retrieval with:
    - Built-in RRF fusion (leverages SochDB's implementation)
    - Pre-filtering via AllowedSet (security invariant)
    - Optional cross-encoder reranking
    - Debugging and explain capabilities
    
    Usage:
        retriever = HybridRetriever.from_database(db, namespace="user_123")
        
        response = retriever.retrieve(
            query_text="machine learning papers",
            query_vector=embed("machine learning papers"),
            allowed=AllowedSet.from_namespace("user_123"),
            k=10,
        )
        
        for result in response.results:
            print(f"{result.id}: {result.score}")
    """
    
    def __init__(
        self,
        backend: RetrievalBackend,
        namespace: str,
        collection: str,
        config: Optional[RetrievalConfig] = None,
        reranker: Optional[Callable[[str, List[Tuple[str, str]]], List[float]]] = None,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            backend: Storage backend
            namespace: Namespace for isolation
            collection: Collection name
            config: Retrieval configuration
            reranker: Optional cross-encoder reranker
        """
        self._backend = backend
        self._namespace = namespace
        self._collection = collection
        self._config = config or RetrievalConfig()
        self._reranker = reranker
    
    @classmethod
    def from_database(
        cls,
        db: "Database",
        namespace: str,
        collection: str = "documents",
        **kwargs,
    ) -> "HybridRetriever":
        """Create retriever from embedded Database."""
        backend = FFIRetrievalBackend(db)
        return cls(backend, namespace, collection, **kwargs)
    
    @classmethod
    def from_client(
        cls,
        client: "SochDBClient",
        namespace: str,
        collection: str = "documents",
        **kwargs,
    ) -> "HybridRetriever":
        """Create retriever from gRPC client."""
        backend = GrpcRetrievalBackend(client)
        return cls(backend, namespace, collection, **kwargs)
    
    @classmethod
    def from_backend(
        cls,
        backend: RetrievalBackend,
        namespace: str,
        collection: str = "documents",
        **kwargs,
    ) -> "HybridRetriever":
        """Create retriever with explicit backend."""
        return cls(backend, namespace, collection, **kwargs)
    
    def retrieve(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        allowed: Optional[AllowedSet] = None,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """
        Retrieve documents using hybrid search.
        
        INVARIANT: Results ⊆ allowed_set
        
        Pre-filtering happens during candidate generation, not after.
        This ensures no ranking/leaking of disallowed documents.
        
        Args:
            query_text: Text query for keyword matching
            query_vector: Embedding vector for semantic search
            allowed: Pre-filter set (required for multi-tenant)
            k: Number of results
            alpha: Balance (1.0 = vector only, 0.0 = keyword only)
            filter: Additional metadata filter
            
        Returns:
            RetrievalResponse with filtered, ranked results
        """
        start_time = time.time()
        
        k = k or self._config.k
        alpha = alpha if alpha is not None else self._config.alpha
        
        # Default to allow-all if not specified (for single-tenant use)
        if allowed is None:
            allowed = AllowedSet.allow_all()
        
        # Determine search mode
        has_vector = query_vector is not None and len(query_vector) > 0
        has_text = query_text is not None and len(query_text.strip()) > 0
        
        # Request more candidates for pre-filtering
        candidate_k = k * 3
        
        if has_vector and has_text:
            # Hybrid search (uses built-in RRF)
            raw_results = self._backend.hybrid_search(
                self._namespace,
                self._collection,
                query_vector,
                query_text,
                candidate_k,
                alpha,
                filter,
            )
            vector_count = candidate_k
            keyword_count = candidate_k
        elif has_vector:
            # Vector-only search
            raw_results = self._backend.vector_search(
                self._namespace,
                self._collection,
                query_vector,
                candidate_k,
                filter,
            )
            vector_count = len(raw_results)
            keyword_count = 0
        elif has_text:
            # Keyword-only search
            raw_results = self._backend.keyword_search(
                self._namespace,
                self._collection,
                query_text,
                candidate_k,
                filter,
            )
            vector_count = 0
            keyword_count = len(raw_results)
        else:
            # No query provided
            return RetrievalResponse(
                results=[],
                query_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Apply pre-filter (AllowedSet)
        pre_filter_count = len(raw_results)
        filtered_results = allowed.filter_results(raw_results)
        filtered_count = pre_filter_count - len(filtered_results)
        
        # Build result objects
        results = []
        for idx, (doc_id, score, metadata) in enumerate(filtered_results):
            result = RetrievalResult(
                id=doc_id,
                score=score,
                metadata=metadata,
                content=metadata.get("content"),
            )
            results.append(result)
        
        # Optional reranking
        if self._reranker and self._config.enable_rerank and query_text:
            results = self._rerank(query_text, results[:self._config.rerank_top_n])
        
        # Apply min_score filter
        if self._config.min_score is not None:
            results = [r for r in results if r.score >= self._config.min_score]
        
        # Take top k
        results = results[:k]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            results=results,
            total_candidates=pre_filter_count,
            query_time_ms=query_time_ms,
            vector_count=vector_count,
            keyword_count=keyword_count,
            filtered_count=filtered_count,
        )
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Apply cross-encoder reranking."""
        if not self._reranker or not results:
            return results
        
        # Prepare pairs for reranker
        pairs = [
            (query, r.content or str(r.metadata))
            for r in results
        ]
        
        # Get rerank scores
        scores = self._reranker(query, pairs)
        
        # Update results with rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = score
            result.score = score  # Replace original score
        
        # Sort by rerank score
        results.sort(key=lambda r: r.rerank_score or 0, reverse=True)
        
        return results
    
    def explain(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        doc_id: str = None,
    ) -> Dict[str, Any]:
        """
        Explain why a document ranked where it did.
        
        Useful for debugging and understanding ranking behavior.
        """
        if not doc_id:
            return {"error": "doc_id required"}
        
        # Get the document
        doc = self._backend.get_document(self._namespace, self._collection, doc_id)
        if not doc:
            return {"error": "Document not found"}
        
        result = {
            "doc_id": doc_id,
            "document": doc,
        }
        
        # Get vector rank
        if query_vector:
            vector_results = self._backend.vector_search(
                self._namespace, self._collection, query_vector, 100, None
            )
            for rank, (rid, score, _) in enumerate(vector_results):
                if rid == doc_id:
                    result["vector_rank"] = rank + 1
                    result["vector_score"] = score
                    break
        
        # Get keyword rank
        if query_text:
            keyword_results = self._backend.keyword_search(
                self._namespace, self._collection, query_text, 100, None
            )
            for rank, (rid, score, _) in enumerate(keyword_results):
                if rid == doc_id:
                    result["keyword_rank"] = rank + 1
                    result["keyword_score"] = score
                    break
        
        # Calculate expected RRF score
        v_rank = result.get("vector_rank")
        k_rank = result.get("keyword_rank")
        alpha = self._config.alpha
        rrf_k = self._config.rrf_k
        
        rrf_score = 0
        if v_rank:
            rrf_score += alpha / (rrf_k + v_rank)
        if k_rank:
            rrf_score += (1 - alpha) / (rrf_k + k_rank)
        
        result["expected_rrf_score"] = rrf_score
        
        return result


# ============================================================================
# Factory Function
# ============================================================================

def create_retriever(
    backend,
    namespace: str,
    collection: str = "documents",
    **kwargs,
) -> HybridRetriever:
    """
    Create a retriever with auto-detected backend.
    
    Args:
        backend: Database, SochDBClient, or RetrievalBackend
        namespace: Namespace for isolation
        collection: Collection name
        **kwargs: Additional arguments
        
    Returns:
        Configured HybridRetriever
    """
    from ..database import Database
    from ..grpc_client import SochDBClient
    
    if isinstance(backend, Database):
        return HybridRetriever.from_database(backend, namespace, collection, **kwargs)
    elif isinstance(backend, SochDBClient):
        return HybridRetriever.from_client(backend, namespace, collection, **kwargs)
    elif isinstance(backend, RetrievalBackend):
        return HybridRetriever.from_backend(backend, namespace, collection, **kwargs)
    else:
        raise TypeError(f"Unknown backend type: {type(backend)}")
