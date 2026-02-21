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
SochDB Namespace Handle (Task 8: First-Class Namespace Handle + Context Manager API)

Provides type-safe namespace isolation with context manager support.

Example:
    # Create and use namespace
    with db.use_namespace("tenant_123") as ns:
        collection = ns.create_collection("documents", dimension=384)
        collection.insert([1.0, 2.0, ...], metadata={"source": "web"})
        results = collection.search(query_vector, k=10)
    
    # Or use the handle directly
    ns = db.namespace("tenant_123")
    collection = ns.collection("documents")
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from enum import Enum

from .errors import (
    NamespaceNotFoundError,
    NamespaceExistsError,
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    ValidationError,
    DimensionMismatchError,
)

# Import VectorIndex for fast HNSW search
from .vector import VectorIndex

if TYPE_CHECKING:
    from .database import Database


# ============================================================================
# Namespace Configuration
# ============================================================================

@dataclass
class NamespaceConfig:
    """Configuration for a namespace."""
    
    name: str
    display_name: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    read_only: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "labels": self.labels,
            "read_only": self.read_only,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NamespaceConfig":
        return cls(
            name=data["name"],
            display_name=data.get("display_name"),
            labels=data.get("labels", {}),
            read_only=data.get("read_only", False),
        )


# ============================================================================
# Collection Configuration (Task 9: Unified Collection Builder)
# ============================================================================

class DistanceMetric(str, Enum):
    """Distance metric for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class QuantizationType(str, Enum):
    """Quantization type for index compression."""
    NONE = "none"
    SCALAR = "scalar"  # int8 quantization
    PQ = "pq"          # Product quantization


@dataclass
class CollectionConfig:
    """
    Collection configuration.
    
    Example:
        # Full specification
        config = CollectionConfig(
            name="documents",
            dimension=384,
            metric=DistanceMetric.COSINE,
        )
        collection = ns.create_collection(config)
        
        # Auto-dimension (inferred from first vector)
        config = CollectionConfig(name="docs")  # dimension=None
        collection = ns.create_collection(config)
        collection.add(embeddings=[[1.0, 2.0, 3.0]])  # dimension auto-set to 3
    """
    
    name: str
    dimension: Optional[int] = None  # None = auto-infer from first vector
    metric: DistanceMetric = DistanceMetric.COSINE
    
    # Index parameters
    m: int = 16                      # HNSW M parameter
    ef_construction: int = 100       # HNSW ef_construction
    quantization: QuantizationType = QuantizationType.NONE
    
    # Optional features
    enable_hybrid_search: bool = False  # Enable BM25 + vector search
    content_field: Optional[str] = None # Field to index for BM25
    
    def __post_init__(self):
        # Coerce string metric to DistanceMetric enum
        if isinstance(self.metric, str):
            object.__setattr__(self, 'metric', DistanceMetric(self.metric))
        # Dimension can be None (auto-infer) or positive
        if self.dimension is not None and self.dimension <= 0:
            raise ValidationError(f"Dimension must be positive, got {self.dimension}")
        if self.m <= 0:
            raise ValidationError(f"M parameter must be positive, got {self.m}")
        if self.ef_construction <= 0:
            raise ValidationError(f"ef_construction must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "metric": self.metric.value,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "quantization": self.quantization.value,
            "enable_hybrid_search": self.enable_hybrid_search,
            "content_field": self.content_field,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: Optional[str] = None) -> "CollectionConfig":
        # Handle Rust FFI compact format (no "name", metric as int)
        coll_name = data.get("name", name or "unnamed")
        
        # Metric may be int (from Rust FFI) or string
        raw_metric = data.get("metric", "cosine")
        if isinstance(raw_metric, int):
            metric_map = {0: "cosine", 1: "euclidean", 2: "dot_product"}
            raw_metric = metric_map.get(raw_metric, "cosine")
        
        return cls(
            name=coll_name,
            dimension=data.get("dimension"),  # Can be None
            metric=DistanceMetric(raw_metric),
            m=data.get("m", 16),
            ef_construction=data.get("ef_construction", 100),
            quantization=QuantizationType(data.get("quantization", "none")),
            enable_hybrid_search=data.get("enable_hybrid_search", False),
            content_field=data.get("content_field"),
        )


# ============================================================================
# Search Request (Task 10: One Search Surface)
# ============================================================================

@dataclass
class SearchRequest:
    """
    Unified search request supporting vector, keyword, and hybrid search.
    
    This is the single entry point for all search operations. Use convenience
    methods for simpler cases.
    
    Example:
        # Full hybrid search
        request = SearchRequest(
            vector=query_embedding,
            text_query="machine learning",
            filter={"category": "tech"},
            k=10,
            alpha=0.7,  # Vector weight for hybrid
        )
        results = collection.search(request)
        
        # Or use convenience methods
        results = collection.vector_search(query_embedding, k=10)
        results = collection.keyword_search("machine learning", k=10)
        results = collection.hybrid_search(query_embedding, "ML", k=10)
    """
    
    # Query inputs (at least one required)
    vector: Optional[List[float]] = None
    text_query: Optional[str] = None
    
    # Result control
    k: int = 10
    min_score: Optional[float] = None
    
    # Filtering
    filter: Optional[Dict[str, Any]] = None
    
    # Hybrid search weights
    alpha: float = 0.5  # 0.0 = pure keyword, 1.0 = pure vector
    rrf_k: float = 60.0  # RRF k parameter
    
    # Multi-vector aggregation
    aggregate: str = "max"  # max | mean | first
    
    # Time-travel (if versioning enabled)
    as_of: Optional[str] = None  # ISO timestamp
    
    # Return options
    include_vectors: bool = False
    include_metadata: bool = True
    include_scores: bool = True
    
    def validate(self, expected_dimension: Optional[int] = None) -> None:
        """Validate the search request."""
        if self.vector is None and self.text_query is None:
            raise ValidationError("At least one of 'vector' or 'text_query' is required")
        
        if self.k <= 0:
            raise ValidationError(f"k must be positive, got {self.k}")
        
        if self.vector is not None and expected_dimension is not None:
            if len(self.vector) != expected_dimension:
                raise DimensionMismatchError(expected_dimension, len(self.vector))
        
        if not 0.0 <= self.alpha <= 1.0:
            raise ValidationError(f"alpha must be between 0 and 1, got {self.alpha}")


@dataclass
class SearchResult:
    """A single search result."""
    
    id: Union[str, int]
    score: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    
    # For multi-vector documents
    matched_chunk: Optional[int] = None


@dataclass  
class SearchResults:
    """Search results with metadata."""
    
    results: List[SearchResult]
    total_count: int
    query_time_ms: float
    
    # Search details
    vector_results: Optional[int] = None
    keyword_results: Optional[int] = None
    
    def __iter__(self) -> Iterator[SearchResult]:
        return iter(self.results)
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __getitem__(self, idx: int) -> SearchResult:
        return self.results[idx]


# ============================================================================
# Collection Handle
# ============================================================================

class Collection:
    """
    A vector collection within a namespace.
    
    Collections store vectors with optional metadata and support:
    - Vector similarity search (ANN) - Powered by HNSW via VectorIndex
    - Keyword search (BM25)
    - Hybrid search (RRF fusion)
    - Metadata filtering
    - Multi-vector documents
    
    All operations are automatically scoped to the parent namespace.
    
    Performance: Uses the native Rust HNSW index (VectorIndex) for
    fast approximate nearest neighbor search with >90% recall.
    """
    
    def __init__(
        self,
        namespace: "Namespace",
        config: CollectionConfig,
    ):
        self._namespace = namespace
        self._config = config
        self._db = namespace._db
        
        # Fast path: Use VectorIndex for HNSW search
        self._vector_index: Optional["VectorIndex"] = None
        self._id_to_internal: Dict[Union[str, int], int] = {}  # doc_id -> internal uint64
        self._internal_to_id: Dict[int, Union[str, int]] = {}  # internal uint64 -> doc_id
        self._metadata_store: Dict[Union[str, int], Dict[str, Any]] = {}  # doc_id -> metadata
        self._next_internal_id: int = 0
        self._raw_vectors: Dict[Union[str, int], Any] = {}  # doc_id -> vector (for snapshot)
        self._ef_search_override: Optional[int] = None  # deferred ef_search setting
    
    # ========================================================================
    # Storage Key Helpers
    # ========================================================================
    
    def _vector_key(self, doc_id: Union[str, int]) -> bytes:
        """Key for storing vector + metadata."""
        return f"{self.namespace_name}/collections/{self.name}/vectors/{doc_id}".encode()
    
    def _vectors_prefix(self) -> bytes:
        """Prefix for all vectors in this collection."""
        return f"{self.namespace_name}/collections/{self.name}/vectors/".encode()
    
    @property
    def name(self) -> str:
        """Collection name."""
        return self._config.name
    
    @property
    def config(self) -> CollectionConfig:
        """Immutable collection configuration."""
        return self._config
    
    @property
    def namespace_name(self) -> str:
        """Parent namespace name."""
        return self._namespace.name
    
    def info(self) -> Dict[str, Any]:
        """Get collection info including frozen config."""
        return {
            "name": self.name,
            "namespace": self.namespace_name,
            "config": self._config.to_dict(),
        }
    
    # ========================================================================
    # Insert Operations
    # ========================================================================
    
    def _ensure_index(self, dimension: int) -> None:
        """Lazily create the VectorIndex when dimension is known."""
        if self._vector_index is None:
            self._vector_index = VectorIndex(
                dimension=dimension,
                max_connections=self._config.m,
                ef_construction=self._config.ef_construction,
            )
            # Apply deferred override if set, otherwise use default
            if self._ef_search_override is not None:
                self._vector_index.ef_search = self._ef_search_override
            else:
                # Default ef_search high enough for good recall@100
                self._vector_index.ef_search = 500
    
    def set_ef_search(self, ef_search: int) -> None:
        """Set the ef_search parameter for HNSW search.
        
        Higher values give better recall at the cost of latency.
        For recall@k, ef_search should be >= 5*k for good results.
        
        Args:
            ef_search: The size of the dynamic candidate list during search.
        """
        if self._vector_index is not None:
            self._vector_index.ef_search = ef_search
        # Store for deferred application (if index not yet created)
        self._ef_search_override = ef_search
    
    def vector_search_exact(
        self,
        vector: List[float],
        k: int = 10,
    ) -> "SearchResults":
        """
        Exact brute-force vector search for perfect recall.
        
        Computes distances to ALL vectors and returns the true k-nearest
        neighbors. Guarantees recall@k = 1.0 but is O(n) per query.
        
        Args:
            vector: Query vector
            k: Number of results
            
        Returns:
            SearchResults with perfect recall
        """
        import time
        start_time = time.time()
        
        if self._vector_index is None:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)
        
        raw_results = self._vector_index.search_exact(vector, k)
        results = []
        for internal_id, distance in raw_results:
            doc_id = self._internal_to_id.get(internal_id)
            if doc_id is None:
                continue
            similarity = max(0.0, 1.0 - distance)
            result = SearchResult(
                id=str(doc_id), score=similarity,
                metadata=None,
                vector=None,
            )
            results.append(result)
        
        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)
    
    def vector_search_exact_f64(
        self,
        vector: List[float],
        k: int = 10,
    ) -> "SearchResults":
        """
        Exact brute-force vector search using f64 precision.
        
        Same as vector_search_exact but computes distances in f64 to match
        ground truth computed with numpy f64 arithmetic. Eliminates f32
        tie-breaking mismatches at the k-th boundary.
        
        Args:
            vector: Query vector
            k: Number of results
            
        Returns:
            SearchResults with perfect precision against f64 ground truth
        """
        import time
        start_time = time.time()
        
        if self._vector_index is None:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)
        
        raw_results = self._vector_index.search_exact_f64(vector, k)
        results = []
        for internal_id, distance in raw_results:
            doc_id = self._internal_to_id.get(internal_id)
            if doc_id is None:
                continue
            similarity = max(0.0, 1.0 - distance)
            result = SearchResult(
                id=str(doc_id), score=similarity,
                metadata=None,
                vector=None,
            )
            results.append(result)
        
        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)
    
    def _get_internal_id(self, doc_id: Union[str, int]) -> int:
        """Get or create internal uint64 ID for a document."""
        if doc_id in self._id_to_internal:
            return self._id_to_internal[doc_id]
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._id_to_internal[doc_id] = internal_id
        self._internal_to_id[internal_id] = doc_id
        return internal_id
    
    def insert(
        self,
        id: Union[str, int],
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """
        Insert a single vector.

        Writes to BOTH the in-memory HNSW index (for fast vector_search /
        vector_search_exact) AND the KV store (for keyword_search / hybrid_search
        BM25 fallback).  Previously only the HNSW path was populated, making
        documents inserted with insert() invisible to keyword/hybrid search.

        Args:
            id: Unique document ID
            vector: Vector embedding
            metadata: Optional metadata dict
            content: Optional text content (for hybrid search)
        """
        # Auto-dimension from first vector
        if self._config.dimension is None:
            object.__setattr__(self._config, 'dimension', len(vector))

        # Validate dimension
        if len(vector) != self._config.dimension:
            raise DimensionMismatchError(self._config.dimension, len(vector))

        # Build metadata dict
        meta = metadata.copy() if metadata else {}
        if content:
            meta["_content"] = content

        # 1. Insert into in-memory HNSW (fast, used for same-session vector search)
        self._ensure_index(self._config.dimension)
        internal_id = self._get_internal_id(id)
        self._vector_index.insert(internal_id, vector)
        self._metadata_store[id] = meta
        self._raw_vectors[id] = vector  # keep for snapshot

        # 2. Persist to KV store so keyword_search / hybrid_search find this doc.
        #    Uses the same JSON schema as insert_multi() so FFI BM25 and the
        #    Python scan fallback can both read it.
        with self._db.transaction() as txn:
            doc_data = {
                "id": str(id),
                "vector": vector,
                "metadata": meta,
                "content": content or meta.get("_content", ""),
                "is_multi_vector": False,
            }
            txn.put(self._vector_key(id), json.dumps(doc_data).encode())
    
    def insert_batch(
        self,
        documents: List[Tuple[Union[str, int], List[float], Optional[Dict[str, Any]], Optional[str]]] = None,
        *,
        ids: List[Union[str, int]] = None,
        vectors: List[List[float]] = None,
        metadatas: List[Optional[Dict[str, Any]]] = None,
    ) -> int:
        """
        Insert multiple vectors in a batch.

        Writes to BOTH in-memory HNSW (for fast vector search) AND KV store
        (for keyword/hybrid search BM25).  Previously only the HNSW path was
        populated, leaving batch-inserted docs invisible to keyword search.

        Supports two calling conventions:
        1. Tuple format: insert_batch([(id, vector, metadata, content), ...])
        2. Keyword format: insert_batch(ids=[...], vectors=[...], metadatas=[...])

        Args:
            documents: List of (id, vector, metadata, content) tuples
            ids: List of document IDs (keyword format)
            vectors: List of vector embeddings (keyword format)
            metadatas: List of metadata dicts (keyword format)

        Returns:
            Number of documents inserted
        """
        import numpy as np

        # Handle keyword argument format
        if ids is not None and vectors is not None:
            if metadatas is None:
                metadatas = [None] * len(ids)
            documents = [(id, vec, meta, None) for id, vec, meta in zip(ids, vectors, metadatas)]

        if not documents:
            return 0

        # Auto-dimension inference from first vector
        first_vec = documents[0][1]
        if self._config.dimension is None:
            object.__setattr__(self._config, 'dimension', len(first_vec))

        # Validate dimensions
        for doc_id, vector, metadata, content in documents:
            if len(vector) != self._config.dimension:
                raise DimensionMismatchError(self._config.dimension, len(vector))

        # Build per-document metadata
        batch_metadatas = []
        for doc_id, vector, metadata, content in documents:
            meta = metadata.copy() if metadata else {}
            if content:
                meta["_content"] = content
            batch_metadatas.append(meta)

        # 1. Fast in-memory HNSW insert (used for same-session vector search)
        self._ensure_index(self._config.dimension)
        internal_ids = np.array([self._get_internal_id(doc[0]) for doc in documents], dtype=np.uint64)
        vectors_array = np.array([doc[1] for doc in documents], dtype=np.float32)
        count = self._vector_index.insert_batch(internal_ids, vectors_array)
        for i, (doc_id, vector, metadata, content) in enumerate(documents):
            self._metadata_store[doc_id] = batch_metadatas[i]
            self._raw_vectors[doc_id] = vector  # keep for snapshot

        # 2. Persist to KV store so keyword_search / hybrid_search find all docs.
        #    Written in one transaction for atomicity.
        with self._db.transaction() as txn:
            for i, (doc_id, vector, metadata, content) in enumerate(documents):
                meta = batch_metadatas[i]
                doc_data = {
                    "id": str(doc_id),
                    "vector": vector,
                    "metadata": meta,
                    "content": content or meta.get("_content", ""),
                    "is_multi_vector": False,
                }
                txn.put(self._vector_key(doc_id), json.dumps(doc_data).encode())

        return count
    
    def add(
        self,
        ids: List[Union[str, int]],
        embeddings: List[List[float]] = None,
        vectors: List[List[float]] = None,
        metadatas: List[Optional[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add vectors to the collection.
        
        Args:
            ids: List of document IDs
            embeddings: List of vector embeddings (standard name)
            vectors: List of vector embeddings (alternative name)
            metadatas: List of metadata dicts
            
        Returns:
            Number of documents added
        """
        # Accept both 'embeddings' and 'vectors' for flexibility
        vecs = embeddings if embeddings is not None else vectors
        if vecs is None:
            raise ValidationError("Either 'embeddings' or 'vectors' must be provided")
        
        return self.insert_batch(ids=ids, vectors=vecs, metadatas=metadatas)
    
    def upsert(
        self,
        ids: List[Union[str, int]],
        embeddings: List[List[float]] = None,
        vectors: List[List[float]] = None,
        metadatas: List[Optional[Dict[str, Any]]] = None,
    ) -> int:
        """
        Insert or update vectors.
        
        Same as add() - overwrites existing IDs.
        """
        return self.add(ids=ids, embeddings=embeddings, vectors=vectors, metadatas=metadatas)
    
    def query(
        self,
        query_embeddings: List[List[float]] = None,
        query_vectors: List[List[float]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the collection.
        
        Args:
            query_embeddings: List of query vectors
            query_vectors: List of query vectors (alternative name)
            n_results: Number of results per query
            where: Metadata filter
            
        Returns:
            Dict with 'ids', 'distances', 'metadatas' keys
        """
        vecs = query_embeddings if query_embeddings is not None else query_vectors
        if vecs is None:
            raise ValidationError("Either 'query_embeddings' or 'query_vectors' must be provided")
        
        all_ids = []
        all_distances = []
        all_metadatas = []
        
        for query_vec in vecs:
            results = self.vector_search(vector=query_vec, k=n_results, filter=where)
            
            ids = [r.id for r in results]
            distances = [r.score for r in results]
            metadatas = [r.metadata for r in results]
            
            all_ids.append(ids)
            all_distances.append(distances)
            all_metadatas.append(metadatas)
        
        return {
            "ids": all_ids,
            "distances": all_distances,
            "metadatas": all_metadatas,
        }
    
    def insert_multi(
        self,
        id: Union[str, int],
        vectors: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None,
        chunk_texts: Optional[List[str]] = None,
        aggregate: str = "max",
    ) -> None:
        """
        Insert a multi-vector document.
        
        Multi-vector documents allow storing multiple embeddings per document
        (e.g., for document chunks). During search, scores are aggregated
        using the specified method.
        
        Args:
            id: Unique document ID
            vectors: List of vector embeddings (one per chunk)
            metadata: Optional document-level metadata
            chunk_texts: Optional text content for each chunk
            aggregate: Aggregation method: "max", "mean", or "first"
        """
        # Validate
        for i, v in enumerate(vectors):
            if self._config.dimension is not None and len(v) != self._config.dimension:
                raise DimensionMismatchError(self._config.dimension, len(v))
        
        if chunk_texts and len(chunk_texts) != len(vectors):
            raise ValidationError(
                f"chunk_texts length ({len(chunk_texts)}) must match vectors length ({len(vectors)})"
            )
        
        # Auto-infer dimension from first vector if not set
        if self._config.dimension is None and vectors:
            object.__setattr__(self._config, 'dimension', len(vectors[0]))
        
        # Store multi-vector document with all chunks
        with self._db.transaction() as txn:
            doc_data = {
                "id": id,
                "vectors": vectors,  # All vectors stored together
                "metadata": metadata or {},
                "chunk_texts": chunk_texts,
                "aggregate": aggregate,
                "is_multi_vector": True,
            }
            key = self._vector_key(id)
            txn.put(key, json.dumps(doc_data).encode())
    
    # ========================================================================
    # Search Operations (Task 10: One Search Surface)
    # ========================================================================
    
    def search(self, request: SearchRequest) -> SearchResults:
        """
        Unified search API.
        
        This is the primary search method supporting vector, keyword,
        and hybrid search modes. Use convenience methods for simpler cases.
        
        Args:
            request: SearchRequest with query parameters
            
        Returns:
            SearchResults with matching documents
        """
        request.validate(self._config.dimension)
        
        # Determine search mode
        has_vector = request.vector is not None
        has_text = request.text_query is not None
        
        if has_vector and has_text:
            # Hybrid search
            return self._hybrid_search(request)
        elif has_vector:
            # Pure vector search
            return self._vector_search(request)
        else:
            # Pure keyword search
            return self._keyword_search(request)
    
    def vector_search(
        self,
        vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> SearchResults:
        """
        Convenience method for vector similarity search.
        
        Args:
            vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            min_score: Minimum similarity score
            
        Returns:
            SearchResults
        """
        request = SearchRequest(
            vector=vector,
            k=k,
            filter=filter,
            min_score=min_score,
        )
        return self.search(request)
    
    def keyword_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """
        Convenience method for keyword (BM25) search.
        
        Requires hybrid search to be enabled on the collection.
        
        Args:
            query: Text query
            k: Number of results
            filter: Optional metadata filter
            
        Returns:
            SearchResults
        """
        if not self._config.enable_hybrid_search:
            raise CollectionConfigError(
                "Keyword search requires enable_hybrid_search=True in collection config",
                remediation="Recreate collection with CollectionConfig(enable_hybrid_search=True)"
            )
        
        request = SearchRequest(
            text_query=query,
            k=k,
            filter=filter,
            alpha=0.0,  # Pure keyword
        )
        return self.search(request)
    
    def hybrid_search(
        self,
        vector: List[float],
        text_query: str,
        k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """
        Convenience method for hybrid (vector + keyword) search.
        
        Uses Reciprocal Rank Fusion (RRF) to combine results.
        
        Args:
            vector: Query vector
            text_query: Text query
            k: Number of results
            alpha: Balance between vector (1.0) and keyword (0.0)
            filter: Optional metadata filter
            
        Returns:
            SearchResults
        """
        request = SearchRequest(
            vector=vector,
            text_query=text_query,
            k=k,
            alpha=alpha,
            filter=filter,
        )
        return self.search(request)
    
    def _vector_search(self, request: SearchRequest) -> SearchResults:
        """Internal vector search implementation.
        
        Uses in-memory HNSW if populated (same session), otherwise
        reloads from KV storage via native Rust FFI.
        """
        import time
        
        start_time = time.time()
        
        if request.vector is None:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)
        
        search_k = request.k * 3 if request.filter else request.k
        
        # Primary path: in-memory HNSW (same session, fast)
        if self._vector_index is not None and len(self._vector_index) > 0:
            return self._search_in_memory(request, search_k, start_time)
        
        # Reload path: load vectors from snapshot into HNSW (fast batch insert)
        if self._config.dimension is not None:
            loaded = self._reload_vectors_from_snapshot()
            if loaded > 0:
                return self._search_in_memory(request, search_k, start_time)
        
        # Fallback: try native Rust FFI search
        ffi_results = self._db.ffi_collection_search(
            namespace=self.namespace_name,
            collection=self.name,
            query_vector=request.vector,
            k=search_k,
        )
        
        if ffi_results is not None and len(ffi_results) > 0:
            results = []
            for r in ffi_results:
                doc_id = r.get("id", "")
                score = r.get("score", 0.0)
                metadata = r.get("metadata", {})
                if request.min_score is not None and score < request.min_score:
                    continue
                if request.filter and not self._matches_filter(metadata, request.filter):
                    continue
                result = SearchResult(
                    id=doc_id, score=score,
                    metadata=metadata if request.include_metadata else None,
                    vector=None,
                )
                results.append(result)
                if len(results) >= request.k:
                    break
            
            elapsed = (time.time() - start_time) * 1000
            return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)
        
        return SearchResults(results=[], total_count=0, query_time_ms=0.0)
    
    def _search_in_memory(self, request: SearchRequest, search_k: int, start_time: float) -> SearchResults:
        """Search using the in-memory HNSW index."""
        import time
        
        raw_results = self._vector_index.search(request.vector, search_k)
        
        results = []
        for internal_id, distance in raw_results:
            doc_id = self._internal_to_id.get(internal_id)
            if doc_id is None:
                continue
            # Convert distance to similarity (higher = better)
            similarity = max(0.0, 1.0 - distance)
            if request.min_score is not None and similarity < request.min_score:
                continue
            metadata = self._metadata_store.get(doc_id, {})
            if request.filter and not self._matches_filter(metadata, request.filter):
                continue
            result = SearchResult(
                id=str(doc_id), score=similarity,
                metadata=metadata if request.include_metadata else None,
                vector=None,
            )
            results.append(result)
            if len(results) >= request.k:
                break
        
        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)
    
    # ── Persistence helpers ──────────────────────────────────────────────

    def _snapshot_dir(self) -> str:
        """Return the directory for numpy vector snapshots."""
        import os
        base = getattr(self._db, '_path', '/tmp/sochdb')
        return os.path.join(base, '_snapshots', self.namespace_name, self.name)

    def _persist_vectors_snapshot(self) -> None:
        """Save all in-memory vectors/ids/metadata to numpy files on disk.
        
        This is called after upload completes (post_upload / checkpoint).
        Much faster to reload than per-vector KV: a single np.load() + insert_batch().
        """
        import os, json, numpy as np

        if self._vector_index is None or len(self._vector_index) == 0:
            return

        snap_dir = self._snapshot_dir()
        os.makedirs(snap_dir, exist_ok=True)

        n = len(self._id_to_internal)
        dim = self._config.dimension

        # Build sorted arrays (sorted by internal_id for determinism)
        doc_ids = []
        internal_ids = np.empty(n, dtype=np.uint64)
        vectors = np.empty((n, dim), dtype=np.float32)
        meta_list = []

        # We need vectors, but they're in the HNSW index (no getter).
        # So we also keep a raw vector store during insert.
        # If _raw_vectors is available, use it; otherwise skip snapshot.
        if not hasattr(self, '_raw_vectors') or len(self._raw_vectors) == 0:
            return

        idx = 0
        for doc_id, iid in self._id_to_internal.items():
            doc_ids.append(str(doc_id))
            internal_ids[idx] = iid
            if doc_id in self._raw_vectors:
                vectors[idx] = self._raw_vectors[doc_id]
            meta_list.append(self._metadata_store.get(doc_id, {}))
            idx += 1

        np.save(os.path.join(snap_dir, 'vectors.npy'), vectors[:idx])
        np.save(os.path.join(snap_dir, 'internal_ids.npy'), internal_ids[:idx])

        # Save doc_ids and metadata as JSON (small relative to vectors)
        with open(os.path.join(snap_dir, 'doc_ids.json'), 'w') as f:
            json.dump(doc_ids[:idx], f)
        with open(os.path.join(snap_dir, 'metadata.json'), 'w') as f:
            json.dump(meta_list[:idx], f)

    def _reload_vectors_from_snapshot(self) -> int:
        """Reload vectors from numpy snapshot files using fast batch insert.
        
        Returns number of vectors loaded. Uses insert_batch (parallel Rust FFI)
        for ~10x faster rebuild compared to one-by-one inserts.
        """
        import os, json, numpy as np

        snap_dir = self._snapshot_dir()
        vectors_path = os.path.join(snap_dir, 'vectors.npy')
        if not os.path.exists(vectors_path):
            return 0

        dim = self._config.dimension
        if dim is None:
            return 0

        try:
            vectors = np.load(vectors_path)
            internal_ids = np.load(os.path.join(snap_dir, 'internal_ids.npy'))
            with open(os.path.join(snap_dir, 'doc_ids.json'), 'r') as f:
                doc_ids = json.load(f)
            with open(os.path.join(snap_dir, 'metadata.json'), 'r') as f:
                meta_list = json.load(f)
        except Exception:
            return 0

        n = len(doc_ids)
        if n == 0:
            return 0

        # Rebuild mappings
        for i, doc_id in enumerate(doc_ids):
            iid = int(internal_ids[i])
            self._id_to_internal[doc_id] = iid
            self._internal_to_id[iid] = doc_id
            self._metadata_store[doc_id] = meta_list[i] if i < len(meta_list) else {}
        self._next_internal_id = int(internal_ids.max()) + 1

        # Rebuild HNSW using parallel batch insert (fast!)
        self._ensure_index(dim)
        self._vector_index.insert_batch(internal_ids, vectors)

        return n

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def _keyword_search(self, request: SearchRequest) -> SearchResults:
        """Internal keyword search implementation."""
        import time
        
        start_time = time.time()
        
        if not request.text_query:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)

        # Basic stopword removal to improve precision
        STOPWORDS = {
            "how", "do", "i", "fix", "in", "tell", "me", "about", 
            "best", "practices", "for", "the", "a", "an", "is", "of"
        }
        
        cleaned_query = " ".join([
            word for word in request.text_query.lower().split()
            if word not in STOPWORDS
        ])
        
        if not cleaned_query:
            cleaned_query = request.text_query # Fallback if empty

        # Try FFI first (Native Rust)
        ffi_results = self._db.ffi_collection_keyword_search(
            self._namespace._name,
            self._config.name,
            cleaned_query,
            request.k
        )
        
        if ffi_results is not None:
             # FFI succeeded
            results = []
            for r in ffi_results:
                # Apply metadata filter
                if request.filter and not self._matches_filter(r.get("metadata", {}), request.filter):
                    continue

                result = SearchResult(
                    id=r["id"],
                    score=r["score"],
                    metadata=r.get("metadata") if request.include_metadata else None,
                    vector=None,
                )
                results.append(result)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return SearchResults(
                results=results,
                total_count=len(results),
                query_time_ms=query_time_ms,
                vector_results=0,
            )

        # Fallback: Python BM25 scan over KV store
        # Uses proper BM25 formula (k1=1.2, b=0.75) instead of raw TF counting.
        # Previously this used simple count(term in doc) with no IDF weighting,
        # which caused popular terms to dominate and domain-specific terms to rank low.
        all_docs = []
        prefix = self._vectors_prefix()
        with self._db.transaction() as txn:
            for key, value in txn.scan_prefix(prefix):
                doc = json.loads(value.decode())
                all_docs.append(doc)

        if not all_docs:
            query_time_ms = (time.time() - start_time) * 1000
            return SearchResults(results=[], total_count=0, query_time_ms=query_time_ms)

        # Tokenise query (stopword-filtered list already computed above)
        query_terms = [w for w in cleaned_query.split() if w]
        if not query_terms:
            query_terms = request.text_query.lower().split()

        # --- BM25 parameters (Lucene / Elasticsearch defaults) ---
        K1   = 1.2  # term-frequency saturation
        B    = 0.75 # length normalisation

        # Build corpus for IDF: token → document-frequency count
        import math
        corpus_texts = []
        for doc in all_docs:
            content  = doc.get("content", "") or ""
            metadata = doc.get("metadata", {})
            text_fields = [content]
            for v in metadata.values():
                if isinstance(v, str):
                    text_fields.append(v)
            corpus_texts.append(" ".join(text_fields).lower())

        N      = len(all_docs)
        avgdl  = sum(len(t.split()) for t in corpus_texts) / N if N else 1.0

        # Document-frequency per query term
        df: dict = {}
        for term in query_terms:
            for text in corpus_texts:
                if term in text.split():
                    df[term] = df.get(term, 0) + 1

        # IDF (Robertson-Spärck Jones formula, +1 to keep non-negative)
        idf: dict = {
            term: math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            for term in query_terms
        }

        scored_docs = []
        for doc, text in zip(all_docs, corpus_texts):
            tokens   = text.split()
            dl       = len(tokens)
            metadata = doc.get("metadata", {})

            # Build TF map
            tf_map: dict = {}
            for tok in tokens:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            # BM25 score
            score = 0.0
            for term in query_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                numerator   = tf * (K1 + 1)
                denominator = tf + K1 * (1 - B + B * dl / avgdl)
                score += idf[term] * numerator / denominator

            if score > 0:
                if request.filter and not self._matches_filter(metadata, request.filter):
                    continue
                scored_docs.append((score, doc))

        # Sort by BM25 score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Take top k
        top_k = scored_docs[:request.k]

        # Build results
        results = []
        for score, doc in top_k:
            result = SearchResult(
                id=doc["id"],
                score=score,
                metadata=doc.get("metadata") if request.include_metadata else None,
                vector=doc.get("vector") if request.include_vectors else None,
            )
            results.append(result)

        query_time_ms = (time.time() - start_time) * 1000

        return SearchResults(
            results=results,
            total_count=len(scored_docs),
            query_time_ms=query_time_ms,
            keyword_results=len(results),
        )
    
    def _hybrid_search(self, request: SearchRequest) -> SearchResults:
        """Internal hybrid search using RRF (Reciprocal Rank Fusion)."""
        import time
        
        start_time = time.time()
        
        # Get vector and keyword results separately
        vector_results = self._vector_search(request)
        keyword_results = self._keyword_search(request)
        
        # RRF fusion
        rrf_k = request.rrf_k
        alpha = request.alpha  # Weight for vector results
        
        # Build score maps
        scores = {}  # id -> (rrf_score, doc_data)
        
        # Add vector results
        for rank, result in enumerate(vector_results.results):
            rrf_score = alpha / (rrf_k + rank + 1)
            scores[result.id] = (rrf_score, result)
        
        # Add keyword results
        for rank, result in enumerate(keyword_results.results):
            rrf_score = (1 - alpha) / (rrf_k + rank + 1)
            if result.id in scores:
                existing_score, existing_result = scores[result.id]
                scores[result.id] = (existing_score + rrf_score, existing_result)
            else:
                scores[result.id] = (rrf_score, result)
        
        # Sort by combined RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        # Take top k and build results
        results = []
        for doc_id, (score, result) in sorted_results[:request.k]:
            results.append(SearchResult(
                id=doc_id,
                score=score,
                metadata=result.metadata,
                vector=result.vector,
            ))
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            results=results,
            total_count=len(sorted_results),
            query_time_ms=query_time_ms,
            vector_results=len(vector_results.results),
            keyword_results=len(keyword_results.results),
        )
    
    # ========================================================================
    # Other Operations
    # ========================================================================
    
    def get(self, id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        key = self._vector_key(id)
        data = self._db.get(key)
        if data is None:
            return None
        return json.loads(data.decode())
    
    def delete(self, id: Union[str, int]) -> bool:
        """
        Delete a document by ID.
        
        Uses tombstone-based logical deletion. The vector remains in the
        index but won't be returned in search results.
        """
        key = self._vector_key(id)
        with self._db.transaction() as txn:
            if txn.get(key) is None:
                return False
            txn.delete(key)
        return True
    
    def count(self) -> int:
        """Get the number of documents (excluding deleted)."""
        prefix = self._vectors_prefix()
        count = 0
        with self._db.transaction() as txn:
            for _ in txn.scan_prefix(prefix):
                count += 1
        return count
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self) -> "Collection":
        """Enter context manager - enables 'with collection:' syntax."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - auto cleanup."""
        # No explicit cleanup needed currently, but ready for future use
        pass
    
    def close(self) -> None:
        """Explicitly close the collection (no-op, for API compatibility)."""
        pass
    
    # ========================================================================
    # Batch API
    # ========================================================================
    
    def add(
        self,
        embeddings: List[List[float]],
        ids: Optional[List[Union[str, int]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> int:
        """
        Add vectors in batch.
        
        This is the recommended way to add vectors - much faster than
        individual inserts.
        
        Args:
            embeddings: List of vector embeddings
            ids: Optional list of IDs (auto-generated UUIDs if not provided)
            metadatas: Optional list of metadata dicts
            documents: Optional list of text documents
            
        Returns:
            Number of vectors added
            
        Example:
            # Full specification
            collection.add(
                embeddings=[[1.0, 2.0], [3.0, 4.0]],
                ids=["doc1", "doc2"],
                metadatas=[{"type": "a"}, {"type": "b"}],
                documents=["hello world", "goodbye"]
            )
            
            # Minimal - just vectors (IDs auto-generated)
            collection.add(embeddings=[[1.0, 2.0], [3.0, 4.0]])
        """
        import uuid
        
        if not embeddings:
            return 0
        
        n = len(embeddings)
        
        # Auto-dimension inference
        if self._config.dimension is None:
            first_dim = len(embeddings[0])
            # Update config with inferred dimension (mutable now)
            object.__setattr__(self._config, 'dimension', first_dim)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        elif len(ids) != n:
            raise ValidationError(f"ids length ({len(ids)}) must match embeddings length ({n})")
        
        # Pad metadatas/documents if not provided
        if metadatas is None:
            metadatas = [None] * n
        elif len(metadatas) != n:
            raise ValidationError(f"metadatas length ({len(metadatas)}) must match embeddings length ({n})")
        
        if documents is None:
            documents = [None] * n
        elif len(documents) != n:
            raise ValidationError(f"documents length ({len(documents)}) must match embeddings length ({n})")
        
        # Validate dimensions
        for i, vec in enumerate(embeddings):
            if len(vec) != self._config.dimension:
                raise DimensionMismatchError(self._config.dimension, len(vec))
        
        # Batch insert
        batch = list(zip(ids, embeddings, metadatas, documents))
        return self.insert_batch(batch)
    
    def upsert(
        self,
        embeddings: List[List[float]],
        ids: List[Union[str, int]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> int:
        """
        Upsert vectors (insert or update).
        
        Args:
            embeddings: List of vector embeddings
            ids: List of IDs (required for upsert)
            metadatas: Optional list of metadata dicts
            documents: Optional list of text documents
            
        Returns:
            Number of vectors upserted
        """
        # For now, upsert is same as add (KV store overwrites)
        return self.add(embeddings=embeddings, ids=ids, metadatas=metadatas, documents=documents)
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[List[Any]]]:
        """
        Query vectors.
        
        Args:
            query_embeddings: List of query vectors (can query multiple at once)
            n_results: Number of results per query
            where: Optional metadata filter
            include: What to include in results ["embeddings", "metadatas", "documents"]
            
        Returns:
            Dictionary with ids, distances, metadatas, embeddings, documents
            Each is a list of lists (one list per query)
            
        Example:
            results = collection.query(
                query_embeddings=[[1.0, 2.0]],
                n_results=5,
                where={"category": "tech"}
            )
            print(results["ids"][0])  # ["doc1", "doc2", ...]
            print(results["distances"][0])  # [0.1, 0.2, ...]
        """
        include = include or ["metadatas", "documents"]
        include_metadata = "metadatas" in include
        include_vectors = "embeddings" in include
        include_documents = "documents" in include
        
        all_ids = []
        all_distances = []
        all_metadatas = [] if include_metadata else None
        all_embeddings = [] if include_vectors else None
        all_documents = [] if include_documents else None
        
        for query_vec in query_embeddings:
            request = SearchRequest(
                vector=query_vec,
                k=n_results,
                filter=where,
                include_metadata=include_metadata,
                include_vectors=include_vectors,
            )
            results = self.search(request)
            
            ids = [r.id for r in results.results]
            # Convert similarity [0,1] to distance [0,1]
            distances = [1.0 - r.score for r in results.results]
            
            all_ids.append(ids)
            all_distances.append(distances)
            
            if include_metadata:
                all_metadatas.append([r.metadata or {} for r in results.results])
            if include_vectors:
                all_embeddings.append([r.vector or [] for r in results.results])
            if include_documents:
                # Extract document from metadata if stored there
                docs = []
                for r in results.results:
                    doc = self.get(r.id)
                    docs.append(doc.get("content") if doc else None)
                all_documents.append(docs)
        
        result = {
            "ids": all_ids,
            "distances": all_distances,
        }
        if include_metadata:
            result["metadatas"] = all_metadatas
        if include_vectors:
            result["embeddings"] = all_embeddings
        if include_documents:
            result["documents"] = all_documents
        
        return result
    
    def __len__(self) -> int:
        """Return collection size (supports len(collection))."""
        return self.count()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Collection(name='{self.name}', namespace='{self.namespace_name}', dimension={self._config.dimension})"


# ============================================================================
# Namespace Handle
# ============================================================================

class Namespace:
    """
    A namespace handle for multi-tenant isolation.
    
    All operations on a namespace are automatically scoped to that namespace,
    making cross-tenant data access impossible by construction.
    
    Use as a context manager for temporary namespace scoping:
    
        with db.use_namespace("tenant_123") as ns:
            # All operations scoped to tenant_123
            collection = ns.collection("documents")
            ...
    
    Or hold a reference for persistent use:
    
        ns = db.namespace("tenant_123")
        collection = ns.collection("documents")
    """
    
    def __init__(self, db: "Database", name: str, config: Optional[NamespaceConfig] = None):
        self._db = db
        self._name = name
        self._config = config
        self._collections: Dict[str, Collection] = {}
    
    @property
    def name(self) -> str:
        """Namespace name."""
        return self._name
    
    @property
    def config(self) -> Optional[NamespaceConfig]:
        """Namespace configuration."""
        return self._config
    
    def __enter__(self) -> "Namespace":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Could flush pending writes here if needed
        pass
    
    # ========================================================================
    # Collection Operations
    # ========================================================================
    
    def create_collection(
        self,
        name_or_config: Union[str, CollectionConfig],
        dimension: Optional[int] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs,
    ) -> Collection:
        """
        Create a collection in this namespace.
        
        Args:
            name_or_config: Collection name or CollectionConfig
            dimension: Vector dimension (required if name provided)
            metric: Distance metric
            **kwargs: Additional config options
            
        Returns:
            Collection handle
            
        Raises:
            CollectionExistsError: If collection already exists
        """
        if isinstance(name_or_config, CollectionConfig):
            config = name_or_config
        else:
            if dimension is None:
                raise ValidationError("dimension is required when creating collection by name")
            config = CollectionConfig(
                name=name_or_config,
                dimension=dimension,
                metric=metric,
                **kwargs,
            )
        
        # Check if exists in memory
        if config.name in self._collections:
            raise CollectionExistsError(config.name, self._name)
        
        # Check if exists in storage
        config_key = f"{self._name}/_collections/{config.name}".encode()
        if self._db.get(config_key) is not None:
            raise CollectionExistsError(config.name, self._name)
        
        # Create native Rust FFI collection first (HNSW index + durable storage)
        # The FFI call also writes a compact config to the same key, so we
        # write our full config AFTER to ensure the proper format is stored.
        metric_str = config.metric.value if hasattr(config.metric, 'value') else str(config.metric)
        self._db.ffi_collection_create(
            namespace=self._name,
            collection=config.name,
            dimension=config.dimension or 0,
            metric=metric_str,
        )
        
        # Persist full config to storage (overwrites compact FFI config)
        self._db.put(config_key, json.dumps(config.to_dict()).encode())
        
        # Create and cache collection handle
        collection = Collection(self, config)
        self._collections[config.name] = collection
        
        return collection
    
    def get_collection(self, name: str) -> Collection:
        """
        Get an existing collection.
        
        Args:
            name: Collection name
            
        Returns:
            Collection handle
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        if name in self._collections:
            return self._collections[name]
        
        # Try loading from storage
        config_key = f"{self._name}/_collections/{name}".encode()
        data = self._db.get(config_key)
        if data is not None:
            config = CollectionConfig.from_dict(json.loads(data.decode()), name=name)
            collection = Collection(self, config)
            self._collections[name] = collection
            return collection
        
        raise CollectionNotFoundError(name, self._name)
    
    def collection(self, name: str) -> Collection:
        """Alias for get_collection."""
        return self.get_collection(name)
    
    def list_collections(self) -> List[str]:
        """List all collections in this namespace."""
        # Scan storage for all collection configs
        prefix = f"{self._name}/_collections/".encode()
        names = set(self._collections.keys())  # Start with cached
        
        with self._db.transaction() as txn:
            for key, _ in txn.scan_prefix(prefix):
                name = key.decode().split("/")[-1]
                names.add(name)
        
        return sorted(names)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its data."""
        # Check if exists (load from storage if needed)
        config_key = f"{self._name}/_collections/{name}".encode()
        if name not in self._collections and self._db.get(config_key) is None:
            raise CollectionNotFoundError(name, self._name)
        
        # Remove from cache
        if name in self._collections:
            del self._collections[name]
        
        # Delete config from storage
        self._db.delete(config_key)
        
        # Delete all vectors in collection
        vectors_prefix = f"{self._name}/collections/{name}/vectors/".encode()
        with self._db.transaction() as txn:
            for key, _ in txn.scan_prefix(vectors_prefix):
                txn.delete(key)
        
        return True
    
    # ========================================================================
    # Key-Value Operations (scoped to namespace)
    # ========================================================================
    
    def put(self, key: str, value: bytes) -> None:
        """Put a key-value pair in this namespace."""
        # Prefix with namespace for isolation
        full_key = f"{self._name}/{key}".encode("utf-8")
        self._db.put(full_key, value)
    
    def get(self, key: str) -> Optional[bytes]:
        """Get a value from this namespace."""
        full_key = f"{self._name}/{key}".encode("utf-8")
        return self._db.get(full_key)
    
    def delete(self, key: str) -> None:
        """Delete a key from this namespace."""
        full_key = f"{self._name}/{key}".encode("utf-8")
        self._db.delete(full_key)
    
    def scan(self, prefix: str = "") -> Iterator[Tuple[str, bytes]]:
        """
        Scan keys in this namespace with optional prefix.
        
        This is safe for multi-tenant use - only returns keys from this namespace.
        """
        full_prefix = f"{self._name}/{prefix}".encode("utf-8")
        namespace_prefix = f"{self._name}/".encode("utf-8")
        
        with self._db.transaction() as txn:
            for key, value in txn.scan_prefix(full_prefix):
                # Strip namespace prefix from returned keys
                relative_key = key[len(namespace_prefix):].decode("utf-8")
                yield relative_key, value
