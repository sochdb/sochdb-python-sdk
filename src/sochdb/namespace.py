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

# Fast JSON serialization for the durable doc-write hot path. Each persisted doc
# embeds the full vector; stdlib json formats every float as a string (~0.15ms per
# 768-d vector -> ~1.5s per 10k batch, the dominant insert overhead). orjson emits
# the SAME valid JSON ~15x faster, so the Rust FFI BM25 reader (serde_json) is
# unaffected. Soft dependency: fall back to stdlib json when orjson is unavailable.
try:
    import orjson as _orjson

    def _dumps_bytes(obj) -> bytes:
        return _orjson.dumps(obj, option=_orjson.OPT_SERIALIZE_NUMPY)

    def _loads(data: bytes):
        # orjson parses bytes directly (no .decode()), ~2-3x faster than stdlib.
        return _orjson.loads(data)
except ImportError:  # pragma: no cover - orjson is an optional accelerator
    def _dumps_bytes(obj) -> bytes:
        return json.dumps(obj).encode()

    def _loads(data: bytes):
        return json.loads(data)
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
    
    # Index parameters — defaults match the engine's HnswConfig::default()
    # (m=32, ef_construction=256) so a collection built without explicit tuning
    # reaches 95+ recall@10 (measured: Cohere-1M 768d cosine = 0.972). The old
    # m=16/efc=100 defaults built cheap graphs that capped recall near 0.90.
    m: int = 32                      # HNSW M parameter (max_connections)
    ef_construction: int = 256       # HNSW ef_construction
    quantization: QuantizationType = QuantizationType.NONE
    
    # Optional features
    enable_hybrid_search: bool = False  # Enable BM25 + vector search
    content_field: Optional[str] = None # Field to index for BM25

    # When True (default) each document's raw embedding is also written into its
    # durable KV doc. That copy is what the cold FFI vector-search path and the
    # Chroma-compat getters read back. Set False to shrink KV docs and speed up
    # inserts when you only ever query in-session — the HNSW graph and the .npy
    # snapshot still retain the vectors, so same-session search is unaffected.
    # CAVEAT: with False, a cold restart that cannot reload the .npy snapshot has
    # no vector source for the FFI-fallback search and will return empty results.
    # Keep the default (True) unless you never rely on cross-restart vector search.
    persist_vector_in_kv: bool = True

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
            "persist_vector_in_kv": self.persist_vector_in_kv,
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
            m=data.get("m", 32),
            ef_construction=data.get("ef_construction", 256),
            quantization=QuantizationType(data.get("quantization", "none")),
            enable_hybrid_search=data.get("enable_hybrid_search", False),
            content_field=data.get("content_field"),
            persist_vector_in_kv=data.get("persist_vector_in_kv", True),
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

    # Keyword leg algorithm for keyword/hybrid search:
    #   "bm25" - BM25 relevance ranking (default, native FFI when available)
    #   "grep" - case-insensitive substring AND-match (all query terms must appear)
    keyword_mode: str = "bm25"
    
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

        if self.keyword_mode not in ("bm25", "grep"):
            raise ValidationError(
                f"keyword_mode must be 'bm25' or 'grep', got {self.keyword_mode!r}"
            )


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
        self._deleted_internal_ids: set = set()  # internal ids tombstoned by delete()
    
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
        include_vectors: bool = False,
    ) -> "SearchResults":
        """
        Exact brute-force vector search for perfect recall.

        Computes distances to ALL vectors and returns the true k-nearest
        neighbors. Guarantees recall@k = 1.0 but is O(n) per query.

        Args:
            vector: Query vector
            k: Number of results
            include_vectors: If True, results carry their stored embedding.

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
                vector=self._raw_vectors.get(doc_id) if include_vectors else None,
            )
            results.append(result)
        
        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)
    
    def vector_search_exact_f64(
        self,
        vector: List[float],
        k: int = 10,
        include_vectors: bool = False,
    ) -> "SearchResults":
        """
        Exact brute-force vector search using f64 precision.

        Same as vector_search_exact but computes distances in f64 to match
        ground truth computed with numpy f64 arithmetic. Eliminates f32
        tie-breaking mismatches at the k-th boundary.

        Args:
            vector: Query vector
            k: Number of results
            include_vectors: If True, results carry their stored embedding.

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
                vector=self._raw_vectors.get(doc_id) if include_vectors else None,
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

    # NOTE on scoring (review Task 5): result scores use the cosine identity
    # `max(0, 1 - distance)`. A metric-aware transform was attempted but reverted:
    # the EMBEDDED FFI path is cosine-only, because `hnsw_new(dimension, M,
    # ef_construction)` (vector.py) takes no metric and builds HnswConfig::default()
    # (metric = Cosine), so CollectionConfig.metric is ignored in-process. Applying
    # a Euclidean/dot transform to a cosine distance would mis-score, hence the
    # revert.
    #
    # The ENGINE itself is metric-complete: HnswConfig.metric +
    # VectorIndex::with_params(DistanceMetric, ...) exist, the distance kernels
    # dispatch on metric (Cosine→cosine, Euclidean→l2, DotProduct→-dot), and the
    # gRPC server path (SochDBClient) already honors all three (tested for L2/dot).
    # So: non-cosine metrics work today via the server path; the embedded fix is
    # thin plumbing — add a metric arg to the `hnsw_new` FFI (set config.metric
    # before HnswIndex::new), rebuild the dylib, pass config.metric from Python,
    # and make this transform metric-aware. Not implementing distance math.

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
                "metadata": meta,
                "content": content or meta.get("_content", ""),
                "is_multi_vector": False,
            }
            if self._config.persist_vector_in_kv:
                doc_data["vector"] = vector
            txn.put(self._vector_key(id), _dumps_bytes(doc_data))
    
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
        # Prefer the zero-allocation flat FFI path (hnsw_insert_batch_flat).
        # vectors_array is already a C-contiguous float32 (N, D) array and
        # internal_ids is C-contiguous uint64, so strict=False adds no copy but
        # avoids the defensive re-copy the generic insert_batch always does.
        try:
            count = self._vector_index.insert_batch_fast(internal_ids, vectors_array, strict=False)
        except AttributeError:
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
                    "metadata": meta,
                    "content": content or meta.get("_content", ""),
                    "is_multi_vector": False,
                }
                if self._config.persist_vector_in_kv:
                    doc_data["vector"] = vector
                txn.put(self._vector_key(doc_id), _dumps_bytes(doc_data))

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
            txn.put(key, _dumps_bytes(doc_data))
    
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
            # Pure keyword search (BM25 or grep)
            if request.keyword_mode == "grep":
                return self._grep_search(request)
            return self._keyword_search(request)
    
    def vector_search(
        self,
        vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
        include_vectors: bool = False,
    ) -> SearchResults:
        """
        Convenience method for vector similarity search.

        Args:
            vector: Query vector
            k: Number of results
            filter: Optional metadata filter
            min_score: Minimum similarity score
            include_vectors: If True, each result carries its stored embedding
                (read from the in-memory index, no re-embedding). Needed for MMR
                / relevance feedback without an extra embed pass.

        Returns:
            SearchResults
        """
        request = SearchRequest(
            vector=vector,
            k=k,
            filter=filter,
            min_score=min_score,
            include_vectors=include_vectors,
        )
        return self.search(request)
    
    def keyword_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> SearchResults:
        """
        Convenience method for keyword (BM25) search.

        Requires hybrid search to be enabled on the collection.

        Args:
            query: Text query
            k: Number of results
            filter: Optional metadata filter
            include_vectors: If True, results carry their stored embedding.

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
            include_vectors=include_vectors,
        )
        return self.search(request)

    def grep_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> SearchResults:
        """
        Convenience method for grep-style keyword search.

        Unlike :meth:`keyword_search` (BM25 relevance ranking), this performs a
        case-insensitive substring AND-match: a document matches only if *every*
        whitespace-separated query term appears as a substring of its content or
        string metadata. Results are ranked by total term-occurrence count.

        This is a precision-oriented exact-match retriever (fast, no scoring
        model). Use it as the keyword leg of a "grep + HNSW" hybrid via
        ``hybrid_search(..., keyword_mode="grep")``.

        Args:
            query: Text query (all terms must appear)
            k: Number of results
            filter: Optional metadata filter

        Returns:
            SearchResults
        """
        request = SearchRequest(
            text_query=query,
            k=k,
            filter=filter,
            alpha=0.0,
            keyword_mode="grep",
            include_vectors=include_vectors,
        )
        return self.search(request)

    def hybrid_search(
        self,
        vector: List[float],
        text_query: str,
        k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        keyword_mode: str = "bm25",
        include_vectors: bool = False,
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
            keyword_mode: Keyword leg algorithm — "bm25" (relevance ranking) or
                "grep" (case-insensitive substring AND-match). Combine with the
                HNSW vector leg to get "BM25 + HNSW" or "grep + HNSW".
            
        Returns:
            SearchResults
        """
        request = SearchRequest(
            vector=vector,
            text_query=text_query,
            k=k,
            alpha=alpha,
            filter=filter,
            keyword_mode=keyword_mode,
            include_vectors=include_vectors,
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

        # Filtered search needs an adaptively-widened candidate pool; a fixed
        # k*3 under-returns when the filter is selective. Route the in-memory
        # path to the dedicated widening driver.
        if request.filter:
            if self._vector_index is not None and len(self._vector_index) > 0:
                return self._search_in_memory_filtered(request, start_time)
            if self._reload_index() > 0:
                return self._search_in_memory_filtered(request, start_time)

        search_k = request.k * 3 if request.filter else request.k

        # Primary path: in-memory HNSW (same session, fast)
        if self._vector_index is not None and len(self._vector_index) > 0:
            return self._search_in_memory(request, search_k, start_time)

        # Reload path: rebuild HNSW from snapshot or KV store (after a reopen).
        loaded = self._reload_index()
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
                    vector=self._raw_vectors.get(doc_id) if request.include_vectors else None,
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

        # Over-fetch to compensate for tombstoned nodes still living in the
        # HNSW graph (deleted docs would otherwise consume top-k slots and make
        # us under-return). Cheap when few deletes; bounded by the index size.
        ndel = len(self._deleted_internal_ids)
        fetch_k = min(search_k + ndel, len(self._vector_index)) if ndel else search_k
        raw_results = self._ann_candidates(request.vector, fetch_k)

        results = []
        seen = set()
        for internal_id, distance in raw_results:
            if internal_id in self._deleted_internal_ids:
                continue
            doc_id = self._internal_to_id.get(internal_id)
            if doc_id is None or doc_id in seen:
                continue
            # Convert distance to similarity (higher = better)
            similarity = max(0.0, 1.0 - distance)
            if request.min_score is not None and similarity < request.min_score:
                continue
            metadata = self._metadata_store.get(doc_id, {})
            if request.filter and not self._matches_filter(metadata, request.filter):
                continue
            seen.add(doc_id)
            result = SearchResult(
                id=str(doc_id), score=similarity,
                metadata=metadata if request.include_metadata else None,
                vector=self._raw_vectors.get(doc_id) if request.include_vectors else None,
            )
            results.append(result)
            if len(results) >= request.k:
                break

        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=results, total_count=len(results), query_time_ms=elapsed)

    def _search_in_memory_filtered(self, request: SearchRequest, start_time: float) -> SearchResults:
        """Vector search with a metadata filter, adaptively widening the ANN
        candidate pool until k survivors are found or the index is exhausted.

        Fixes under-return: a fixed k*3 pool drops below k results when the
        filter is selective. Widens geometrically with a latency cap, lifting
        ef_search to match (and restoring it afterwards so later unfiltered
        queries are not slowed).
        """
        import time

        vi = self._vector_index
        n_total = len(vi)
        k = request.k
        MAX_SEARCH_K = max(2000, k * 50)
        EF_CEILING = 4000

        try:
            prev_ef = vi.ef_search
        except Exception:
            prev_ef = None

        search_k = min(max(k * 4, 16), n_total) if n_total else k
        prev_count = -1
        best: list = []
        try:
            while True:
                # The HNSW never returns more than ef_search neighbours; lift it.
                if prev_ef is not None:
                    desired = min(max(prev_ef, search_k), EF_CEILING)
                    if desired > (prev_ef or 0):
                        try:
                            vi.ef_search = desired
                        except Exception:
                            pass
                raw = self._ann_candidates(request.vector, search_k)
                results = []
                seen = set()
                for internal_id, distance in raw:
                    if internal_id in self._deleted_internal_ids:
                        continue
                    doc_id = self._internal_to_id.get(internal_id)
                    if doc_id is None or doc_id in seen:
                        continue
                    similarity = max(0.0, 1.0 - distance)
                    if request.min_score is not None and similarity < request.min_score:
                        continue
                    metadata = self._metadata_store.get(doc_id, {})
                    if not self._matches_filter(metadata, request.filter):
                        continue
                    seen.add(doc_id)
                    results.append(SearchResult(
                        id=str(doc_id), score=similarity,
                        metadata=metadata if request.include_metadata else None,
                        vector=self._raw_vectors.get(doc_id) if request.include_vectors else None,
                    ))
                    if len(results) >= k:
                        break
                best = results
                if len(results) >= k:
                    break
                if not n_total or search_k >= n_total or search_k >= MAX_SEARCH_K:
                    break
                if len(raw) <= prev_count:  # index returned no new candidates (ef ceiling)
                    break
                prev_count = len(raw)
                search_k = min(search_k * 2, n_total, MAX_SEARCH_K)
        finally:
            # Restore ef_search so later unfiltered queries are not slowed.
            if prev_ef is not None:
                try:
                    vi.ef_search = prev_ef
                except Exception:
                    pass

        elapsed = (time.time() - start_time) * 1000
        return SearchResults(results=best, total_count=len(best), query_time_ms=elapsed)

    def _ann_candidates(self, vector, search_k: int):
        """Fetch ANN candidates, size-gating to the faster HNSW path.

        Above the Rust flat-scan threshold, search_fast (lock-light HNSW) is
        faster than search() at identical recall; below it, search() uses a
        perfect-recall brute-force scan that is faster, so keep it there.
        Returns a list of (internal_id, distance).
        """
        vi = self._vector_index
        n = len(vi)
        dim = self._config.dimension or getattr(vi, "dimension", 0) or 128
        flat_threshold = 10000 if dim <= 128 else (4000 if dim <= 384 else 1000)
        if n > flat_threshold and hasattr(vi, "search_fast"):
            try:
                return vi.search_fast(vector, search_k)
            except Exception:
                return vi.search(vector, search_k)
        return vi.search(vector, search_k)
    
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

    def _reload_vectors_from_kv(self) -> int:
        """Rebuild the in-memory HNSW index from vectors persisted in the KV store.

        Robust cross-session reload path used when no numpy snapshot exists.
        ``insert``/``insert_batch`` write each document — including its vector
        when ``persist_vector_in_kv`` is set (the default) — to the KV store in
        the same transaction as the HNSW insert, so the KV store is an
        always-current, crash-safe source of truth that, unlike the snapshot,
        does NOT depend on ``close()`` ever being called. This is what makes a
        ``persist_directory`` collection searchable again after a reopen.

        Returns the number of vectors loaded (0 if none are recoverable, e.g.
        the collection was created with ``persist_vector_in_kv=False``, in which
        case the vectors live only in the snapshot).
        """
        import numpy as np

        dim = self._config.dimension
        prefix = self._vectors_prefix()
        doc_ids = []
        internal_ids = []
        vectors = []
        with self._db.transaction() as txn:
            for _key, value in txn.scan_prefix(prefix):
                try:
                    doc = _loads(value)
                except Exception:
                    continue
                vec = doc.get("vector")
                # Skip records with no recoverable single vector: persist was
                # disabled, or this is a multi-vector doc handled elsewhere.
                if vec is None or doc.get("is_multi_vector"):
                    continue
                doc_id = doc.get("id")
                if doc_id is None:
                    continue
                if dim is None:
                    dim = len(vec)
                    object.__setattr__(self._config, "dimension", dim)
                if len(vec) != dim:
                    continue
                iid = self._get_internal_id(doc_id)
                self._metadata_store[doc_id] = doc.get("metadata", {}) or {}
                self._raw_vectors[doc_id] = vec
                doc_ids.append(doc_id)
                internal_ids.append(iid)
                vectors.append(vec)

        if not doc_ids:
            return 0

        self._ensure_index(dim)
        ids_arr = np.asarray(internal_ids, dtype=np.uint64)
        vecs_arr = np.asarray(vectors, dtype=np.float32)
        try:
            self._vector_index.insert_batch_fast(ids_arr, vecs_arr, strict=False)
        except AttributeError:
            self._vector_index.insert_batch(ids_arr, vecs_arr)
        return len(doc_ids)

    def _reload_index(self) -> int:
        """Repopulate the in-memory HNSW after a reopen.

        Tries the fast numpy snapshot first, then falls back to rebuilding from
        the KV store. The KV fallback is what keeps ``persist_directory``
        collections searchable across sessions; the snapshot is only an
        opportunistic accelerator that may be absent.
        """
        loaded = self._reload_vectors_from_snapshot()
        if loaded > 0:
            return loaded
        return self._reload_vectors_from_kv()

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
                    vector=self._raw_vectors.get(r["id"]) if request.include_vectors else None,
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

    def _grep_search(self, request: SearchRequest) -> SearchResults:
        """Internal grep-style keyword search.

        Case-insensitive substring AND-match: a document is a hit only if every
        whitespace-separated query term appears as a substring of the combined
        searchable text (content + string metadata values). Results are ranked
        by the total number of term occurrences (a simple, model-free signal).

        Unlike :meth:`_keyword_search`, there is no IDF/BM25 weighting and no
        stopword removal — this is an exact-match retriever optimised for
        precision and predictability.
        """
        import time

        start_time = time.time()

        if not request.text_query:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)

        terms = [t for t in request.text_query.lower().split() if t]
        if not terms:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)

        # --- Primary path: scan the in-memory metadata store -----------------
        # The collection already keeps every document's content + metadata in
        # `self._metadata_store` (content lives under the injected `_content`
        # key). That is everything grep needs — and crucially it holds NO vector
        # payload, so we avoid the real bottleneck: pulling all stored documents
        # (each embedding its full vector) across the FFI boundary and JSON-
        # parsing them on every query. This makes grep a pure in-process dict
        # scan over small text, mirroring how vector search prefers the
        # in-memory HNSW before touching storage.
        if not self._metadata_store and self._config.dimension is not None:
            # Same lazy-rebuild trigger used by vector search: repopulate the
            # in-memory maps (incl. metadata/content) from snapshot or KV store.
            self._reload_index()

        if self._metadata_store:
            return self._grep_search_in_memory(request, terms, start_time)

        # --- Fallback path: scan the KV store --------------------------------
        # Reached only when nothing is loaded in memory (e.g. a freshly reopened
        # collection with no snapshot). Uses a raw-bytes prefilter + deferred
        # parse to limit the cost of reading full doc payloads.
        return self._grep_search_kv(request, terms, start_time)

    def _grep_search_in_memory(
        self, request: SearchRequest, terms: List[str], start_time: float
    ) -> SearchResults:
        """grep over the in-memory metadata/content store (no FFI, no parse)."""
        import time

        has_filter = request.filter is not None
        scored = []  # (score, doc_id, metadata)
        for doc_id, meta in self._metadata_store.items():
            # Build the searchable text from string metadata values (this
            # includes the injected `_content` field when content was supplied).
            parts = [v for v in meta.values() if isinstance(v, str)]
            if not parts:
                continue
            haystack = " ".join(parts).lower()

            if not all(term in haystack for term in terms):
                continue
            if has_filter and not self._matches_filter(meta, request.filter):
                continue

            score = float(sum(haystack.count(term) for term in terms))
            scored.append((score, doc_id, meta))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc_id, meta in scored[: request.k]:
            results.append(SearchResult(
                id=str(doc_id),
                score=score,
                metadata=meta if request.include_metadata else None,
                vector=self._raw_vectors.get(doc_id) if request.include_vectors else None,
            ))

        query_time_ms = (time.time() - start_time) * 1000
        return SearchResults(
            results=results,
            total_count=len(scored),
            query_time_ms=query_time_ms,
            keyword_results=len(results),
        )

    def _grep_search_kv(
        self, request: SearchRequest, terms: List[str], start_time: float
    ) -> SearchResults:
        """grep fallback that scans the KV store when nothing is in memory."""
        import time

        # --- Fast path: rank on raw bytes, defer JSON parsing to top-k -------
        # Each stored doc embeds its full vector, so json.loads() is by far the
        # dominant cost (the vector payload dwarfs the text). Two optimisations:
        #
        #   1. Prefilter on the lowercased raw bytes. grep is a substring
        #      AND-match, so a term absent from the raw bytes cannot match the
        #      parsed text either. ASCII query terms (the common case) get a
        #      sound prefilter with no false negatives; non-ASCII terms bypass it
        #      and are validated after parsing.
        #   2. Score candidates from the raw bytes too, then sort and parse ONLY
        #      the documents we actually return. We never parse the long tail of
        #      matches that fall outside the top-k, turning ~O(matches) parses
        #      into ~O(k).
        prefilter = [t.encode("utf-8") for t in terms if t.isascii()]

        prefix = self._vectors_prefix()
        has_filter = request.filter is not None

        # Collect (raw_score, value_bytes) candidates without parsing.
        candidates = []
        with self._db.transaction() as txn:
            for _key, value in txn.scan_prefix(prefix):
                low = value.lower()
                # Cheap necessary-condition check; skips the vast majority.
                if not all(tb in low for tb in prefilter):
                    continue
                # Approximate occurrence score straight off the raw bytes. This
                # can include JSON keys/structure but is a strong, cheap proxy
                # for ranking; the exact score is recomputed for returned docs.
                raw_score = sum(low.count(tb) for tb in prefilter)
                candidates.append((raw_score, value))

        # Best candidates first, then parse lazily until we have k results.
        candidates.sort(key=lambda c: c[0], reverse=True)

        results = []
        for _raw_score, value in candidates:
            if len(results) >= request.k:
                break

            doc = _loads(value)
            metadata = doc.get("metadata", {}) or {}

            # Build the searchable text (content + string metadata values).
            text_fields = [doc.get("content", "") or ""]
            for v in metadata.values():
                if isinstance(v, str):
                    text_fields.append(v)
            haystack = " ".join(text_fields).lower()

            # Authoritative AND-match on the real text fields only.
            if not all(term in haystack for term in terms):
                continue

            if has_filter and not self._matches_filter(metadata, request.filter):
                continue

            # Exact occurrence score over the text fields (model-free ranking).
            score = float(sum(haystack.count(term) for term in terms))
            results.append(SearchResult(
                id=doc["id"],
                score=score,
                metadata=doc.get("metadata") if request.include_metadata else None,
                vector=doc.get("vector") if request.include_vectors else None,
            ))

        # Re-sort the returned page by exact score (raw-byte order is approximate).
        results.sort(key=lambda r: r.score, reverse=True)

        query_time_ms = (time.time() - start_time) * 1000

        return SearchResults(
            results=results,
            total_count=len(candidates),
            query_time_ms=query_time_ms,
            keyword_results=len(results),
        )

    def _hybrid_search(self, request: SearchRequest) -> SearchResults:
        """Internal hybrid search using RRF (Reciprocal Rank Fusion)."""
        import time
        
        start_time = time.time()
        
        # Get vector and keyword results separately
        vector_results = self._vector_search(request)
        if request.keyword_mode == "grep":
            keyword_results = self._grep_search(request)
        else:
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

        Removes the document from the KV store AND evicts it from the
        in-memory HNSW index state so it is no longer returned by vector
        search. (Previously only the KV key was tombstoned, so the vector
        remained searchable and surfaced stale content.)
        """
        key = self._vector_key(id)
        with self._db.transaction() as txn:
            if txn.get(key) is None:
                return False
            txn.delete(key)
        # Evict from in-memory state so vector search stops returning it.
        iid = self._id_to_internal.get(id)
        if iid is not None:
            self._deleted_internal_ids.add(iid)
            # Tombstone in the Rust HNSW too, if the loaded dylib supports it
            # (no-op on older dylibs; the Python maps already fix the session).
            remove = getattr(self._vector_index, "remove", None)
            if remove is not None:
                try:
                    remove(iid)
                except Exception:
                    pass
            self._internal_to_id.pop(iid, None)
            self._id_to_internal.pop(id, None)
        self._metadata_store.pop(id, None)
        self._raw_vectors.pop(id, None)
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
