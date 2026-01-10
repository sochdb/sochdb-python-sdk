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
ToonDB Namespace Handle (Task 8: First-Class Namespace Handle + Context Manager API)

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


@dataclass(frozen=True)
class CollectionConfig:
    """
    Immutable collection configuration.
    
    Once a collection is created, its configuration is frozen.
    This prevents "works on my machine" drift and ensures reproducibility.
    
    Example:
        config = CollectionConfig(
            name="documents",
            dimension=384,
            metric=DistanceMetric.COSINE,
        )
        collection = ns.create_collection(config)
        
        # Access frozen config
        print(collection.config.dimension)  # 384
    """
    
    name: str
    dimension: int
    metric: DistanceMetric = DistanceMetric.COSINE
    
    # Index parameters
    m: int = 16                      # HNSW M parameter
    ef_construction: int = 100       # HNSW ef_construction
    quantization: QuantizationType = QuantizationType.NONE
    
    # Optional features
    enable_hybrid_search: bool = False  # Enable BM25 + vector search
    content_field: Optional[str] = None # Field to index for BM25
    
    def __post_init__(self):
        if self.dimension <= 0:
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
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionConfig":
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            metric=DistanceMetric(data.get("metric", "cosine")),
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
    - Vector similarity search (ANN)
    - Keyword search (BM25)
    - Hybrid search (RRF fusion)
    - Metadata filtering
    - Multi-vector documents
    
    All operations are automatically scoped to the parent namespace.
    """
    
    def __init__(
        self,
        namespace: "Namespace",
        config: CollectionConfig,
    ):
        self._namespace = namespace
        self._config = config
        self._db = namespace._db
    
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
    
    def insert(
        self,
        id: Union[str, int],
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """
        Insert a single vector.
        
        Args:
            id: Unique document ID
            vector: Vector embedding
            metadata: Optional metadata dict
            content: Optional text content (for hybrid search)
        """
        self.insert_batch([(id, vector, metadata, content)])
    
    def insert_batch(
        self,
        documents: List[Tuple[Union[str, int], List[float], Optional[Dict[str, Any]], Optional[str]]],
    ) -> int:
        """
        Insert multiple vectors in a batch.
        
        This is more efficient than individual inserts.
        
        Args:
            documents: List of (id, vector, metadata, content) tuples
            
        Returns:
            Number of documents inserted
        """
        # Validate dimensions
        for doc_id, vector, metadata, content in documents:
            if len(vector) != self._config.dimension:
                raise DimensionMismatchError(self._config.dimension, len(vector))
        
        # Store via namespace-scoped key in KV layer
        with self._db.transaction() as txn:
            for doc_id, vector, metadata, content in documents:
                doc_data = {
                    "id": doc_id,
                    "vector": vector,
                    "metadata": metadata or {},
                    "content": content,
                }
                key = self._vector_key(doc_id)
                txn.put(key, json.dumps(doc_data).encode())
        
        return len(documents)
    
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
            if len(v) != self._config.dimension:
                raise DimensionMismatchError(self._config.dimension, len(v))
        
        if chunk_texts and len(chunk_texts) != len(vectors):
            raise ValidationError(
                f"chunk_texts length ({len(chunk_texts)}) must match vectors length ({len(vectors)})"
            )
        
        # Store multi-vector document
        # (Implementation would use multi_vector mapping)
    
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
        
        Tries native Rust FFI first for O(log N) performance.
        Falls back to Python brute-force if FFI unavailable.
        """
        import time
        
        start_time = time.time()
        
        # Try FFI first (40x faster - native Rust)
        ffi_results = self._db.ffi_collection_search(
            self._namespace._name,
            self._config.name,
            request.vector,
            request.k
        )
        
        if ffi_results is not None:
            # print("DEBUG: Using FFI for vector search")
            # FFI succeeded - convert to SearchResults
            results = []
            for r in ffi_results:
                # Apply min_score filter
                if request.min_score is not None and r["score"] < request.min_score:
                    continue
                
                # Apply metadata filter
                if request.filter:
                    if not self._matches_filter(r.get("metadata", {}), request.filter):
                        continue
                
                result = SearchResult(
                    id=r["id"],
                    score=r["score"],
                    metadata=r.get("metadata") if request.include_metadata else None,
                    vector=None,  # FFI doesn't return vectors
                )
                results.append(result)
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return SearchResults(
                results=results,
                total_count=len(results),
                query_time_ms=query_time_ms,
                vector_results=len(results),
            )
        
        # Fallback: Python brute-force (slower but always works)
        import math
        
        # Get all documents from collection
        all_docs = []
        prefix = self._vectors_prefix()
        with self._db.transaction() as txn:
            for key, value in txn.scan_prefix(prefix):
                doc = json.loads(value.decode())
                all_docs.append(doc)
        
        if not all_docs or request.vector is None:
            return SearchResults(results=[], total_count=0, query_time_ms=0.0)
        
        # Compute cosine similarity for each document
        query_vec = request.vector
        query_norm = math.sqrt(sum(x * x for x in query_vec))
        
        scored_docs = []
        for doc in all_docs:
            doc_vec = doc.get("vector", [])
            if len(doc_vec) != len(query_vec):
                continue
            
            # Cosine similarity
            dot_product = sum(q * d for q, d in zip(query_vec, doc_vec))
            doc_norm = math.sqrt(sum(x * x for x in doc_vec))
            
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
                # Normalize from [-1, 1] to [0, 1] for threshold comparisons
                similarity = (similarity + 1.0) / 2.0
            else:
                similarity = 0.0
            
            # Apply min_score filter
            if request.min_score is not None and similarity < request.min_score:
                continue
            
            # Apply metadata filter
            if request.filter:
                metadata = doc.get("metadata", {})
                if not self._matches_filter(metadata, request.filter):
                    continue
            
            scored_docs.append((similarity, doc))
        
        # Sort by score descending
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
            vector_results=len(results),
        )
    
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

        # Fallback: Python scan
        all_docs = []
        prefix = self._vectors_prefix()
        with self._db.transaction() as txn:
            for key, value in txn.scan_prefix(prefix):
                doc = json.loads(value.decode())
                all_docs.append(doc)
        
        # Simple keyword matching on content and metadata
        query_lower = request.text_query.lower()
        query_terms = query_lower.split()
        
        scored_docs = []
        for doc in all_docs:
            # Search in content field
            content = doc.get("content", "") or ""
            metadata = doc.get("metadata", {})
            
            # Also search in metadata text fields
            text_fields = [content]
            for v in metadata.values():
                if isinstance(v, str):
                    text_fields.append(v)
            
            combined_text = " ".join(text_fields).lower()
            
            # Simple term frequency scoring
            score = 0.0
            for term in query_terms:
                if term in combined_text:
                    score += combined_text.count(term)
            
            if score > 0:
                # Apply metadata filter
                if request.filter:
                    if not self._matches_filter(metadata, request.filter):
                        continue
                scored_docs.append((score, doc))
        
        # Sort by score descending
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
        
        # Persist config to storage
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
            config = CollectionConfig.from_dict(json.loads(data.decode()))
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
