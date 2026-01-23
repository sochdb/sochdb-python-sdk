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
Consolidation as Event-Sourced Canonicalization

This module implements memory consolidation without destructive updates:

- Raw assertions are stored as immutable "events" (append-only)
- A derived canonical view maintains merged/deduplicated facts
- Contradictions are handled via temporal interval updates
- Full provenance and auditability is preserved

Key principles:
1. Never delete raw events - they are evidence
2. Canonical view is derived, not source of truth
3. Contradictions update intervals, not overwrite facts
4. Union-find clustering for efficient deduplication

Supports both embedded (FFI) and server (gRPC) modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Iterator
)
import hashlib
import json
import time
from collections import defaultdict


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class RawAssertion:
    """
    Immutable raw assertion (event).
    
    These are never deleted, only marked as superseded.
    
    Attributes:
        id: Unique assertion ID
        fact: The assertion content (subject, predicate, object)
        embedding: Vector embedding for similarity
        timestamp: When the assertion was recorded
        source: Source information (document, conversation, etc.)
        confidence: Extraction confidence
        superseded_by: ID of superseding assertion (if any)
    """
    id: str
    fact: Dict[str, Any]
    embedding: Optional[List[float]] = None
    timestamp: int = 0
    source: Optional[str] = None
    confidence: float = 1.0
    superseded_by: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = int(time.time() * 1000)
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from fact + timestamp."""
        content = json.dumps(self.fact, sort_keys=True) + str(self.timestamp)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fact": self.fact,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence,
            "superseded_by": self.superseded_by,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawAssertion":
        return cls(
            id=data.get("id", ""),
            fact=data["fact"],
            embedding=data.get("embedding"),
            timestamp=data.get("timestamp", 0),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            superseded_by=data.get("superseded_by"),
        )


@dataclass
class CanonicalFact:
    """
    Derived canonical fact (merged/deduplicated view).
    
    Attributes:
        id: Canonical fact ID
        merged_fact: The consolidated fact representation
        support_set: IDs of raw assertions supporting this fact
        last_updated: When the canonical view was last updated
        confidence: Aggregated confidence score
        valid_from: Start of validity period
        valid_until: End of validity period (0 = still valid)
    """
    id: str
    merged_fact: Dict[str, Any]
    support_set: Set[str] = field(default_factory=set)
    last_updated: int = 0
    confidence: float = 1.0
    valid_from: int = 0
    valid_until: int = 0
    
    def __post_init__(self):
        if self.last_updated == 0:
            self.last_updated = int(time.time() * 1000)
        if self.valid_from == 0:
            self.valid_from = self.last_updated
    
    @property
    def is_current(self) -> bool:
        """Check if fact is currently valid."""
        now = int(time.time() * 1000)
        if now < self.valid_from:
            return False
        if self.valid_until > 0 and now >= self.valid_until:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "merged_fact": self.merged_fact,
            "support_set": list(self.support_set),
            "last_updated": self.last_updated,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalFact":
        return cls(
            id=data["id"],
            merged_fact=data["merged_fact"],
            support_set=set(data.get("support_set", [])),
            last_updated=data.get("last_updated", 0),
            confidence=data.get("confidence", 1.0),
            valid_from=data.get("valid_from", 0),
            valid_until=data.get("valid_until", 0),
        )


@dataclass
class ConsolidationConfig:
    """
    Configuration for consolidation behavior.
    
    Attributes:
        similarity_threshold: Min similarity to consider for merging
        max_cluster_size: Max assertions in a single canonical cluster
        min_confidence: Min confidence to include in canonical view
        use_temporal_updates: Use interval updates for contradictions
        embedding_dim: Dimension of embeddings (for validation)
    """
    similarity_threshold: float = 0.85
    max_cluster_size: int = 100
    min_confidence: float = 0.5
    use_temporal_updates: bool = True
    embedding_dim: int = 384


# ============================================================================
# Union-Find for Clustering
# ============================================================================

class UnionFind:
    """
    Union-Find data structure for efficient clustering.
    
    Supports path compression and union by rank.
    Complexity: O(α(n)) per operation where α is inverse Ackermann.
    """
    
    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}
    
    def find(self, x: str) -> str:
        """Find root with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # Path compression
        return self._parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """Union by rank. Returns True if merged, False if already same set."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self._rank[root_x] < self._rank[root_y]:
            self._parent[root_x] = root_y
        elif self._rank[root_x] > self._rank[root_y]:
            self._parent[root_y] = root_x
        else:
            self._parent[root_y] = root_x
            self._rank[root_x] += 1
        
        return True
    
    def connected(self, x: str, y: str) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)
    
    def get_clusters(self) -> Dict[str, Set[str]]:
        """Get all clusters."""
        clusters: Dict[str, Set[str]] = defaultdict(set)
        for item in self._parent.keys():
            root = self.find(item)
            clusters[root].add(item)
        return dict(clusters)


# ============================================================================
# Consolidation Backend Interface
# ============================================================================

class ConsolidationBackend(ABC):
    """
    Abstract interface for consolidation storage.
    """
    
    @abstractmethod
    def store_raw_assertion(self, namespace: str, assertion: RawAssertion) -> None:
        """Store a raw assertion (append-only)."""
        pass
    
    @abstractmethod
    def get_raw_assertion(self, namespace: str, assertion_id: str) -> Optional[RawAssertion]:
        """Get a raw assertion by ID."""
        pass
    
    @abstractmethod
    def scan_raw_assertions(self, namespace: str, 
                             since: Optional[int] = None) -> Iterator[RawAssertion]:
        """Scan all raw assertions, optionally since a timestamp."""
        pass
    
    @abstractmethod
    def mark_superseded(self, namespace: str, assertion_id: str, 
                        superseded_by: str) -> None:
        """Mark an assertion as superseded."""
        pass
    
    @abstractmethod
    def store_canonical_fact(self, namespace: str, fact: CanonicalFact) -> None:
        """Store or update a canonical fact."""
        pass
    
    @abstractmethod
    def get_canonical_fact(self, namespace: str, fact_id: str) -> Optional[CanonicalFact]:
        """Get a canonical fact by ID."""
        pass
    
    @abstractmethod
    def scan_canonical_facts(self, namespace: str,
                              current_only: bool = True) -> Iterator[CanonicalFact]:
        """Scan canonical facts."""
        pass
    
    @abstractmethod
    def update_temporal_interval(self, namespace: str, fact_id: str,
                                   valid_until: int) -> None:
        """Close a temporal interval (for contradictions)."""
        pass
    
    @abstractmethod
    def search_similar(self, namespace: str, embedding: List[float],
                       k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar assertions by embedding."""
        pass


# ============================================================================
# FFI Backend
# ============================================================================

class FFIConsolidationBackend(ConsolidationBackend):
    """
    Consolidation backend using embedded database via FFI.
    """
    
    def __init__(self, db: "Database", collection: str = "raw_assertions"):
        self._db = db
        self._collection = collection
    
    def _raw_key(self, namespace: str, assertion_id: str) -> bytes:
        return f"raw:{namespace}:{assertion_id}".encode()
    
    def _canonical_key(self, namespace: str, fact_id: str) -> bytes:
        return f"canonical:{namespace}:{fact_id}".encode()
    
    def store_raw_assertion(self, namespace: str, assertion: RawAssertion) -> None:
        key = self._raw_key(namespace, assertion.id)
        value = json.dumps(assertion.to_dict()).encode()
        self._db.put(key, value)
        
        # Also add to vector collection if embedding exists
        if assertion.embedding:
            try:
                ns = self._db.namespace(namespace)
                coll = ns.collection(self._collection)
                coll.add(
                    id=assertion.id,
                    vector=assertion.embedding,
                    metadata={
                        "fact": json.dumps(assertion.fact),
                        "timestamp": assertion.timestamp,
                        "confidence": assertion.confidence,
                    },
                )
            except Exception:
                pass  # Collection might not exist
    
    def get_raw_assertion(self, namespace: str, assertion_id: str) -> Optional[RawAssertion]:
        key = self._raw_key(namespace, assertion_id)
        data = self._db.get(key)
        if data is None:
            return None
        return RawAssertion.from_dict(json.loads(data.decode()))
    
    def scan_raw_assertions(self, namespace: str,
                             since: Optional[int] = None) -> Iterator[RawAssertion]:
        prefix = f"raw:{namespace}:".encode()
        for key, value in self._db.scan_prefix(prefix):
            assertion = RawAssertion.from_dict(json.loads(value.decode()))
            if since is None or assertion.timestamp >= since:
                yield assertion
    
    def mark_superseded(self, namespace: str, assertion_id: str,
                        superseded_by: str) -> None:
        assertion = self.get_raw_assertion(namespace, assertion_id)
        if assertion:
            assertion.superseded_by = superseded_by
            key = self._raw_key(namespace, assertion_id)
            self._db.put(key, json.dumps(assertion.to_dict()).encode())
    
    def store_canonical_fact(self, namespace: str, fact: CanonicalFact) -> None:
        key = self._canonical_key(namespace, fact.id)
        value = json.dumps(fact.to_dict()).encode()
        self._db.put(key, value)
    
    def get_canonical_fact(self, namespace: str, fact_id: str) -> Optional[CanonicalFact]:
        key = self._canonical_key(namespace, fact_id)
        data = self._db.get(key)
        if data is None:
            return None
        return CanonicalFact.from_dict(json.loads(data.decode()))
    
    def scan_canonical_facts(self, namespace: str,
                              current_only: bool = True) -> Iterator[CanonicalFact]:
        prefix = f"canonical:{namespace}:".encode()
        for key, value in self._db.scan_prefix(prefix):
            fact = CanonicalFact.from_dict(json.loads(value.decode()))
            if current_only and not fact.is_current:
                continue
            yield fact
    
    def update_temporal_interval(self, namespace: str, fact_id: str,
                                   valid_until: int) -> None:
        fact = self.get_canonical_fact(namespace, fact_id)
        if fact:
            fact.valid_until = valid_until
            fact.last_updated = int(time.time() * 1000)
            self.store_canonical_fact(namespace, fact)
    
    def search_similar(self, namespace: str, embedding: List[float],
                       k: int = 10) -> List[Tuple[str, float]]:
        try:
            ns = self._db.namespace(namespace)
            coll = ns.collection(self._collection)
            results = coll.search(vector=embedding, k=k)
            return [(r.id, r.score) for r in results.results]
        except Exception:
            return []


# ============================================================================
# gRPC Backend
# ============================================================================

class GrpcConsolidationBackend(ConsolidationBackend):
    """
    Consolidation backend using gRPC client.
    """
    
    def __init__(self, client: "SochDBClient", collection: str = "raw_assertions"):
        self._client = client
        self._collection = collection
    
    def _raw_key(self, namespace: str, assertion_id: str) -> str:
        return f"raw:{namespace}:{assertion_id}"
    
    def _canonical_key(self, namespace: str, fact_id: str) -> str:
        return f"canonical:{namespace}:{fact_id}"
    
    def store_raw_assertion(self, namespace: str, assertion: RawAssertion) -> None:
        key = self._raw_key(namespace, assertion.id)
        value = json.dumps(assertion.to_dict()).encode()
        self._client.put_kv(key, value)
        
        # Also add to vector collection if embedding exists
        if assertion.embedding:
            try:
                self._client.add_documents(
                    collection=f"{namespace}/{self._collection}",
                    documents=[{
                        "id": assertion.id,
                        "content": json.dumps(assertion.fact),
                        "embedding": assertion.embedding,
                        "metadata": {
                            "timestamp": assertion.timestamp,
                            "confidence": assertion.confidence,
                        },
                    }],
                )
            except Exception:
                pass
    
    def get_raw_assertion(self, namespace: str, assertion_id: str) -> Optional[RawAssertion]:
        key = self._raw_key(namespace, assertion_id)
        data = self._client.get_kv(key)
        if data is None:
            return None
        return RawAssertion.from_dict(json.loads(data.decode() if isinstance(data, bytes) else data))
    
    def scan_raw_assertions(self, namespace: str,
                             since: Optional[int] = None) -> Iterator[RawAssertion]:
        prefix = f"raw:{namespace}:"
        for key, value in self._client.scan_kv(prefix):
            assertion = RawAssertion.from_dict(
                json.loads(value.decode() if isinstance(value, bytes) else value)
            )
            if since is None or assertion.timestamp >= since:
                yield assertion
    
    def mark_superseded(self, namespace: str, assertion_id: str,
                        superseded_by: str) -> None:
        assertion = self.get_raw_assertion(namespace, assertion_id)
        if assertion:
            assertion.superseded_by = superseded_by
            key = self._raw_key(namespace, assertion_id)
            self._client.put_kv(key, json.dumps(assertion.to_dict()).encode())
    
    def store_canonical_fact(self, namespace: str, fact: CanonicalFact) -> None:
        key = self._canonical_key(namespace, fact.id)
        value = json.dumps(fact.to_dict()).encode()
        self._client.put_kv(key, value)
    
    def get_canonical_fact(self, namespace: str, fact_id: str) -> Optional[CanonicalFact]:
        key = self._canonical_key(namespace, fact_id)
        data = self._client.get_kv(key)
        if data is None:
            return None
        return CanonicalFact.from_dict(json.loads(data.decode() if isinstance(data, bytes) else data))
    
    def scan_canonical_facts(self, namespace: str,
                              current_only: bool = True) -> Iterator[CanonicalFact]:
        prefix = f"canonical:{namespace}:"
        for key, value in self._client.scan_kv(prefix):
            fact = CanonicalFact.from_dict(
                json.loads(value.decode() if isinstance(value, bytes) else value)
            )
            if current_only and not fact.is_current:
                continue
            yield fact
    
    def update_temporal_interval(self, namespace: str, fact_id: str,
                                   valid_until: int) -> None:
        fact = self.get_canonical_fact(namespace, fact_id)
        if fact:
            fact.valid_until = valid_until
            fact.last_updated = int(time.time() * 1000)
            self.store_canonical_fact(namespace, fact)
    
    def search_similar(self, namespace: str, embedding: List[float],
                       k: int = 10) -> List[Tuple[str, float]]:
        try:
            results = self._client.search(
                collection=f"{namespace}/{self._collection}",
                query_vector=embedding,
                k=k,
            )
            return [(r.id, r.distance) for r in results]
        except Exception:
            return []


# ============================================================================
# In-Memory Backend
# ============================================================================

class InMemoryConsolidationBackend(ConsolidationBackend):
    """
    In-memory backend for testing.
    """
    
    def __init__(self):
        self._raw: Dict[str, Dict[str, RawAssertion]] = defaultdict(dict)
        self._canonical: Dict[str, Dict[str, CanonicalFact]] = defaultdict(dict)
    
    def store_raw_assertion(self, namespace: str, assertion: RawAssertion) -> None:
        self._raw[namespace][assertion.id] = assertion
    
    def get_raw_assertion(self, namespace: str, assertion_id: str) -> Optional[RawAssertion]:
        return self._raw.get(namespace, {}).get(assertion_id)
    
    def scan_raw_assertions(self, namespace: str,
                             since: Optional[int] = None) -> Iterator[RawAssertion]:
        for assertion in self._raw.get(namespace, {}).values():
            if since is None or assertion.timestamp >= since:
                yield assertion
    
    def mark_superseded(self, namespace: str, assertion_id: str,
                        superseded_by: str) -> None:
        assertion = self.get_raw_assertion(namespace, assertion_id)
        if assertion:
            assertion.superseded_by = superseded_by
    
    def store_canonical_fact(self, namespace: str, fact: CanonicalFact) -> None:
        self._canonical[namespace][fact.id] = fact
    
    def get_canonical_fact(self, namespace: str, fact_id: str) -> Optional[CanonicalFact]:
        return self._canonical.get(namespace, {}).get(fact_id)
    
    def scan_canonical_facts(self, namespace: str,
                              current_only: bool = True) -> Iterator[CanonicalFact]:
        for fact in self._canonical.get(namespace, {}).values():
            if current_only and not fact.is_current:
                continue
            yield fact
    
    def update_temporal_interval(self, namespace: str, fact_id: str,
                                   valid_until: int) -> None:
        fact = self.get_canonical_fact(namespace, fact_id)
        if fact:
            fact.valid_until = valid_until
            fact.last_updated = int(time.time() * 1000)
    
    def search_similar(self, namespace: str, embedding: List[float],
                       k: int = 10) -> List[Tuple[str, float]]:
        """Simple cosine similarity search."""
        import math
        
        results = []
        for assertion in self._raw.get(namespace, {}).values():
            if assertion.embedding:
                # Compute cosine similarity
                dot = sum(a * b for a, b in zip(embedding, assertion.embedding))
                norm_a = math.sqrt(sum(a * a for a in embedding))
                norm_b = math.sqrt(sum(b * b for b in assertion.embedding))
                if norm_a > 0 and norm_b > 0:
                    similarity = dot / (norm_a * norm_b)
                    results.append((assertion.id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ============================================================================
# Consolidator
# ============================================================================

class Consolidator:
    """
    Event-sourced memory consolidation.
    
    Maintains append-only raw assertions and derives a canonical view.
    
    Features:
    - Append-only raw events (never delete)
    - Derived canonical view via clustering
    - Temporal interval updates for contradictions
    - Union-find for efficient deduplication
    
    Usage:
        consolidator = Consolidator.from_database(db, namespace="user_123")
        
        # Add new assertion
        consolidator.add(RawAssertion(
            fact={"subject": "alice", "predicate": "lives_in", "object": "SF"},
            embedding=[...],
        ))
        
        # Consolidate (update canonical view)
        consolidator.consolidate()
        
        # Handle contradiction (alice moved to NYC)
        consolidator.add_with_contradiction(
            new_fact={"subject": "alice", "predicate": "lives_in", "object": "NYC"},
            contradicts=["previous_assertion_id"],
        )
    """
    
    def __init__(
        self,
        backend: ConsolidationBackend,
        namespace: str,
        config: Optional[ConsolidationConfig] = None,
    ):
        self._backend = backend
        self._namespace = namespace
        self._config = config or ConsolidationConfig()
        self._union_find = UnionFind()
    
    @classmethod
    def from_database(
        cls,
        db: "Database",
        namespace: str,
        **kwargs,
    ) -> "Consolidator":
        """Create consolidator from embedded Database."""
        backend = FFIConsolidationBackend(db)
        return cls(backend, namespace, **kwargs)
    
    @classmethod
    def from_client(
        cls,
        client: "SochDBClient",
        namespace: str,
        **kwargs,
    ) -> "Consolidator":
        """Create consolidator from gRPC client."""
        backend = GrpcConsolidationBackend(client)
        return cls(backend, namespace, **kwargs)
    
    @classmethod
    def from_backend(
        cls,
        backend: ConsolidationBackend,
        namespace: str,
        **kwargs,
    ) -> "Consolidator":
        """Create consolidator with explicit backend."""
        return cls(backend, namespace, **kwargs)
    
    def add(self, assertion: RawAssertion) -> str:
        """
        Add a new raw assertion.
        
        Returns the assertion ID.
        """
        self._backend.store_raw_assertion(self._namespace, assertion)
        return assertion.id
    
    def add_with_contradiction(
        self,
        new_assertion: RawAssertion,
        contradicts: List[str],
    ) -> str:
        """
        Add an assertion that contradicts existing ones.
        
        Uses temporal interval updates rather than deletion.
        
        Args:
            new_assertion: The new assertion
            contradicts: IDs of contradicted assertions
            
        Returns:
            New assertion ID
        """
        now = int(time.time() * 1000)
        
        # Close intervals on contradicted assertions
        for old_id in contradicts:
            # Mark raw assertion as superseded
            self._backend.mark_superseded(
                self._namespace, old_id, new_assertion.id
            )
            
            # Close temporal interval on canonical fact
            if self._config.use_temporal_updates:
                # Find canonical fact containing this assertion
                for fact in self._backend.scan_canonical_facts(
                    self._namespace, current_only=True
                ):
                    if old_id in fact.support_set:
                        self._backend.update_temporal_interval(
                            self._namespace, fact.id, now
                        )
        
        # Store new assertion
        return self.add(new_assertion)
    
    def consolidate(self, incremental: bool = True) -> int:
        """
        Update the canonical view.
        
        Args:
            incremental: Only process new assertions
            
        Returns:
            Number of canonical facts updated
        """
        # Get assertions to process
        if incremental:
            # Find last consolidation timestamp
            last_ts = self._get_last_consolidation_ts()
            assertions = list(self._backend.scan_raw_assertions(
                self._namespace, since=last_ts
            ))
        else:
            assertions = list(self._backend.scan_raw_assertions(self._namespace))
        
        if not assertions:
            return 0
        
        # Build similarity clusters using union-find
        for assertion in assertions:
            if assertion.is_superseded:
                continue
            
            if assertion.embedding:
                # Find similar existing assertions
                similar = self._backend.search_similar(
                    self._namespace,
                    assertion.embedding,
                    k=10,
                )
                
                for other_id, similarity in similar:
                    if other_id == assertion.id:
                        continue
                    
                    if similarity >= self._config.similarity_threshold:
                        self._union_find.union(assertion.id, other_id)
        
        # Build canonical facts from clusters
        clusters = self._union_find.get_clusters()
        updated = 0
        
        for canonical_id, member_ids in clusters.items():
            if len(member_ids) > self._config.max_cluster_size:
                # Skip overly large clusters
                continue
            
            # Get all assertions in cluster
            members = []
            for mid in member_ids:
                assertion = self._backend.get_raw_assertion(self._namespace, mid)
                if assertion and not assertion.is_superseded:
                    if assertion.confidence >= self._config.min_confidence:
                        members.append(assertion)
            
            if not members:
                continue
            
            # Merge facts (use most recent as canonical)
            members.sort(key=lambda a: a.timestamp, reverse=True)
            canonical = members[0]
            
            # Calculate aggregated confidence
            avg_confidence = sum(m.confidence for m in members) / len(members)
            
            # Create/update canonical fact
            fact = CanonicalFact(
                id=canonical_id,
                merged_fact=canonical.fact,
                support_set=set(m.id for m in members),
                confidence=avg_confidence,
            )
            
            self._backend.store_canonical_fact(self._namespace, fact)
            updated += 1
        
        # Update consolidation timestamp
        self._set_last_consolidation_ts(int(time.time() * 1000))
        
        return updated
    
    def _get_last_consolidation_ts(self) -> int:
        """Get last consolidation timestamp."""
        key = f"consolidation_ts:{self._namespace}".encode()
        data = None
        
        # Try to get from backend (varies by type)
        if hasattr(self._backend, '_db'):
            data = self._backend._db.get(key)
        elif hasattr(self._backend, '_client'):
            data = self._backend._client.get_kv(key.decode())
        elif hasattr(self._backend, '_raw'):
            # In-memory backend
            return 0
        
        if data:
            return int(data.decode() if isinstance(data, bytes) else data)
        return 0
    
    def _set_last_consolidation_ts(self, ts: int) -> None:
        """Set last consolidation timestamp."""
        key = f"consolidation_ts:{self._namespace}".encode()
        value = str(ts).encode()
        
        if hasattr(self._backend, '_db'):
            self._backend._db.put(key, value)
        elif hasattr(self._backend, '_client'):
            self._backend._client.put_kv(key.decode(), value)
    
    def get_canonical_facts(self, current_only: bool = True) -> List[CanonicalFact]:
        """Get all canonical facts."""
        return list(self._backend.scan_canonical_facts(
            self._namespace, current_only=current_only
        ))
    
    def get_support(self, fact_id: str) -> List[RawAssertion]:
        """Get supporting assertions for a canonical fact."""
        fact = self._backend.get_canonical_fact(self._namespace, fact_id)
        if not fact:
            return []
        
        return [
            self._backend.get_raw_assertion(self._namespace, aid)
            for aid in fact.support_set
            if self._backend.get_raw_assertion(self._namespace, aid)
        ]
    
    def explain(self, fact_id: str) -> Dict[str, Any]:
        """Explain why we believe a fact (provenance + evidence)."""
        fact = self._backend.get_canonical_fact(self._namespace, fact_id)
        if not fact:
            return {"error": "Fact not found"}
        
        support = self.get_support(fact_id)
        
        return {
            "fact": fact.merged_fact,
            "confidence": fact.confidence,
            "valid_from": fact.valid_from,
            "valid_until": fact.valid_until,
            "is_current": fact.is_current,
            "evidence_count": len(support),
            "evidence": [
                {
                    "id": a.id,
                    "source": a.source,
                    "timestamp": a.timestamp,
                    "confidence": a.confidence,
                }
                for a in support
            ],
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_consolidator(
    backend,
    namespace: str,
    **kwargs,
) -> Consolidator:
    """
    Create a consolidator with auto-detected backend.
    
    Args:
        backend: Database, SochDBClient, or ConsolidationBackend
        namespace: Namespace for isolation
        **kwargs: Additional arguments
        
    Returns:
        Configured Consolidator
    """
    from ..database import Database
    from ..grpc_client import SochDBClient
    
    if isinstance(backend, Database):
        return Consolidator.from_database(backend, namespace, **kwargs)
    elif isinstance(backend, SochDBClient):
        return Consolidator.from_client(backend, namespace, **kwargs)
    elif isinstance(backend, ConsolidationBackend):
        return Consolidator.from_backend(backend, namespace, **kwargs)
    else:
        raise TypeError(f"Unknown backend type: {type(backend)}")
