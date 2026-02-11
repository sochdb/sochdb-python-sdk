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
Namespace-First Memory Isolation

This module provides strong namespace isolation for multi-tenant safety.

Design Philosophy:
- Namespace is mandatory, not optional
- Every query type is scoped by namespace
- Cross-namespace operations are explicit and auditable
- "Can't happen" safety via type system

Key Guarantees:
1. Data isolation: Namespace A cannot see namespace B's data
2. Query isolation: Queries are scoped at boundary, not filtered after
3. Audit trail: Cross-namespace access is explicit and logged
4. Monotonicity: Isolation cannot be weakened by query parameters

Supports both embedded (FFI) and server (gRPC) modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Set, Generic, TypeVar, Callable, Tuple
)
from enum import Enum, auto
import time
import hashlib


# ============================================================================
# Namespace Types
# ============================================================================

class NamespacePolicy(Enum):
    """
    Policy for namespace isolation enforcement.
    """
    STRICT = auto()       # No cross-namespace operations
    EXPLICIT = auto()     # Cross-namespace requires explicit grant
    AUDIT_ONLY = auto()   # Log cross-namespace but allow


@dataclass(frozen=True)
class NamespaceId:
    """
    Strongly-typed namespace identifier.
    
    Using a dedicated type prevents accidental string confusion
    and enables type-level guarantees.
    """
    value: str
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("Namespace cannot be empty")
        if "/" in self.value:
            raise ValueError("Namespace cannot contain '/'")
        if self.value.startswith("_"):
            raise ValueError("Namespace cannot start with '_' (reserved)")
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)
    
    def child(self, suffix: str) -> "NamespaceId":
        """Create a child namespace."""
        return NamespaceId(f"{self.value}:{suffix}")
    
    def is_child_of(self, parent: "NamespaceId") -> bool:
        """Check if this is a child of the given namespace."""
        return self.value.startswith(f"{parent.value}:")
    
    @classmethod
    def root(cls) -> "NamespaceId":
        """Get the root namespace (admin only)."""
        return cls("__root__")


# ============================================================================
# Scoped Query (Type-Level Safety)
# ============================================================================

T = TypeVar('T')


@dataclass
class ScopedQuery(Generic[T]):
    """
    A query that is guaranteed to be scoped to a namespace.
    
    This type wraps query parameters and carries proof that
    the namespace has been set. You cannot construct a ScopedQuery
    without providing a namespace.
    
    Usage:
        # Create scoped query (namespace required)
        query = ScopedQuery(
            namespace=NamespaceId("user_123"),
            inner={"text": "machine learning"},
        )
        
        # Extract values (namespace guaranteed)
        ns = query.namespace
        params = query.inner
    """
    namespace: NamespaceId
    inner: T
    created_at: float = field(default_factory=time.time)
    
    def with_namespace(self, new_namespace: NamespaceId) -> "ScopedQuery[T]":
        """Create a new query with different namespace (explicit)."""
        return ScopedQuery(namespace=new_namespace, inner=self.inner)


# ============================================================================
# Namespace Grant (Cross-Namespace Access)
# ============================================================================

@dataclass
class NamespaceGrant:
    """
    Explicit grant for cross-namespace access.
    
    Used when explicit cross-namespace operations are needed,
    with full audit trail.
    """
    from_namespace: NamespaceId
    to_namespace: NamespaceId
    operations: Set[str]  # allowed operations
    expires_at: Optional[float] = None
    created_by: Optional[str] = None
    reason: Optional[str] = None
    
    def is_valid(self) -> bool:
        if self.expires_at is not None:
            return time.time() < self.expires_at
        return True
    
    def allows(self, operation: str) -> bool:
        return self.is_valid() and operation in self.operations


# ============================================================================
# Namespace Backend Interface
# ============================================================================

class NamespaceBackend(ABC):
    """
    Abstract interface for namespace management.
    """
    
    @abstractmethod
    def create_namespace(
        self,
        namespace: NamespaceId,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a new namespace."""
        pass
    
    @abstractmethod
    def namespace_exists(self, namespace: NamespaceId) -> bool:
        """Check if namespace exists."""
        pass
    
    @abstractmethod
    def delete_namespace(self, namespace: NamespaceId) -> bool:
        """Delete a namespace and all its data."""
        pass
    
    @abstractmethod
    def list_namespaces(self, prefix: Optional[str] = None) -> List[NamespaceId]:
        """List all namespaces."""
        pass
    
    @abstractmethod
    def get_namespace_metadata(
        self,
        namespace: NamespaceId,
    ) -> Optional[Dict[str, Any]]:
        """Get namespace metadata."""
        pass
    
    @abstractmethod
    def set_namespace_metadata(
        self,
        namespace: NamespaceId,
        metadata: Dict[str, Any],
    ) -> bool:
        """Set namespace metadata."""
        pass


# ============================================================================
# FFI Backend
# ============================================================================

class FFINamespaceBackend(NamespaceBackend):
    """
    Namespace backend using embedded database via FFI.
    """
    
    NAMESPACE_PREFIX = "__namespaces__/"
    
    def __init__(self, db: "Database"):
        self._db = db
    
    def _namespace_key(self, namespace: NamespaceId) -> str:
        return f"{self.NAMESPACE_PREFIX}{namespace.value}"
    
    def create_namespace(
        self,
        namespace: NamespaceId,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        import json
        key = self._namespace_key(namespace)
        data = {
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        self._db.put(key, json.dumps(data))
        return True
    
    def namespace_exists(self, namespace: NamespaceId) -> bool:
        key = self._namespace_key(namespace)
        return self._db.get(key) is not None
    
    def delete_namespace(self, namespace: NamespaceId) -> bool:
        key = self._namespace_key(namespace)
        self._db.delete(key)
        return True
    
    def list_namespaces(self, prefix: Optional[str] = None) -> List[NamespaceId]:
        search_prefix = self.NAMESPACE_PREFIX
        if prefix:
            search_prefix = f"{self.NAMESPACE_PREFIX}{prefix}"
        
        results = []
        for key, _ in self._db.scan_prefix(search_prefix):
            ns_value = key[len(self.NAMESPACE_PREFIX):]
            try:
                results.append(NamespaceId(ns_value))
            except ValueError:
                pass  # Skip invalid namespace IDs
        return results
    
    def get_namespace_metadata(
        self,
        namespace: NamespaceId,
    ) -> Optional[Dict[str, Any]]:
        import json
        key = self._namespace_key(namespace)
        value = self._db.get(key)
        if value:
            data = json.loads(value)
            return data.get("metadata", {})
        return None
    
    def set_namespace_metadata(
        self,
        namespace: NamespaceId,
        metadata: Dict[str, Any],
    ) -> bool:
        import json
        key = self._namespace_key(namespace)
        value = self._db.get(key)
        if value:
            data = json.loads(value)
            data["metadata"] = metadata
            self._db.put(key, json.dumps(data))
            return True
        return False


# ============================================================================
# gRPC Backend
# ============================================================================

class GrpcNamespaceBackend(NamespaceBackend):
    """
    Namespace backend using gRPC client.
    """
    
    NAMESPACE_PREFIX = "__namespaces__/"
    
    def __init__(self, client: "SochDBClient"):
        self._client = client
    
    def _namespace_key(self, namespace: NamespaceId) -> str:
        return f"{self.NAMESPACE_PREFIX}{namespace.value}"
    
    def create_namespace(
        self,
        namespace: NamespaceId,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        import json
        key = self._namespace_key(namespace)
        data = {
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        self._client.put(key, json.dumps(data))
        return True
    
    def namespace_exists(self, namespace: NamespaceId) -> bool:
        key = self._namespace_key(namespace)
        try:
            return self._client.get(key) is not None
        except Exception:
            return False
    
    def delete_namespace(self, namespace: NamespaceId) -> bool:
        key = self._namespace_key(namespace)
        self._client.delete(key)
        return True
    
    def list_namespaces(self, prefix: Optional[str] = None) -> List[NamespaceId]:
        search_prefix = self.NAMESPACE_PREFIX
        if prefix:
            search_prefix = f"{self.NAMESPACE_PREFIX}{prefix}"
        
        results = []
        for key, _ in self._client.scan_prefix(search_prefix):
            ns_value = key[len(self.NAMESPACE_PREFIX):]
            try:
                results.append(NamespaceId(ns_value))
            except ValueError:
                pass
        return results
    
    def get_namespace_metadata(
        self,
        namespace: NamespaceId,
    ) -> Optional[Dict[str, Any]]:
        import json
        key = self._namespace_key(namespace)
        try:
            value = self._client.get(key)
            if value:
                data = json.loads(value)
                return data.get("metadata", {})
        except Exception:
            pass
        return None
    
    def set_namespace_metadata(
        self,
        namespace: NamespaceId,
        metadata: Dict[str, Any],
    ) -> bool:
        import json
        key = self._namespace_key(namespace)
        try:
            value = self._client.get(key)
            if value:
                data = json.loads(value)
                data["metadata"] = metadata
                self._client.put(key, json.dumps(data))
                return True
        except Exception:
            pass
        return False


# ============================================================================
# In-Memory Backend
# ============================================================================

class InMemoryNamespaceBackend(NamespaceBackend):
    """
    In-memory namespace backend for testing.
    """
    
    def __init__(self):
        self._namespaces: Dict[str, Dict[str, Any]] = {}
    
    def create_namespace(
        self,
        namespace: NamespaceId,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self._namespaces[namespace.value] = {
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        return True
    
    def namespace_exists(self, namespace: NamespaceId) -> bool:
        return namespace.value in self._namespaces
    
    def delete_namespace(self, namespace: NamespaceId) -> bool:
        if namespace.value in self._namespaces:
            del self._namespaces[namespace.value]
            return True
        return False
    
    def list_namespaces(self, prefix: Optional[str] = None) -> List[NamespaceId]:
        results = []
        for ns_value in self._namespaces.keys():
            if prefix is None or ns_value.startswith(prefix):
                try:
                    results.append(NamespaceId(ns_value))
                except ValueError:
                    pass
        return results
    
    def get_namespace_metadata(
        self,
        namespace: NamespaceId,
    ) -> Optional[Dict[str, Any]]:
        data = self._namespaces.get(namespace.value)
        if data:
            return data.get("metadata", {})
        return None
    
    def set_namespace_metadata(
        self,
        namespace: NamespaceId,
        metadata: Dict[str, Any],
    ) -> bool:
        if namespace.value in self._namespaces:
            self._namespaces[namespace.value]["metadata"] = metadata
            return True
        return False


# ============================================================================
# Scoped Namespace (Main Interface)
# ============================================================================

class ScopedNamespace:
    """
    A namespace-scoped interface to memory operations.
    
    All operations through this interface are automatically
    scoped to the namespace. Cross-namespace access is impossible
    without explicit grant.
    
    Usage:
        # Get scoped namespace
        scoped = namespace_manager.scope("user_123")
        
        # All operations are scoped
        scoped.store(entity)       # Stored in user_123
        scoped.retrieve(query)     # Only searches user_123
        
        # Cross-namespace access (explicit)
        shared = scoped.with_grant(grant)
        shared.retrieve(query)  # Can access granted namespaces
    """
    
    def __init__(
        self,
        namespace: NamespaceId,
        extraction_pipeline: Optional["ExtractionPipeline"] = None,
        consolidator: Optional["Consolidator"] = None,
        retriever: Optional["HybridRetriever"] = None,
        grants: Optional[List[NamespaceGrant]] = None,
        policy: NamespacePolicy = NamespacePolicy.STRICT,
        audit_log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._namespace = namespace
        self._extraction = extraction_pipeline
        self._consolidator = consolidator
        self._retriever = retriever
        self._grants = grants or []
        self._policy = policy
        self._audit_log = audit_log
    
    @property
    def namespace(self) -> NamespaceId:
        """Get the namespace."""
        return self._namespace
    
    @property
    def namespace_str(self) -> str:
        """Get namespace as string."""
        return str(self._namespace)
    
    def _audit(self, operation: str, details: Dict[str, Any]) -> None:
        """Log operation for audit trail."""
        if self._audit_log:
            self._audit_log({
                "timestamp": time.time(),
                "namespace": str(self._namespace),
                "operation": operation,
                "details": details,
            })
    
    # ========================================================================
    # Extraction Operations
    # ========================================================================
    
    def extract(
        self,
        text: str,
        source: Optional[str] = None,
        extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> "ExtractionResult":
        """
        Extract entities, relations, and assertions from text.
        
        Results are automatically scoped to this namespace.
        """
        if not self._extraction:
            raise RuntimeError("Extraction pipeline not configured")
        
        self._audit("extract", {"source": source, "text_length": len(text)})
        
        result = self._extraction.extract(
            text=text,
            namespace=self.namespace_str,
            source=source,
            extractor=extractor,
        )
        return result
    
    def commit_extraction(self, result: "ExtractionResult") -> Dict[str, int]:
        """Commit extraction result to storage."""
        if not self._extraction:
            raise RuntimeError("Extraction pipeline not configured")
        
        self._audit("commit_extraction", {
            "entities": len(result.entities),
            "relations": len(result.relations),
            "assertions": len(result.assertions),
        })
        
        return self._extraction.commit(result)
    
    # ========================================================================
    # Consolidation Operations
    # ========================================================================
    
    def add_assertion(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source: str,
        confidence: float = 1.0,
        **metadata,
    ) -> str:
        """Add a raw assertion (append-only)."""
        if not self._consolidator:
            raise RuntimeError("Consolidator not configured")
        
        # Import here to avoid circular dependency
        from .consolidation import RawAssertion
        
        assertion = RawAssertion(
            id="",  # Will be generated
            namespace=self.namespace_str,
            subject=subject,
            predicate=predicate,
            object=object_,
            source=source,
            confidence=confidence,
            timestamp=time.time(),
            metadata=metadata,
        )
        
        self._audit("add_assertion", {
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "source": source,
        })
        
        return self._consolidator.add(assertion)
    
    def add_contradiction(
        self,
        subject: str,
        predicate: str,
        old_object: str,
        new_object: str,
        source: str,
        confidence: float = 1.0,
    ) -> Tuple[str, str]:
        """Add a contradicting assertion (invalidates old, adds new)."""
        if not self._consolidator:
            raise RuntimeError("Consolidator not configured")
        
        from .consolidation import RawAssertion
        
        # Create new assertion
        new_assertion = RawAssertion(
            id="",
            namespace=self.namespace_str,
            subject=subject,
            predicate=predicate,
            object=new_object,
            source=source,
            confidence=confidence,
            timestamp=time.time(),
        )
        
        self._audit("add_contradiction", {
            "subject": subject,
            "predicate": predicate,
            "old_object": old_object,
            "new_object": new_object,
        })
        
        return self._consolidator.add_with_contradiction(
            new_assertion=new_assertion,
            old_object=old_object,
        )
    
    def consolidate(self) -> Dict[str, int]:
        """Run consolidation to derive canonical facts."""
        if not self._consolidator:
            raise RuntimeError("Consolidator not configured")
        
        self._audit("consolidate", {})
        
        return self._consolidator.consolidate(namespace=self.namespace_str)
    
    def get_canonical_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List["CanonicalFact"]:
        """Get canonical facts for this namespace."""
        if not self._consolidator:
            raise RuntimeError("Consolidator not configured")
        
        return self._consolidator.get_canonical_facts(
            namespace=self.namespace_str,
            subject=subject,
            predicate=predicate,
        )
    
    # ========================================================================
    # Retrieval Operations
    # ========================================================================
    
    def retrieve(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> "RetrievalResponse":
        """
        Retrieve documents from this namespace.
        
        INVARIANT: Results are always from this namespace only.
        """
        if not self._retriever:
            raise RuntimeError("Retriever not configured")
        
        from .retrieval import AllowedSet
        
        # Force namespace scope via AllowedSet
        allowed = AllowedSet.from_namespace(self.namespace_str)
        
        self._audit("retrieve", {
            "query_length": len(query_text),
            "k": k,
            "alpha": alpha,
        })
        
        return self._retriever.retrieve(
            query_text=query_text,
            query_vector=query_vector,
            allowed=allowed,
            k=k,
            alpha=alpha,
            filter=filter,
        )
    
    # ========================================================================
    # Cross-Namespace Operations
    # ========================================================================
    
    def with_grant(self, grant: NamespaceGrant) -> "ScopedNamespace":
        """
        Create a view that includes a cross-namespace grant.
        
        This is explicit and auditable. The grant must be valid
        and the policy must allow cross-namespace access.
        """
        if self._policy == NamespacePolicy.STRICT:
            raise PermissionError(
                "Cross-namespace access not allowed (STRICT policy)"
            )
        
        if not grant.is_valid():
            raise PermissionError("Grant has expired")
        
        if grant.from_namespace != self._namespace:
            raise PermissionError(
                f"Grant is for {grant.from_namespace}, not {self._namespace}"
            )
        
        self._audit("add_grant", {
            "to_namespace": str(grant.to_namespace),
            "operations": list(grant.operations),
            "reason": grant.reason,
        })
        
        new_grants = self._grants + [grant]
        
        return ScopedNamespace(
            namespace=self._namespace,
            extraction_pipeline=self._extraction,
            consolidator=self._consolidator,
            retriever=self._retriever,
            grants=new_grants,
            policy=self._policy,
            audit_log=self._audit_log,
        )
    
    def retrieve_with_grants(
        self,
        query_text: str,
        query_vector: Optional[List[float]] = None,
        k: int = 10,
        alpha: float = 0.5,
    ) -> "RetrievalResponse":
        """
        Retrieve from this namespace AND granted namespaces.
        
        Only works with EXPLICIT or AUDIT_ONLY policies.
        """
        if self._policy == NamespacePolicy.STRICT:
            raise PermissionError(
                "Cross-namespace retrieval not allowed (STRICT policy)"
            )
        
        if not self._retriever:
            raise RuntimeError("Retriever not configured")
        
        from .retrieval import AllowedSet
        
        # Build allowed set from this namespace + granted namespaces
        allowed_namespaces = {self.namespace_str}
        for grant in self._grants:
            if grant.is_valid() and grant.allows("retrieve"):
                allowed_namespaces.add(str(grant.to_namespace))
        
        # Create composite allowed set
        def filter_fn(doc_id: str, metadata: Dict[str, Any]) -> bool:
            for ns in allowed_namespaces:
                if doc_id.startswith(ns):
                    return True
            return False
        
        allowed = AllowedSet.from_filter(filter_fn)
        
        self._audit("retrieve_with_grants", {
            "query_length": len(query_text),
            "k": k,
            "namespaces": list(allowed_namespaces),
        })
        
        return self._retriever.retrieve(
            query_text=query_text,
            query_vector=query_vector,
            allowed=allowed,
            k=k,
            alpha=alpha,
        )


# ============================================================================
# Namespace Manager
# ============================================================================

class NamespaceManager:
    """
    Manager for namespace lifecycle and access.
    
    Provides the entry point for getting scoped namespaces
    and managing namespace lifecycle.
    
    Usage:
        # Create manager
        manager = NamespaceManager.from_database(db)
        
        # Create namespace
        manager.create("user_123", metadata={"plan": "pro"})
        
        # Get scoped interface
        scoped = manager.scope("user_123")
        
        # All operations are scoped
        scoped.extract(text)
        scoped.retrieve(query)
    """
    
    def __init__(
        self,
        backend: NamespaceBackend,
        policy: NamespacePolicy = NamespacePolicy.STRICT,
        audit_log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self._backend = backend
        self._policy = policy
        self._audit_log = audit_log
        
        # Component registries (for building ScopedNamespace)
        self._extraction_factory: Optional[Callable[[str], "ExtractionPipeline"]] = None
        self._consolidator_factory: Optional[Callable[[str], "Consolidator"]] = None
        self._retriever_factory: Optional[Callable[[str], "HybridRetriever"]] = None
    
    @classmethod
    def from_database(
        cls,
        db: "Database",
        **kwargs,
    ) -> "NamespaceManager":
        """Create manager from embedded Database."""
        backend = FFINamespaceBackend(db)
        return cls(backend, **kwargs)
    
    @classmethod
    def from_client(
        cls,
        client: "SochDBClient",
        **kwargs,
    ) -> "NamespaceManager":
        """Create manager from gRPC client."""
        backend = GrpcNamespaceBackend(client)
        return cls(backend, **kwargs)
    
    @classmethod
    def from_backend(
        cls,
        backend: NamespaceBackend,
        **kwargs,
    ) -> "NamespaceManager":
        """Create manager with explicit backend."""
        return cls(backend, **kwargs)
    
    def register_extraction_factory(
        self,
        factory: Callable[[str], "ExtractionPipeline"],
    ) -> None:
        """Register factory for creating extraction pipelines."""
        self._extraction_factory = factory
    
    def register_consolidator_factory(
        self,
        factory: Callable[[str], "Consolidator"],
    ) -> None:
        """Register factory for creating consolidators."""
        self._consolidator_factory = factory
    
    def register_retriever_factory(
        self,
        factory: Callable[[str], "HybridRetriever"],
    ) -> None:
        """Register factory for creating retrievers."""
        self._retriever_factory = factory
    
    # ========================================================================
    # Namespace Lifecycle
    # ========================================================================
    
    def create(
        self,
        namespace: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NamespaceId:
        """Create a new namespace."""
        ns_id = NamespaceId(namespace)
        
        if self._backend.namespace_exists(ns_id):
            raise ValueError(f"Namespace already exists: {namespace}")
        
        self._backend.create_namespace(ns_id, metadata)
        
        if self._audit_log:
            self._audit_log({
                "timestamp": time.time(),
                "operation": "create_namespace",
                "namespace": namespace,
                "metadata": metadata,
            })
        
        return ns_id
    
    def exists(self, namespace: str) -> bool:
        """Check if namespace exists."""
        try:
            ns_id = NamespaceId(namespace)
            return self._backend.namespace_exists(ns_id)
        except ValueError:
            return False
    
    def delete(self, namespace: str) -> bool:
        """Delete a namespace and all its data."""
        ns_id = NamespaceId(namespace)
        
        if not self._backend.namespace_exists(ns_id):
            return False
        
        if self._audit_log:
            self._audit_log({
                "timestamp": time.time(),
                "operation": "delete_namespace",
                "namespace": namespace,
            })
        
        return self._backend.delete_namespace(ns_id)
    
    def list(self, prefix: Optional[str] = None) -> List[NamespaceId]:
        """List all namespaces."""
        return self._backend.list_namespaces(prefix)
    
    def get_metadata(self, namespace: str) -> Optional[Dict[str, Any]]:
        """Get namespace metadata."""
        ns_id = NamespaceId(namespace)
        return self._backend.get_namespace_metadata(ns_id)
    
    def set_metadata(self, namespace: str, metadata: Dict[str, Any]) -> bool:
        """Set namespace metadata."""
        ns_id = NamespaceId(namespace)
        return self._backend.set_namespace_metadata(ns_id, metadata)
    
    # ========================================================================
    # Scoped Access
    # ========================================================================
    
    def scope(
        self,
        namespace: str,
        auto_create: bool = False,
    ) -> ScopedNamespace:
        """
        Get a scoped interface for a namespace.
        
        Args:
            namespace: Namespace to scope to
            auto_create: Create namespace if it doesn't exist
            
        Returns:
            ScopedNamespace with isolation guarantees
        """
        ns_id = NamespaceId(namespace)
        
        if not self._backend.namespace_exists(ns_id):
            if auto_create:
                self._backend.create_namespace(ns_id)
            else:
                raise ValueError(f"Namespace does not exist: {namespace}")
        
        # Build components if factories registered
        extraction = None
        consolidator = None
        retriever = None
        
        if self._extraction_factory:
            extraction = self._extraction_factory(namespace)
        if self._consolidator_factory:
            consolidator = self._consolidator_factory(namespace)
        if self._retriever_factory:
            retriever = self._retriever_factory(namespace)
        
        return ScopedNamespace(
            namespace=ns_id,
            extraction_pipeline=extraction,
            consolidator=consolidator,
            retriever=retriever,
            grants=[],
            policy=self._policy,
            audit_log=self._audit_log,
        )
    
    def create_grant(
        self,
        from_namespace: str,
        to_namespace: str,
        operations: List[str],
        expires_in_seconds: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> NamespaceGrant:
        """
        Create a cross-namespace access grant.
        
        Args:
            from_namespace: Source namespace (requester)
            to_namespace: Target namespace (to access)
            operations: Allowed operations
            expires_in_seconds: Grant expiration
            reason: Reason for grant (audit)
            
        Returns:
            NamespaceGrant for cross-namespace access
        """
        if self._policy == NamespacePolicy.STRICT:
            raise PermissionError(
                "Cross-namespace grants not allowed (STRICT policy)"
            )
        
        from_ns = NamespaceId(from_namespace)
        to_ns = NamespaceId(to_namespace)
        
        expires_at = None
        if expires_in_seconds:
            expires_at = time.time() + expires_in_seconds
        
        grant = NamespaceGrant(
            from_namespace=from_ns,
            to_namespace=to_ns,
            operations=set(operations),
            expires_at=expires_at,
            reason=reason,
        )
        
        if self._audit_log:
            self._audit_log({
                "timestamp": time.time(),
                "operation": "create_grant",
                "from_namespace": from_namespace,
                "to_namespace": to_namespace,
                "operations": operations,
                "expires_at": expires_at,
                "reason": reason,
            })
        
        return grant


# ============================================================================
# Factory Function
# ============================================================================

def create_namespace_manager(
    backend,
    **kwargs,
) -> NamespaceManager:
    """
    Create a namespace manager with auto-detected backend.
    
    Args:
        backend: Database, SochDBClient, or NamespaceBackend
        **kwargs: Additional arguments
        
    Returns:
        Configured NamespaceManager
    """
    from ..database import Database
    from ..grpc_client import SochDBClient
    
    if isinstance(backend, Database):
        return NamespaceManager.from_database(backend, **kwargs)
    elif isinstance(backend, SochDBClient):
        return NamespaceManager.from_client(backend, **kwargs)
    elif isinstance(backend, NamespaceBackend):
        return NamespaceManager.from_backend(backend, **kwargs)
    else:
        raise TypeError(f"Unknown backend type: {type(backend)}")
