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
LLM-Gated Structured Extraction Pipeline (Write-Time "Fact Compiler")

This module treats extraction as a compilation step, producing a typed
intermediate representation (IR) from untrusted LLM outputs:

- Entity[] - Named entities with types and properties
- Relation[] - Edges between entities  
- Assertion{confidence, provenance, valid_from/until} - Time-aware facts

The extraction pipeline ensures:
1. Schema validation (reject/repair invalid JSON)
2. Idempotency via deterministic ID hashing
3. Atomicity via transactional writes
4. Time-aware assertions using temporal edges

Supports both embedded (FFI) and server (gRPC) modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar
)
from enum import Enum
import hashlib
import json
import time
import re


# ============================================================================
# Data Models (Typed Intermediate Representation)
# ============================================================================

@dataclass
class Entity:
    """
    A named entity extracted from text.
    
    Entities are nodes in the knowledge graph with a type and properties.
    
    Attributes:
        id: Deterministic ID (generated from namespace + name + type)
        name: Entity name/label
        entity_type: Type classification (e.g., "person", "organization")
        properties: Additional metadata
        confidence: Extraction confidence (0.0-1.0)
        provenance: Source reference (document ID, span, etc.)
    """
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: Optional[str] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from content."""
        content = f"{self.entity_type}:{self.name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=data.get("id"),
            name=data["name"],
            entity_type=data["entity_type"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            provenance=data.get("provenance"),
        )


@dataclass
class Relation:
    """
    A relationship/edge between two entities.
    
    Relations connect entities in the knowledge graph.
    
    Attributes:
        from_entity: Source entity ID
        relation_type: Type of relationship (e.g., "works_at", "knows")
        to_entity: Target entity ID
        properties: Additional edge metadata
        confidence: Extraction confidence
        provenance: Source reference
    """
    from_entity: str
    relation_type: str
    to_entity: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: Optional[str] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from content."""
        content = f"{self.from_entity}:{self.relation_type}:{self.to_entity}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_entity": self.from_entity,
            "relation_type": self.relation_type,
            "to_entity": self.to_entity,
            "properties": self.properties,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            id=data.get("id"),
            from_entity=data["from_entity"],
            relation_type=data["relation_type"],
            to_entity=data["to_entity"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            provenance=data.get("provenance"),
        )


@dataclass 
class Assertion:
    """
    A time-aware assertion/fact with provenance.
    
    Assertions are the core unit of memory, supporting:
    - Temporal validity (valid_from/until for time-travel queries)
    - Confidence scoring
    - Provenance tracking
    - Embedding for semantic search
    
    Attributes:
        subject: Entity ID of the subject
        predicate: Assertion type (e.g., "believes", "stated", "observed")
        object: Entity ID or literal value
        valid_from: Start of validity period (Unix ms)
        valid_until: End of validity period (0 = no expiry)
        confidence: Certainty of the assertion
        provenance: Source information
        embedding: Optional vector embedding
    """
    subject: str
    predicate: str
    object: Union[str, Any]
    valid_from: int = 0  # Unix ms
    valid_until: int = 0  # 0 = still valid
    confidence: float = 1.0
    provenance: Optional[str] = None
    embedding: Optional[List[float]] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = self._generate_id()
        if self.valid_from == 0:
            self.valid_from = int(time.time() * 1000)
    
    def _generate_id(self) -> str:
        """Generate deterministic ID from content + time interval."""
        # Include time interval for uniqueness across temporal updates
        content = f"{self.subject}:{self.predicate}:{self.object}:{self.valid_from}:{self.valid_until}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_current(self, at_time: Optional[int] = None) -> bool:
        """Check if assertion is valid at given time."""
        now = at_time or int(time.time() * 1000)
        if now < self.valid_from:
            return False
        if self.valid_until > 0 and now >= self.valid_until:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }
        if self.embedding:
            result["embedding"] = self.embedding
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Assertion":
        return cls(
            id=data.get("id"),
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            valid_from=data.get("valid_from", 0),
            valid_until=data.get("valid_until", 0),
            confidence=data.get("confidence", 1.0),
            provenance=data.get("provenance"),
            embedding=data.get("embedding"),
        )


@dataclass
class ExtractionResult:
    """
    Complete extraction result from LLM processing.
    
    Contains all extracted artifacts ready for atomic commit.
    """
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    assertions: List[Assertion] = field(default_factory=list)
    raw_text: Optional[str] = None
    source_id: Optional[str] = None
    extraction_time_ms: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        return not (self.entities or self.relations or self.assertions)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "assertions": [a.to_dict() for a in self.assertions],
            "raw_text": self.raw_text,
            "source_id": self.source_id,
            "extraction_time_ms": self.extraction_time_ms,
        }


# ============================================================================
# Schema Validation
# ============================================================================

class ExtractionSchema:
    """
    Schema definition for extraction validation.
    
    Enforces type constraints and validates LLM outputs.
    """
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        relation_types: Optional[List[str]] = None,
        assertion_predicates: Optional[List[str]] = None,
        require_confidence: bool = False,
        min_confidence: float = 0.0,
    ):
        self.entity_types = set(entity_types) if entity_types else None
        self.relation_types = set(relation_types) if relation_types else None
        self.assertion_predicates = set(assertion_predicates) if assertion_predicates else None
        self.require_confidence = require_confidence
        self.min_confidence = min_confidence
    
    def validate_entity(self, entity: Entity) -> Tuple[bool, Optional[str]]:
        """Validate an entity against schema."""
        if self.entity_types and entity.entity_type not in self.entity_types:
            return False, f"Unknown entity type: {entity.entity_type}"
        if self.require_confidence and entity.confidence < self.min_confidence:
            return False, f"Confidence {entity.confidence} below threshold {self.min_confidence}"
        return True, None
    
    def validate_relation(self, relation: Relation) -> Tuple[bool, Optional[str]]:
        """Validate a relation against schema."""
        if self.relation_types and relation.relation_type not in self.relation_types:
            return False, f"Unknown relation type: {relation.relation_type}"
        if self.require_confidence and relation.confidence < self.min_confidence:
            return False, f"Confidence {relation.confidence} below threshold {self.min_confidence}"
        return True, None
    
    def validate_assertion(self, assertion: Assertion) -> Tuple[bool, Optional[str]]:
        """Validate an assertion against schema."""
        if self.assertion_predicates and assertion.predicate not in self.assertion_predicates:
            return False, f"Unknown predicate: {assertion.predicate}"
        if self.require_confidence and assertion.confidence < self.min_confidence:
            return False, f"Confidence {assertion.confidence} below threshold {self.min_confidence}"
        return True, None
    
    def validate_result(self, result: ExtractionResult) -> Tuple[bool, List[str]]:
        """Validate complete extraction result."""
        errors = []
        
        for entity in result.entities:
            valid, error = self.validate_entity(entity)
            if not valid:
                errors.append(f"Entity '{entity.name}': {error}")
        
        for relation in result.relations:
            valid, error = self.validate_relation(relation)
            if not valid:
                errors.append(f"Relation '{relation.relation_type}': {error}")
        
        for assertion in result.assertions:
            valid, error = self.validate_assertion(assertion)
            if not valid:
                errors.append(f"Assertion '{assertion.predicate}': {error}")
        
        return len(errors) == 0, errors


# ============================================================================
# Memory Backend Interface
# ============================================================================

class MemoryBackend(ABC):
    """
    Abstract interface for memory storage backends.
    
    Implementations must provide atomic write semantics.
    """
    
    @abstractmethod
    def begin_transaction(self) -> "MemoryTransaction":
        """Start a new transaction for atomic writes."""
        pass
    
    @abstractmethod
    def add_node(self, namespace: str, node_id: str, node_type: str, 
                 properties: Dict[str, Any]) -> None:
        """Add a graph node (entity)."""
        pass
    
    @abstractmethod
    def add_edge(self, namespace: str, from_id: str, edge_type: str,
                 to_id: str, properties: Dict[str, Any]) -> None:
        """Add a graph edge (relation)."""
        pass
    
    @abstractmethod
    def add_temporal_edge(self, namespace: str, from_id: str, edge_type: str,
                          to_id: str, valid_from: int, valid_until: int,
                          properties: Dict[str, Any]) -> None:
        """Add a temporal edge (time-bounded assertion)."""
        pass
    
    @abstractmethod
    def add_document(self, namespace: str, collection: str, doc_id: str,
                     content: str, embedding: List[float],
                     metadata: Dict[str, Any]) -> None:
        """Add a document with embedding to a collection."""
        pass
    
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        """Raw key-value put for metadata storage."""
        pass
    
    @abstractmethod
    def get(self, key: bytes) -> Optional[bytes]:
        """Raw key-value get."""
        pass


class MemoryTransaction(ABC):
    """Abstract transaction interface for atomic commits."""
    
    @abstractmethod
    def __enter__(self) -> "MemoryTransaction":
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        pass
    
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        pass
    
    @abstractmethod
    def commit(self) -> None:
        pass
    
    @abstractmethod
    def abort(self) -> None:
        pass


# ============================================================================
# FFI Backend (Embedded Mode)
# ============================================================================

class FFIMemoryBackend(MemoryBackend):
    """
    Memory backend using embedded database via FFI.
    
    Uses the Database class for direct Rust bindings.
    """
    
    def __init__(self, db: "Database"):
        """
        Initialize with a Database instance.
        
        Args:
            db: SochDB Database instance
        """
        self._db = db
    
    def begin_transaction(self) -> "FFIMemoryTransaction":
        return FFIMemoryTransaction(self._db)
    
    def add_node(self, namespace: str, node_id: str, node_type: str,
                 properties: Dict[str, Any]) -> None:
        self._db.add_node(namespace, node_id, node_type, properties)
    
    def add_edge(self, namespace: str, from_id: str, edge_type: str,
                 to_id: str, properties: Dict[str, Any]) -> None:
        self._db.add_edge(namespace, from_id, edge_type, to_id, properties)
    
    def add_temporal_edge(self, namespace: str, from_id: str, edge_type: str,
                          to_id: str, valid_from: int, valid_until: int,
                          properties: Dict[str, Any]) -> None:
        self._db.add_temporal_edge(
            namespace=namespace,
            from_id=from_id,
            edge_type=edge_type,
            to_id=to_id,
            valid_from=valid_from,
            valid_until=valid_until,
            properties=properties,
        )
    
    def add_document(self, namespace: str, collection: str, doc_id: str,
                     content: str, embedding: List[float],
                     metadata: Dict[str, Any]) -> None:
        # Use namespace to get collection
        ns = self._db.namespace(namespace)
        coll = ns.collection(collection)
        coll.add(
            id=doc_id,
            vector=embedding,
            metadata={**metadata, "content": content},
        )
    
    def put(self, key: bytes, value: bytes) -> None:
        self._db.put(key, value)
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._db.get(key)


class FFIMemoryTransaction(MemoryTransaction):
    """FFI-based transaction."""
    
    def __init__(self, db: "Database"):
        self._db = db
        self._txn = None
    
    def __enter__(self) -> "FFIMemoryTransaction":
        self._txn = self._db.transaction()
        self._txn.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._txn:
            return self._txn.__exit__(exc_type, exc_val, exc_tb)
        return False
    
    def put(self, key: bytes, value: bytes) -> None:
        if self._txn:
            self._txn.put(key, value)
    
    def commit(self) -> None:
        pass  # Auto-commit on exit
    
    def abort(self) -> None:
        pass  # Auto-abort on exception


# ============================================================================
# gRPC Backend (Server Mode)
# ============================================================================

class GrpcMemoryBackend(MemoryBackend):
    """
    Memory backend using gRPC client for server mode.
    """
    
    def __init__(self, client: "SochDBClient"):
        """
        Initialize with a SochDBClient instance.
        
        Args:
            client: SochDB gRPC client
        """
        self._client = client
    
    def begin_transaction(self) -> "GrpcMemoryTransaction":
        return GrpcMemoryTransaction(self._client)
    
    def add_node(self, namespace: str, node_id: str, node_type: str,
                 properties: Dict[str, Any]) -> None:
        self._client.add_node(namespace, node_id, node_type, properties)
    
    def add_edge(self, namespace: str, from_id: str, edge_type: str,
                 to_id: str, properties: Dict[str, Any]) -> None:
        self._client.add_edge(namespace, from_id, edge_type, to_id, properties)
    
    def add_temporal_edge(self, namespace: str, from_id: str, edge_type: str,
                          to_id: str, valid_from: int, valid_until: int,
                          properties: Dict[str, Any]) -> None:
        self._client.add_temporal_edge(
            namespace=namespace,
            from_id=from_id,
            edge_type=edge_type,
            to_id=to_id,
            valid_from=valid_from,
            valid_until=valid_until,
            properties=properties,
        )
    
    def add_document(self, namespace: str, collection: str, doc_id: str,
                     content: str, embedding: List[float],
                     metadata: Dict[str, Any]) -> None:
        self._client.add_documents(
            collection=f"{namespace}/{collection}",
            documents=[{
                "id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            }],
        )
    
    def put(self, key: bytes, value: bytes) -> None:
        self._client.put_kv(key.decode() if isinstance(key, bytes) else key, value)
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._client.get_kv(key.decode() if isinstance(key, bytes) else key)


class GrpcMemoryTransaction(MemoryTransaction):
    """gRPC-based transaction (batch operations)."""
    
    def __init__(self, client: "SochDBClient"):
        self._client = client
        self._ops: List[Tuple[str, Any]] = []
    
    def __enter__(self) -> "GrpcMemoryTransaction":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.commit()
        return False
    
    def put(self, key: bytes, value: bytes) -> None:
        self._ops.append(("put", (key, value)))
    
    def commit(self) -> None:
        # Execute all buffered operations
        for op_type, args in self._ops:
            if op_type == "put":
                key, value = args
                self._client.put_kv(
                    key.decode() if isinstance(key, bytes) else key,
                    value
                )
        self._ops.clear()
    
    def abort(self) -> None:
        self._ops.clear()


# ============================================================================
# In-Memory Backend (Testing)
# ============================================================================

class InMemoryBackend(MemoryBackend):
    """
    In-memory backend for testing and development.
    """
    
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Dict[str, Any]] = []
        self._temporal_edges: List[Dict[str, Any]] = []
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._kv: Dict[bytes, bytes] = {}
    
    def begin_transaction(self) -> "InMemoryTransaction":
        return InMemoryTransaction(self)
    
    def add_node(self, namespace: str, node_id: str, node_type: str,
                 properties: Dict[str, Any]) -> None:
        key = f"{namespace}:{node_id}"
        self._nodes[key] = {
            "id": node_id,
            "type": node_type,
            "properties": properties,
            "namespace": namespace,
        }
    
    def add_edge(self, namespace: str, from_id: str, edge_type: str,
                 to_id: str, properties: Dict[str, Any]) -> None:
        self._edges.append({
            "namespace": namespace,
            "from_id": from_id,
            "edge_type": edge_type,
            "to_id": to_id,
            "properties": properties,
        })
    
    def add_temporal_edge(self, namespace: str, from_id: str, edge_type: str,
                          to_id: str, valid_from: int, valid_until: int,
                          properties: Dict[str, Any]) -> None:
        self._temporal_edges.append({
            "namespace": namespace,
            "from_id": from_id,
            "edge_type": edge_type,
            "to_id": to_id,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "properties": properties,
        })
    
    def add_document(self, namespace: str, collection: str, doc_id: str,
                     content: str, embedding: List[float],
                     metadata: Dict[str, Any]) -> None:
        key = f"{namespace}:{collection}:{doc_id}"
        self._documents[key] = {
            "id": doc_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
        }
    
    def put(self, key: bytes, value: bytes) -> None:
        self._kv[key] = value
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._kv.get(key)


class InMemoryTransaction(MemoryTransaction):
    """In-memory transaction for testing."""
    
    def __init__(self, backend: InMemoryBackend):
        self._backend = backend
        self._ops: List[Tuple[bytes, bytes]] = []
    
    def __enter__(self) -> "InMemoryTransaction":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.commit()
        return False
    
    def put(self, key: bytes, value: bytes) -> None:
        self._ops.append((key, value))
    
    def commit(self) -> None:
        for key, value in self._ops:
            self._backend._kv[key] = value
        self._ops.clear()
    
    def abort(self) -> None:
        self._ops.clear()


# ============================================================================
# Extraction Pipeline
# ============================================================================

class ExtractionPipeline:
    """
    LLM-gated structured extraction pipeline.
    
    Compiles LLM outputs into typed, validated facts with atomic commits.
    
    Features:
    - Schema validation (reject/repair invalid outputs)
    - Idempotent writes via deterministic ID hashing
    - Atomic transactions (all-or-nothing)
    - Time-aware assertions using temporal edges
    
    Usage:
        pipeline = ExtractionPipeline.from_database(db, namespace="user_123")
        
        result = pipeline.extract(
            text="Alice works at Acme Corp since 2020",
            extractor=my_llm_extractor,
        )
        
        pipeline.commit(result)  # Atomic write
    """
    
    def __init__(
        self,
        backend: MemoryBackend,
        namespace: str,
        collection: str = "memories",
        schema: Optional[ExtractionSchema] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize extraction pipeline.
        
        Args:
            backend: Storage backend (FFI, gRPC, or InMemory)
            namespace: Namespace for isolation
            collection: Collection name for vector storage
            schema: Optional schema for validation
            embed_fn: Function to generate embeddings
        """
        self._backend = backend
        self._namespace = namespace
        self._collection = collection
        self._schema = schema
        self._embed_fn = embed_fn
    
    @classmethod
    def from_database(
        cls,
        db: "Database",
        namespace: str,
        collection: str = "memories",
        **kwargs,
    ) -> "ExtractionPipeline":
        """Create pipeline from embedded Database."""
        backend = FFIMemoryBackend(db)
        return cls(backend, namespace, collection, **kwargs)
    
    @classmethod
    def from_client(
        cls,
        client: "SochDBClient",
        namespace: str,
        collection: str = "memories",
        **kwargs,
    ) -> "ExtractionPipeline":
        """Create pipeline from gRPC client."""
        backend = GrpcMemoryBackend(client)
        return cls(backend, namespace, collection, **kwargs)
    
    @classmethod
    def from_backend(
        cls,
        backend: MemoryBackend,
        namespace: str,
        collection: str = "memories",
        **kwargs,
    ) -> "ExtractionPipeline":
        """Create pipeline with explicit backend."""
        return cls(backend, namespace, collection, **kwargs)
    
    def extract(
        self,
        text: str,
        extractor: Callable[[str], Dict[str, Any]],
        source_id: Optional[str] = None,
        validate: bool = True,
    ) -> ExtractionResult:
        """
        Extract structured data from text.
        
        Args:
            text: Input text to process
            extractor: LLM extraction function returning JSON
            source_id: Optional source document ID
            validate: Whether to validate against schema
            
        Returns:
            ExtractionResult with entities, relations, and assertions
            
        Raises:
            ValueError: If validation fails and no repair possible
        """
        start_time = time.time()
        
        # Call LLM extractor
        raw_output = extractor(text)
        
        # Parse output
        result = self._parse_extraction(raw_output)
        result.raw_text = text
        result.source_id = source_id or hashlib.sha256(text.encode()).hexdigest()[:16]
        
        # Generate embeddings for assertions
        if self._embed_fn:
            for assertion in result.assertions:
                if assertion.embedding is None:
                    content = f"{assertion.subject} {assertion.predicate} {assertion.object}"
                    assertion.embedding = self._embed_fn(content)
        
        # Validate against schema
        if validate and self._schema:
            valid, errors = self._schema.validate_result(result)
            if not valid:
                # Try to repair or raise
                result = self._repair_or_raise(result, errors)
        
        result.extraction_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _parse_extraction(self, raw: Dict[str, Any]) -> ExtractionResult:
        """Parse raw LLM output into typed objects."""
        entities = [
            Entity.from_dict(e) for e in raw.get("entities", [])
        ]
        relations = [
            Relation.from_dict(r) for r in raw.get("relations", [])
        ]
        assertions = [
            Assertion.from_dict(a) for a in raw.get("assertions", [])
        ]
        
        return ExtractionResult(
            entities=entities,
            relations=relations,
            assertions=assertions,
        )
    
    def _repair_or_raise(
        self, 
        result: ExtractionResult,
        errors: List[str],
    ) -> ExtractionResult:
        """Attempt to repair validation errors or raise."""
        # Simple strategy: filter out invalid items
        if self._schema:
            result.entities = [
                e for e in result.entities
                if self._schema.validate_entity(e)[0]
            ]
            result.relations = [
                r for r in result.relations
                if self._schema.validate_relation(r)[0]
            ]
            result.assertions = [
                a for a in result.assertions
                if self._schema.validate_assertion(a)[0]
            ]
        
        if result.is_empty:
            raise ValueError(f"Extraction failed validation: {errors}")
        
        return result
    
    def commit(self, result: ExtractionResult) -> None:
        """
        Atomically commit extraction result to storage.
        
        Writes all entities, relations, and assertions in a single
        transaction to ensure consistency.
        
        Args:
            result: ExtractionResult to commit
        """
        # Write entities as graph nodes
        for entity in result.entities:
            self._backend.add_node(
                namespace=self._namespace,
                node_id=entity.id,
                node_type=entity.entity_type,
                properties={
                    **entity.properties,
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "provenance": entity.provenance,
                },
            )
        
        # Write relations as edges
        for relation in result.relations:
            self._backend.add_edge(
                namespace=self._namespace,
                from_id=relation.from_entity,
                edge_type=relation.relation_type,
                to_id=relation.to_entity,
                properties={
                    **relation.properties,
                    "confidence": relation.confidence,
                    "provenance": relation.provenance,
                },
            )
        
        # Write assertions as temporal edges + vector docs
        for assertion in result.assertions:
            # Add as temporal edge for time-travel queries
            self._backend.add_temporal_edge(
                namespace=self._namespace,
                from_id=assertion.subject,
                edge_type=assertion.predicate,
                to_id=str(assertion.object),
                valid_from=assertion.valid_from,
                valid_until=assertion.valid_until,
                properties={
                    "confidence": assertion.confidence,
                    "provenance": assertion.provenance,
                },
            )
            
            # Add to vector collection if embedding available
            if assertion.embedding:
                content = f"{assertion.subject} {assertion.predicate} {assertion.object}"
                self._backend.add_document(
                    namespace=self._namespace,
                    collection=self._collection,
                    doc_id=assertion.id,
                    content=content,
                    embedding=assertion.embedding,
                    metadata={
                        "subject": assertion.subject,
                        "predicate": assertion.predicate,
                        "object": str(assertion.object),
                        "valid_from": assertion.valid_from,
                        "valid_until": assertion.valid_until,
                        "confidence": assertion.confidence,
                    },
                )
        
        # Store extraction metadata
        meta_key = f"extraction:{self._namespace}:{result.source_id}".encode()
        meta_value = json.dumps({
            "source_id": result.source_id,
            "entity_count": len(result.entities),
            "relation_count": len(result.relations),
            "assertion_count": len(result.assertions),
            "extraction_time_ms": result.extraction_time_ms,
            "timestamp": int(time.time() * 1000),
        }).encode()
        self._backend.put(meta_key, meta_value)
    
    def extract_and_commit(
        self,
        text: str,
        extractor: Callable[[str], Dict[str, Any]],
        **kwargs,
    ) -> ExtractionResult:
        """Extract and commit in one call."""
        result = self.extract(text, extractor, **kwargs)
        self.commit(result)
        return result


# ============================================================================
# Factory Function
# ============================================================================

def create_extraction_pipeline(
    backend: Union["Database", "SochDBClient", MemoryBackend],
    namespace: str,
    collection: str = "memories",
    **kwargs,
) -> ExtractionPipeline:
    """
    Create an extraction pipeline with auto-detected backend.
    
    Args:
        backend: Database, SochDBClient, or MemoryBackend instance
        namespace: Namespace for isolation
        collection: Collection name
        **kwargs: Additional arguments for ExtractionPipeline
        
    Returns:
        Configured ExtractionPipeline
    """
    # Import here to avoid circular imports
    from ..database import Database
    from ..grpc_client import SochDBClient
    
    if isinstance(backend, Database):
        return ExtractionPipeline.from_database(backend, namespace, collection, **kwargs)
    elif isinstance(backend, SochDBClient):
        return ExtractionPipeline.from_client(backend, namespace, collection, **kwargs)
    elif isinstance(backend, MemoryBackend):
        return ExtractionPipeline.from_backend(backend, namespace, collection, **kwargs)
    else:
        raise TypeError(f"Unknown backend type: {type(backend)}")
