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
SochDB Memory Module - LLM-Native Memory Management

This module provides a complete memory system for AI agents:

1. **Extraction Pipeline** - Compile LLM outputs into typed, validated facts
2. **Consolidation** - Event-sourced canonicalization without data loss
3. **Hybrid Retrieval** - RRF-based retrieval with pre-filtering
4. **Namespace Isolation** - Multi-tenant memory with strict scoping

All components support both embedded (FFI) and server (gRPC) modes.
"""

from .extraction import (
    Entity,
    Relation,
    Assertion,
    ExtractionResult,
    ExtractionSchema,
    MemoryBackend,
    FFIMemoryBackend,
    GrpcMemoryBackend,
    InMemoryBackend,
    ExtractionPipeline,
    create_extraction_pipeline,
)

from .consolidation import (
    RawAssertion,
    CanonicalFact,
    ConsolidationConfig,
    ConsolidationBackend,
    FFIConsolidationBackend,
    GrpcConsolidationBackend,
    InMemoryConsolidationBackend,
    Consolidator,
    create_consolidator,
)

from .retrieval import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalResponse,
    RetrievalBackend,
    FFIRetrievalBackend,
    GrpcRetrievalBackend,
    InMemoryRetrievalBackend,
    HybridRetriever,
    AllowedSet,
    create_retriever,
)

from .isolation import (
    NamespaceId,
    NamespacePolicy,
    NamespaceGrant,
    ScopedQuery,
    ScopedNamespace,
    NamespaceBackend,
    FFINamespaceBackend,
    GrpcNamespaceBackend,
    InMemoryNamespaceBackend,
    NamespaceManager,
    create_namespace_manager,
)

__all__ = [
    # Extraction
    "Entity",
    "Relation", 
    "Assertion",
    "ExtractionResult",
    "ExtractionSchema",
    "MemoryBackend",
    "FFIMemoryBackend",
    "GrpcMemoryBackend",
    "InMemoryBackend",
    "ExtractionPipeline",
    "create_extraction_pipeline",
    # Consolidation
    "RawAssertion",
    "CanonicalFact",
    "ConsolidationConfig",
    "ConsolidationBackend",
    "FFIConsolidationBackend",
    "GrpcConsolidationBackend",
    "InMemoryConsolidationBackend",
    "Consolidator",
    "create_consolidator",
    # Retrieval
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalResponse",
    "RetrievalBackend",
    "FFIRetrievalBackend",
    "GrpcRetrievalBackend",
    "InMemoryRetrievalBackend",
    "HybridRetriever",
    "AllowedSet",
    "create_retriever",
    # Namespace Isolation
    "NamespaceId",
    "NamespacePolicy",
    "NamespaceGrant",
    "ScopedQuery",
    "ScopedNamespace",
    "NamespaceBackend",
    "FFINamespaceBackend",
    "GrpcNamespaceBackend",
    "InMemoryNamespaceBackend",
    "NamespaceManager",
    "create_namespace_manager",
]
