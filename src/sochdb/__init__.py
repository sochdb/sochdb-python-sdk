"""
SochDB Python SDK v0.4.0

Dual-mode architecture: Embedded (FFI) + Server (gRPC/IPC)

Architecture: Flexible Deployment
=================================
This SDK supports BOTH modes:

1. Embedded Mode (FFI) - For single-process apps:
   - Direct FFI bindings to Rust libraries
   - No server required - just pip install and run
   - Best for: Local development, simple apps, notebooks
   
2. Server Mode (gRPC/IPC) - For distributed systems:
   - Thin client connecting to sochdb-grpc server
   - Best for: Production, multi-language, scalability

Example (Embedded Mode):
    from sochdb import Database
    
    # Direct FFI - no server needed
    with Database.open("./mydb") as db:
        db.put(b"key", b"value")
        value = db.get(b"key")

Example (Server Mode):
    from sochdb import SochDBClient
    
    # Connect to server
    client = SochDBClient("localhost:50051")
    client.put_kv("key", b"value")
"""

__version__ = "0.4.1"

# Embedded mode (FFI)
from .database import Database, Transaction
from .namespace import (
    Namespace,
    NamespaceConfig,
    Collection,
    CollectionConfig,
    DistanceMetric,
    SearchRequest,
    SearchResults,
)
from .vector import VectorIndex

# Server mode (gRPC/IPC)
from .grpc_client import SochDBClient, SearchResult, Document, GraphNode, GraphEdge, TemporalEdge
from .ipc_client import IpcClient

# Format utilities
from .format import (
    WireFormat,
    ContextFormat,
    CanonicalFormat,
    FormatCapabilities,
    FormatConversionError,
)

# Type definitions
from .errors import (
    SochDBError,
    ConnectionError,
    TransactionError,
    ProtocolError,
    DatabaseError,
    ErrorCode,
    NamespaceNotFoundError,
    NamespaceExistsError,
    # Lock errors (v0.4.1)
    LockError,
    DatabaseLockedError,
    LockTimeoutError,
    EpochMismatchError,
    SplitBrainError,
)
from .query import Query, SQLQueryResult

# Convenience aliases
GrpcClient = SochDBClient

__all__ = [
    # Version
    "__version__",
    
    # Embedded mode (FFI)
    "Database",
    "Transaction",
    "Namespace",
    "NamespaceConfig",
    "Collection",
    "CollectionConfig",
    "DistanceMetric",
    "SearchRequest",
    "SearchResults",
    "VectorIndex",
    
    # Server mode (thin clients)
    "SochDBClient",
    "GrpcClient",
    "IpcClient",
    
    # Format utilities
    "WireFormat",
    "ContextFormat",
    "CanonicalFormat",
    "FormatCapabilities",
    "FormatConversionError",
    
    # Data types
    "SearchResult",
    "Document",
    "GraphNode",
    "GraphEdge",
    "Query",
    "SQLQueryResult",
    
    # Errors
    "SochDBError",
    "ConnectionError",
    "TransactionError",
    "ProtocolError",
    "DatabaseError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "ErrorCode",
    # Lock errors (v0.4.1)
    "LockError",
    "DatabaseLockedError",
    "LockTimeoutError",
    "EpochMismatchError",
    "SplitBrainError",
]
