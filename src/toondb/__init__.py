"""
ToonDB Python SDK v0.3.4

Dual-mode architecture: Embedded (FFI) + Server (gRPC/IPC)

Architecture: Flexible Deployment
=================================
This SDK supports BOTH modes:

1. Embedded Mode (FFI) - For single-process apps:
   - Direct FFI bindings to Rust libraries
   - No server required - just pip install and run
   - Best for: Local development, simple apps, notebooks
   
2. Server Mode (gRPC/IPC) - For distributed systems:
   - Thin client connecting to toondb-grpc server
   - Best for: Production, multi-language, scalability

Example (Embedded Mode):
    from toondb import Database
    
    # Direct FFI - no server needed
    with Database.open("./mydb") as db:
        db.put(b"key", b"value")
        value = db.get(b"key")

Example (Server Mode):
    from toondb import ToonDBClient
    
    # Connect to server
    client = ToonDBClient("localhost:50051")
    client.put_kv("key", b"value")
"""

__version__ = "0.3.4"

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
from .grpc_client import ToonDBClient, SearchResult, Document, GraphNode, GraphEdge, TemporalEdge
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
    ToonDBError,
    ConnectionError,
    TransactionError,
    ProtocolError,
    DatabaseError,
    ErrorCode,
    NamespaceNotFoundError,
    NamespaceExistsError,
)
from .query import Query, SQLQueryResult

# Convenience aliases
GrpcClient = ToonDBClient

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
    "ToonDBClient",
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
    "ToonDBError",
    "ConnectionError",
    "TransactionError",
    "ProtocolError",
    "DatabaseError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "ErrorCode",
]
