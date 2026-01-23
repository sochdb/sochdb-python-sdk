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

__version__ = "0.4.4"

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

# Queue API (v0.4.3)
from .queue import (
    PriorityQueue,
    QueueConfig,
    QueueKey,
    Task,
    TaskState,
    QueueStats,
    StreamingTopK,
    create_queue,
    # Backend interfaces for custom implementations
    QueueBackend,
    QueueTransaction,
    FFIQueueBackend,
    GrpcQueueBackend,
    InMemoryQueueBackend,
)

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
    
    # Queue API (v0.4.3)
    "PriorityQueue",
    "QueueConfig",
    "QueueKey",
    "Task",
    "TaskState",
    "QueueStats",
    "StreamingTopK",
    "create_queue",
    "QueueBackend",
    "QueueTransaction",
    "FFIQueueBackend",
    "GrpcQueueBackend",
    "InMemoryQueueBackend",
    
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
    
    # Convenience functions
    "open_collection",
    "Client",
]


# ============================================================================
# Convenience Client API
# ============================================================================

class Client:
    """
    High-level client for SochDB.
    
    Provides a simple API for common vector database operations.
    
    Example:
        import sochdb
        
        # Create client
        client = sochdb.Client()
        collection = client.get_or_create_collection("my_vectors")
        
        # Add vectors
        collection.add(
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ids=["doc1", "doc2"],
            metadatas=[{"type": "a"}, {"type": "b"}]
        )
        
        # Query
        results = collection.query(
            query_embeddings=[[1.0, 2.0, 3.0]],
            n_results=5
        )
    """
    
    def __init__(self, path: str = ":memory:"):
        """
        Create a SochDB client.
        
        Args:
            path: Database path. Use ":memory:" for in-memory (default)
        """
        import tempfile
        import os
        
        if path == ":memory:":
            # Create temp directory for in-memory-like usage
            self._temp_dir = tempfile.mkdtemp(prefix="sochdb_")
            self._path = self._temp_dir
        else:
            self._temp_dir = None
            self._path = path
        
        self._db = Database.open(self._path)
        
        # Create default namespace if it doesn't exist
        try:
            self._default_ns = self._db.namespace("default")
        except NamespaceNotFoundError:
            self._default_ns = self._db.create_namespace("default")
    
    def get_or_create_collection(
        self,
        name: str,
        dimension: int = None,
        metadata: dict = None,
    ) -> Collection:
        """
        Get or create a collection.
        
        Args:
            name: Collection name
            dimension: Vector dimension (optional, can be inferred)
            metadata: Collection metadata (optional)
            
        Returns:
            Collection handle
        """
        config = CollectionConfig(name=name, dimension=dimension)
        try:
            return self._default_ns.create_collection(config)
        except:
            return self._default_ns.collection(name)
    
    def create_collection(self, name: str, dimension: int = None, **kwargs) -> Collection:
        """Create a new collection."""
        return self.get_or_create_collection(name, dimension)
    
    def get_collection(self, name: str) -> Collection:
        """Get an existing collection."""
        return self._default_ns.collection(name)
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        return self._default_ns.delete_collection(name)
    
    def list_collections(self) -> list:
        """List all collections."""
        return self._default_ns.list_collections()
    
    def close(self):
        """Close the client."""
        if self._temp_dir:
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def open_collection(
    name: str,
    path: str = ":memory:",
    dimension: int = None,
) -> Collection:
    """
    Open a collection directly (convenience function).
    
    This is the simplest way to get started with SochDB.
    
    Args:
        name: Collection name
        path: Database path (default: in-memory)
        dimension: Vector dimension (optional, auto-inferred)
        
    Returns:
        Collection handle
        
    Example:
        import sochdb
        
        # One-liner to get started
        collection = sochdb.open_collection("vectors")
        collection.add(embeddings=[[1.0, 2.0, 3.0]])
    """
    client = Client(path=path)
    return client.get_or_create_collection(name, dimension=dimension)
