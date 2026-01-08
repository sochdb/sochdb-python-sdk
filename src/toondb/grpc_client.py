"""
ToonDB gRPC Client - Thin SDK Wrapper

This module provides a thin gRPC client wrapper for the ToonDB server.
All business logic runs on the server (Thick Server / Thin Client architecture).

The client is approximately ~200 lines of code, delegating all operations to the server.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple
import grpc
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Vector search result."""
    id: int
    distance: float


@dataclass
class Document:
    """Document with embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, str]


@dataclass
class GraphNode:
    """Node in the graph overlay."""
    id: str
    node_type: str
    properties: Dict[str, str]


@dataclass
class GraphEdge:
    """Edge in the graph overlay."""
    from_id: str
    edge_type: str
    to_id: str
    properties: Dict[str, str]


@dataclass
class TemporalEdge:
    """Temporal edge with validity interval."""
    from_id: str
    edge_type: str
    to_id: str
    valid_from: int  # Unix timestamp (ms)
    valid_until: int  # Unix timestamp (ms), 0 = no expiry
    properties: Dict[str, str]


class ToonDBClient:
    """
    Thin gRPC client for ToonDB.
    
    All operations are delegated to the ToonDB gRPC server.
    This client provides a Pythonic interface over the gRPC protocol.
    
    Usage:
        client = ToonDBClient("localhost:50051")
        
        # Create collection
        client.create_collection("docs", dimension=384)
        
        # Add documents
        client.add_documents("docs", [
            {"id": "1", "content": "Hello", "embedding": [...]}
        ])
        
        # Search
        results = client.search("docs", query_vector, k=5)
    """
    
    def __init__(self, address: str = "localhost:50051", secure: bool = False):
        """
        Connect to ToonDB gRPC server.
        
        Args:
            address: Server address in host:port format
            secure: Use TLS if True
        """
        self.address = address
        
        if secure:
            self.channel = grpc.secure_channel(address, grpc.ssl_channel_credentials())
        else:
            self.channel = grpc.insecure_channel(address)
        
        # Lazy-load stubs to avoid import issues
        self._stubs: Dict[str, Any] = {}
    
    def _get_stub(self, service_name: str) -> Any:
        """Get or create a gRPC stub for a service."""
        if service_name not in self._stubs:
            # Import proto modules lazily
            try:
                from . import toondb_pb2_grpc
                stub_class = getattr(toondb_pb2_grpc, f"{service_name}Stub")
                self._stubs[service_name] = stub_class(self.channel)
            except (ImportError, AttributeError):
                raise RuntimeError(
                    f"gRPC proto files not generated. Run: python -m grpc_tools.protoc "
                    f"-I. --python_out=. --grpc_python_out=. proto/toondb.proto"
                )
        return self._stubs[service_name]
    
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # Vector Index Operations (VectorIndexService)
    # =========================================================================
    
    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 200
    ) -> bool:
        """Create a new vector index."""
        stub = self._get_stub("VectorIndexService")
        from . import toondb_pb2
        
        response = stub.CreateIndex(toondb_pb2.CreateIndexRequest(
            name=name,
            dimension=dimension,
            metric=getattr(toondb_pb2, f"DISTANCE_METRIC_{metric.upper()}", 2),
            config=toondb_pb2.HnswConfig(
                max_connections=m,
                ef_construction=ef_construction
            )
        ))
        return response.success
    
    def insert_vectors(
        self,
        index_name: str,
        ids: List[int],
        vectors: List[List[float]]
    ) -> int:
        """Insert vectors into an index."""
        stub = self._get_stub("VectorIndexService")
        from . import toondb_pb2
        
        # Flatten vectors
        flat_vectors = [v for vec in vectors for v in vec]
        
        response = stub.InsertBatch(toondb_pb2.InsertBatchRequest(
            index_name=index_name,
            ids=ids,
            vectors=flat_vectors
        ))
        return response.inserted_count
    
    def search(
        self,
        index_name: str,
        query: List[float],
        k: int = 10,
        ef: int = 50
    ) -> List[SearchResult]:
        """Search for k-nearest neighbors."""
        stub = self._get_stub("VectorIndexService")
        from . import toondb_pb2
        
        response = stub.Search(toondb_pb2.SearchRequest(
            index_name=index_name,
            query=query,
            k=k,
            ef=ef
        ))
        
        return [SearchResult(id=r.id, distance=r.distance) for r in response.results]
    
    # =========================================================================
    # Collection Operations (CollectionService)
    # =========================================================================
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        namespace: str = "default",
        metric: str = "cosine"
    ) -> bool:
        """Create a new collection."""
        stub = self._get_stub("CollectionService")
        from . import toondb_pb2
        
        response = stub.CreateCollection(toondb_pb2.CreateCollectionRequest(
            name=name,
            namespace=namespace,
            dimension=dimension,
            metric=getattr(toondb_pb2, f"DISTANCE_METRIC_{metric.upper()}", 2)
        ))
        return response.success
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict],
        namespace: str = "default"
    ) -> List[str]:
        """Add documents to a collection."""
        stub = self._get_stub("CollectionService")
        from . import toondb_pb2
        
        doc_protos = [
            toondb_pb2.Document(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                embedding=doc.get("embedding", []),
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]
        
        response = stub.AddDocuments(toondb_pb2.AddDocumentsRequest(
            collection_name=collection_name,
            namespace=namespace,
            documents=doc_protos
        ))
        return list(response.ids)
    
    def search_collection(
        self,
        collection_name: str,
        query: List[float],
        k: int = 10,
        namespace: str = "default",
        filter: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """Search a collection for similar documents."""
        stub = self._get_stub("CollectionService")
        from . import toondb_pb2
        
        response = stub.SearchCollection(toondb_pb2.SearchCollectionRequest(
            collection_name=collection_name,
            namespace=namespace,
            query=query,
            k=k,
            filter=filter or {}
        ))
        
        return [
            Document(
                id=r.document.id,
                content=r.document.content,
                embedding=list(r.document.embedding),
                metadata=dict(r.document.metadata)
            )
            for r in response.results
        ]
    
    # =========================================================================
    # Graph Operations (GraphService)
    # =========================================================================
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, str]] = None,
        namespace: str = "default"
    ) -> bool:
        """Add a node to the graph."""
        stub = self._get_stub("GraphService")
        from . import toondb_pb2
        
        response = stub.AddNode(toondb_pb2.AddNodeRequest(
            namespace=namespace,
            node=toondb_pb2.GraphNode(
                id=node_id,
                node_type=node_type,
                properties=properties or {}
            )
        ))
        return response.success
    
    def add_edge(
        self,
        from_id: str,
        edge_type: str,
        to_id: str,
        properties: Optional[Dict[str, str]] = None,
        namespace: str = "default"
    ) -> bool:
        """Add an edge between nodes."""
        stub = self._get_stub("GraphService")
        from . import toondb_pb2
        
        response = stub.AddEdge(toondb_pb2.AddEdgeRequest(
            namespace=namespace,
            edge=toondb_pb2.GraphEdge(
                from_id=from_id,
                edge_type=edge_type,
                to_id=to_id,
                properties=properties or {}
            )
        ))
        return response.success
    
    def traverse(
        self,
        start_node: str,
        max_depth: int = 10,
        order: str = "bfs",
        namespace: str = "default"
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Traverse the graph from a starting node."""
        stub = self._get_stub("GraphService")
        from . import toondb_pb2
        
        response = stub.Traverse(toondb_pb2.TraverseRequest(
            namespace=namespace,
            start_node_id=start_node,
            order=0 if order == "bfs" else 1,
            max_depth=max_depth
        ))
        
        nodes = [
            GraphNode(id=n.id, node_type=n.node_type, properties=dict(n.properties))
            for n in response.nodes
        ]
        edges = [
            GraphEdge(
                from_id=e.from_id,
                edge_type=e.edge_type,
                to_id=e.to_id,
                properties=dict(e.properties)
            )
            for e in response.edges
        ]
        return nodes, edges
    
    # =========================================================================
    # Semantic Cache Operations (SemanticCacheService)
    # =========================================================================
    
    def cache_get(
        self,
        cache_name: str,
        query_embedding: List[float],
        threshold: float = 0.85
    ) -> Optional[str]:
        """Get from semantic cache by similarity."""
        stub = self._get_stub("SemanticCacheService")
        from . import toondb_pb2
        
        response = stub.Get(toondb_pb2.SemanticCacheGetRequest(
            cache_name=cache_name,
            query_embedding=query_embedding,
            similarity_threshold=threshold
        ))
        
        return response.cached_value if response.hit else None
    
    def cache_put(
        self,
        cache_name: str,
        key: str,
        value: str,
        key_embedding: List[float],
        ttl_seconds: int = 0
    ) -> bool:
        """Put a value in the semantic cache."""
        stub = self._get_stub("SemanticCacheService")
        from . import toondb_pb2
        
        response = stub.Put(toondb_pb2.SemanticCachePutRequest(
            cache_name=cache_name,
            key=key,
            value=value,
            key_embedding=key_embedding,
            ttl_seconds=ttl_seconds
        ))
        return response.success
    
    # =========================================================================
    # Context Operations (ContextService)
    # =========================================================================
    
    def query_context(
        self,
        session_id: str,
        sections: List[Dict],
        token_limit: int = 2048,
        format: str = "toon"
    ) -> str:
        """Assemble LLM context with token budget."""
        stub = self._get_stub("ContextService")
        from . import toondb_pb2
        
        format_map = {"toon": 0, "json": 1, "markdown": 2, "text": 3}
        
        section_protos = [
            toondb_pb2.ContextSection(
                name=s.get("name", ""),
                priority=s.get("priority", 0),
                section_type=s.get("type", 0),
                query=s.get("query", "")
            )
            for s in sections
        ]
        
        response = stub.Query(toondb_pb2.ContextQueryRequest(
            session_id=session_id,
            token_limit=token_limit,
            sections=section_protos,
            format=format_map.get(format, 0)
        ))
        
        return response.context
    
    # =========================================================================
    # Trace Operations (TraceService)
    # =========================================================================
    
    def start_trace(self, name: str) -> Tuple[str, str]:
        """Start a new trace. Returns (trace_id, root_span_id)."""
        stub = self._get_stub("TraceService")
        from . import toondb_pb2
        
        response = stub.StartTrace(toondb_pb2.StartTraceRequest(name=name))
        return response.trace_id, response.root_span_id
    
    def start_span(
        self,
        trace_id: str,
        parent_span_id: str,
        name: str
    ) -> str:
        """Start a span within a trace. Returns span_id."""
        stub = self._get_stub("TraceService")
        from . import toondb_pb2
        
        response = stub.StartSpan(toondb_pb2.StartSpanRequest(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name
        ))
        return response.span_id
    
    def end_span(self, trace_id: str, span_id: str, status: str = "ok") -> int:
        """End a span. Returns duration in microseconds."""
        stub = self._get_stub("TraceService")
        from . import toondb_pb2
        
        status_map = {"unset": 0, "ok": 1, "error": 2}
        response = stub.EndSpan(toondb_pb2.EndSpanRequest(
            trace_id=trace_id,
            span_id=span_id,
            status=status_map.get(status, 0)
        ))
        return response.duration_us
    
    # =========================================================================
    # KV Operations (KvService)  
    # =========================================================================
    
    def get(self, key: bytes, namespace: str = "default") -> Optional[bytes]:
        """Get a value by key."""
        stub = self._get_stub("KvService")
        from . import toondb_pb2
        
        response = stub.Get(toondb_pb2.KvGetRequest(namespace=namespace, key=key))
        return response.value if response.found else None
    
    def put(
        self,
        key: bytes,
        value: bytes,
        namespace: str = "default",
        ttl_seconds: int = 0
    ) -> bool:
        """Put a value."""
        stub = self._get_stub("KvService")
        from . import toondb_pb2
        
        response = stub.Put(toondb_pb2.KvPutRequest(
            namespace=namespace,
            key=key,
            value=value,
            ttl_seconds=ttl_seconds
        ))
        return response.success
    
    def delete(self, key: bytes, namespace: str = "default") -> bool:
        """Delete a key."""
        stub = self._get_stub("KvService")
        from . import toondb_pb2
        
        response = stub.Delete(toondb_pb2.KvDeleteRequest(namespace=namespace, key=key))
        return response.success

    # =========================================================================
    # Temporal Graph Operations
    # =========================================================================
    
    def add_temporal_edge(
        self,
        namespace: str,
        from_id: str,
        edge_type: str,
        to_id: str,
        valid_from: int,
        valid_until: int = 0,
        properties: Optional[Dict[str, str]] = None
    ) -> bool:
        """Add a temporal edge with validity interval.
        
        Args:
            namespace: Namespace for the edge
            from_id: Source node ID
            edge_type: Type of the edge
            to_id: Target node ID
            valid_from: Start timestamp (Unix ms)
            valid_until: End timestamp (Unix ms), 0 = no expiry
            properties: Optional edge properties
            
        Returns:
            True if successful
        """
        stub = self._get_stub("GraphService")
        from . import toondb_pb2
        
        response = stub.AddTemporalEdge(toondb_pb2.AddTemporalEdgeRequest(
            namespace=namespace,
            from_id=from_id,
            edge_type=edge_type,
            to_id=to_id,
            valid_from=valid_from,
            valid_until=valid_until,
            properties=properties or {}
        ))
        return response.success
    
    def query_temporal_graph(
        self,
        namespace: str,
        node_id: str,
        mode: str = "CURRENT",
        timestamp: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        edge_types: Optional[List[str]] = None
    ) -> List[TemporalEdge]:
        """Query temporal graph.
        
        Args:
            namespace: Namespace to query
            node_id: Node to query from
            mode: Query mode - POINT_IN_TIME, RANGE, or CURRENT
            timestamp: For POINT_IN_TIME mode
            start_time: For RANGE mode
            end_time: For RANGE mode
            edge_types: Optional filter for edge types
            
        Returns:
            List of temporal edges
        """
        stub = self._get_stub("GraphService")
        from . import toondb_pb2
        
        # Map string mode to enum
        mode_map = {
            "POINT_IN_TIME": toondb_pb2.TemporalQueryMode.POINT_IN_TIME,
            "RANGE": toondb_pb2.TemporalQueryMode.RANGE,
            "CURRENT": toondb_pb2.TemporalQueryMode.CURRENT,
        }
        
        response = stub.QueryTemporalGraph(toondb_pb2.QueryTemporalGraphRequest(
            namespace=namespace,
            node_id=node_id,
            mode=mode_map.get(mode, toondb_pb2.TemporalQueryMode.CURRENT),
            timestamp=timestamp or 0,
            start_time=start_time or 0,
            end_time=end_time or 0,
            edge_types=edge_types or []
        ))
        
        return [
            TemporalEdge(
                from_id=e.from_id,
                edge_type=e.edge_type,
                to_id=e.to_id,
                valid_from=e.valid_from,
                valid_until=e.valid_until,
                properties=dict(e.properties)
            )
            for e in response.edges
        ]


# Convenience function
def connect(address: str = "localhost:50051", **kwargs) -> ToonDBClient:
    """
    Connect to ToonDB gRPC server.
    
    Args:
        address: Server address (host:port or grpc://host:port)
        **kwargs: Additional options (secure=True for TLS)
    
    Returns:
        ToonDBClient instance
    """
    if address.startswith("grpc://"):
        address = address[7:]
    return ToonDBClient(address, **kwargs)


# Alias for backwards compatibility
GrpcClient = ToonDBClient
