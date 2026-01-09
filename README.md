# ToonDB Python SDK v0.3.5

**Dual-mode architecture: Embedded (FFI) + Server (gRPC/IPC)**  
Choose the deployment mode that fits your needs.

## What's New in 0.3.5

### 🔢 Vector Operations in Database Class
No need for separate `VectorIndex` class anymore - vector operations are now directly available on the `Database` class:

```python
from toondb import Database

db = Database.open('./mydb')

# Create vector index
db.create_index('embeddings', dimension=384)

# Insert vectors
db.insert_vectors('embeddings', [1, 2, 3], [
    [0.1, 0.2, ...],  # vector 1
    [0.3, 0.4, ...],  # vector 2
    [0.5, 0.6, ...],  # vector 3
])

# Search
results = db.search('embeddings', [0.1, 0.2, ...], k=5)
for id, distance in results:
    print(f'ID: {id}, Distance: {distance:.4f}')
```

### 🏗️ Works with Tokio-Optional Architecture
- Supports both sync and async Rust backend (v0.3.5)
- No breaking changes to existing code
- Smaller binaries (~500KB reduction)

## Architecture: Flexible Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT OPTIONS                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. EMBEDDED MODE (FFI)          2. SERVER MODE (gRPC)      │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │   Python App        │         │   Python App        │   │
│  │   ├─ Database.open()│         │   ├─ ToonDBClient() │   │
│  │   └─ Direct FFI     │         │   └─ gRPC calls     │   │
│  │         │           │         │         │           │   │
│  │         ▼           │         │         ▼           │   │
│  │   libtoondb_storage │         │   toondb-grpc       │   │
│  │   (Rust native)     │         │   (Rust server)     │   │
│  └─────────────────────┘         └─────────────────────┘   │
│                                                               │
│  ✅ No server needed               ✅ Multi-language          │
│  ✅ Local files                    ✅ Centralized logic      │
│  ✅ Simple deployment              ✅ Production scale       │
└─────────────────────────────────────────────────────────────┘
```

### When to Use Each Mode

**Embedded Mode (FFI):**
- ✅ Local development and testing
- ✅ Jupyter notebooks and data science
- ✅ Single-process applications
- ✅ Edge deployments without network
- ✅ No server setup required

**Server Mode (gRPC):**
- ✅ Production deployments
- ✅ Multi-language teams (Python, Node.js, Go)
- ✅ Distributed systems
- ✅ Centralized business logic
- ✅ Horizontal scaling

---

## Installation

```bash
pip install toondb-client
```

Or from source:
```bash
cd toondb-python-sdk
pip install -e .
```

---

## Quick Start

### Mode 1: Embedded (FFI) - No Server Required

```python
from toondb import Database

# Open database with direct FFI bindings
with Database.open("./mydb") as db:
    # Key-value operations
    db.put(b"key", b"value")
    value = db.get(b"key")
    
    # Vector operations (NEW in 0.3.5)
    db.create_index('embeddings', dimension=384)
    db.insert_vectors('embeddings', [1, 2, 3], [
        [0.1, 0.2, ...],  # vector 1
        [0.3, 0.4, ...],  # vector 2
        [0.5, 0.6, ...],  # vector 3
    ])
    
    # Search for similar vectors
    results = db.search('embeddings', [0.1, 0.2, ...], k=5)
    for id, distance in results:
        print(f'ID: {id}, Distance: {distance}')
    
    # Namespaces
    ns = db.namespace("tenant_123")
    collection = ns.collection("documents", dimension=384)
    
    # Temporal graphs (NEW in 0.3.4)
    import time
    now = int(time.time() * 1000)
    
    db.add_temporal_edge(
        namespace="smart_home",
        from_id="door_front",
        edge_type="STATE",
        to_id="open",
        valid_from=now - 3600000,  # 1 hour ago
        valid_until=now,
        properties={"sensor": "motion_1"}
    )
    
    # Time-travel query: "Was door open 30 minutes ago?"
    edges = db.query_temporal_graph(
        namespace="smart_home",
        node_id="door_front",
        mode="POINT_IN_TIME",
        timestamp=now - 1800000  # 30 minutes ago
    )
```

### Mode 2: Server (gRPC) - For Production

### 2.1. Start ToonDB Server

```bash
# Start the gRPC server
cd toondb
cargo run -p toondb-grpc --release

# Server listens on localhost:50051
```

### 2.2. Connect from Python

```python
from toondb import ToonDBClient

# Connect to server
client = ToonDBClient("localhost:50051")

# Create a vector collection
client.create_collection("documents", dimension=384)

# Add documents with embeddings
documents = [
    {
        "id": "doc1",
        "content": "Machine learning tutorial",
        "embedding": [0.1, 0.2, ...],  # 384-dimensional vector
        "metadata": {"category": "AI"}
    }
]
client.add_documents("documents", documents)

# Search for similar documents
query_vector = [0.15, 0.25, ...]  # 384-dimensional
results = client.search_collection("documents", query_vector, k=5)

for result in results:
    print(f"Score: {result.score}, Content: {result.content}")
```

---

## API Reference

### ToonDBClient (gRPC Transport)

**Constructor:**
```python
client = ToonDBClient(address: str = "localhost:50051", secure: bool = False)
```

**Vector Operations:**
```python
# Create vector index
client.create_index(
    name: str,
    dimension: int,
    metric: str = "cosine"  # cosine, euclidean, dot
) -> bool

# Insert vectors
client.insert_vectors(
    index_name: str,
    ids: List[int],
    vectors: List[List[float]]
) -> bool

# Search vectors
client.search(
    index_name: str,
    query: List[float],
    k: int = 10
) -> List[SearchResult]
```

**Collection Operations:**
```python
# Create collection
client.create_collection(
    name: str,
    dimension: int,
    namespace: str = "default"
) -> bool

# Add documents
client.add_documents(
    collection_name: str,
    documents: List[Dict],
    namespace: str = "default"
) -> List[str]

# Search collection
client.search_collection(
    collection_name: str,
    query: List[float],
    k: int = 10,
    namespace: str = "default",
    filter: Optional[Dict] = None
) -> List[Document]
```

**Graph Operations:**
```python
# Add graph node
client.add_node(
    node_id: str,
    node_type: str,
    properties: Optional[Dict] = None,
    namespace: str = "default"
) -> bool

# Add graph edge
client.add_edge(
    from_id: str,
    edge_type: str,
    to_id: str,
    properties: Optional[Dict] = None,
    namespace: str = "default"
) -> bool

# Traverse graph
client.traverse(
    start_node: str,
    max_depth: int = 3,
    edge_types: Optional[List[str]] = None,
    namespace: str = "default"
) -> Tuple[List[GraphNode], List[GraphEdge]]
```

**Namespace Operations:**
```python
# Create namespace
client.create_namespace(
    name: str,
    metadata: Optional[Dict] = None
) -> bool

# List namespaces
client.list_namespaces() -> List[str]
```

**Key-Value Operations:**
```python
# Put key-value
client.put_kv(
    key: str,
    value: bytes,
    namespace: str = "default"
) -> bool

# Get value
client.get_kv(
    key: str,
    namespace: str = "default"
) -> Optional[bytes]

# Batch operations (atomic)
client.batch_put([
    (b"key1", b"value1"),
    (b"key2", b"value2"),
]) -> bool
```

**Temporal Graph Operations:**
```python
# Add time-bounded edge (gRPC)
client.add_temporal_edge(
    namespace: str,
    from_id: str,
    edge_type: str,
    to_id: str,
    valid_from: int,  # Unix timestamp (ms)
    valid_until: int = 0,  # 0 = no expiry
    properties: Optional[Dict] = None
) -> bool

# Query at specific point in time (gRPC)
edges = client.query_temporal_graph(
    namespace: str,
    node_id: str,
    mode: str = "POINT_IN_TIME",  # POINT_IN_TIME, RANGE, CURRENT
    timestamp: int = None,  # For POINT_IN_TIME
    start_time: int = None,  # For RANGE
    end_time: int = None,    # For RANGE
    edge_types: List[str] = None
) -> List[TemporalEdge]

# Same API available in embedded mode via Database class
db.add_temporal_edge(...)  # Direct FFI, no server needed
db.query_temporal_graph(...)  # Direct FFI, no server needed
```

**Use Cases for Temporal Graphs:**
- 🧠 **Agent Memory**: "Was door open 30 minutes ago?"
- 📊 **Audit Trail**: Track all state changes over time
- 🔍 **Time-Travel Debugging**: Query historical system state
- 🤖 **Multi-Agent Systems**: Each agent tracks beliefs over time

**Format Utilities:**
```python
from toondb import WireFormat, ContextFormat, FormatCapabilities

# Parse format from string
wire = WireFormat.from_string("json")  # WireFormat.JSON

# Convert between formats
ctx = FormatCapabilities.wire_to_context(WireFormat.JSON)
# Returns: ContextFormat.JSON

# Check round-trip support
supports = FormatCapabilities.supports_round_trip(WireFormat.TOON)
# Returns: True (TOON and JSON support round-trip)
```

### IpcClient (Unix Socket Transport)

For local inter-process communication:

```python
from toondb import IpcClient

# Connect via Unix socket
client = IpcClient.connect("/tmp/toondb.sock")

# Same API as ToonDBClient
client.put(b"key", b"value")
value = client.get(b"key")
```

---

## Data Types

### SearchResult
```python
@dataclass
class SearchResult:
    id: int           # Vector ID
    distance: float   # Similarity distance
```

### Document
```python
@dataclass
class Document:
    id: str                      # Document ID
    content: str                 # Text content
    embedding: List[float]       # Vector embedding
    metadata: Dict[str, str]     # Metadata
```

### GraphNode
```python
@dataclass
class GraphNode:
    id: str                      # Node ID
    node_type: str               # Node type
    properties: Dict[str, str]   # Properties
```

### GraphEdge
```python
@dataclass
class GraphEdge:
    from_id: str                 # Source node
    edge_type: str               # Edge type
    to_id: str                   # Target node
    properties: Dict[str, str]   # Properties
```

### TemporalEdge
```python
@dataclass
class TemporalEdge:
    from_id: str                 # Source node
    edge_type: str               # Edge type
    to_id: str                   # Target node
    valid_from: int              # Unix timestamp (ms)
    valid_until: int             # Unix timestamp (ms), 0 = no expiry
    properties: Dict[str, str]   # Properties
```

### WireFormat
```python
class WireFormat(Enum):
    TOON = "toon"        # 40-66% fewer tokens than JSON
    JSON = "json"        # Standard compatibility
    COLUMNAR = "columnar"  # Analytics optimized
```

### ContextFormat
```python
class ContextFormat(Enum):
    TOON = "toon"        # Token-efficient for LLMs
    JSON = "json"        # Structured data
    MARKDOWN = "markdown"  # Human-readable
```

---

## Advanced Features

### Temporal Graph Queries

Temporal graphs allow you to query "What did the system know at time T?"

**Use Case: Agent Memory with Time Travel**
```python
import time
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")

# Record that door was open from 10:00 to 11:00
now = int(time.time() * 1000)
one_hour = 60 * 60 * 1000

client.add_temporal_edge(
    namespace="agent_memory",
    from_id="door_1",
    edge_type="is_open",
    to_id="room_5",
    valid_from=now,
    valid_until=now + one_hour
)

# Query: "Was door_1 open 30 minutes ago?"
thirty_min_ago = now - (30 * 60 * 1000)
edges = client.query_temporal_graph(
    namespace="agent_memory",
    node_id="door_1",
    mode="POINT_IN_TIME",
    timestamp=thirty_min_ago
)

print(f"Door was open: {len(edges) > 0}")

# Query: "What changed in the last hour?"
edges = client.query_temporal_graph(
    namespace="agent_memory",
    node_id="door_1",
    mode="RANGE",
    start_time=now - one_hour,
    end_time=now
)
```

**Query Modes:**
- `POINT_IN_TIME`: Edges valid at specific timestamp
- `RANGE`: Edges overlapping a time range
- `CURRENT`: Edges valid right now

### Atomic Multi-Operation Writes

Ensure all-or-nothing semantics across multiple operations:

```python
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")

# All operations succeed or all fail atomically
client.batch_put([
    (b"user:alice:email", b"alice@example.com"),
    (b"user:alice:age", b"30"),
    (b"user:alice:created", b"2026-01-07"),
])

# If server crashes mid-batch, none of the writes persist
```

### Format Conversion for LLM Context

Optimize token usage when sending data to LLMs:

```python
from toondb import WireFormat, ContextFormat, FormatCapabilities

# Query results come in WireFormat
query_format = WireFormat.TOON  # 40-66% fewer tokens than JSON

# Convert to ContextFormat for LLM prompt
ctx_format = FormatCapabilities.wire_to_context(query_format)
# Returns: ContextFormat.TOON

# TOON format example:
# user:alice|email:alice@example.com,age:30
# vs JSON:
# {"user":"alice","email":"alice@example.com","age":30}

# Check if format supports decode(encode(x)) = x
is_lossless = FormatCapabilities.supports_round_trip(WireFormat.TOON)
# Returns: True (TOON and JSON are lossless)
```

**Format Benefits:**
- **TOON format**: 40-66% fewer tokens than JSON → Lower LLM API costs
- **Round-trip guarantee**: `decode(encode(x)) = x` for TOON and JSON
- **Columnar format**: Optimized for analytics queries with projections

---

## Error Handling

```python
from toondb import ToonDBError, ConnectionError

try:
    client = ToonDBClient("localhost:50051")
    client.create_collection("test", dimension=128)
except ConnectionError as e:
    print(f"Cannot connect to server: {e}")
except ToonDBError as e:
    print(f"ToonDB error: {e}")
```

**Error Types:**
- `ToonDBError` - Base exception
- `ConnectionError` - Cannot connect to server
- `TransactionError` - Transaction failed
- `ProtocolError` - Protocol mismatch
- `DatabaseError` - Server-side error

---

## Advanced Usage

### Connection with TLS
```python
client = ToonDBClient("api.example.com:50051", secure=True)
```

### Batch Operations
```python
# Insert multiple vectors at once
ids = list(range(1000))
vectors = [[...] for _ in range(1000)]  # 1000 vectors
client.insert_vectors("my_index", ids, vectors)
```

### Filtered Search
```python
# Search with metadata filtering
results = client.search_collection(
    "documents",
    query_vector,
    k=10,
    filter={"category": "AI", "year": "2024"}
)
```

---

## Performance

**Network Overhead:**
- gRPC: ~100-200 μs per request (local)
- IPC: ~50-100 μs per request (Unix socket)

**Batch Operations:**
- Vector insert: 50,000 vectors/sec (batch mode)
- Vector search: 20,000 queries/sec (47 μs/query)

**Recommendation:**
- Use **batch operations** for high throughput
- Use **IPC** for same-machine communication
- Use **gRPC** for distributed systems

---

## FAQ

**Q: Which mode should I use?**  
A: 
- **Embedded (FFI)**: For local dev, notebooks, single-process apps
- **Server (gRPC)**: For production, multi-language, distributed systems

**Q: Can I switch between modes?**  
A: Yes! Both modes have the same API. Change `Database.open()` to `ToonDBClient()` and vice versa.

**Q: Do temporal graphs work in embedded mode?**  
A: Yes! As of v0.3.4, temporal graphs work in both embedded and server modes with identical APIs.

**Q: Is embedded mode slower than server mode?**  
A: Embedded mode is faster for single-process use (no network overhead). Server mode is better for distributed deployments.

**Q: Where is the business logic?**  
A: All business logic is in Rust. Embedded mode uses FFI bindings, server mode uses gRPC. Same Rust code, different transport.

**Q: What about the old "fat client" Database class?**  
A: It's still here as embedded mode! We now support dual-mode: embedded FFI + server gRPC.

---

## Examples

See the [examples/](examples/) directory for complete working examples:

**Embedded Mode (FFI - No Server):**
- [23_collections_embedded.py](examples/23_collections_embedded.py) - Document storage, JSON, transactions
- [22_namespaces.py](examples/22_namespaces.py) - Multi-tenant isolation with key prefixes
- [24_batch_operations.py](examples/24_batch_operations.py) - Atomic writes, rollback, conditional updates
- [25_temporal_graph_embedded.py](examples/25_temporal_graph_embedded.py) - Time-travel queries (NEW!)

**Server Mode (gRPC - Requires Server):**
- [21_temporal_graph.py](examples/21_temporal_graph.py) - Temporal graphs via gRPC

---

## Getting Help

- **Documentation**: https://toondb.dev
- **GitHub Issues**: https://github.com/sushanthpy/toondb/issues
- **Examples**: See [examples/](examples/) directory

---

## Contributing

Interested in contributing? See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development environment setup
- Building from source  
- Running tests
- Code style guidelines
- Pull request process

---

## License

Apache License 2.0
