# ToonDB Python SDK v0.3.5

**Dual-mode architecture: Embedded (FFI) + Server (gRPC/IPC)**  
Choose the deployment mode that fits your needs.

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

# ToonDB Python SDK Documentation

**Version 0.3.4** | LLM-Optimized Embedded Database with Native Vector Search

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Key-Value Operations](#4-core-key-value-operations)
5. [Transactions (ACID with SSI)](#5-transactions-acid-with-ssi)
6. [Query Builder](#6-query-builder)
7. [Prefix Scanning](#7-prefix-scanning)
8. [SQL Operations](#8-sql-operations)
9. [Table Management & Index Policies](#9-table-management--index-policies)
10. [Namespaces & Multi-Tenancy](#10-namespaces--multi-tenancy)
11. [Collections & Vector Search](#11-collections--vector-search)
12. [Hybrid Search (Vector + BM25)](#12-hybrid-search-vector--bm25)
13. [Graph Operations](#13-graph-operations)
14. [Temporal Graph (Time-Travel)](#14-temporal-graph-time-travel)
15. [Semantic Cache](#15-semantic-cache)
16. [Context Query Builder (LLM Optimization) and Session](#16-context-query-builder-llm-optimization)
17. [Atomic Multi-Index Writes](#17-atomic-multi-index-writes)
18. [Recovery & WAL Management](#18-recovery--wal-management)
19. [Checkpoints & Snapshots](#19-checkpoints--snapshots)
20. [Compression & Storage](#20-compression--storage)
21. [Statistics & Monitoring](#21-statistics--monitoring)
22. [Distributed Tracing](#22-distributed-tracing)
23. [Workflow & Run Tracking](#23-workflow--run-tracking)
24. [Server Mode (gRPC Client)](#24-server-mode-grpc-client)
25. [IPC Client (Unix Sockets)](#25-ipc-client-unix-sockets)
26. [Standalone VectorIndex](#26-standalone-vectorindex)
27. [Vector Utilities](#27-vector-utilities)
28. [Data Formats (TOON/JSON/Columnar)](#28-data-formats-toonjsoncolumnar)
29. [Policy Service](#29-policy-service)
30. [MCP (Model Context Protocol)](#30-mcp-model-context-protocol)
31. [Configuration Reference](#31-configuration-reference)
32. [Error Handling](#32-error-handling)
33. [Async Support](#33-async-support)
34. [Building & Development](#34-building--development)
35. [Complete Examples](#35-complete-examples)
36. [Migration Guide](#36-migration-guide)

---

## 1. Quick Start

```python
from toondb import Database

# Open (or create) a database
db = Database.open("./my_database")

# Store and retrieve data
db.put(b"hello", b"world")
value = db.get(b"hello")  # b"world"

# Use transactions for atomic operations
with db.transaction() as txn:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    # Auto-commits on success, auto-rollbacks on exception

# Clean up
db.delete(b"hello")
db.close()
```

**30-Second Overview:**
- **Key-Value**: Fast reads/writes with `get`/`put`/`delete`
- **Transactions**: ACID with SSI isolation
- **Vector Search**: HNSW-based semantic search
- **Hybrid Search**: Combine vectors with BM25 keyword search
- **Graph**: Build and traverse knowledge graphs
- **LLM-Optimized**: TOON format uses 40-60% fewer tokens than JSON

---

## 2. Installation

```bash
pip install toondb
```

**Platform Support:**
| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x86_64, aarch64 | ✅ Full support |
| macOS | x86_64, arm64 | ✅ Full support |
| Windows | x86_64 | ✅ Full support |

**Optional Dependencies:**
```bash
# For async support
pip install toondb[async]

# For server mode
pip install toondb[grpc]

# Everything
pip install toondb[all]
```

---

## 3. Architecture Overview

ToonDB supports two deployment modes:

### Embedded Mode (Default)

Direct Rust bindings via FFI. No server required.

```python
from toondb import Database

with Database.open("./mydb") as db:
    db.put(b"key", b"value")
    value = db.get(b"key")
```

**Best for:** Local development, notebooks, single-process applications.

### Server Mode (gRPC)

Thin client connecting to `toondb-grpc` server.

```python
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")
client.put(b"key", b"value", namespace="default")
value = client.get(b"key", namespace="default")
```

**Best for:** Production, multi-process, distributed systems.

### Feature Comparison

| Feature | Embedded | Server |
|---------|----------|--------|
| Setup | `pip install` only | Server + client |
| Performance | Fastest (in-process) | Network overhead |
| Multi-process | ❌ | ✅ |
| Horizontal scaling | ❌ | ✅ |
| Vector search | ✅ | ✅ |
| Graph operations | ✅ | ✅ |
| Semantic cache | ✅ | ✅ |
| Context service | Limited | ✅ Full |
| MCP integration | ❌ | ✅ |

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT OPTIONS                        │
├─────────────────────────────────────────────────────────────┤
│  EMBEDDED MODE (FFI)             SERVER MODE (gRPC)         │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │   Python App        │         │   Python App        │   │
│  │   ├─ Database.open()│         │   ├─ ToonDBClient() │   │
│  │   └─ Direct FFI     │         │   └─ gRPC calls     │   │
│  │         │           │         │         │           │   │
│  │         ▼           │         │         ▼           │   │
│  │   libtoondb_storage │         │   toondb-grpc       │   │
│  │   (Rust native)     │         │   (Rust server)     │   │
│  └─────────────────────┘         └─────────────────────┘   │
│                                                              │
│  ✅ No server needed             ✅ Multi-language          │
│  ✅ Local files                  ✅ Centralized logic       │
│  ✅ Simple deployment            ✅ Production scale        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Core Key-Value Operations

All keys and values are **bytes**.

### Basic Operations

```python
from toondb import Database

db = Database.open("./my_db")

# Store data
db.put(b"user:1", b"Alice")
db.put(b"user:2", b"Bob")

# Retrieve data
user = db.get(b"user:1")  # Returns b"Alice" or None

# Check existence
exists = db.exists(b"user:1")  # True

# Delete data
db.delete(b"user:1")

db.close()
```

### Path-Based Keys (Hierarchical)

Organize data hierarchically with path-based access:

```python
# Store with path (strings auto-converted to bytes internally)
db.put_path("users/alice/name", b"Alice Smith")
db.put_path("users/alice/email", b"alice@example.com")
db.put_path("users/bob/name", b"Bob Jones")

# Retrieve by path
name = db.get_path("users/alice/name")  # b"Alice Smith"

# Delete by path
db.delete_path("users/alice/email")

# List at path (like listing directory)
children = db.list_path("users/")  # ["alice", "bob"]
```

### With TTL (Time-To-Live)

```python
# Store with expiration (seconds)
db.put(b"session:abc123", b"user_data", ttl_seconds=3600)  # Expires in 1 hour

# TTL of 0 means no expiration
db.put(b"permanent_key", b"value", ttl_seconds=0)
```

### Batch Operations

```python
# Batch put (more efficient than individual puts)
db.put_batch([
    (b"key1", b"value1"),
    (b"key2", b"value2"),
    (b"key3", b"value3"),
])

# Batch get
values = db.get_batch([b"key1", b"key2", b"key3"])
# Returns: [b"value1", b"value2", b"value3"] (None for missing keys)

# Batch delete
db.delete_batch([b"key1", b"key2", b"key3"])
```

### Context Manager

```python
with Database.open("./my_db") as db:
    db.put(b"key", b"value")
    # Automatically closes when exiting
```

---

## 5. Transactions (ACID with SSI)

ToonDB provides full ACID transactions with **Serializable Snapshot Isolation (SSI)**.

### Context Manager Pattern (Recommended)

```python
# Auto-commits on success, auto-rollbacks on exception
with db.transaction() as txn:
    txn.put(b"accounts/alice", b"1000")
    txn.put(b"accounts/bob", b"500")
    
    # Read within transaction sees your writes
    balance = txn.get(b"accounts/alice")  # b"1000"
    
    # If exception occurs, rolls back automatically
```

### Closure Pattern (Rust-Style)

```python
# Using with_transaction for automatic commit/rollback
def transfer_funds(txn):
    alice = int(txn.get(b"accounts/alice") or b"0")
    bob = int(txn.get(b"accounts/bob") or b"0")
    
    txn.put(b"accounts/alice", str(alice - 100).encode())
    txn.put(b"accounts/bob", str(bob + 100).encode())
    
    return "Transfer complete"

result = db.with_transaction(transfer_funds)
```

### Manual Transaction Control

```python
txn = db.begin_transaction()
try:
    txn.put(b"key1", b"value1")
    txn.put(b"key2", b"value2")
    
    commit_ts = txn.commit()  # Returns HLC timestamp
    print(f"Committed at: {commit_ts}")
except Exception as e:
    txn.abort()
    raise
```

### Transaction Properties

```python
txn = db.transaction()
print(f"Transaction ID: {txn.id}")      # Unique identifier
print(f"Start timestamp: {txn.start_ts}")  # HLC start time
print(f"Isolation: {txn.isolation}")    # "serializable"
```

### SSI Conflict Handling

```python
from toondb import TransactionConflictError

MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    try:
        with db.transaction() as txn:
            # Read and modify
            value = int(txn.get(b"counter") or b"0")
            txn.put(b"counter", str(value + 1).encode())
        break  # Success
    except TransactionConflictError:
        if attempt == MAX_RETRIES - 1:
            raise
        # Retry on conflict
        continue
```

### All Transaction Operations

```python
with db.transaction() as txn:
    # Key-value
    txn.put(key, value)
    txn.get(key)
    txn.delete(key)
    txn.exists(key)
    
    # Path-based
    txn.put_path(path, value)
    txn.get_path(path)
    
    # Batch operations
    txn.put_batch(pairs)
    txn.get_batch(keys)
    
    # Scanning
    for k, v in txn.scan_prefix(b"prefix/"):
        print(k, v)
    
    # SQL (within transaction isolation)
    result = txn.execute("SELECT * FROM users WHERE id = 1")
```

### Isolation Levels

```python
from toondb import IsolationLevel

# Default: Serializable (strongest)
with db.transaction(isolation=IsolationLevel.SERIALIZABLE) as txn:
    pass

# Snapshot isolation (faster, allows some anomalies)
with db.transaction(isolation=IsolationLevel.SNAPSHOT) as txn:
    pass

# Read committed (fastest, least isolation)
with db.transaction(isolation=IsolationLevel.READ_COMMITTED) as txn:
    pass
```

---

## 6. Query Builder

Fluent API for building efficient queries with predicate pushdown.

### Basic Query

```python
# Query with prefix and limit
results = db.query("users/")
    .limit(10)
    .execute()

for key, value in results:
    print(f"{key.decode()}: {value.decode()}")
```

### Filtered Query

```python
from toondb import CompareOp

# Query with filters
results = db.query("orders/")
    .where("status", CompareOp.EQ, "pending")
    .where("amount", CompareOp.GT, 100)
    .order_by("created_at", descending=True)
    .limit(50)
    .offset(10)
    .execute()
```

### Column Selection

```python
# Select specific fields only
results = db.query("users/")
    .select(["name", "email"])  # Only fetch these columns
    .where("active", CompareOp.EQ, True)
    .execute()
```

### Aggregate Queries

```python
# Count
count = db.query("orders/")
    .where("status", CompareOp.EQ, "completed")
    .count()

# Sum (for numeric columns)
total = db.query("orders/")
    .sum("amount")

# Group by
results = db.query("orders/")
    .select(["status", "COUNT(*)", "SUM(amount)"])
    .group_by("status")
    .execute()
```

### Query in Transaction

```python
with db.transaction() as txn:
    results = txn.query("users/")
        .where("role", CompareOp.EQ, "admin")
        .execute()
```

---

## 7. Prefix Scanning

Iterate over keys with common prefixes efficiently.

### Safe Prefix Scan (Recommended)

```python
# Requires minimum 2-byte prefix (prevents accidental full scans)
for key, value in db.scan_prefix(b"users/"):
    print(f"{key.decode()}: {value.decode()}")

# Raises ValueError if prefix < 2 bytes
```

### Unchecked Prefix Scan

```python
# For internal operations needing empty/short prefixes
# WARNING: Can cause expensive full-database scans
for key, value in db.scan_prefix_unchecked(b""):
    print(f"All keys: {key}")
```

### Batched Scanning (1000x Faster)

```python
# Fetches 1000 results per FFI call instead of 1
# Performance: 10,000 results = 10 FFI calls vs 10,000 calls

for key, value in db.scan_batched(b"prefix/", batch_size=1000):
    process(key, value)
```

### Reverse Scan

```python
# Scan in reverse order (newest first)
for key, value in db.scan_prefix(b"logs/", reverse=True):
    print(key, value)
```

### Range Scan

```python
# Scan within a specific range
for key, value in db.scan_range(b"users/a", b"users/m"):
    print(key, value)  # All users from "a" to "m"
```

### Streaming Large Results

```python
# For very large result sets, use streaming to avoid memory issues
for batch in db.scan_stream(b"logs/", batch_size=10000):
    for key, value in batch:
        process(key, value)
    # Memory is freed after processing each batch
```

---

## 8. SQL Operations

Execute SQL queries for familiar relational patterns.

### Creating Tables

```python
db.execute_sql("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        age INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")

db.execute_sql("""
    CREATE TABLE posts (
        id INTEGER PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        title TEXT NOT NULL,
        content TEXT,
        likes INTEGER DEFAULT 0
    )
""")
```

### CRUD Operations

```python
# Insert
db.execute_sql("""
    INSERT INTO users (id, name, email, age) 
    VALUES (1, 'Alice', 'alice@example.com', 30)
""")

# Insert with parameters (prevents SQL injection)
db.execute_sql(
    "INSERT INTO users (id, name, email, age) VALUES (?, ?, ?, ?)",
    params=[2, "Bob", "bob@example.com", 25]
)

# Select
result = db.execute_sql("SELECT * FROM users WHERE age > 25")
for row in result.rows:
    print(row)  # {'id': 1, 'name': 'Alice', ...}

# Update
db.execute_sql("UPDATE users SET email = 'alice.new@example.com' WHERE id = 1")

# Delete
db.execute_sql("DELETE FROM users WHERE id = 2")
```

### Upsert (Insert or Update)

```python
# Insert or update on conflict
db.execute_sql("""
    INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')
    ON CONFLICT (id) DO UPDATE SET 
        name = excluded.name,
        email = excluded.email
""")
```

### Query Results

```python
from toondb import SQLQueryResult

result = db.execute_sql("SELECT id, name FROM users")

print(f"Columns: {result.columns}")      # ['id', 'name']
print(f"Row count: {len(result.rows)}")
print(f"Execution time: {result.execution_time_ms}ms")

for row in result.rows:
    print(f"ID: {row['id']}, Name: {row['name']}")

# Convert to different formats
df = result.to_dataframe()  # pandas DataFrame
json_data = result.to_json()
```

### Index Management

```python
# Create index
db.execute_sql("CREATE INDEX idx_users_email ON users(email)")

# Create unique index
db.execute_sql("CREATE UNIQUE INDEX idx_users_email ON users(email)")

# Drop index
db.execute_sql("DROP INDEX IF EXISTS idx_users_email")

# List indexes
indexes = db.list_indexes("users")
```

### Prepared Statements

```python
# Prepare once, execute many times
stmt = db.prepare("SELECT * FROM users WHERE age > ? AND status = ?")

# Execute with different parameters
young_active = stmt.execute([25, "active"])
old_active = stmt.execute([50, "active"])

# Close when done
stmt.close()
```

### Dialect Support

ToonDB auto-detects SQL dialects:

```python
# PostgreSQL style
db.execute_sql("INSERT INTO users VALUES (1, 'Alice') ON CONFLICT DO NOTHING")

# MySQL style
db.execute_sql("INSERT IGNORE INTO users VALUES (1, 'Alice')")

# SQLite style  
db.execute_sql("INSERT OR IGNORE INTO users VALUES (1, 'Alice')")
```

---

## 9. Table Management & Index Policies

### Table Information

```python
# Get table schema
schema = db.get_table_schema("users")
print(f"Columns: {schema.columns}")
print(f"Primary key: {schema.primary_key}")
print(f"Indexes: {schema.indexes}")

# List all tables
tables = db.list_tables()

# Drop table
db.execute_sql("DROP TABLE IF EXISTS old_table")
```

### Index Policies

Configure per-table indexing strategies for optimal performance:

```python
# Policy constants
Database.INDEX_WRITE_OPTIMIZED  # 0 - O(1) insert, O(N) scan
Database.INDEX_BALANCED         # 1 - O(1) amortized insert, O(log K) scan
Database.INDEX_SCAN_OPTIMIZED   # 2 - O(log N) insert, O(log N + K) scan
Database.INDEX_APPEND_ONLY      # 3 - O(1) insert, O(N) scan (time-series)

# Set by constant
db.set_table_index_policy("logs", Database.INDEX_APPEND_ONLY)

# Set by string
db.set_table_index_policy("users", "scan_optimized")

# Get current policy
policy = db.get_table_index_policy("users")
print(f"Policy: {policy}")  # "scan_optimized"
```

### Policy Selection Guide

| Policy | Insert | Scan | Best For |
|--------|--------|------|----------|
| `write_optimized` | O(1) | O(N) | High-write ingestion |
| `balanced` | O(1) amortized | O(log K) | General use (default) |
| `scan_optimized` | O(log N) | O(log N + K) | Analytics, read-heavy |
| `append_only` | O(1) | O(N) | Time-series, logs |

---

## 10. Namespaces & Multi-Tenancy

Organize data into logical namespaces for tenant isolation.

### Creating Namespaces

```python
from toondb import NamespaceConfig

# Create namespace with metadata
ns = db.create_namespace(
    name="tenant_123",
    display_name="Acme Corp",
    labels={"tier": "premium", "region": "us-east"}
)

# Simple creation
ns = db.create_namespace("tenant_456")
```

### Getting Namespaces

```python
# Get existing namespace
ns = db.namespace("tenant_123")

# Get or create (idempotent)
ns = db.get_or_create_namespace("tenant_123")

# Check if exists
exists = db.namespace_exists("tenant_123")
```

### Context Manager for Scoped Operations

```python
with db.use_namespace("tenant_123") as ns:
    # All operations automatically scoped to tenant_123
    collection = ns.collection("documents")
    ns.put("config/key", b"value")
    
    # No need to specify namespace in each call
```

### Namespace Operations

```python
# List all namespaces
namespaces = db.list_namespaces()
print(namespaces)  # ['tenant_123', 'tenant_456']

# Get namespace info
info = db.namespace_info("tenant_123")
print(f"Created: {info['created_at']}")
print(f"Labels: {info['labels']}")
print(f"Size: {info['size_bytes']}")

# Update labels
db.update_namespace("tenant_123", labels={"tier": "enterprise"})

# Delete namespace (WARNING: deletes all data in namespace)
db.delete_namespace("old_tenant", force=True)
```

### Namespace-Scoped Key-Value

```python
ns = db.namespace("tenant_123")

# Operations automatically prefixed with namespace
ns.put("users/alice", b"data")      # Actually: tenant_123/users/alice
ns.get("users/alice")
ns.delete("users/alice")

# Scan within namespace
for key, value in ns.scan("users/"):
    print(key, value)  # Keys shown without namespace prefix
```

### Cross-Namespace Operations

```python
# Copy data between namespaces
db.copy_between_namespaces(
    source_ns="tenant_123",
    target_ns="tenant_456",
    prefix="shared/"
)
```

---

## 11. Collections & Vector Search

Collections store documents with embeddings for semantic search using HNSW.

### Collection Configuration

```python
from toondb import (
    CollectionConfig,
    DistanceMetric,
    QuantizationType,
)

config = CollectionConfig(
    name="documents",
    dimension=384,                          # Embedding dimension (must match your model)
    metric=DistanceMetric.COSINE,           # COSINE, EUCLIDEAN, DOT_PRODUCT
    m=16,                                   # HNSW M parameter (connections per node)
    ef_construction=100,                    # HNSW construction quality
    ef_search=50,                           # HNSW search quality (higher = slower but better)
    quantization=QuantizationType.NONE,     # NONE, SCALAR (int8), PQ (product quantization)
    enable_hybrid_search=False,             # Enable BM25 + vector
    content_field=None,                     # Field for BM25 indexing
)
```

### Creating Collections

```python
ns = db.namespace("default")

# With config object
collection = ns.create_collection(config)

# With parameters (simpler)
collection = ns.create_collection(
    name="documents",
    dimension=384,
    metric=DistanceMetric.COSINE
)

# Get existing collection
collection = ns.collection("documents")
```

### Inserting Documents

```python
# Single insert
collection.insert(
    id="doc1",
    vector=[0.1, 0.2, ...],  # 384-dim float array
    metadata={"title": "Introduction", "author": "Alice", "category": "tech"}
)

# Batch insert (more efficient for bulk loading)
collection.insert_batch(
    ids=["doc1", "doc2", "doc3"],
    vectors=[[...], [...], [...]],  # List of vectors
    metadata=[
        {"title": "Doc 1"},
        {"title": "Doc 2"},
        {"title": "Doc 3"}
    ]
)

# Multi-vector insert (multiple vectors per document, e.g., chunks)
collection.insert_multi(
    id="long_doc",
    vectors=[[...], [...], [...]],  # Multiple vectors for same doc
    metadata={"title": "Long Document"}
)
```

### Vector Search

```python
from toondb import SearchRequest

# Using SearchRequest (full control)
request = SearchRequest(
    vector=[0.15, 0.25, ...],       # Query vector
    k=10,                           # Number of results
    ef_search=100,                  # Search quality (overrides collection default)
    filter={"author": "Alice"},     # Metadata filter
    min_score=0.7,                  # Minimum similarity score
    include_vectors=False,          # Include vectors in results
    include_metadata=True,          # Include metadata in results
)
results = collection.search(request)

# Convenience method (simpler)
results = collection.vector_search(
    vector=[0.15, 0.25, ...],
    k=10,
    filter={"author": "Alice"}
)

# Process results
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score:.4f}")  # Similarity score
    print(f"Metadata: {result.metadata}")
```

### Metadata Filtering

```python
# Equality
filter={"author": "Alice"}

# Comparison operators
filter={"age": {"$gt": 30}}              # Greater than
filter={"age": {"$gte": 30}}             # Greater than or equal
filter={"age": {"$lt": 30}}              # Less than
filter={"age": {"$lte": 30}}             # Less than or equal
filter={"author": {"$ne": "Alice"}}      # Not equal

# Array operators
filter={"category": {"$in": ["tech", "science"]}}    # In array
filter={"category": {"$nin": ["sports"]}}            # Not in array

# Logical operators
filter={"$and": [{"author": "Alice"}, {"year": 2024}]}
filter={"$or": [{"category": "tech"}, {"category": "science"}]}
filter={"$not": {"author": "Bob"}}

# Nested filters
filter={
    "$and": [
        {"$or": [{"category": "tech"}, {"category": "science"}]},
        {"year": {"$gte": 2020}}
    ]
}
```

### Collection Management

```python
# Get collection
collection = ns.get_collection("documents")
# or
collection = ns.collection("documents")

# List collections
collections = ns.list_collections()

# Collection info
info = collection.info()
print(f"Name: {info['name']}")
print(f"Dimension: {info['dimension']}")
print(f"Count: {info['count']}")
print(f"Metric: {info['metric']}")
print(f"Index size: {info['index_size_bytes']}")

# Delete collection
ns.delete_collection("old_collection")

# Individual document operations
doc = collection.get("doc1")
collection.delete("doc1")
collection.update("doc1", metadata={"category": "updated"})
count = collection.count()
```

### Quantization for Memory Efficiency

```python
# Scalar quantization (int8) - 4x memory reduction
config = CollectionConfig(
    name="documents",
    dimension=384,
    quantization=QuantizationType.SCALAR
)

# Product quantization - 32x memory reduction
config = CollectionConfig(
    name="documents",
    dimension=768,
    quantization=QuantizationType.PQ,
    pq_num_subvectors=96,   # 768/96 = 8 dimensions per subvector
    pq_num_centroids=256    # 8-bit codes
)
```

---

## 12. Hybrid Search (Vector + BM25)

Combine vector similarity with keyword matching for best results.

### Enable Hybrid Search

```python
config = CollectionConfig(
    name="articles",
    dimension=384,
    enable_hybrid_search=True,      # Enable BM25 indexing
    content_field="text"            # Field to index for BM25
)
collection = ns.create_collection(config)

# Insert with text content
collection.insert(
    id="article1",
    vector=[...],
    metadata={
        "title": "Machine Learning Tutorial",
        "text": "This tutorial covers the basics of machine learning...",
        "category": "tech"
    }
)
```

### Keyword Search (BM25 Only)

```python
results = collection.keyword_search(
    query="machine learning tutorial",
    k=10,
    filter={"category": "tech"}
)
```

### Hybrid Search (Vector + BM25)

```python
# Combine vector and keyword search
results = collection.hybrid_search(
    vector=[0.1, 0.2, ...],        # Query embedding
    text_query="machine learning",  # Keyword query
    k=10,
    alpha=0.7,  # 0.0 = pure keyword, 1.0 = pure vector, 0.5 = balanced
    filter={"category": "tech"}
)
```

### Full SearchRequest for Hybrid

```python
request = SearchRequest(
    vector=[0.1, 0.2, ...],
    text_query="machine learning",
    k=10,
    alpha=0.7,                      # Blend factor
    rrf_k=60.0,                     # RRF k parameter (Reciprocal Rank Fusion)
    filter={"category": "tech"},
    aggregate="max",                # max | mean | first (for multi-vector docs)
    as_of="2024-01-01T00:00:00Z",   # Time-travel query
    include_vectors=False,
    include_metadata=True,
    include_scores=True,
)
results = collection.search(request)

# Access detailed results
print(f"Query time: {results.query_time_ms}ms")
print(f"Total matches: {results.total_count}")
print(f"Vector results: {results.vector_results}")    # Results from vector search
print(f"Keyword results: {results.keyword_results}")  # Results from BM25
print(f"Fused results: {results.fused_results}")      # Combined results
```

---

## 13. Graph Operations

Build and query knowledge graphs.

### Adding Nodes

```python
# Add a node
db.add_node(
    namespace="default",
    node_id="alice",
    node_type="person",
    properties={"role": "engineer", "team": "ml", "level": "senior"}
)

db.add_node("default", "project_x", "project", {"status": "active", "priority": "high"})
db.add_node("default", "bob", "person", {"role": "manager", "team": "ml"})
```

### Adding Edges

```python
# Add directed edge
db.add_edge(
    namespace="default",
    from_id="alice",
    edge_type="works_on",
    to_id="project_x",
    properties={"role": "lead", "since": "2024-01"}
)

db.add_edge("default", "alice", "reports_to", "bob")
db.add_edge("default", "bob", "manages", "project_x")
```

### Graph Traversal

```python
# BFS traversal from a starting node
nodes, edges = db.traverse(
    namespace="default",
    start_node="alice",
    max_depth=3,
    order="bfs"  # "bfs" or "dfs"
)

for node in nodes:
    print(f"Node: {node['id']} ({node['node_type']})")
    print(f"  Properties: {node['properties']}")
    
for edge in edges:
    print(f"{edge['from_id']} --{edge['edge_type']}--> {edge['to_id']}")
```

### Filtered Traversal

```python
# Traverse with filters
nodes, edges = db.traverse(
    namespace="default",
    start_node="alice",
    max_depth=2,
    edge_types=["works_on", "reports_to"],  # Only follow these edge types
    node_types=["person", "project"],        # Only include these node types
    node_filter={"team": "ml"}               # Filter nodes by properties
)
```

### Graph Queries

```python
# Find shortest path
path = db.find_path(
    namespace="default",
    from_id="alice",
    to_id="project_y",
    max_depth=5
)

# Get neighbors
neighbors = db.get_neighbors(
    namespace="default",
    node_id="alice",
    direction="outgoing"  # "outgoing", "incoming", "both"
)

# Get specific edge
edge = db.get_edge("default", "alice", "works_on", "project_x")

# Delete node (and all connected edges)
db.delete_node("default", "old_node")

# Delete edge
db.delete_edge("default", "alice", "works_on", "project_old")
```

---

## 14. Temporal Graph (Time-Travel)

Track state changes over time with temporal edges.

### Adding Temporal Edges

```python
import time

now = int(time.time() * 1000)  # milliseconds since epoch
one_hour = 60 * 60 * 1000

# Record: Door was open from 10:00 to 11:00
db.add_temporal_edge(
    namespace="smart_home",
    from_id="door_front",
    edge_type="STATE",
    to_id="open",
    valid_from=now - one_hour,  # Start time (ms)
    valid_until=now,             # End time (ms)
    properties={"sensor": "motion_1", "confidence": 0.95}
)

# Record: Light is currently on (no end time yet)
db.add_temporal_edge(
    namespace="smart_home",
    from_id="light_living",
    edge_type="STATE",
    to_id="on",
    valid_from=now,
    valid_until=0,  # 0 = still valid (no end time)
    properties={"brightness": "80%", "color": "warm"}
)
```

### Time-Travel Queries

```python
# Query modes:
# - "CURRENT": Edges valid right now
# - "POINT_IN_TIME": Edges valid at specific timestamp
# - "RANGE": All edges within a time range

# What is the current state?
edges = db.query_temporal_graph(
    namespace="smart_home",
    node_id="door_front",
    mode="CURRENT",
    edge_type="STATE"
)
current_state = edges[0]["to_id"] if edges else "unknown"

# Was the door open 1.5 hours ago?
edges = db.query_temporal_graph(
    namespace="smart_home",
    node_id="door_front",
    mode="POINT_IN_TIME",
    timestamp=now - int(1.5 * 60 * 60 * 1000)
)
was_open = any(e["to_id"] == "open" for e in edges)

# All state changes in last hour
edges = db.query_temporal_graph(
    namespace="smart_home",
    node_id="door_front",
    mode="RANGE",
    start_time=now - one_hour,
    end_time=now
)
for edge in edges:
    print(f"State: {edge['to_id']} from {edge['valid_from']} to {edge['valid_until']}")
```

### End a Temporal Edge

```python
# Close the current "on" state
db.end_temporal_edge(
    namespace="smart_home",
    from_id="light_living",
    edge_type="STATE",
    to_id="on",
    end_time=int(time.time() * 1000)
)
```

---

## 15. Semantic Cache

Cache LLM responses with similarity-based retrieval for cost savings.

### Storing Cached Responses

```python
# Store response with embedding
db.cache_put(
    cache_name="llm_responses",
    key="What is Python?",           # Original query (for display/debugging)
    value="Python is a high-level programming language...",
    embedding=[0.1, 0.2, ...],       # Query embedding (384-dim)
    ttl_seconds=3600,                # Expire in 1 hour (0 = no expiry)
    metadata={"model": "claude-3", "tokens": 150}
)
```

### Cache Lookup

```python
# Check cache before calling LLM
cached = db.cache_get(
    cache_name="llm_responses",
    query_embedding=[0.12, 0.18, ...],  # Embed the new query
    threshold=0.85                       # Cosine similarity threshold
)

if cached:
    print(f"Cache HIT!")
    print(f"Original query: {cached['key']}")
    print(f"Response: {cached['value']}")
    print(f"Similarity: {cached['score']:.4f}")
else:
    print("Cache MISS - calling LLM...")
    # Call LLM and cache the result
```

### Cache Management

```python
# Delete specific entry
db.cache_delete("llm_responses", key="What is Python?")

# Clear entire cache
db.cache_clear("llm_responses")

# Get cache statistics
stats = db.cache_stats("llm_responses")
print(f"Total entries: {stats['count']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats['size_bytes']}")
```

### Full Usage Pattern

```python
def get_llm_response(query: str, embed_fn, llm_fn):
    """Get response from cache or LLM."""
    query_embedding = embed_fn(query)
    
    # Try cache first
    cached = db.cache_get(
        cache_name="llm_responses",
        query_embedding=query_embedding,
        threshold=0.90
    )
    
    if cached:
        return cached['value']
    
    # Cache miss - call LLM
    response = llm_fn(query)
    
    # Store in cache
    db.cache_put(
        cache_name="llm_responses",
        key=query,
        value=response,
        embedding=query_embedding,
        ttl_seconds=86400  # 24 hours
    )
    
    return response
```

---

## 16. Context Query Builder (LLM Optimization)

Assemble LLM context with token budgeting and priority-based truncation.

### Basic Context Query

```python
from toondb import ContextQueryBuilder, ContextFormat, TruncationStrategy

# Build context for LLM
context = ContextQueryBuilder() \
    .for_session("session_123") \
    .with_budget(4096) \
    .format(ContextFormat.TOON) \
    .literal("SYSTEM", priority=0, text="You are a helpful assistant.") \
    .section("USER_PROFILE", priority=1) \
        .get("user.profile.{name, preferences}") \
        .done() \
    .section("HISTORY", priority=2) \
        .last(10, "messages") \
        .where_eq("session_id", "session_123") \
        .done() \
    .section("KNOWLEDGE", priority=3) \
        .search("documents", "$query_embedding", k=5) \
        .done() \
    .execute()

print(f"Token count: {context.token_count}")
print(f"Context:\n{context.text}")
```

### Section Types

| Type | Method | Description |
|------|--------|-------------|
| `literal` | `.literal(name, priority, text)` | Static text content |
| `get` | `.get(path)` | Fetch specific data by path |
| `last` | `.last(n, table)` | Most recent N records from table |
| `search` | `.search(collection, embedding, k)` | Vector similarity search |
| `sql` | `.sql(query)` | SQL query results |

### Truncation Strategies

```python
# Drop from end (keep beginning) - default
.truncation(TruncationStrategy.TAIL_DROP)

# Drop from beginning (keep end)
.truncation(TruncationStrategy.HEAD_DROP)

# Proportionally truncate across sections
.truncation(TruncationStrategy.PROPORTIONAL)

# Fail if budget exceeded
.truncation(TruncationStrategy.STRICT)
```

### Variables and Bindings

```python
from toondb import ContextValue

context = ContextQueryBuilder() \
    .for_session("session_123") \
    .set_var("query_embedding", ContextValue.Embedding([0.1, 0.2, ...])) \
    .set_var("user_id", ContextValue.String("user_456")) \
    .section("KNOWLEDGE", priority=2) \
        .search("documents", "$query_embedding", k=5) \
        .done() \
    .execute()
```

### Output Formats

```python
# TOON format (40-60% fewer tokens)
.format(ContextFormat.TOON)

# JSON format
.format(ContextFormat.JSON)

# Markdown format (human-readable)
.format(ContextFormat.MARKDOWN)

# Plain text
.format(ContextFormat.TEXT)
```


## Session Management (Agent Context)

Stateful session management for agentic use cases with permissions, sandboxing, audit logging, and budget tracking.

### Session Overview

```
Agent session abc123:
  cwd: /agents/abc123
  vars: $model = "gpt-4", $budget = 1000
  permissions: fs:rw, db:rw, calc:*
  audit: [read /data/users, write /agents/abc123/cache]
```

### Creating Sessions

```python
from toondb import SessionManager, AgentContext
from datetime import timedelta

# Create session manager with idle timeout
session_mgr = SessionManager(idle_timeout=timedelta(hours=1))

# Create a new session
session = session_mgr.create_session("session_abc123")

# Get existing session
session = session_mgr.get_session("session_abc123")

# Get or create (idempotent)
session = session_mgr.get_or_create("session_abc123")

# Remove session
session_mgr.remove_session("session_abc123")

# Cleanup expired sessions
removed_count = session_mgr.cleanup_expired()

# Get active session count
count = session_mgr.session_count()
```

### Agent Context

```python
from toondb import AgentContext, ContextValue

# Create agent context
ctx = AgentContext("session_abc123")
print(f"Session ID: {ctx.session_id}")
print(f"Working dir: {ctx.working_dir}")  # /agents/session_abc123

# Create with custom working directory
ctx = AgentContext.with_working_dir("session_abc123", "/custom/path")

# Create with full permissions (trusted agents)
ctx = AgentContext.with_full_permissions("session_abc123")
```

### Session Variables

```python
# Set variables
ctx.set_var("model", ContextValue.String("gpt-4"))
ctx.set_var("budget", ContextValue.Number(1000.0))
ctx.set_var("debug", ContextValue.Bool(True))
ctx.set_var("tags", ContextValue.List([
    ContextValue.String("ml"),
    ContextValue.String("production")
]))

# Get variables
model = ctx.get_var("model")  # Returns ContextValue or None
budget = ctx.get_var("budget")

# Peek (read-only, no audit)
value = ctx.peek_var("model")

# Variable substitution in strings
text = ctx.substitute_vars("Using $model with budget $budget")
# Result: "Using gpt-4 with budget 1000"
```

### Context Value Types

```python
from toondb import ContextValue

# String
ContextValue.String("hello")

# Number (float)
ContextValue.Number(42.5)

# Boolean
ContextValue.Bool(True)

# List
ContextValue.List([
    ContextValue.String("a"),
    ContextValue.Number(1.0)
])

# Object (dict)
ContextValue.Object({
    "key": ContextValue.String("value"),
    "count": ContextValue.Number(10.0)
})

# Null
ContextValue.Null()
```

### Permissions

```python
from toondb import (
    AgentPermissions,
    FsPermissions,
    DbPermissions,
    NetworkPermissions
)

# Configure permissions
ctx.permissions = AgentPermissions(
    filesystem=FsPermissions(
        read=True,
        write=True,
        mkdir=True,
        delete=False,
        allowed_paths=["/agents/session_abc123", "/shared/data"]
    ),
    database=DbPermissions(
        read=True,
        write=True,
        create=False,
        drop=False,
        allowed_tables=["user_*", "cache_*"]  # Pattern matching
    ),
    calculator=True,
    network=NetworkPermissions(
        http=True,
        allowed_domains=["api.example.com", "*.internal.net"]
    )
)

# Check permissions before operations
try:
    ctx.check_fs_permission("/agents/session_abc123/data.json", AuditOperation.FS_READ)
    # Permission granted
except ContextError as e:
    print(f"Permission denied: {e}")

try:
    ctx.check_db_permission("user_profiles", AuditOperation.DB_QUERY)
    # Permission granted
except ContextError as e:
    print(f"Permission denied: {e}")
```

### Budget Tracking

```python
from toondb import OperationBudget

# Configure budget limits
ctx.budget = OperationBudget(
    max_tokens=100000,        # Maximum tokens (input + output)
    max_cost=5000,            # Maximum cost in millicents ($50.00)
    max_operations=10000      # Maximum operation count
)

# Consume budget (called automatically by operations)
try:
    ctx.consume_budget(tokens=500, cost=10)  # 500 tokens, $0.10
except ContextError as e:
    if "Budget exceeded" in str(e):
        print("Budget limit reached!")

# Check budget status
print(f"Tokens used: {ctx.budget.tokens_used}/{ctx.budget.max_tokens}")
print(f"Cost used: ${ctx.budget.cost_used / 100:.2f}/${ctx.budget.max_cost / 100:.2f}")
print(f"Operations: {ctx.budget.operations_used}/{ctx.budget.max_operations}")
```

### Session Transactions

```python
# Begin transaction within session
ctx.begin_transaction(tx_id=12345)

# Create savepoint
ctx.savepoint("before_update")

# Record pending writes (for rollback)
ctx.record_pending_write(
    resource_type=ResourceType.FILE,
    resource_key="/agents/session_abc123/data.json",
    original_value=b'{"old": "data"}'
)

# Commit transaction
ctx.commit_transaction()

# Or rollback
pending_writes = ctx.rollback_transaction()
for write in pending_writes:
    print(f"Rolling back: {write.resource_key}")
    # Restore original_value
```

### Path Resolution

```python
# Paths are resolved relative to working directory
ctx = AgentContext.with_working_dir("session_abc123", "/home/agent")

# Relative paths
resolved = ctx.resolve_path("data.json")  # /home/agent/data.json

# Absolute paths pass through
resolved = ctx.resolve_path("/absolute/path")  # /absolute/path
```

### Audit Trail

```python
# All operations are automatically logged
# Audit entry includes: timestamp, operation, resource, result, metadata

# Export audit log
audit_log = ctx.export_audit()
for entry in audit_log:
    print(f"[{entry['timestamp']}] {entry['operation']}: {entry['resource']} -> {entry['result']}")

# Example output:
# [1705312345] var.set: model -> success
# [1705312346] fs.read: /data/config.json -> success
# [1705312347] db.query: users -> success
# [1705312348] fs.write: /forbidden/file -> denied:path not in allowed paths
```

### Audit Operations

```python
from toondb import AuditOperation

# Filesystem operations
AuditOperation.FS_READ
AuditOperation.FS_WRITE
AuditOperation.FS_MKDIR
AuditOperation.FS_DELETE
AuditOperation.FS_LIST

# Database operations
AuditOperation.DB_QUERY
AuditOperation.DB_INSERT
AuditOperation.DB_UPDATE
AuditOperation.DB_DELETE

# Other operations
AuditOperation.CALCULATE
AuditOperation.VAR_SET
AuditOperation.VAR_GET
AuditOperation.TX_BEGIN
AuditOperation.TX_COMMIT
AuditOperation.TX_ROLLBACK
```

### Tool Registry

```python
from toondb import ToolDefinition, ToolCallRecord
from datetime import datetime

# Register tools available to the agent
ctx.register_tool(ToolDefinition(
    name="search_documents",
    description="Search documents by semantic similarity",
    parameters_schema='{"type": "object", "properties": {"query": {"type": "string"}}}',
    requires_confirmation=False
))

ctx.register_tool(ToolDefinition(
    name="delete_file",
    description="Delete a file from the filesystem",
    parameters_schema='{"type": "object", "properties": {"path": {"type": "string"}}}',
    requires_confirmation=True  # Requires user confirmation
))

# Record tool calls
ctx.record_tool_call(ToolCallRecord(
    call_id="call_001",
    tool_name="search_documents",
    arguments='{"query": "machine learning"}',
    result='[{"id": "doc1", "score": 0.95}]',
    error=None,
    timestamp=datetime.now()
))

# Access tool call history
for call in ctx.tool_calls:
    print(f"{call.tool_name}: {call.result or call.error}")
```

### Session Lifecycle

```python
# Check session age
age = ctx.age()
print(f"Session age: {age}")

# Check idle time
idle = ctx.idle_time()
print(f"Idle time: {idle}")

# Check if expired
if ctx.is_expired(idle_timeout=timedelta(hours=1)):
    print("Session has expired!")
```

### Complete Session Example

```python
from toondb import (
    SessionManager, AgentContext, ContextValue,
    AgentPermissions, FsPermissions, DbPermissions,
    OperationBudget, ToolDefinition, AuditOperation
)
from datetime import timedelta

# Initialize session manager
session_mgr = SessionManager(idle_timeout=timedelta(hours=2))

# Create session for an agent
session_id = "agent_session_12345"
ctx = session_mgr.get_or_create(session_id)

# Configure the agent
ctx.permissions = AgentPermissions(
    filesystem=FsPermissions(
        read=True,
        write=True,
        allowed_paths=[f"/agents/{session_id}", "/shared"]
    ),
    database=DbPermissions(
        read=True,
        write=True,
        allowed_tables=["documents", "cache_*"]
    ),
    calculator=True
)

ctx.budget = OperationBudget(
    max_tokens=50000,
    max_cost=1000,  # $10.00
    max_operations=1000
)

# Set initial variables
ctx.set_var("model", ContextValue.String("claude-3-sonnet"))
ctx.set_var("temperature", ContextValue.Number(0.7))

# Register available tools
ctx.register_tool(ToolDefinition(
    name="vector_search",
    description="Search vectors by similarity",
    parameters_schema='{"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}}',
    requires_confirmation=False
))

# Perform operations with permission checks
def safe_read_file(ctx: AgentContext, path: str) -> bytes:
    resolved = ctx.resolve_path(path)
    ctx.check_fs_permission(resolved, AuditOperation.FS_READ)
    ctx.consume_budget(tokens=100, cost=1)
    # ... actual file read ...
    return b"file contents"

def safe_db_query(ctx: AgentContext, table: str, query: str):
    ctx.check_db_permission(table, AuditOperation.DB_QUERY)
    ctx.consume_budget(tokens=500, cost=5)
    # ... actual query ...
    return []

# Use in transaction
ctx.begin_transaction(tx_id=1)
try:
    # Operations here...
    ctx.commit_transaction()
except Exception as e:
    ctx.rollback_transaction()
    raise

# Export audit trail for debugging/compliance
audit = ctx.export_audit()
print(f"Session performed {len(audit)} operations")

# Cleanup
session_mgr.cleanup_expired()
```

### Session Errors

```python
from toondb import ContextError

try:
    ctx.check_fs_permission("/forbidden", AuditOperation.FS_READ)
except ContextError as e:
    if e.is_permission_denied():
        print(f"Permission denied: {e.message}")
    elif e.is_variable_not_found():
        print(f"Variable not found: {e.variable_name}")
    elif e.is_budget_exceeded():
        print(f"Budget exceeded: {e.budget_type}")
    elif e.is_transaction_error():
        print(f"Transaction error: {e.message}")
    elif e.is_invalid_path():
        print(f"Invalid path: {e.path}")
    elif e.is_session_expired():
        print("Session has expired")
```
---

## 17. Atomic Multi-Index Writes

Ensure consistency across KV storage, vectors, and graphs with atomic operations.

### Problem Without Atomicity

```
# Without atomic writes, a crash can leave:
# - Embedding exists but graph edges don't
# - KV data exists but embedding is missing
# - Partial graph relationships
```

### Atomic Memory Writer

```python
from toondb import AtomicMemoryWriter, MemoryOp

writer = AtomicMemoryWriter(db)

# Build atomic operation set
result = writer.write_atomic(
    memory_id="memory_123",
    ops=[
        # Store the blob/content
        MemoryOp.PutBlob(
            key=b"memories/memory_123/content",
            value=b"Meeting notes: discussed project timeline..."
        ),
        
        # Store the embedding
        MemoryOp.PutEmbedding(
            collection="memories",
            id="memory_123",
            embedding=[0.1, 0.2, ...],
            metadata={"type": "meeting", "date": "2024-01-15"}
        ),
        
        # Create graph nodes
        MemoryOp.CreateNode(
            namespace="default",
            node_id="memory_123",
            node_type="memory",
            properties={"importance": "high"}
        ),
        
        # Create graph edges
        MemoryOp.CreateEdge(
            namespace="default",
            from_id="memory_123",
            edge_type="relates_to",
            to_id="project_x",
            properties={}
        ),
    ]
)

print(f"Intent ID: {result.intent_id}")
print(f"Operations applied: {result.ops_applied}")
print(f"Status: {result.status}")  # "committed"
```

### How It Works

```
1. Write intent(id, ops...) to WAL    ← Crash-safe
2. Apply ops one-by-one
3. Write commit(id) to WAL            ← All-or-nothing
4. Recovery replays incomplete intents
```

---

## 18. Recovery & WAL Management

ToonDB uses Write-Ahead Logging (WAL) for durability with automatic recovery.

### Recovery Manager

```python
from toondb import RecoveryManager

recovery = db.recovery()

# Check if recovery is needed
if recovery.needs_recovery():
    result = recovery.recover()
    print(f"Status: {result.status}")
    print(f"Replayed entries: {result.replayed_entries}")
```

### WAL Verification

```python
# Verify WAL integrity
result = recovery.verify_wal()

print(f"Valid: {result.is_valid}")
print(f"Total entries: {result.total_entries}")
print(f"Valid entries: {result.valid_entries}")
print(f"Corrupted: {result.corrupted_entries}")
print(f"Last valid LSN: {result.last_valid_lsn}")

if result.checksum_errors:
    for error in result.checksum_errors:
        print(f"Checksum error at LSN {error.lsn}: expected {error.expected}, got {error.actual}")
```

### Force Checkpoint

```python
# Force a checkpoint (flush memtable to disk)
result = recovery.checkpoint()

print(f"Checkpoint LSN: {result.checkpoint_lsn}")
print(f"Duration: {result.duration_ms}ms")
```

### WAL Statistics

```python
stats = recovery.wal_stats()

print(f"Total size: {stats.total_size_bytes} bytes")
print(f"Active size: {stats.active_size_bytes} bytes")
print(f"Archived size: {stats.archived_size_bytes} bytes")
print(f"Entry count: {stats.entry_count}")
print(f"Oldest LSN: {stats.oldest_entry_lsn}")
print(f"Newest LSN: {stats.newest_entry_lsn}")
```

### WAL Truncation

```python
# Truncate WAL after checkpoint (reclaim disk space)
result = recovery.truncate_wal(up_to_lsn=12345)

print(f"Truncated to LSN: {result.up_to_lsn}")
print(f"Bytes freed: {result.bytes_freed}")
```

### Open with Auto-Recovery

```python
from toondb import open_with_recovery

# Automatically recovers if needed
db = open_with_recovery("./my_database")
```

---

## 19. Checkpoints & Snapshots

### Application Checkpoints

Save and restore application state for workflow interruption/resumption.

```python
from toondb import CheckpointService

checkpoint_svc = db.checkpoint_service()

# Create a checkpoint
checkpoint_id = checkpoint_svc.create(
    name="workflow_step_3",
    state=serialized_state,  # bytes
    metadata={"step": "3", "user": "alice", "workflow": "data_pipeline"}
)

# Restore checkpoint
state = checkpoint_svc.restore(checkpoint_id)

# List checkpoints
checkpoints = checkpoint_svc.list()
for cp in checkpoints:
    print(f"{cp.name}: {cp.created_at}, {cp.state_size} bytes")

# Delete checkpoint
checkpoint_svc.delete(checkpoint_id)
```

### Workflow Checkpointing

```python
# Create a workflow run
run_id = checkpoint_svc.create_run(
    workflow="data_pipeline",
    params={"input_file": "data.csv", "batch_size": 1000}
)

# Save checkpoint at each node/step
checkpoint_svc.save_node_checkpoint(
    run_id=run_id,
    node_id="transform_step",
    state=step_state,
    metadata={"rows_processed": 5000}
)

# Load latest checkpoint for a node
checkpoint = checkpoint_svc.load_node_checkpoint(run_id, "transform_step")

# List all checkpoints for a run
node_checkpoints = checkpoint_svc.list_run_checkpoints(run_id)
```

### Snapshot Reader (Point-in-Time)

```python
# Create a consistent snapshot for reading
snapshot = db.snapshot()

# Read from snapshot (doesn't see newer writes)
value = snapshot.get(b"key")

# All reads within snapshot see consistent state
with db.snapshot() as snap:
    v1 = snap.get(b"key1")
    v2 = snap.get(b"key2")  # Same consistent view

# Meanwhile, writes continue in main DB
db.put(b"key1", b"new_value")  # Snapshot doesn't see this
```

---

## 20. Compression & Storage

### Compression Settings

```python
from toondb import CompressionType

db = Database.open("./my_db", config={
    # Compression for SST files
    "compression": CompressionType.LZ4,      # LZ4 (fast), ZSTD (better ratio), NONE
    "compression_level": 3,                   # ZSTD: 1-22, LZ4: ignored
    
    # Compression for WAL
    "wal_compression": CompressionType.NONE,  # Usually NONE for WAL (already sequential)
})
```

### Compression Comparison

| Type | Ratio | Compress Speed | Decompress Speed | Use Case |
|------|-------|----------------|------------------|----------|
| `NONE` | 1x | N/A | N/A | Already compressed data |
| `LZ4` | ~2.5x | ~780 MB/s | ~4500 MB/s | General use (default) |
| `ZSTD` | ~3.5x | ~520 MB/s | ~1800 MB/s | Cold storage, large datasets |

### Storage Statistics

```python
stats = db.storage_stats()

print(f"Data size: {stats.data_size_bytes}")
print(f"Index size: {stats.index_size_bytes}")
print(f"WAL size: {stats.wal_size_bytes}")
print(f"Compression ratio: {stats.compression_ratio:.2f}x")
print(f"SST files: {stats.sst_file_count}")
print(f"Levels: {stats.level_stats}")
```

### Compaction Control

```python
# Manual compaction (reclaim space, optimize reads)
db.compact()

# Compact specific level
db.compact_level(level=0)

# Get compaction stats
stats = db.compaction_stats()
print(f"Pending compactions: {stats.pending_compactions}")
print(f"Running compactions: {stats.running_compactions}")
```

---

## 21. Statistics & Monitoring

### Database Statistics

```python
stats = db.stats()

# Transaction stats
print(f"Active transactions: {stats.active_transactions}")
print(f"Committed transactions: {stats.committed_transactions}")
print(f"Aborted transactions: {stats.aborted_transactions}")
print(f"Conflict rate: {stats.conflict_rate:.2%}")

# Operation stats
print(f"Total reads: {stats.total_reads}")
print(f"Total writes: {stats.total_writes}")
print(f"Cache hit rate: {stats.cache_hit_rate:.2%}")

# Storage stats
print(f"Key count: {stats.key_count}")
print(f"Total data size: {stats.total_data_bytes}")
```

### Token Statistics (LLM Optimization)

```python
stats = db.token_stats()

print(f"TOON tokens emitted: {stats.toon_tokens_emitted}")
print(f"Equivalent JSON tokens: {stats.json_tokens_equivalent}")
print(f"Token savings: {stats.token_savings_percent:.1f}%")
```

### Performance Metrics

```python
metrics = db.performance_metrics()

# Latency percentiles
print(f"Read P50: {metrics.read_latency_p50_us}µs")
print(f"Read P99: {metrics.read_latency_p99_us}µs")
print(f"Write P50: {metrics.write_latency_p50_us}µs")
print(f"Write P99: {metrics.write_latency_p99_us}µs")

# Throughput
print(f"Reads/sec: {metrics.reads_per_second}")
print(f"Writes/sec: {metrics.writes_per_second}")
```

---

## 22. Distributed Tracing

Track operations for debugging and performance analysis.

### Starting Traces

```python
from toondb import TraceStore

traces = TraceStore(db)

# Start a trace run
run = traces.start_run(
    name="user_request",
    resource={"service": "api", "version": "1.0.0"}
)
trace_id = run.trace_id
```

### Creating Spans

```python
from toondb import SpanKind, SpanStatusCode

# Start root span
root_span = traces.start_span(
    trace_id=trace_id,
    name="handle_request",
    parent_span_id=None,
    kind=SpanKind.SERVER
)

# Start child span
db_span = traces.start_span(
    trace_id=trace_id,
    name="database_query",
    parent_span_id=root_span.span_id,
    kind=SpanKind.CLIENT
)

# Add attributes
traces.set_span_attributes(trace_id, db_span.span_id, {
    "db.system": "toondb",
    "db.operation": "SELECT",
    "db.table": "users"
})

# End spans
traces.end_span(trace_id, db_span.span_id, SpanStatusCode.OK)
traces.end_span(trace_id, root_span.span_id, SpanStatusCode.OK)

# End the trace run
traces.end_run(trace_id, TraceStatus.COMPLETED)
```

### Domain Events

```python
# Log retrieval (for RAG debugging)
traces.log_retrieval(
    trace_id=trace_id,
    query="user query",
    results=[{"id": "doc1", "score": 0.95}],
    latency_ms=15
)

# Log LLM call
traces.log_llm_call(
    trace_id=trace_id,
    model="claude-3-sonnet",
    input_tokens=500,
    output_tokens=200,
    latency_ms=1200
)
```

---

## 23. Workflow & Run Tracking

Track long-running workflows with events and state.

### Creating Workflow Runs

```python
from toondb import WorkflowService, RunStatus

workflow_svc = db.workflow_service()

# Create a new run
run = workflow_svc.create_run(
    run_id="run_123",
    workflow="data_pipeline",
    params={"input": "data.csv", "output": "results.json"}
)

print(f"Run ID: {run.run_id}")
print(f"Status: {run.status}")
print(f"Created: {run.created_at}")
```

### Appending Events

```python
from toondb import WorkflowEvent, EventType

# Append events as workflow progresses
workflow_svc.append_event(WorkflowEvent(
    run_id="run_123",
    event_type=EventType.NODE_STARTED,
    node_id="extract",
    data={"input_file": "data.csv"}
))

workflow_svc.append_event(WorkflowEvent(
    run_id="run_123",
    event_type=EventType.NODE_COMPLETED,
    node_id="extract",
    data={"rows_extracted": 10000}
))
```

### Querying Events

```python
# Get all events for a run
events = workflow_svc.get_events("run_123")

# Get events since a sequence number
new_events = workflow_svc.get_events("run_123", since_seq=10, limit=100)

# Stream events (for real-time monitoring)
for event in workflow_svc.stream_events("run_123"):
    print(f"[{event.seq}] {event.event_type}: {event.node_id}")
```

### Update Run Status

```python
# Update status
workflow_svc.update_run_status("run_123", RunStatus.COMPLETED)

# Or mark as failed
workflow_svc.update_run_status("run_123", RunStatus.FAILED)
```

---

## 24. Server Mode (gRPC Client)

Full-featured client for distributed deployments.

### Connection

```python
from toondb import ToonDBClient

# Basic connection
client = ToonDBClient("localhost:50051")

# With TLS
client = ToonDBClient("localhost:50051", secure=True, ca_cert="ca.pem")

# With authentication
client = ToonDBClient("localhost:50051", api_key="your_api_key")

# Context manager
with ToonDBClient("localhost:50051") as client:
    client.put(b"key", b"value")
```

### Key-Value Operations

```python
# Put with TTL
client.put(b"key", b"value", namespace="default", ttl_seconds=3600)

# Get
value = client.get(b"key", namespace="default")

# Delete
client.delete(b"key", namespace="default")

# Batch operations
client.put_batch([
    (b"key1", b"value1"),
    (b"key2", b"value2"),
], namespace="default")
```

### Vector Operations (Server Mode)

```python
# Create index
client.create_index(
    name="embeddings",
    dimension=384,
    metric="cosine",
    m=16,
    ef_construction=200
)

# Insert vectors
client.insert_vectors(
    index_name="embeddings",
    ids=[1, 2, 3],
    vectors=[[...], [...], [...]]
)

# Search
results = client.search(
    index_name="embeddings",
    query=[0.1, 0.2, ...],
    k=10,
    ef_search=50
)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")
```

### Collection Operations (Server Mode)

```python
# Create collection
client.create_collection(
    name="documents",
    dimension=384,
    namespace="default",
    metric="cosine"
)

# Add documents
client.add_documents(
    collection_name="documents",
    documents=[
        {"id": "1", "content": "Hello", "embedding": [...], "metadata": {...}},
        {"id": "2", "content": "World", "embedding": [...], "metadata": {...}}
    ],
    namespace="default"
)

# Search
results = client.search_collection(
    collection_name="documents",
    query_vector=[...],
    k=10,
    namespace="default",
    filter={"author": "Alice"}
)
```

### Context Service (Server Mode)

```python
# Query context for LLM
context = client.query_context(
    session_id="session_123",
    sections=[
        {"name": "system", "priority": 0, "type": "literal", 
         "content": "You are a helpful assistant."},
        {"name": "history", "priority": 1, "type": "recent", 
         "table": "messages", "top_k": 10},
        {"name": "knowledge", "priority": 2, "type": "search", 
         "collection": "documents", "embedding": [...], "top_k": 5}
    ],
    token_limit=4096,
    format="toon"
)

print(context.text)
print(f"Tokens used: {context.token_count}")
```

---

## 25. IPC Client (Unix Sockets)

Local server communication via Unix sockets (lower latency than gRPC).

```python
from toondb import IpcClient

# Connect
client = IpcClient.connect("/tmp/toondb.sock", timeout=30.0)

# Basic operations
client.put(b"key", b"value")
value = client.get(b"key")
client.delete(b"key")

# Path operations
client.put_path(["users", "alice"], b"data")
value = client.get_path(["users", "alice"])

# Query
result = client.query("users/", limit=100)

# Scan
results = client.scan("prefix/")

# Transactions
txn_id = client.begin_transaction()
# ... operations ...
commit_ts = client.commit(txn_id)
# or client.abort(txn_id)

# Admin
client.ping()
client.checkpoint()
stats = client.stats()

client.close()
```

---

## 26. Standalone VectorIndex

Direct HNSW index operations without collections.

```python
from toondb import VectorIndex, VectorIndexConfig, DistanceMetric
import numpy as np

# Create index
config = VectorIndexConfig(
    dimension=384,
    metric=DistanceMetric.COSINE,
    m=16,
    ef_construction=200,
    ef_search=50,
    max_elements=100000
)
index = VectorIndex(config)

# Insert single vector
index.insert(id=1, vector=np.array([0.1, 0.2, ...], dtype=np.float32))

# Batch insert
ids = np.array([1, 2, 3], dtype=np.uint64)
vectors = np.array([[...], [...], [...]], dtype=np.float32)
count = index.insert_batch(ids, vectors)

# Fast batch insert (returns failures)
inserted, failed = index.insert_batch_fast(ids, vectors)

# Search
query = np.array([0.1, 0.2, ...], dtype=np.float32)
results = index.search(query, k=10, ef_search=100)

for id, distance in results:
    print(f"ID: {id}, Distance: {distance}")

# Properties
print(f"Size: {len(index)}")
print(f"Dimension: {index.dimension}")

# Save/load
index.save("./index.bin")
index = VectorIndex.load("./index.bin")
```

---

## 27. Vector Utilities

Standalone vector operations for preprocessing and analysis.

```python
from toondb import vector

# Distance calculations
a = [1.0, 0.0, 0.0]
b = [0.707, 0.707, 0.0]

cosine_dist = vector.cosine_distance(a, b)
euclidean_dist = vector.euclidean_distance(a, b)
dot_product = vector.dot_product(a, b)

print(f"Cosine distance: {cosine_dist:.4f}")
print(f"Euclidean distance: {euclidean_dist:.4f}")
print(f"Dot product: {dot_product:.4f}")

# Normalize a vector
v = [3.0, 4.0]
normalized = vector.normalize(v)
print(f"Normalized: {normalized}")  # [0.6, 0.8]

# Batch normalize
vectors = [[3.0, 4.0], [1.0, 0.0]]
normalized_batch = vector.normalize_batch(vectors)

# Compute centroid
vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
centroid = vector.centroid(vectors)

# Cosine similarity (1 - distance)
similarity = vector.cosine_similarity(a, b)
```

---

## 28. Data Formats (TOON/JSON/Columnar)

### Wire Formats

```python
from toondb import WireFormat

# Available formats
WireFormat.TOON      # Token-efficient (40-66% fewer tokens)
WireFormat.JSON      # Standard JSON
WireFormat.COLUMNAR  # Raw columnar for analytics

# Parse from string
fmt = WireFormat.from_string("toon")

# Convert between formats
data = {"users": [{"id": 1, "name": "Alice"}]}
toon_data = WireFormat.to_toon(data)
json_data = WireFormat.to_json(data)
```

### TOON Format Benefits

TOON uses **40-60% fewer tokens** than JSON:

```
# JSON (15 tokens)
{"users": [{"id": 1, "name": "Alice"}]}

# TOON (9 tokens)
users:
  - id: 1
    name: Alice
```

### Context Formats

```python
from toondb import ContextFormat

ContextFormat.TOON      # Token-efficient
ContextFormat.JSON      # Structured data
ContextFormat.MARKDOWN  # Human-readable

# Format capabilities
from toondb import FormatCapabilities

# Convert between formats
ctx_fmt = FormatCapabilities.wire_to_context(WireFormat.TOON)
wire_fmt = FormatCapabilities.context_to_wire(ContextFormat.JSON)

# Check round-trip support
if FormatCapabilities.supports_round_trip(WireFormat.TOON):
    print("Safe for decode(encode(x)) = x")
```

---

## 29. Policy Service

Register and evaluate access control policies.

```python
from toondb import PolicyService

policy_svc = db.policy_service()

# Register a policy
policy_svc.register(
    policy_id="read_own_data",
    name="Users can read their own data",
    trigger="READ",
    action="ALLOW",
    condition="resource.owner == user.id"
)

# Register another policy
policy_svc.register(
    policy_id="admin_all",
    name="Admins can do everything",
    trigger="*",
    action="ALLOW",
    condition="user.role == 'admin'"
)

# Evaluate policy
result = policy_svc.evaluate(
    action="READ",
    resource="documents/123",
    context={"user.id": "alice", "user.role": "user", "resource.owner": "alice"}
)

if result.allowed:
    print("Access granted")
else:
    print(f"Access denied: {result.reason}")
    print(f"Denying policy: {result.policy_id}")

# List policies
policies = policy_svc.list()
for p in policies:
    print(f"{p.policy_id}: {p.name}")

# Delete policy
policy_svc.delete("old_policy")
```

---

## 30. MCP (Model Context Protocol)

Integrate ToonDB as an MCP tool provider.

### Built-in MCP Tools

| Tool | Description |
|------|-------------|
| `toondb_query` | Execute ToonQL/SQL queries |
| `toondb_context_query` | Fetch AI-optimized context |
| `toondb_put` | Store key-value data |
| `toondb_get` | Retrieve data by key |
| `toondb_search` | Vector similarity search |

### Using MCP Tools (Server Mode)

```python
# List available tools
tools = client.list_mcp_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")

# Get tool schema
schema = client.get_mcp_tool_schema("toondb_search")
print(schema)

# Execute tool
result = client.execute_mcp_tool(
    name="toondb_query",
    arguments={"query": "SELECT * FROM users", "format": "toon"}
)
print(result)
```

### Register Custom Tool

```python
# Register a custom tool
client.register_mcp_tool(
    name="search_documents",
    description="Search documents by semantic similarity",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results", "default": 10}
        },
        "required": ["query"]
    }
)
```

---

## 31. Configuration Reference

### Database Configuration

```python
from toondb import Database, CompressionType, SyncMode

db = Database.open("./my_db", config={
    # Durability
    "wal_enabled": True,                      # Write-ahead logging
    "sync_mode": SyncMode.NORMAL,             # FULL, NORMAL, OFF
    
    # Performance
    "memtable_size_bytes": 64 * 1024 * 1024,  # 64MB (flush threshold)
    "block_cache_size_bytes": 256 * 1024 * 1024,  # 256MB
    "group_commit": True,                      # Batch commits
    
    # Compression
    "compression": CompressionType.LZ4,
    
    # Index policy
    "index_policy": "balanced",
    
    # Background workers
    "compaction_threads": 2,
    "flush_threads": 1,
})
```

### Sync Modes

| Mode | Speed | Safety | Use Case |
|------|-------|--------|----------|
| `OFF` | ~10x faster | Risk of data loss | Development, caches |
| `NORMAL` | Balanced | Fsync at checkpoints | Default |
| `FULL` | Slowest | Fsync every commit | Financial data |

### CollectionConfig Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | required | Collection name |
| `dimension` | int | required | Vector dimension |
| `metric` | DistanceMetric | COSINE | COSINE, EUCLIDEAN, DOT_PRODUCT |
| `m` | int | 16 | HNSW M parameter |
| `ef_construction` | int | 100 | HNSW build quality |
| `ef_search` | int | 50 | HNSW search quality |
| `quantization` | QuantizationType | NONE | NONE, SCALAR, PQ |
| `enable_hybrid_search` | bool | False | Enable BM25 |
| `content_field` | str | None | Field for BM25 indexing |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `TOONDB_LIB_PATH` | Custom path to native library |
| `TOONDB_DISABLE_ANALYTICS` | Disable anonymous usage tracking |
| `TOONDB_LOG_LEVEL` | Log level (DEBUG, INFO, WARN, ERROR) |

---

## 32. Error Handling

### Error Types

```python
from toondb import (
    # Base
    ToonDBError,
    
    # Connection
    ConnectionError,
    ConnectionTimeoutError,
    
    # Transaction
    TransactionError,
    TransactionConflictError,  # SSI conflict - retry
    TransactionTimeoutError,
    
    # Storage
    DatabaseError,
    CorruptionError,
    DiskFullError,
    
    # Namespace
    NamespaceNotFoundError,
    NamespaceExistsError,
    NamespaceAccessError,
    
    # Collection
    CollectionNotFoundError,
    CollectionExistsError,
    CollectionConfigError,
    
    # Validation
    ValidationError,
    DimensionMismatchError,
    InvalidMetadataError,
    
    # Query
    QueryError,
    QuerySyntaxError,
    QueryTimeoutError,
)
```

### Error Handling Pattern

```python
from toondb import (
    ToonDBError,
    TransactionConflictError,
    DimensionMismatchError,
    CollectionNotFoundError,
)

try:
    with db.transaction() as txn:
        txn.put(b"key", b"value")
        
except TransactionConflictError as e:
    # SSI conflict - safe to retry
    print(f"Conflict detected: {e}")
    
except DimensionMismatchError as e:
    # Vector dimension wrong
    print(f"Expected {e.expected} dimensions, got {e.actual}")
    
except CollectionNotFoundError as e:
    # Collection doesn't exist
    print(f"Collection not found: {e.collection}")
    
except ToonDBError as e:
    # All other ToonDB errors
    print(f"Error: {e}")
    print(f"Code: {e.code}")
    print(f"Remediation: {e.remediation}")
```

### Error Information

```python
try:
    # ...
except ToonDBError as e:
    print(f"Message: {e.message}")
    print(f"Code: {e.code}")           # ErrorCode enum
    print(f"Details: {e.details}")      # Additional context
    print(f"Remediation: {e.remediation}")  # How to fix
    print(f"Retryable: {e.retryable}")  # Safe to retry?
```

---

## 33. Async Support

Optional async/await support for non-blocking operations.

```python
from toondb import AsyncDatabase

async def main():
    # Open async database
    db = await AsyncDatabase.open("./my_db")
    
    # Async operations
    await db.put(b"key", b"value")
    value = await db.get(b"key")
    
    # Async transactions
    async with db.transaction() as txn:
        await txn.put(b"key1", b"value1")
        await txn.put(b"key2", b"value2")
    
    # Async vector search
    results = await db.collection("docs").search(SearchRequest(
        vector=[0.1, 0.2, ...],
        k=10
    ))
    
    await db.close()

# Run
import asyncio
asyncio.run(main())
```

**Note:** Requires `pip install toondb[async]`

---

## 34. Building & Development

### Building Native Extensions

```bash
# Build for current platform
python build_native.py

# Build only FFI libraries
python build_native.py --libs

# Build for all platforms
python build_native.py --all

# Clean
python build_native.py --clean
```

### Library Discovery

The SDK looks for native libraries in this order:
1. `TOONDB_LIB_PATH` environment variable
2. Bundled in wheel: `lib/{target}/`
3. Package directory
4. Development builds: `target/release/`, `target/debug/`
5. System paths: `/usr/local/lib`, `/usr/lib`

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_vector_search.py

# With coverage
pytest --cov=toondb

# Performance tests
pytest tests/perf/ --benchmark
```

### Package Structure

```
toondb/
├── __init__.py          # Public API exports
├── database.py          # Database, Transaction
├── namespace.py         # Namespace, Collection
├── vector.py            # VectorIndex, utilities
├── grpc_client.py       # ToonDBClient (server mode)
├── ipc_client.py        # IpcClient (Unix sockets)
├── context.py           # ContextQueryBuilder
├── atomic.py            # AtomicMemoryWriter
├── recovery.py          # RecoveryManager
├── checkpoint.py        # CheckpointService
├── workflow.py          # WorkflowService
├── trace.py             # TraceStore
├── policy.py            # PolicyService
├── format.py            # WireFormat, ContextFormat
├── errors.py            # All error types
├── _bin/                # Bundled binaries
└── lib/                 # FFI libraries
```

---

## 35. Complete Examples

### RAG Pipeline Example

```python
from toondb import Database, CollectionConfig, DistanceMetric, SearchRequest

# Setup
db = Database.open("./rag_db")
ns = db.get_or_create_namespace("rag")

# Create collection for documents
collection = ns.create_collection(CollectionConfig(
    name="documents",
    dimension=384,
    metric=DistanceMetric.COSINE,
    enable_hybrid_search=True,
    content_field="text"
))

# Index documents
def index_document(doc_id: str, text: str, embed_fn):
    embedding = embed_fn(text)
    collection.insert(
        id=doc_id,
        vector=embedding,
        metadata={"text": text, "indexed_at": "2024-01-15"}
    )

# Retrieve relevant context
def retrieve_context(query: str, embed_fn, k: int = 5) -> list:
    query_embedding = embed_fn(query)
    
    results = collection.hybrid_search(
        vector=query_embedding,
        text_query=query,
        k=k,
        alpha=0.7  # 70% vector, 30% keyword
    )
    
    return [r.metadata["text"] for r in results]

# Full RAG pipeline
def rag_query(query: str, embed_fn, llm_fn):
    # 1. Retrieve
    context_docs = retrieve_context(query, embed_fn)
    
    # 2. Build context
    from toondb import ContextQueryBuilder, ContextFormat
    
    context = ContextQueryBuilder() \
        .for_session("rag_session") \
        .with_budget(4096) \
        .literal("SYSTEM", 0, "Answer based on the provided context.") \
        .literal("CONTEXT", 1, "\n\n".join(context_docs)) \
        .literal("QUESTION", 2, query) \
        .execute()
    
    # 3. Generate
    response = llm_fn(context.text)
    
    return response

db.close()
```

### Knowledge Graph Example

```python
from toondb import Database
import time

db = Database.open("./knowledge_graph")

# Build a knowledge graph
db.add_node("kg", "alice", "person", {"role": "engineer", "level": "senior"})
db.add_node("kg", "bob", "person", {"role": "manager"})
db.add_node("kg", "project_ai", "project", {"status": "active", "budget": 100000})
db.add_node("kg", "ml_team", "team", {"size": 5})

db.add_edge("kg", "alice", "works_on", "project_ai", {"role": "lead"})
db.add_edge("kg", "alice", "member_of", "ml_team")
db.add_edge("kg", "bob", "manages", "project_ai")
db.add_edge("kg", "bob", "leads", "ml_team")

# Query: Find all projects Alice works on
nodes, edges = db.traverse("kg", "alice", max_depth=1)
projects = [n for n in nodes if n["node_type"] == "project"]
print(f"Alice's projects: {[p['id'] for p in projects]}")

# Query: Who manages Alice's projects?
for project in projects:
    nodes, edges = db.traverse("kg", project["id"], max_depth=1)
    managers = [e["from_id"] for e in edges if e["edge_type"] == "manages"]
    print(f"{project['id']} managed by: {managers}")

db.close()
```

### Multi-Tenant SaaS Example

```python
from toondb import Database

db = Database.open("./saas_db")

# Create tenant namespaces
for tenant in ["acme_corp", "globex", "initech"]:
    ns = db.create_namespace(
        name=tenant,
        labels={"tier": "premium" if tenant == "acme_corp" else "standard"}
    )
    
    # Create tenant-specific collections
    ns.create_collection(
        name="documents",
        dimension=384
    )

# Tenant-scoped operations
with db.use_namespace("acme_corp") as ns:
    collection = ns.collection("documents")
    
    # All operations isolated to acme_corp
    collection.insert(
        id="doc1",
        vector=[0.1] * 384,
        metadata={"title": "Acme Internal Doc"}
    )
    
    # Search only searches acme_corp's documents
    results = collection.vector_search(
        vector=[0.1] * 384,
        k=10
    )

# Cleanup
db.close()
```

---

## 36. Migration Guide

### From v0.2.x to v0.3.x

```python
# Old: scan() with range
for k, v in db.scan(b"users/", b"users0"):  # DEPRECATED
    pass

# New: scan_prefix()
for k, v in db.scan_prefix(b"users/"):
    pass

# Old: execute_sql returns tuple
columns, rows = db.execute_sql("SELECT * FROM users")

# New: execute_sql returns SQLQueryResult
result = db.execute_sql("SELECT * FROM users")
columns = result.columns
rows = result.rows
```

### From SQLite/PostgreSQL

```python
# SQLite
# conn = sqlite3.connect("app.db")
# cursor = conn.execute("SELECT * FROM users")

# ToonDB (same SQL, embedded)
db = Database.open("./app_db")
result = db.execute_sql("SELECT * FROM users")
```

### From Redis

```python
# Redis
# r = redis.Redis()
# r.set("key", "value")
# r.get("key")

# ToonDB
db = Database.open("./cache_db")
db.put(b"key", b"value")
db.get(b"key")

# With TTL
db.put(b"session:123", b"data", ttl_seconds=3600)
```

### From Pinecone/Weaviate

```python
# Pinecone
# index.upsert(vectors=[(id, embedding, metadata)])
# results = index.query(vector=query, top_k=10)

# ToonDB
collection = db.namespace("default").collection("vectors")
collection.insert(id=id, vector=embedding, metadata=metadata)
results = collection.vector_search(vector=query, k=10)
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
- **GitHub Issues**: https://github.com/toondb/toondb/issues
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
