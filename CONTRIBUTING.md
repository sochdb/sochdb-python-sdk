# Contributing to ToonDB Python SDK

Thank you for your interest in contributing to the ToonDB Python SDK! This guide provides all the information you need to build, test, and contribute to the project.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Server Setup for Development](#server-setup-for-development)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Architecture Overview](#architecture-overview)
- [Migration Guide](#migration-guide)

---

## Development Setup

### Prerequisites

- Python 3.8+
- Rust toolchain (for building from source)
- Git

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/toondb/toondb-python-sdk.git
cd toondb-python-sdk

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

---

## Building from Source

### Python SDK Only

```bash
cd toondb-python-sdk
pip install -e .
```

### With Rust FFI Library

If you need to rebuild the native library:

```bash
# Build Rust library
cd toondb
cargo build --release -p toondb-storage

# Copy to Python SDK
cp target/release/libtoondb_storage.dylib \
   toondb-python-sdk/src/toondb/_bin/darwin-arm64/

# For Linux
cp target/release/libtoondb_storage.so \
   toondb-python-sdk/src/toondb/_bin/linux-x86_64/
```

---

## Running Tests

### Unit Tests

```bash
# Run Python tests
cd toondb-python-sdk
pytest tests/

# Run with coverage
pytest --cov=toondb tests/
```

### Integration Tests

```bash
# Start ToonDB server first
cd toondb
cargo run -p toondb-grpc

# In another terminal, run integration tests
cd toondb-python-sdk
pytest tests/integration/
```

### Run All Examples

```bash
# Test all examples
cd toondb-python-sdk
./run_examples.sh

# Test specific example
python3 examples/25_temporal_graph_embedded.py
```

---

## Server Setup for Development

### Starting the Server

```bash
# Development mode
cd toondb
cargo run -p toondb-grpc

# Production mode (optimized)
cargo build --release -p toondb-grpc
./target/release/toondb-grpc --host 0.0.0.0 --port 50051
```

### Server Configuration

The server runs all business logic including:
- ✅ HNSW vector indexing (15x faster than ChromaDB)
- ✅ SQL query parsing and execution
- ✅ Graph traversal algorithms
- ✅ Policy evaluation
- ✅ Multi-tenant namespace isolation
- ✅ Collection management

### Configuration File

Create `toondb-server-config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 50051

[storage]
data_dir = "./data"

[logging]
level = "info"
```

---

## Code Style

### Python

We follow PEP 8 with some modifications:

```bash
# Format code
black src/ tests/

# Check types
mypy src/

# Lint
ruff check src/ tests/
```

### Commit Messages

Follow conventional commits:

```
feat: Add temporal graph support
fix: Handle connection timeout
docs: Update API reference
test: Add integration tests for graphs
```

### Code Review Checklist

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Examples added/updated if needed
- [ ] No breaking changes (or documented in CHANGELOG)

---

## Pull Request Process

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/toondb-python-sdk.git
   cd toondb-python-sdk
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

4. **Test Locally**
   ```bash
   pytest tests/
   black src/
   ruff check src/
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: Your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub
   - Create PR from your branch
   - Fill out PR template
   - Wait for review

---

## Architecture Overview

### Dual-Mode Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  PYTHON SDK ARCHITECTURE                  │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  1. EMBEDDED MODE              2. SERVER MODE             │
│  ┌─────────────────┐          ┌─────────────────┐       │
│  │  Database       │          │  ToonDBClient   │       │
│  │  (FFI bindings) │          │  (gRPC client)  │       │
│  └────────┬────────┘          └────────┬────────┘       │
│           │                             │                 │
│           ▼                             ▼                 │
│  libtoondb_storage.dylib      toondb-grpc server        │
│  (Rust native library)        (Rust gRPC service)       │
└──────────────────────────────────────────────────────────┘
```

### Key Components

**database.py** (1,974 lines)
- `Database` class with FFI bindings
- Key-value operations
- Temporal graph operations
- Namespace management
- Collection management

**grpc_client.py** (630 lines)
- `ToonDBClient` class for gRPC
- All server-based operations
- Connection management
- Error handling

**format.py** (162 lines)
- `WireFormat` enum (TOON, JSON, Columnar)
- `ContextFormat` enum (TOON, JSON, Markdown)
- `FormatCapabilities` utilities

### Comparison with Old Architecture

| Feature | Old (Fat Client) | New (Dual-Mode) |
|---------|------------------|-----------------|
| SDK Size | 15,872 LOC | 5,400 LOC (-66%) |
| Business Logic | In SDK (Python) | In Server (Rust) |
| Deployment | Single mode only | Embedded + Server |
| Bug Fixes | Per language | Once in server |
| Semantic Drift | High risk | Zero risk |
| Performance | FFI only | FFI + gRPC |
| Maintenance | 3x effort | 1x effort |

---

## Migration Guide

### From v0.3.3 to v0.3.4

**No breaking changes!** We added embedded FFI support while keeping server mode.

**New in 0.3.4:**
```python
# NEW: Temporal graphs in embedded mode
from toondb import Database

db = Database.open("./mydb")
db.add_temporal_edge(...)
db.query_temporal_graph(...)

# NEW: Same temporal graph API in server mode
from toondb import ToonDBClient

client = ToonDBClient("localhost:50051")
client.add_temporal_edge(...)
client.query_temporal_graph(...)
```

### From Fat Client (v0.3.2 or earlier)

**Old Code:**
```python
from toondb import Database, GraphOverlay

db = Database.open("./data")
graph = GraphOverlay(db)
graph.add_node("alice", "person", {"name": "Alice"})
```

**New Code (Server Mode):**
```python
from toondb import ToonDBClient

# Start server first: cargo run -p toondb-grpc
client = ToonDBClient("localhost:50051")
client.add_node("alice", "person", {"name": "Alice"})
```

**New Code (Embedded Mode):**
```python
from toondb import Database

db = Database.open("./data")
# GraphOverlay is now built into Database
db.add_node("alice", "person", {"name": "Alice"})
```

**Key Changes:**
1. `GraphOverlay` removed - features merged into `Database` and `ToonDBClient`
2. Server mode requires running `toondb-grpc` server
3. Embedded mode uses direct FFI bindings (faster, no network)
4. Same API for both modes

---

## Release Process

### Version Bumping

```bash
# Update version in setup.py
vim setup.py

# Update version in __init__.py
vim src/toondb/__init__.py

# Update CHANGELOG.md
vim CHANGELOG.md
```

### Building Distribution

```bash
# Build wheel
python3 setup.py bdist_wheel

# Check distribution
twine check dist/*
```

### Publishing to PyPI

```bash
# Test PyPI first
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Production PyPI
twine upload dist/*
```

---

## Testing Checklist

Before submitting a PR, ensure:

- [ ] All unit tests pass: `pytest tests/unit/`
- [ ] All integration tests pass: `pytest tests/integration/`
- [ ] All examples run: `./run_examples.sh`
- [ ] Code formatted: `black src/`
- [ ] Linting passes: `ruff check src/`
- [ ] Type checking passes: `mypy src/`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated

---

## Getting Help

- **Main Repo**: https://github.com/toondb/toondb
- **Python SDK Issues**: https://github.com/toondb/toondb-python-sdk/issues
- **Discussions**: https://github.com/toondb/toondb/discussions
- **Contributing Guide**: See main repo [CONTRIBUTING.md](https://github.com/toondb/toondb/blob/main/CONTRIBUTING.md)

---

## License

By contributing to ToonDB Python SDK, you agree that your contributions will be licensed under the Apache License 2.0.
