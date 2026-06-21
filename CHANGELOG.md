# Changelog

All notable changes to the SochDB Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-06-21

### Added

- **`Collection.optimize()` and `VectorIndex.optimize()`** — finalize a vector
  index for maximum recall after a bulk load. Runs the engine's exact layer-0
  rebuild (NN-descent + exact-f32 rerank) and connectivity repair, lifting recall
  toward exact-kNN quality, which is most noticeable at high dimension where the
  as-built HNSW graph can fall a few points short. Call once after bulk insert;
  it's a no-op above the engine's exact-rebuild scale cap, serialized against
  concurrent inserts, and (on `Collection`) rebuilds the index from KV first if
  the collection was just reopened. Bundles native engine **2.0.11**
  (`__core_version__` 2.0.10 → 2.0.11), which adds the `hnsw_optimize` C FFI
  export this binds to. Regression test in `tests/test_optimize.py`.

## [0.7.1] - 2026-06-21

### Fixed

- **`persist_directory` collections were not searchable after reopening.** Vector
  search returned 0 results once a process reopened a persisted collection, even
  though the data was intact (`count()` and `get_by_ids()` worked). Root cause:
  the in-memory HNSW was rebuilt only from a numpy snapshot that was never written
  (`_persist_vectors_snapshot` had no callers), so the lazy reload always found
  nothing. The reload path now falls back to rebuilding the HNSW from the
  KV-persisted vectors — which are written transactionally on every insert, so the
  rebuild is crash-safe and does not depend on `close()` being called. Vector,
  filtered, and keyword search all work again across sessions. Added regression
  tests (`tests/test_persist_reload.py`).
  - Note: collections created with `persist_vector_in_kv=False` store vectors only
    in the (currently unwritten) snapshot, so they remain non-reloadable; use the
    default `persist_vector_in_kv=True` when you need cross-session vector search.

## [0.7.0] - 2026-06-21

### Changed

- **Bundles the SochDB 2.0.10 native engine** (`__core_version__` 2.0.9 → 2.0.10),
  which brings substantial vector-search correctness and performance fixes:
  - High-dimensional recall is no longer a build-seed coin flip — `optimize()`
    repairs HNSW graph fragmentation (orphans → 0, recall@10 ≈ 0.994–1.0).
  - `optimize()` is **2.4–7.1× faster** at high dimension via a sub-quadratic
    NN-descent rebuild (e.g. dim 3072 euclidean 80.6s → 11.3s) with recall held.
  - Search returns **correct lower-is-better distances** for every metric
    (cosine `1 - similarity`, euclidean true `sqrt(Σ diff²)`, dot `-dot`), fixing
    a cosine result-ordering bug and squared-L2 distance values.
  - Vector IDs use a checked `u128 → u64` conversion (no silent truncation).



### Added

- **Agent memory release** — consolidates the `sochdb.memory.agent_memory` API
  (`AgentMemory`, `QueryLanes`, `build_search_section`, `build_ingest_section`,
  `create_agent_memory`) and the `ContextService` client methods
  (`write_episode`, `estimate_tokens`, `format_context`) backed by regenerated
  gRPC stubs that include `WriteEpisode`.
- **Example:** `examples/30_agent_memory.py`
- **`as_of` point-in-time search** carried over from 0.5.9.

## [0.5.9] - 2026-06-10

### Added

- **`as_of` point-in-time search** — `AgentMemory.search(as_of=<unix_ms>)` and
  `build_search_section(..., as_of=...)` pass bi-temporal query time via SEARCH
  section options to `ContextService` / `sochdb-memory`.

### Tests

- `tests/test_asof_integration.py` — gRPC integration (skipped if server down)

## [0.5.8] - 2026-06-10

### Added

- **Agent memory API** (`sochdb.memory.agent_memory`, exported from `sochdb`):
  - `AgentMemory` — high-level wrapper for `ContextService` + sochdb-memory backend
  - `QueryLanes` — `lexical`, `three_lane`, `hybrid`, `bm25`, `trigram`
  - `build_search_section`, `build_ingest_section`, `create_agent_memory`

- **ContextService gRPC client methods** on `SochDBClient`:
  - `write_episode` — preferred episode ingest (lexical index + async vector enrich)
  - `estimate_tokens` — exact BPE token count
  - `format_context` — toon/json/markdown/text formatting

- **Response dataclasses:** `ContextQueryResult`, `ContextSectionResult`, `EpisodeWriteResult`

- **Example:** `examples/30_agent_memory.py`

- **Proto sync:** `proto/sochdb.proto` aligned with main repo; `scripts/generate_proto.sh` for stub regeneration

### Changed

- **`query_context` return type** — now returns `ContextQueryResult` instead of `str`. Use `.context` for the assembled string.
- **`query_context` parameters** — added `format` and `include_schema`; sections support `options` dict (`lanes`, `namespace`, `episode_text`, `doc_id`, …)

## [0.5.5] - 2026-02-21

### Fixed

- **`insert()` now writes to KV store** — previously `insert()` only populated
  the in-memory HNSW index, leaving single-vector inserts invisible to
  `keyword_search()`, `hybrid_search()`, and the FFI BM25 path.  Docs are now
  persisted to the KV store using the same JSON schema as `insert_multi()`.

- **`insert_batch()` now writes to KV store** — same issue as `insert()`.
  Batch-inserted documents were not written to KV, so any follow-up keyword or
  hybrid search would miss them.  All docs in a batch are now written in a
  single atomic transaction.

- **Python BM25 fallback now uses proper BM25 formula** — the fallback
  `_keyword_search()` (used when the native FFI call returns `None`) previously
  scored documents by raw term-frequency count with no IDF weighting and no
  length normalisation.  It now implements the Robertson–Spärck Jones BM25
  formula (k1=1.2, b=0.75 — Lucene / Elasticsearch defaults), matching the
  behaviour of the native Rust `bm25.rs` implementation.

## [0.5.4] - 2026-02-15

### Changed

- Version bump aligned with SochDB core 0.5.0

## [0.5.3] - 2026-02-10

### Changed

- Version bump aligned with SochDB core 0.4.9
- Added Engine Internals status table to README (§4 Architecture Overview)
- Cost-based optimizer documented as production-ready
- Adaptive group commit documented as implemented
- WAL compaction documented as partially implemented

### Fixed

- Added missing `QuantizationType` export to `__init__.py`

## [0.4.5] - 2026-01-23

### Added

#### LLM-Native Memory System

A complete memory management system for AI agents with FFI/gRPC dual-mode support:

**Extraction Pipeline** (`sochdb.memory.extraction`):
- `Entity`, `Relation`, `Assertion` typed intermediate representation
- `ExtractionSchema` for validation with type constraints and confidence thresholds
- `ExtractionPipeline` with atomic commits
- Deterministic ID generation via content hashing

**Event-Sourced Consolidation** (`sochdb.memory.consolidation`):
- `RawAssertion` immutable events (append-only, never deleted)
- `CanonicalFact` derived view (merged, deduplicated)
- `UnionFind` clustering with O(α(n)) operations
- Temporal interval updates for contradictions (not destructive edits)
- Full provenance tracking with `explain()` method

**Hybrid Retrieval** (`sochdb.memory.retrieval`):
- `AllowedSet` for pre-filtering (security invariant: Results ⊆ allowed_set)
- RRF fusion leveraging SochDB's built-in implementation
- Optional cross-encoder reranking support
- `HybridRetriever` with `explain()` for ranking debugging

**Namespace Isolation** (`sochdb.memory.isolation`):
- `NamespaceId` strongly-typed identifier with validation
- `ScopedQuery` for type-level safety guarantees
- `NamespaceGrant` for explicit, auditable cross-namespace access
- `ScopedNamespace` with full audit logging
- `NamespaceManager` for namespace lifecycle management
- Policy modes: `STRICT`, `EXPLICIT`, `AUDIT_ONLY`

All modules include:
- FFI backend (embedded mode)
- gRPC backend (server mode)
- In-memory backend (testing)
- Factory functions with auto-detection

### Documentation
- Added comprehensive Memory System section (Section 18) to README
- Full API documentation with usage examples
- Updated Table of Contents

## [0.2.3] - 2025-01-xx

### Fixed
- **Platform detection bug**: Fixed binary resolution using Rust target triple format (`aarch64-apple-darwin`) instead of Python platform tag format (`darwin-aarch64`)
- Improved documentation accuracy across all doc files

### Changed
## [0.3.2] - 2026-01-04

### Repository Update
- 📦 **Moved Python SDK** to its own repository: [https://github.com/sochdb/sochdb-python-sdk](https://github.com/sochdb/sochdb-python-sdk)
- This allows for independent versioning and faster CI/CD pipelines.

### Infrastructure
- **New Release Workflow**: Now pulls pre-built binaries directly from [sochdb/sochdb](https://github.com/sochdb/sochdb) releases.
  - Supports Python 3.9 through 3.13
  - Automatically creates GitHub releases with all wheel packages attached
  - Each wheel bundles platform-specific binaries and FFI libraries
  - See [RELEASE.md](RELEASE.md) for detailed release process documentation
- **Trusted Publishing**: Configured PyPI Trusted Publisher (OIDC) security.
- **Platform Bundles**: 
  - Linux x86_64 (manylinux_2_17)
  - macOS ARM64 (Apple Silicon)
  - Windows x64

### Documentation
- Added comprehensive [RELEASE.md](RELEASE.md) explaining how binaries are sourced from sochdb/sochdb
- Updated README with binary source information
- Enhanced release workflow with detailed summaries and status reporting

## [0.2.9] - 2026-01-02

### Added

#### Production-Grade CLI Tools

CLI commands now available globally after `pip install sochdb-client`:

```bash
sochdb-server      # IPC server for multi-process access
sochdb-bulk        # High-performance vector operations
sochdb-grpc-server # gRPC server for remote vector search
```

**sochdb-server features:**
- **Stale socket detection** - Auto-cleans orphaned socket files
- **Health checks** - Waits for server ready before returning
- **Graceful shutdown** - Handles SIGTERM/SIGINT/SIGHUP
- **PID tracking** - Writes PID file for process management
- **Permission checks** - Validates directory writable before starting
- **stop/status commands** - Built-in process management

**sochdb-bulk features:**
- **Input validation** - Checks file exists, readable, correct extension
- **Output validation** - Checks directory writable, handles overwrites
- **Progress reporting** - Shows file sizes during operations
- **Structured subcommands** - build-index, query, info, convert

**sochdb-grpc-server features:**
- **Port checking** - Verifies port available before binding
- **Process detection** - Identifies what process is using a port
- **Privileged port check** - Warns about ports < 1024 requiring root
- **status command** - Check if server is running

#### Consistent Exit Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | Operation completed |
| 1 | GENERAL_ERROR | General error |
| 2 | BINARY_NOT_FOUND | Native binary not found |
| 3 | PORT/SOCKET_IN_USE | Port or socket in use |
| 4 | PERMISSION_DENIED | Permission denied |
| 5 | STARTUP_FAILED | Server startup failed |
| 130 | INTERRUPTED | Interrupted by Ctrl+C |

#### Environment Variable Overrides

- `SOCHDB_SERVER_PATH` - Override sochdb-server binary path
- `SOCHDB_BULK_PATH` - Override sochdb-bulk binary path  
- `SOCHDB_GRPC_SERVER_PATH` - Override sochdb-grpc-server binary path

### Changed

- CLI wrappers now provide actionable error messages with fix suggestions
- Binary resolution searches multiple locations with clear fallback chain
- Signal handlers for graceful shutdown on all platforms

## [0.2.3] - 2025-01-xx

### Added

#### Cross-Platform Binary Distribution
- **Zero-compile installation**: Pre-built `sochdb-bulk` binaries bundled in wheels
- **Platform support matrix**:
  - `manylinux_2_17_x86_64` - Linux x86_64 (glibc ≥ 2.17)
  - `manylinux_2_17_aarch64` - Linux ARM64 (AWS Graviton, etc.)
  - `macosx_11_0_universal2` - macOS Intel + Apple Silicon
  - `win_amd64` - Windows x64
- **Automatic binary resolution** with fallback chain:
  1. Bundled in wheel (`_bin/<platform>/sochdb-bulk`)
  2. System PATH (`which sochdb-bulk`)
  3. Cargo target directory (development mode)

#### Bulk API Enhancements
- `bulk_query_index()` - Query HNSW indexes for k nearest neighbors
- `bulk_info()` - Get index metadata (vector count, dimension, etc.)
- `get_sochdb_bulk_path()` - Get resolved path to sochdb-bulk binary
- `_get_platform_tag()` - Platform detection (linux-x86_64, darwin-aarch64, etc.)
- `_find_bundled_binary()` - Uses `importlib.resources` for installed packages

#### CI/CD Infrastructure
- GitHub Actions workflow for building platform-specific wheels
- cibuildwheel configuration for cross-platform builds
- QEMU emulation for ARM64 Linux builds
- PyPI publishing with trusted publishing

#### Documentation
- [PYTHON_DISTRIBUTION.md](../docs/PYTHON_DISTRIBUTION.md) - Full distribution architecture
- Updated [BULK_OPERATIONS.md](../docs/BULK_OPERATIONS.md) with troubleshooting
- Updated [SDK_DOCUMENTATION.md](docs/SDK_DOCUMENTATION.md) with Bulk API reference
- Updated [ARCHITECTURE.md](../docs/ARCHITECTURE.md) with Python SDK section

### Changed

- Package renamed from `sochdb-client` to `sochdb`
- Wheel tags changed from `any` to platform-specific (`py3-none-<platform>`)
- Binary resolution now uses `importlib.resources` instead of `__file__` paths

### Technical Details

#### Distribution Model
Follows the "uv-style" approach where:
- Wheels are tagged `py3-none-<platform>` (not CPython-ABI-tied)
- One wheel per platform (not per Python version)
- Artifact count: O(P·A) where P=platforms, A=architectures

#### Linux Compatibility
- **manylinux_2_17** baseline (glibc ≥ 2.17)
- Covers: CentOS 7+, RHEL 7+, Ubuntu 14.04+, Debian 8+
- Same baseline used by `uv` for production deployments

#### macOS Strategy
- **universal2** fat binaries containing both x86_64 and arm64
- Created with `lipo -create` during build
- Minimum macOS 11.0 (Big Sur)

## [0.1.0] - 2024-12-XX

### Added

- Initial release
- Embedded mode with FFI access to SochDB
- IPC client mode for multi-process access
- Path-native API with O(|path|) lookups
- ACID transactions with snapshot isolation
- Range scans and prefix queries
- TOON format output for LLM context optimization
- Bulk API for high-throughput vector ingestion
  - `bulk_build_index()` - Build HNSW indexes at ~1,600 vec/s
  - `convert_embeddings_to_raw()` - Convert numpy to raw f32
- Support for raw f32 and NumPy .npy input formats

### Performance

| Method | 768D Throughput | Notes |
|--------|-----------------|-------|
| Python FFI | ~130 vec/s | Direct FFI calls |
| Bulk API | ~1,600 vec/s | Subprocess to sochdb-bulk |

FFI overhead eliminated by subprocess approach for bulk operations.
