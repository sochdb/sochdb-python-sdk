# SochDB Python SDK — Stress Test Findings

**Test:** `test_stress_gaps.py` | **Result:** 119 PASS, 16 FAIL, 0 SKIP across 135 checks in 16 sections (A–P)

Methodology: Jepsen-inspired correctness testing (write skew, lost updates, phantom/dirty reads), concurrency stress, resource exhaustion, API contract verification, and crash recovery — informed by [Jepsen consistency analyses](https://jepsen.io/analyses), [PostgreSQL SSI docs](https://www.postgresql.org/docs/current/transaction-iso.html), and [Brendan Gregg's performance methodology](https://www.brendangregg.com/methodology.html).

---

## CRITICAL — Correctness Bugs

### 1. SSI Does Not Prevent Write Skew (A2)
**Severity: CRITICAL** — Violates serializability guarantee

Two transactions read overlapping keys (alice, bob balances), then each writes a *different* key. Under true SSI, one must be rejected because the read set overlaps and the combined writes violate the invariant.

```
T1: reads alice=100, bob=100 → writes alice = 100-150 = -50
T2: reads alice=100, bob=100 → writes bob  = 100-150 = -50
Both COMMIT → alice=-50, bob=-50, sum=-100 (invariant: sum≥0 violated!)
```

**Impact:** Any application relying on SSI for multi-key invariants (bank transfers, inventory, resource allocation) will produce **corrupt data** under concurrent access.

### 2. SSI Does Not Detect Lost Updates (A1)
**Severity: CRITICAL** — Silent data loss

Two transactions read the same counter=0, both increment to 1, both commit successfully. Final value is 1, not 2. Neither transaction is rejected.

```
T1: reads counter=0, writes counter=1 → COMMIT OK
T2: reads counter=0, writes counter=1 → COMMIT OK  ← should have been rejected
Final counter = 1 (one increment silently lost)
```

**Impact:** Counters, sequence generators, inventory decrement — any read-modify-write pattern loses updates silently.

### 3. Concurrent Transactions Silently Lose Writes (B2)
**Severity: CRITICAL** — Data integrity violation

Under 20-thread concurrent increment with per-txn retry logic:
- 196 transactions report **successful commit**
- Counter reaches only **134** (62 committed writes silently lost)

**Impact:** `commit()` returns success but the write is not durably applied. Applications cannot trust commit acknowledgements.

### 4. Queue Double-Claiming (D1)
**Severity: HIGH** — At-least-once guarantee broken for workers

Under 10 concurrent worker threads dequeuing from a 50-task queue:
- 61 total claims for 48 unique tasks (13 double-claimed)
- 2 tasks never claimed at all

**Impact:** Work items executed multiple times. In payment processing, billing, or idempotent-expected workflows, this causes duplicate processing.

---

## HIGH — API Contract Violations

### 5. README Documents 6 Non-Existent APIs (K1)
**Severity: HIGH** — Developer trust / documentation debt

| README Claims | Reality |
|---|---|
| `db.put(key, value, ttl_seconds=60)` | `put()` has no `ttl_seconds` parameter |
| `db.with_transaction(fn)` | Method does not exist |
| `from sochdb import IsolationLevel` | Not importable |
| `txn.start_ts` | Property does not exist |
| `txn.isolation` | Property does not exist |
| `db.begin_transaction()` | Method doesn't exist (`transaction()` does) |

### 6. `stats()` Method Shadowed — Returns Placeholder (L1)
**Severity: HIGH** — Observability broken

`stats()` is defined twice in `database.py`. The second definition (line ~2020) shadows the FFI-backed version (line ~1648) and returns `keys_count=-1` placeholder.

| Method | Works? |
|---|---|
| `db.stats()` | Returns `{keys_count: -1, ...}` (placeholder) |
| `db.stats_full()` | Returns real FFI data (memtable_size, wal_size, etc.) |

---

## MEDIUM — Edge Cases & Gaps

### 7. Cache Delete Doesn't Immediately Invalidate (G3.2)
After `cache_delete("ops_cache", "key_5")`, `cache_get()` with the exact same embedding still returns the deleted entry. Cache deletion may be deferred or the similarity index is not updated.

### 8. Null Byte in Database Path Silently Accepted (J3.1)
`Database.open("/tmp/bad\x00path")` succeeds instead of raising an error. Null bytes in paths are invalid on all platforms and can cause C-string truncation bugs in the Rust FFI layer.

### 9. `scan_prefix_unchecked("")` Returns 0 Keys (L2.1)
Despite having 100+ keys in the database, `scan_prefix_unchecked(b"")` returns empty. This method is supposed to bypass the 2-byte minimum prefix check and scan all keys — but it doesn't find anything. Either the method is broken or the internal key encoding doesn't match empty-prefix iteration.

---

## What Passed Well

| Area | Tests | Notes |
|---|---|---|
| **Dirty Read Prevention** | A4 ✅ | Readers never see uncommitted writes |
| **Phantom Read Prevention** | A3 ✅ | Scan within txn is snapshot-consistent |
| **Read-Your-Writes** | A5 ✅ | Txn reads its own uncommitted writes |
| **Transaction Lifecycle** | A6 ✅ | Double-commit, commit-after-abort all properly rejected |
| **Concurrent KV Writes** | B1 ✅ | 4000 writes across 40 threads, zero corruption |
| **100 Simultaneous Txns** | B3 ✅ | All committed cleanly |
| **SQL Engine** | C1-C5 ✅ | Injection blocked, type coercion, ORDER BY, LIKE patterns |
| **Queue Visibility Timeout** | D2 ✅ | Timeout, invisibility, re-visibility all correct |
| **Queue NACK / DLQ** | D3 ✅ | Max-attempts exhaustion works |
| **Graph Topology** | E1-E3 ✅ | Self-loops, cycles, dangling edges, Unicode IDs all handled |
| **Temporal Graph** | F1 ✅ | Inverted intervals, boundary queries, zero/future timestamps |
| **Cache TTL** | G1 ✅ | Short TTL expires, zero TTL never expires |
| **Cache Similarity** | G2 ✅ | Threshold, orthogonal rejection, zero-vec, dim mismatch |
| **Vector Search** | H1-H3 ✅ | Zero vectors, large K, duplicates, 5000-vector HNSW stress |
| **Large Values** | I1 ✅ | 10MB and 50MB round-trip |
| **100K Keys** | I2 ✅ | Insert 0.6s, scan 0.25s, point reads correct |
| **Deep Nesting** | I3 ✅ | 50-level paths, 2KB keys |
| **Crash Recovery** | J1-J4 ✅ | Close/reopen durable, closed-DB ops raise properly |
| **Compression** | M1 ✅ | Switch codecs mid-stream, old data still readable |
| **Binary Roundtrip** | M2 ✅ | Null bytes, full byte range, UTF-8, 16KB binary |
| **Backup Under Load** | N1 ✅ | Writes during backup, backup verifies clean |
| **Namespace Isolation** | O1 ✅ | 10 tenants × 100 keys, scan isolation confirmed |
| **Batch Ops** | P1 ✅ | Empty batch, 10K batch (17ms), duplicate-key last-write-wins |

---

## Priority Recommendations

1. **Fix SSI conflict detection** — The Rust MVCC layer appears to use snapshot isolation (SI) not serializable snapshot isolation (SSI). Write skew and lost updates are the canonical SI anomalies that SSI was designed to prevent. Either implement proper read-set tracking with rw-dependency detection, or downgrade the documentation to "Snapshot Isolation."

2. **Fix queue dequeue atomicity** — The check-then-act in `dequeue()` must be atomic. Options: use compare-and-swap at the KV layer, or a Rust-side `dequeue` FFI function that atomically claims.

3. **Remove or implement README-documented APIs** — 6 missing APIs erode developer trust. Either implement `ttl_seconds`, `with_transaction`, `IsolationLevel`, etc., or remove them from documentation.

4. **Fix `stats()` method shadowing** — Delete the placeholder second definition so the FFI-backed version is used.

5. **Fix `cache_delete` index invalidation** — Deleted cache entries should not be returned by similarity search.
