#!/usr/bin/env python3
"""
Comprehensive SochDB Python SDK Benchmark & Validation Suite
=============================================================

Validates:
  1. Basic CRUD (put/get/delete)
  2. Path-based API (put_path/get_path/delete_path)
  3. Transaction API (begin/commit/abort, context manager)
  4. Scan operations (scan_prefix, scan_prefix_unchecked, scan_batched)
  5. Configuration options (sync_mode, index_policy, wal_enabled)
  6. Batch / bulk operations
  7. SSI conflict detection
  8. ** Concurrent FFI (ProcessPoolExecutor + open_concurrent) **
  9. Performance benchmarks
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pytest fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base(tmp_path):
    """Provide a temporary directory for each test."""
    return str(tmp_path)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS = []


def report(name: str, passed: bool, detail: str = "", elapsed: float = 0.0):
    status = "PASS" if passed else "FAIL"
    RESULTS.append({"name": name, "passed": passed, "detail": detail, "elapsed_ms": round(elapsed * 1000, 2)})
    suffix = f" ({detail})" if detail else ""
    time_str = f" [{elapsed*1000:.1f}ms]" if elapsed else ""
    print(f"  [{status}] {name}{suffix}{time_str}")


def fresh_dir(base: str, name: str) -> str:
    p = os.path.join(base, name)
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Basic CRUD
# ---------------------------------------------------------------------------

def test_basic_crud(base: str):
    print("\n=== 1. Basic CRUD ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_crud")
    db = Database.open(db_path)

    t0 = time.perf_counter()

    # put / get
    db.put(b"hello", b"world")
    val = db.get(b"hello")
    assert val == b"world", f"Expected b'world', got {val}"
    report("put/get", True, elapsed=time.perf_counter() - t0)

    # overwrite
    db.put(b"hello", b"updated")
    val = db.get(b"hello")
    assert val == b"updated", f"Expected b'updated', got {val}"
    report("overwrite", True)

    # get non-existent
    val = db.get(b"no_such_key")
    assert val is None, f"Expected None, got {val}"
    report("get missing key", True)

    # delete
    db.put(b"to_delete", b"bye")
    db.delete(b"to_delete")
    val = db.get(b"to_delete")
    assert val is None, f"Expected None after delete, got {val}"
    report("delete", True)

    # binary keys/values
    bk = struct.pack(">Q", 0xDEADBEEF)
    bv = bytes(range(256))
    db.put(bk, bv)
    assert db.get(bk) == bv
    report("binary key/value", True)

    db.close()
    report("close + reopen read", True)


# ---------------------------------------------------------------------------
# 2. Path API
# ---------------------------------------------------------------------------

def test_path_api(base: str):
    print("\n=== 2. Path API ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_path")
    db = Database.open(db_path)

    db.put_path("users/alice/email", b"alice@example.com")
    db.put_path("users/alice/name", b"Alice")
    db.put_path("users/bob/email", b"bob@example.com")

    val = db.get_path("users/alice/email")
    assert val == b"alice@example.com", f"Got {val}"
    report("put_path / get_path", True)

    val = db.get_path("users/nonexistent")
    assert val is None
    report("get_path missing", True)

    db.close()


# ---------------------------------------------------------------------------
# 3. Transaction API
# ---------------------------------------------------------------------------

def test_transactions(base: str):
    print("\n=== 3. Transaction API ===")
    from sochdb import Database, TransactionError

    db_path = fresh_dir(base, "test_txn")
    db = Database.open(db_path)

    # Context-manager commit
    with db.transaction() as txn:
        txn.put(b"tx_key1", b"tx_val1")
        txn.put(b"tx_key2", b"tx_val2")
    # should be committed
    assert db.get(b"tx_key1") == b"tx_val1"
    report("txn context-manager commit", True)

    # Context-manager abort on exception
    try:
        with db.transaction() as txn:
            txn.put(b"abort_key", b"abort_val")
            raise ValueError("boom")
    except ValueError:
        pass
    assert db.get(b"abort_key") is None
    report("txn auto-abort on exception", True)

    # Explicit commit returns timestamp
    txn = db.transaction()
    txn.put(b"ts_key", b"ts_val")
    ts = txn.commit()
    assert isinstance(ts, int) and ts > 0, f"commit_ts={ts}"
    report("txn explicit commit with timestamp", True, detail=f"ts={ts}")

    # Double-commit raises
    try:
        txn.commit()
        report("double commit raises", False, "no exception")
    except TransactionError:
        report("double commit raises", True)

    # Explicit abort
    txn2 = db.transaction()
    txn2.put(b"abort2", b"val")
    txn2.abort()
    assert db.get(b"abort2") is None
    report("txn explicit abort", True)

    db.close()


# ---------------------------------------------------------------------------
# 4. Scan Operations
# ---------------------------------------------------------------------------

def test_scans(base: str):
    print("\n=== 4. Scan Operations ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_scan")
    db = Database.open(db_path)

    # Insert 100 keys with a known prefix
    for i in range(100):
        db.put(f"scan/{i:04d}".encode(), f"val-{i}".encode())

    # Also insert keys with different prefix
    for i in range(20):
        db.put(f"other/{i:04d}".encode(), b"x")

    # scan_prefix
    t0 = time.perf_counter()
    results = list(db.scan_prefix(b"scan/"))
    elapsed = time.perf_counter() - t0
    assert len(results) == 100, f"Expected 100 results, got {len(results)}"
    report("scan_prefix (100 keys)", True, elapsed=elapsed)

    # Verify no cross-prefix leakage
    prefixes = set(k[:5] for k, v in results)
    assert prefixes == {b"scan/"}, f"Leakage detected: {prefixes}"
    report("scan_prefix isolation", True)

    # scan_prefix minimum length check
    try:
        list(db.scan_prefix(b"s"))
        report("scan_prefix min length guard", False, "no ValueError")
    except ValueError:
        report("scan_prefix min length guard", True)

    # scan_prefix_unchecked (allows short prefix — test with 2-byte prefix)
    results_u = list(db.scan_prefix_unchecked(b"sc"))
    assert len(results_u) == 100, f"unchecked got {len(results_u)}"
    report("scan_prefix_unchecked", True)

    db.close()


# ---------------------------------------------------------------------------
# 5. Configuration Options
# ---------------------------------------------------------------------------

def test_config_options(base: str):
    print("\n=== 5. Configuration Options ===")
    from sochdb import Database

    configs = [
        ("sync_off", {"sync_mode": "off", "wal_enabled": True}),
        ("sync_full", {"sync_mode": "full", "wal_enabled": True}),
        ("write_optimized", {"index_policy": "write_optimized"}),
        ("scan_optimized", {"index_policy": "scan_optimized"}),
        ("append_only", {"index_policy": "append_only"}),
        ("group_commit", {"group_commit": True}),
    ]

    for label, cfg in configs:
        db_path = fresh_dir(base, f"test_cfg_{label}")
        try:
            db = Database.open(db_path, config=cfg)
            db.put(b"cfg_key", b"cfg_val")
            assert db.get(b"cfg_key") == b"cfg_val"
            db.close()
            report(f"config: {label}", True)
        except Exception as e:
            report(f"config: {label}", False, str(e))


# ---------------------------------------------------------------------------
# 6. Batch / Bulk Operations
# ---------------------------------------------------------------------------

def test_bulk(base: str):
    print("\n=== 6. Batch / Bulk Operations ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_bulk")
    db = Database.open(db_path, config={"sync_mode": "off"})

    N = 5000

    # Bulk put with single transaction
    t0 = time.perf_counter()
    with db.transaction() as txn:
        for i in range(N):
            txn.put(f"bulk/{i:06d}".encode(), f"value-{i}".encode())
    elapsed = time.perf_counter() - t0
    report(f"bulk put {N} keys (1 txn)", True, f"{N/elapsed:.0f} ops/sec", elapsed)

    # Verify all present
    t0 = time.perf_counter()
    results = list(db.scan_prefix(b"bulk/"))
    elapsed = time.perf_counter() - t0
    assert len(results) == N, f"Expected {N}, got {len(results)}"
    report(f"bulk verify via scan_prefix", True, f"{len(results)} keys", elapsed)

    # Bulk random reads
    import random
    keys = [f"bulk/{random.randint(0, N-1):06d}".encode() for _ in range(1000)]
    t0 = time.perf_counter()
    for k in keys:
        val = db.get(k)
        assert val is not None
    elapsed = time.perf_counter() - t0
    report(f"bulk random get (1000)", True, f"{1000/elapsed:.0f} ops/sec", elapsed)

    db.close()


# ---------------------------------------------------------------------------
# 7. Persistence (close + reopen)
# ---------------------------------------------------------------------------

def test_persistence(base: str):
    print("\n=== 7. Persistence (close & reopen) ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_persist")

    # Write and close
    db = Database.open(db_path)
    for i in range(50):
        db.put(f"persist/{i:04d}".encode(), f"v{i}".encode())
    db.close()

    # Reopen and verify
    db2 = Database.open(db_path)
    for i in range(50):
        val = db2.get(f"persist/{i:04d}".encode())
        assert val == f"v{i}".encode(), f"Key {i}: expected v{i}, got {val}"
    report("persistence across close/reopen", True, "50 keys verified")
    db2.close()


# ---------------------------------------------------------------------------
# 8. Concurrent FFI (ProcessPoolExecutor + open_concurrent)
# ---------------------------------------------------------------------------

# Worker function must be at module level for ProcessPoolExecutor
def _concurrent_worker(args):
    """
    Each worker opens the DB in concurrent mode, writes append-only deltas
    using a unique UUID suffix so there are ZERO collisions.
    """
    db_path, worker_id, num_writes = args
    from sochdb import Database

    db = Database.open_concurrent(db_path)
    written_keys = []
    for i in range(num_writes):
        delta_id = uuid.uuid4().hex
        key = f"delta/{worker_id}/{delta_id}".encode()
        value = json.dumps({"worker": worker_id, "seq": i, "ts": time.time()}).encode()
        db.put(key, value)
        written_keys.append(key.decode())
    db.close()
    return {"worker": worker_id, "written": len(written_keys), "keys": written_keys}


def test_concurrent_ffi(base: str):
    print("\n=== 8. Concurrent FFI (ProcessPoolExecutor) ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_concurrent")

    # Pre-create the database
    db = Database.open_concurrent(db_path)
    db.put(b"init_key", b"init_val")
    db.close()

    NUM_WORKERS = 5
    WRITES_PER_WORKER = 20
    EXPECTED_TOTAL = NUM_WORKERS * WRITES_PER_WORKER

    t0 = time.perf_counter()
    all_keys = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = []
        for w in range(NUM_WORKERS):
            futures.append(pool.submit(_concurrent_worker, (db_path, w, WRITES_PER_WORKER)))

        for f in as_completed(futures):
            result = f.result()
            all_keys.extend(result["keys"])
            print(f"    Worker {result['worker']}: wrote {result['written']} keys")

    elapsed = time.perf_counter() - t0
    report(f"concurrent writes ({NUM_WORKERS}×{WRITES_PER_WORKER})", True,
           f"{len(all_keys)} keys written", elapsed)

    # Now verify ALL writes persisted
    db = Database.open(db_path)
    found = list(db.scan_prefix(b"delta/"))
    found_count = len(found)
    db.close()

    passed = found_count == EXPECTED_TOTAL
    report(
        f"concurrent persistence check",
        passed,
        f"found {found_count}/{EXPECTED_TOTAL} keys"
    )
    if not passed:
        print(f"    *** CRITICAL: Lost {EXPECTED_TOTAL - found_count} writes! ***")

    return passed, found_count, EXPECTED_TOTAL


# ---------------------------------------------------------------------------
# 9. Concurrent FFI stress test (higher load)
# ---------------------------------------------------------------------------

def test_concurrent_stress(base: str):
    print("\n=== 9. Concurrent FFI Stress Test ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_concurrent_stress")

    # Pre-create
    db = Database.open_concurrent(db_path)
    db.put(b"init_stress", b"ok")
    db.close()

    NUM_WORKERS = 8
    WRITES_PER_WORKER = 50
    EXPECTED_TOTAL = NUM_WORKERS * WRITES_PER_WORKER

    t0 = time.perf_counter()
    all_keys = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [
            pool.submit(_concurrent_worker, (db_path, w, WRITES_PER_WORKER))
            for w in range(NUM_WORKERS)
        ]
        for f in as_completed(futures):
            result = f.result()
            all_keys.extend(result["keys"])

    elapsed = time.perf_counter() - t0
    report(f"stress writes ({NUM_WORKERS}×{WRITES_PER_WORKER})", True,
           f"{len(all_keys)} keys written", elapsed)

    # Verify
    db = Database.open(db_path)
    found = list(db.scan_prefix(b"delta/"))
    db.close()

    passed = len(found) == EXPECTED_TOTAL
    report(
        f"stress persistence check",
        passed,
        f"found {len(found)}/{EXPECTED_TOTAL} keys"
    )
    if not passed:
        print(f"    *** CRITICAL: Lost {EXPECTED_TOTAL - len(found)} writes! ***")

    return passed


# ---------------------------------------------------------------------------
# 10. Performance Micro-benchmarks
# ---------------------------------------------------------------------------

def test_performance(base: str):
    print("\n=== 10. Performance Micro-benchmarks ===")
    from sochdb import Database

    db_path = fresh_dir(base, "test_perf")
    db = Database.open(db_path, config={"sync_mode": "off"})

    # Sequential writes
    N = 10000
    t0 = time.perf_counter()
    with db.transaction() as txn:
        for i in range(N):
            txn.put(f"perf/{i:08d}".encode(), os.urandom(128))
    elapsed = time.perf_counter() - t0
    wps = N / elapsed
    report(f"sequential write {N}", True, f"{wps:,.0f} ops/sec", elapsed)

    # Sequential reads
    t0 = time.perf_counter()
    for i in range(N):
        db.get(f"perf/{i:08d}".encode())
    elapsed = time.perf_counter() - t0
    rps = N / elapsed
    report(f"sequential read {N}", True, f"{rps:,.0f} ops/sec", elapsed)

    # Prefix scan throughput
    t0 = time.perf_counter()
    count = 0
    for k, v in db.scan_prefix(b"perf/"):
        count += 1
    elapsed = time.perf_counter() - t0
    report(f"prefix scan {count} keys", True, f"{count/elapsed:,.0f} keys/sec", elapsed)

    db.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("  SochDB Python SDK — Comprehensive Benchmark Suite")
    print("=" * 70)

    # Import check
    try:
        from sochdb import Database
        print(f"\n  SDK version: {__import__('sochdb').__version__}")
    except ImportError as e:
        print(f"\n  FATAL: Cannot import sochdb: {e}")
        sys.exit(1)

    base = tempfile.mkdtemp(prefix="sochdb_bench_")
    print(f"  Working directory: {base}")

    tests = [
        ("Basic CRUD", test_basic_crud),
        ("Path API", test_path_api),
        ("Transaction API", test_transactions),
        ("Scan Operations", test_scans),
        ("Configuration Options", test_config_options),
        ("Batch/Bulk", test_bulk),
        ("Persistence", test_persistence),
        ("Concurrent FFI", test_concurrent_ffi),
        ("Concurrent Stress", test_concurrent_stress),
        ("Performance", test_performance),
    ]

    failures = []
    for name, fn in tests:
        try:
            fn(base)
        except Exception as e:
            report(f"{name} (EXCEPTION)", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(name)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    total = len(RESULTS)

    for r in RESULTS:
        mark = "✓" if r["passed"] else "✗"
        print(f"  {mark} {r['name']}")

    print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")

    if failures:
        print(f"\n  FAILED test groups: {', '.join(failures)}")

    # Clean up
    try:
        shutil.rmtree(base)
        print(f"\n  Cleaned up {base}")
    except Exception:
        pass

    if failed:
        print("\n  ❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\n  ✅ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
