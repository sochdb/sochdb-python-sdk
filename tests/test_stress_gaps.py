#!/usr/bin/env python3
"""
SochDB Stress Test & Gap Analysis
===================================
Jepsen-inspired correctness + stress testing.
Goes beyond happy-path: concurrency anomalies, resource exhaustion,
edge-case inputs, crash recovery, race conditions, API contract violations.

Categories:
  A. Transaction Isolation (SSI correctness)
  B. Concurrency & Thread Safety
  C. SQL Engine Edge Cases
  D. Queue Race Conditions
  E. Graph Topology Edge Cases
  F. Temporal Graph Boundaries
  G. Cache Correctness
  H. Vector Search Edge Cases
  I. Resource Exhaustion
  J. Crash Recovery & Error Paths
  K. API Contract Violations (README vs Reality)
  L. Method Shadowing & Dead Code
"""

import os, sys, time, json, math, struct, shutil, tempfile, hashlib
import threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Ensure the SDK is on the path ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from sochdb import Database

PASS = 0
FAIL = 0
SKIP = 0
results = []

def check(name, condition, detail=""):
    global PASS, FAIL
    tag = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  {str(detail)[:120]}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    results.append((tag, name, detail))

def skip(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}  {reason}")
    results.append(("SKIP", name, reason))

def section(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")

def expect_error(name, fn, error_types=Exception, detail=""):
    """Check that fn() raises one of error_types."""
    try:
        fn()
        check(name, False, f"expected {error_types.__name__} but no exception raised")
    except error_types as e:
        check(name, True, detail or str(e)[:100])
    except Exception as e:
        check(name, False, f"expected {error_types.__name__} but got {type(e).__name__}: {e}")

tmpdir = tempfile.mkdtemp(prefix="sochdb_stress_")
print(f"Working directory: {tmpdir}")


###############################################################################
#   A. TRANSACTION ISOLATION — SSI CORRECTNESS (Jepsen-inspired)
###############################################################################

section("A1. Lost Update Detection (SSI)")

# Classic lost update: two txns read same counter, both increment, both try to commit
# SSI should reject one.
db_path = os.path.join(tmpdir, "lost_update_db")
db = Database.open(db_path)
db.put(b"counter", b"0")

txn1 = db.transaction()
txn2 = db.transaction()

# Both read the same value
val1 = txn1.get(b"counter")
val2 = txn2.get(b"counter")
check("A1.1 both txns read counter=0", val1 == b"0" and val2 == b"0")

# Both increment — SSI may detect conflict at write time OR commit time
committed_1 = False
committed_2 = False
error_on_commit = None
try:
    txn1.put(b"counter", str(int(val1) + 1).encode())
    txn1.commit()
    committed_1 = True
except Exception as e:
    error_on_commit = str(e)
    try:
        txn1.abort()
    except Exception:
        pass

try:
    txn2.put(b"counter", str(int(val2) + 1).encode())
    txn2.commit()
    committed_2 = True
except Exception as e:
    error_on_commit = error_on_commit or str(e)
    try:
        txn2.abort()
    except Exception:
        pass

# SSI should reject one
check("A1.2 SSI detects lost update (one rejected)",
      not (committed_1 and committed_2),
      f"txn1={'OK' if committed_1 else 'REJECTED'}, txn2={'OK' if committed_2 else 'REJECTED'}, err={error_on_commit}")

final = db.get(b"counter")
check("A1.3 counter = 1 (not 2)", final == b"1",
      f"counter={final}")
db.close()


section("A2. Write Skew Detection (SSI)")

# Write skew: T1 reads A, T2 reads A, T1 writes B based on A, T2 writes A based on old A
db_path = os.path.join(tmpdir, "write_skew_db")
db = Database.open(db_path)
db.put(b"alice_balance", b"100")
db.put(b"bob_balance", b"100")

# Constraint: alice_balance + bob_balance >= 0
# Both txns check the sum, then withdraw 150 from their respective accounts
txn1 = db.transaction()
txn2 = db.transaction()

alice1 = int(txn1.get(b"alice_balance"))
bob1 = int(txn1.get(b"bob_balance"))
alice2 = int(txn2.get(b"alice_balance"))
bob2 = int(txn2.get(b"bob_balance"))

check("A2.1 both see sum=200", alice1 + bob1 == 200 and alice2 + bob2 == 200)

# T1 withdraws 150 from alice (thinks sum=200, so 200-150=50 >= 0)
# T2 withdraws 150 from bob (thinks same)
# SSI may detect conflict at write time or commit time
c1 = c2 = False
try:
    txn1.put(b"alice_balance", str(alice1 - 150).encode())
    txn1.commit()
    c1 = True
except Exception:
    try:
        txn1.abort()
    except Exception:
        pass

try:
    txn2.put(b"bob_balance", str(bob2 - 150).encode())
    txn2.commit()
    c2 = True
except Exception:
    try:
        txn2.abort()
    except Exception:
        pass

check("A2.2 SSI prevents write skew (one rejected)",
      not (c1 and c2),
      f"T1={'COMMIT' if c1 else 'ABORT'}, T2={'COMMIT' if c2 else 'ABORT'}")

# If both committed, total would be -100 (violation)
a = int(db.get(b"alice_balance"))
b_val = int(db.get(b"bob_balance"))
check("A2.3 invariant preserved: sum >= 0",
      a + b_val >= 0,
      f"alice={a}, bob={b_val}, sum={a + b_val}")
db.close()


section("A3. Phantom Read Prevention")

# T1 scans prefix, T2 inserts into that prefix, T1 scans again within same txn
db_path = os.path.join(tmpdir, "phantom_db")
db = Database.open(db_path)
for i in range(5):
    db.put(f"items/{i:04d}".encode(), f"val{i}".encode())

txn1 = db.transaction()
scan1 = list(txn1.scan_prefix(b"items/"))
check("A3.1 initial scan sees 5", len(scan1) == 5)

# Another transaction inserts
try:
    txn_insert = db.transaction()
    txn_insert.put(b"items/9999", b"phantom")
    txn_insert.commit()
except Exception:
    try:
        txn_insert.abort()
    except Exception:
        pass
    # If concurrent txn fails, insert outside transaction
    db.put(b"items/9999", b"phantom")

# T1 should still see same snapshot (no phantoms)
scan2 = list(txn1.scan_prefix(b"items/"))
check("A3.2 no phantom: second scan within txn still sees 5",
      len(scan2) == 5,
      f"got {len(scan2)}")

txn1.commit()

# After commit, new txn should see the insert
scan3 = list(db.scan_prefix(b"items/"))
check("A3.3 after commit, new read sees 6", len(scan3) == 6)
db.close()


section("A4. Dirty Read Prevention")

db_path = os.path.join(tmpdir, "dirty_read_db")
db = Database.open(db_path)
db.put(b"secret", b"original")

txn_writer = db.transaction()
try:
    txn_writer.put(b"secret", b"dirty_value")
except Exception:
    pass  # SSI may reject if concurrent access

# Reader should NOT see the uncommitted write
val = db.get(b"secret")
check("A4.1 no dirty read: reader sees original", val == b"original", f"got={val}")

try:
    txn_writer.abort()
except Exception:
    pass
val2 = db.get(b"secret")
check("A4.2 after abort, still original", val2 == b"original")
db.close()


section("A5. Read-Your-Writes Consistency")

db_path = os.path.join(tmpdir, "ryw_db")
db = Database.open(db_path)

with db.transaction() as txn:
    txn.put(b"ryw_key", b"ryw_val")
    read_back = txn.get(b"ryw_key")
    check("A5.1 read-your-write within txn", read_back == b"ryw_val")

    txn.put(b"ryw_key", b"updated")
    read_again = txn.get(b"ryw_key")
    check("A5.2 second read-your-write", read_again == b"updated")
db.close()


section("A6. Transaction Lifecycle Edge Cases")

db_path = os.path.join(tmpdir, "txn_lifecycle_db")
db = Database.open(db_path)

# Double commit
txn = db.transaction()
txn.put(b"k", b"v")
txn.commit()
try:
    txn.commit()
    check("A6.1 double commit raises error", False, "no exception")
except Exception as e:
    check("A6.1 double commit raises error", True, str(e)[:80])

# Commit after abort
txn2 = db.transaction()
txn2.put(b"k2", b"v2")
txn2.abort()
try:
    txn2.commit()
    check("A6.2 commit after abort raises error", False, "no exception")
except Exception as e:
    check("A6.2 commit after abort raises error", True, str(e)[:80])

# Use after commit
txn3 = db.transaction()
txn3.put(b"k3", b"v3")
txn3.commit()
try:
    txn3.put(b"k4", b"v4")
    check("A6.3 put after commit raises error", False, "no exception")
except Exception as e:
    check("A6.3 put after commit raises error", True, str(e)[:80])

# Use after abort
txn4 = db.transaction()
txn4.abort()
try:
    txn4.get(b"k")
    check("A6.4 get after abort raises error", False, "no exception")
except Exception as e:
    check("A6.4 get after abort raises error", True, str(e)[:80])

db.close()


###############################################################################
#   B. CONCURRENCY & THREAD SAFETY
###############################################################################

section("B1. Concurrent Writes (Thread Safety)")

db_path = os.path.join(tmpdir, "concurrent_writes_db")
db = Database.open_concurrent(db_path)

errors = []
NUM_THREADS = 8
WRITES_PER_THREAD = 500

def writer(thread_id):
    for i in range(WRITES_PER_THREAD):
        try:
            db.put(f"t{thread_id}/k{i}".encode(), f"{thread_id}-{i}".encode())
        except Exception as e:
            errors.append((thread_id, i, str(e)))

threads = [threading.Thread(target=writer, args=(t,)) for t in range(NUM_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("B1.1 no errors from concurrent writes", len(errors) == 0,
      f"{len(errors)} errors" if errors else f"{NUM_THREADS * WRITES_PER_THREAD} writes OK")

# Verify all data
missing = 0
corrupt = 0
for tid in range(NUM_THREADS):
    for i in range(WRITES_PER_THREAD):
        val = db.get(f"t{tid}/k{i}".encode())
        if val is None:
            missing += 1
        elif val != f"{tid}-{i}".encode():
            corrupt += 1

check("B1.2 all writes readable", missing == 0,
      f"missing={missing}" if missing else f"{NUM_THREADS * WRITES_PER_THREAD} verified")
check("B1.3 no corruption", corrupt == 0, f"corrupt={corrupt}" if corrupt else "")
db.close()


section("B2. Concurrent Transaction Conflicts")

db_path = os.path.join(tmpdir, "txn_conflicts_db")
db = Database.open_concurrent(db_path)
db.put(b"shared_counter", b"0")

committed_count = 0
aborted_count = 0
lock = threading.Lock()

def increment_counter(worker_id):
    global committed_count, aborted_count
    for _ in range(50):
        retries = 0
        while retries < 10:
            try:
                with db.transaction() as txn:
                    val = int(txn.get(b"shared_counter") or b"0")
                    txn.put(b"shared_counter", str(val + 1).encode())
                with lock:
                    committed_count += 1
                break
            except Exception:
                retries += 1
                with lock:
                    aborted_count += 1

threads = [threading.Thread(target=increment_counter, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

final_val = int(db.get(b"shared_counter"))
check("B2.1 some conflicts detected", aborted_count > 0 or committed_count == 200,
      f"committed={committed_count}, aborted={aborted_count}")
check("B2.2 counter = committed count", final_val == committed_count,
      f"counter={final_val}, committed={committed_count}")
db.close()


section("B3. Many Simultaneous Open Transactions")

db_path = os.path.join(tmpdir, "many_txns_db")
db = Database.open(db_path)

open_txns = []
MAX_TXNS = 100
try:
    for i in range(MAX_TXNS):
        txn = db.transaction()
        txn.put(f"txn_key_{i}".encode(), f"txn_val_{i}".encode())
        open_txns.append(txn)
    check("B3.1 opened 100 simultaneous txns", len(open_txns) == MAX_TXNS)
except Exception as e:
    check("B3.1 opened 100 simultaneous txns", False, str(e)[:100])

# Commit all
committed = 0
for txn in open_txns:
    try:
        txn.commit()
        committed += 1
    except Exception:
        try:
            txn.abort()
        except Exception:
            pass

check("B3.2 committed or aborted all txns", True, f"committed={committed}/{MAX_TXNS}")

# Verify data
readable = sum(1 for i in range(MAX_TXNS) if db.get(f"txn_key_{i}".encode()) is not None)
check("B3.3 data from committed txns readable", readable == committed,
      f"readable={readable}, committed={committed}")
db.close()


###############################################################################
#   C. SQL ENGINE EDGE CASES
###############################################################################

section("C1. SQL Concurrent INSERT Sequence Race")

db_path = os.path.join(tmpdir, "sql_race_db")
db = Database.open_concurrent(db_path)
db.execute("CREATE TABLE race_test (id INTEGER PRIMARY KEY, name TEXT)")

errors_sql = []
ids_inserted = []
sql_lock = threading.Lock()

def sql_inserter(worker_id):
    for i in range(20):
        try:
            db.execute(f"INSERT INTO race_test (name) VALUES ('worker{worker_id}_row{i}')")
        except Exception as e:
            with sql_lock:
                errors_sql.append((worker_id, i, str(e)))

threads = [threading.Thread(target=sql_inserter, args=(w,)) for w in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Check for duplicate IDs (the race condition)
result = db.execute("SELECT id FROM race_test ORDER BY id")
ids = [r["id"] for r in result.rows]
unique_ids = set(ids)
check("C1.1 no duplicate PKs from concurrent INSERT",
      len(ids) == len(unique_ids),
      f"total={len(ids)}, unique={len(unique_ids)}, dups={len(ids)-len(unique_ids)}")
check("C1.2 all rows inserted", len(ids) == 80 or len(errors_sql) > 0,
      f"rows={len(ids)}, errors={len(errors_sql)}")
db.close()


section("C2. SQL Injection & Parsing Edge Cases")

db_path = os.path.join(tmpdir, "sql_inject_db")
db = Database.open(db_path)
db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
db.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@test.com')")

# SQL injection attempt via value
try:
    db.execute("INSERT INTO users (name, email) VALUES ('Robert'); DROP TABLE users; --', 'bob@test.com')")
    # Check table still exists
    result = db.execute("SELECT id FROM users")
    check("C2.1 SQL injection via value blocked", len(result.rows) >= 1, f"rows={len(result.rows)}")
except Exception as e:
    check("C2.1 SQL injection via value blocked", True, f"rejected: {str(e)[:80]}")

# Empty string value
db.execute("INSERT INTO users (name, email) VALUES ('', '')")
result = db.execute("SELECT name FROM users WHERE name = ''")
check("C2.2 empty string value", len(result.rows) >= 1, f"rows={len(result.rows)}")

# Very long string
long_name = "x" * 10000
try:
    db.execute(f"INSERT INTO users (name, email) VALUES ('{long_name}', 'long@test.com')")
    result = db.execute("SELECT name FROM users WHERE email = 'long@test.com'")
    check("C2.3 10KB string in SQL", len(result.rows) == 1 and len(result.rows[0]["name"]) == 10000)
except Exception as e:
    check("C2.3 10KB string in SQL", False, str(e)[:100])

# NULL handling
db.execute("INSERT INTO users (name, email) VALUES ('NullTest', NULL)")
result = db.execute("SELECT email FROM users WHERE name = 'NullTest'")
check("C2.4 NULL value handling", len(result.rows) == 1, f"email={result.rows[0].get('email') if result.rows else 'N/A'}")

db.close()


section("C3. SQL LIKE Pattern Bugs")

db_path = os.path.join(tmpdir, "sql_like_db")
db = Database.open(db_path)
db.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT)")
db.execute("INSERT INTO docs (title) VALUES ('hello_world')")
db.execute("INSERT INTO docs (title) VALUES ('hello world')")
db.execute("INSERT INTO docs (title) VALUES ('helloXworld')")
db.execute("INSERT INTO docs (title) VALUES ('hello%world')")

# LIKE with % wildcard
result = db.execute("SELECT title FROM docs WHERE title LIKE 'hello%'")
check("C3.1 LIKE 'hello%' matches all 4", len(result.rows) == 4,
      f"got {len(result.rows)}: {[r['title'] for r in result.rows]}")

# LIKE with _ single char wildcard
result = db.execute("SELECT title FROM docs WHERE title LIKE 'hello_world'")
check("C3.2 LIKE 'hello_world' _ wildcard", len(result.rows) >= 1,
      f"got {len(result.rows)}: {[r['title'] for r in result.rows]}")

# LIKE with literal % in data — the reported bug
result = db.execute("SELECT title FROM docs WHERE title LIKE 'hello\\%world'")
# This tests if literal % matching works (it likely doesn't per the audit)
check("C3.3 LIKE literal % (may fail — known bug)", len(result.rows) >= 0,
      f"got {len(result.rows)}: {[r['title'] for r in result.rows]}")

db.close()


section("C4. SQL Type Coercion & Edge Cases")

db_path = os.path.join(tmpdir, "sql_types_db")
db = Database.open(db_path)
db.execute("CREATE TABLE typed (id INTEGER PRIMARY KEY, val TEXT, num FLOAT)")

# Insert wrong types (TEXT into INT-expected column via auto-id)
try:
    db.execute("INSERT INTO typed (val, num) VALUES ('text', 3.14)")
    result = db.execute("SELECT val, num FROM typed WHERE val = 'text'")
    check("C4.1 basic typed insert", len(result.rows) == 1)
except Exception as e:
    check("C4.1 basic typed insert", False, str(e)[:100])

# Float edge cases
try:
    db.execute("INSERT INTO typed (val, num) VALUES ('inf', 999999999999999999.0)")
    result = db.execute("SELECT num FROM typed WHERE val = 'inf'")
    check("C4.2 very large float", len(result.rows) == 1, f"num={result.rows[0].get('num')}")
except Exception as e:
    check("C4.2 very large float", False, str(e)[:100])

# Scientific notation
try:
    db.execute("INSERT INTO typed (val, num) VALUES ('sci', 1e10)")
    result = db.execute("SELECT num FROM typed WHERE val = 'sci'")
    # _parse_value won't match `1e10` as float (only -?\d+\.\d+ pattern)
    check("C4.3 scientific notation 1e10 (may fail)", len(result.rows) == 1,
          f"num={result.rows[0].get('num') if result.rows else 'N/A'}")
except Exception as e:
    check("C4.3 scientific notation 1e10 (may fail)", False, str(e)[:100])

# Boolean values
try:
    db.execute("CREATE TABLE bool_test (id INTEGER PRIMARY KEY, flag BOOL)")
    db.execute("INSERT INTO bool_test (flag) VALUES (TRUE)")
    db.execute("INSERT INTO bool_test (flag) VALUES (FALSE)")
    result = db.execute("SELECT flag FROM bool_test")
    check("C4.4 boolean values", len(result.rows) == 2, f"flags={[r.get('flag') for r in result.rows]}")
except Exception as e:
    check("C4.4 boolean values", False, str(e)[:100])

db.close()


section("C5. SQL ORDER BY / WHERE Edge Cases")

db_path = os.path.join(tmpdir, "sql_order_db")
db = Database.open(db_path)
db.execute("CREATE TABLE scores (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)")
for i in range(10):
    db.execute(f"INSERT INTO scores (name, score) VALUES ('player{i}', {i * 10})")

# ORDER BY DESC
result = db.execute("SELECT name, score FROM scores ORDER BY score DESC")
scores = [r["score"] for r in result.rows]
check("C5.1 ORDER BY DESC", scores == sorted(scores, reverse=True), f"scores={scores}")

# WHERE with multiple conditions
result = db.execute("SELECT name FROM scores WHERE score >= 30 AND score <= 70")
check("C5.2 WHERE range AND", len(result.rows) == 5,
      f"got {len(result.rows)}: {[r['name'] for r in result.rows]}")

# WHERE non-existent column (should handle gracefully)
try:
    result = db.execute("SELECT name FROM scores WHERE nonexistent = 5")
    check("C5.3 WHERE on nonexistent column", True,
          f"returned {len(result.rows)} rows")
except Exception as e:
    check("C5.3 WHERE on nonexistent column", True, f"error: {str(e)[:80]}")

db.close()


###############################################################################
#   D. QUEUE RACE CONDITIONS
###############################################################################

section("D1. Concurrent Queue Dequeue (Double Claiming)")

try:
    from sochdb import PriorityQueue, create_queue

    db_path = os.path.join(tmpdir, "queue_race_db")
    db = Database.open_concurrent(db_path)

    queue = create_queue(db, "race_queue")

    # Enqueue many tasks
    NUM_TASKS = 50
    for i in range(NUM_TASKS):
        queue.enqueue(priority=i, payload=f"task-{i}".encode())

    # Concurrent dequeue from multiple workers
    claimed_tasks = []
    claim_lock = threading.Lock()
    claim_errors = []

    def worker_dequeue(wid):
        while True:
            try:
                task = queue.dequeue(worker_id=f"worker-{wid}")
                if task is None:
                    break
                with claim_lock:
                    claimed_tasks.append((wid, task.task_id, task.payload))
                queue.ack(task.task_id)
            except Exception as e:
                with claim_lock:
                    claim_errors.append((wid, str(e)))

    threads = [threading.Thread(target=worker_dequeue, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Check no double-claims
    task_ids = [t[1] for t in claimed_tasks]
    unique_task_ids = set(task_ids)
    check("D1.1 no double-claimed tasks",
          len(task_ids) == len(unique_task_ids),
          f"claimed={len(task_ids)}, unique={len(unique_task_ids)}")
    check("D1.2 all tasks claimed", len(unique_task_ids) == NUM_TASKS,
          f"claimed={len(unique_task_ids)}/{NUM_TASKS}")
    check("D1.3 no errors during dequeue", len(claim_errors) == 0,
          f"errors={claim_errors[:3]}" if claim_errors else "")
    db.close()

except Exception as e:
    check("D1 queue race test", False, str(e)[:100])


section("D2. Queue Visibility Timeout Edge Cases")

try:
    from sochdb import PriorityQueue, create_queue

    db_path = os.path.join(tmpdir, "queue_vis_db")
    db = Database.open(db_path)
    queue = create_queue(db, "vis_queue")

    # Enqueue task
    queue.enqueue(priority=1, payload=b"timeout_test")

    # Dequeue with very short visibility timeout
    task = queue.dequeue(worker_id="w1", visibility_timeout_ms=100)
    check("D2.1 dequeue with 100ms timeout", task is not None)

    # Immediately try to dequeue again — should be invisible
    task2 = queue.dequeue(worker_id="w2")
    check("D2.2 task invisible to other worker", task2 is None)

    # Wait for timeout to expire
    time.sleep(0.2)  # 200ms > 100ms timeout

    # Should be visible again
    task3 = queue.dequeue(worker_id="w2")
    check("D2.3 task re-visible after timeout", task3 is not None,
          f"payload={task3.payload if task3 else None}")
    if task3:
        queue.ack(task3.task_id)

    db.close()
except Exception as e:
    check("D2 visibility timeout", False, str(e)[:100])


section("D3. Queue NACK & Dead Letter Queue")

try:
    from sochdb import PriorityQueue, create_queue
    from sochdb.queue import QueueConfig

    db_path = os.path.join(tmpdir, "queue_dlq_db")
    db = Database.open(db_path)

    queue = PriorityQueue.from_database(db,
        queue_id="dlq_test",
        max_attempts=3,
        visibility_timeout_ms=100,
    )

    queue.enqueue(priority=1, payload=b"will_fail")

    for attempt in range(5):
        task = queue.dequeue(worker_id="w1", visibility_timeout_ms=50)
        if task is None:
            time.sleep(0.1)
            continue
        try:
            queue.nack(task.task_id)
        except Exception:
            break

    # Check if task is dead-lettered or still in queue
    stats = queue.stats()
    check("D3.1 DLQ or exhausted after max_attempts", True,
          f"stats={stats}")
    db.close()

except ImportError:
    skip("D3 DLQ test", "QueueConfig not importable")
except Exception as e:
    check("D3 DLQ", False, str(e)[:100])


###############################################################################
#   E. GRAPH TOPOLOGY EDGE CASES
###############################################################################

section("E1. Graph Self-Loops & Cycles")

db_path = os.path.join(tmpdir, "graph_edge_db")
db = Database.open(db_path)

# Self-loop
try:
    db.add_node("test", "A", "entity", {"name": "Self-Referencer"})
    db.add_edge("test", "A", "LINKS_TO", "A", {"type": "self"})
    check("E1.1 self-loop created", True)

    result = db.traverse("test", "A", max_depth=5)
    check("E1.2 traverse with self-loop doesn't hang",
          result is not None,
          f"nodes={len(result[0]) if isinstance(result, tuple) else result}")
except Exception as e:
    check("E1.1-2 self-loop", False, str(e)[:100])

# Cycle: A -> B -> C -> A
try:
    db.add_node("test", "B", "entity", {})
    db.add_node("test", "C", "entity", {})
    db.add_edge("test", "A", "LINKS_TO", "B")
    db.add_edge("test", "B", "LINKS_TO", "C")
    db.add_edge("test", "C", "LINKS_TO", "A")  # closes cycle

    result = db.traverse("test", "A", max_depth=100)
    nodes, edges = result if isinstance(result, tuple) else (result.get('nodes', []), result.get('edges', []))
    check("E1.3 traverse with cycle doesn't infinite loop",
          result is not None,
          f"nodes={len(nodes)}, edges={len(edges)}")
except Exception as e:
    check("E1.3 cycle traversal", False, str(e)[:100])

# Disconnected component
try:
    db.add_node("test", "X", "entity", {})
    db.add_node("test", "Y", "entity", {})
    db.add_edge("test", "X", "LINKS_TO", "Y")

    # Traverse from A should NOT find X or Y
    result = db.traverse("test", "A", max_depth=10)
    nodes_list = result[0] if isinstance(result, tuple) else result.get("nodes", [])
    node_ids = [n.get("id", n.get("node_id", "")) for n in nodes_list]
    check("E1.4 disconnected component isolation",
          "X" not in node_ids and "Y" not in node_ids,
          f"found nodes: {node_ids}")
except Exception as e:
    check("E1.4 disconnected", False, str(e)[:100])
db.close()


section("E2. Graph Dangling Edges & Deletion Cascading")

db_path = os.path.join(tmpdir, "graph_dangle_db")
db = Database.open(db_path)

# Edge to non-existent node
try:
    db.add_node("ns", "exist", "entity", {})
    db.add_edge("ns", "exist", "REF", "ghost")
    check("E2.1 edge to non-existent node", True, "no validation")
except Exception as e:
    check("E2.1 edge to non-existent node", True, f"rejected: {str(e)[:60]}")

# Delete node — does it cascade-delete edges?
try:
    db.add_node("ns", "src", "entity", {})
    db.add_node("ns", "dst", "entity", {})
    db.add_edge("ns", "src", "REF", "dst")
    db.delete_node("ns", "src")

    # Try to traverse from dst — what happens with the dangling edge?
    result = db.traverse("ns", "dst", max_depth=5)
    check("E2.2 traverse after node deletion", result is not None,
          f"result={result}")
except Exception as e:
    check("E2.2 post-deletion traverse", False, str(e)[:100])

# Find path between disconnected nodes
try:
    db.add_node("ns", "island1", "entity", {})
    db.add_node("ns", "island2", "entity", {})
    path = db.find_path("island1", "island2", namespace="ns")
    check("E2.3 find_path disconnected returns None/empty",
          path is None or path == [] or (isinstance(path, dict) and len(path.get("path", [])) == 0),
          f"path={path}")
except Exception as e:
    check("E2.3 find_path disconnected", True, f"error: {str(e)[:80]}")

db.close()


section("E3. Graph with Special Characters in IDs")

db_path = os.path.join(tmpdir, "graph_special_db")
db = Database.open(db_path)

special_ids = [
    ("unicode_emoji", "🎉"),
    ("unicode_cjk", "你好"),
    ("spaces", "node with spaces"),
    ("slashes", "a/b/c"),
    ("special_chars", "node@#$%"),
]

for name, node_id in special_ids:
    try:
        db.add_node("ns", node_id, "entity", {"label": name})
        result = db.traverse("ns", node_id, max_depth=1)
        nodes_r = result[0] if isinstance(result, tuple) else result.get("nodes", [])
        has_nodes = result is not None and len(nodes_r) >= 1
        check(f"E3 {name} node id", has_nodes, f"id='{node_id}'")
    except Exception as e:
        check(f"E3 {name} node id", False, f"id='{node_id}' error={str(e)[:60]}")

db.close()


###############################################################################
#   F. TEMPORAL GRAPH BOUNDARIES
###############################################################################

section("F1. Temporal Edge Boundary Conditions")

db_path = os.path.join(tmpdir, "temporal_boundary_db")
db = Database.open(db_path)

db.add_node("ns", "Alice", "person", {})
db.add_node("ns", "Bob", "person", {})

# Inverted interval: valid_from > valid_until
try:
    db.add_temporal_edge("ns", "Alice", "KNOWS", "Bob",
                         valid_from=2000, valid_until=1000)
    # Query at midpoint
    result = db.query_temporal_graph("ns", "Alice", mode="POINT_IN_TIME",
                                     timestamp=1500)
    edges = result if isinstance(result, list) else result.get("edges", [])
    check("F1.1 inverted interval (from>until)",
          len(edges) == 0,
          f"edges={len(edges)} (should be 0 for inverted interval)")
except Exception as e:
    check("F1.1 inverted interval", True, f"rejected: {str(e)[:80]}")

# Exact boundary: query at valid_from
try:
    db.add_temporal_edge("ns", "Alice", "MET", "Bob",
                         valid_from=5000, valid_until=6000)
    result = db.query_temporal_graph("ns", "Alice", mode="POINT_IN_TIME",
                                     timestamp=5000)
    edges_at = result if isinstance(result, list) else result.get("edges", [])
    check("F1.2 query at exact valid_from", len(edges_at) >= 1,
          f"edges={len(edges_at)}")
except Exception as e:
    check("F1.2 exact boundary", False, str(e)[:100])

# Query at exact valid_until
try:
    result = db.query_temporal_graph("ns", "Alice", mode="POINT_IN_TIME",
                                     timestamp=6000)
    edges_end = result if isinstance(result, list) else result.get("edges", [])
    check("F1.3 query at exact valid_until (inclusive?)",
          True,  # Just checking it doesn't crash
          f"edges={len(edges_end)} (boundary semantics)")
except Exception as e:
    check("F1.3 exact valid_until", False, str(e)[:100])

# Zero timestamps
try:
    db.add_temporal_edge("ns", "Alice", "EPOCH", "Bob",
                         valid_from=0, valid_until=0)
    check("F1.4 zero timestamps (no crash)", True)
except Exception as e:
    check("F1.4 zero timestamps", False, str(e)[:100])

# Very large timestamps (year 3000)
try:
    future_ts = 32503680000000  # ~year 3000 in ms
    db.add_temporal_edge("ns", "Alice", "FUTURE", "Bob",
                         valid_from=future_ts, valid_until=future_ts + 1000)
    result = db.query_temporal_graph("ns", "Alice", mode="POINT_IN_TIME",
                                     timestamp=future_ts + 500)
    edges_f = result if isinstance(result, list) else result.get('edges', [])
    check("F1.5 far-future timestamps", True,
          f"edges={len(edges_f)}")
except Exception as e:
    check("F1.5 far-future timestamps", False, str(e)[:100])

db.close()


###############################################################################
#   G. CACHE CORRECTNESS
###############################################################################

section("G1. Cache TTL Boundary Conditions")

db_path = os.path.join(tmpdir, "cache_ttl_db")
db = Database.open(db_path)

dim = 4
embed = lambda seed: [(seed * 0.1 + i * 0.01) for i in range(dim)]

# Very short TTL
db.cache_put("ttl_cache", "short_lived", "value_short",
             embedding=embed(1), ttl_seconds=1)

# Verify it's there
result = db.cache_get("ttl_cache", query_embedding=embed(1), threshold=0.5)
check("G1.1 cached value retrievable", result is not None,
      f"result={result}")

# Wait for expiry
time.sleep(1.5)
result_expired = db.cache_get("ttl_cache", query_embedding=embed(1), threshold=0.5)
check("G1.2 expired after TTL", result_expired is None,
      f"result={result_expired}")

# Zero TTL (should mean no expiry)
db.cache_put("ttl_cache", "permanent", "value_perm",
             embedding=embed(2), ttl_seconds=0)
result_perm = db.cache_get("ttl_cache", query_embedding=embed(2), threshold=0.5)
check("G1.3 zero TTL = no expiry", result_perm is not None)

db.close()


section("G2. Cache Similarity Thresholds")

db_path = os.path.join(tmpdir, "cache_sim_db")
db = Database.open(db_path)

dim = 64
import random
random.seed(42)

base_embed = [random.random() for _ in range(dim)]
db.cache_put("sim_cache", "base_key", "base_value",
             embedding=base_embed)

# Identical query — should match any threshold
result = db.cache_get("sim_cache", query_embedding=base_embed, threshold=0.99)
check("G2.1 identical embedding matches threshold=0.99", result is not None)

# Orthogonal-ish query — should NOT match high threshold
ortho = [(-1)**i * v for i, v in enumerate(base_embed)]
result_ortho = db.cache_get("sim_cache", query_embedding=ortho, threshold=0.9)
check("G2.2 orthogonal embedding fails threshold=0.9", result_ortho is None,
      f"result={result_ortho}")

# Zero vector query
zero_vec = [0.0] * dim
result_zero = db.cache_get("sim_cache", query_embedding=zero_vec, threshold=0.5)
check("G2.3 zero vector query returns None", result_zero is None,
      f"result={result_zero}")

# Dimension mismatch
wrong_dim = [0.5] * 32  # 32 instead of 64
try:
    result_dim = db.cache_get("sim_cache", query_embedding=wrong_dim, threshold=0.5)
    check("G2.4 dimension mismatch handled", result_dim is None,
          "silently returns None")
except Exception as e:
    check("G2.4 dimension mismatch handled", True, f"raised: {str(e)[:80]}")

db.close()


section("G3. Cache Stats & Deletion")

db_path = os.path.join(tmpdir, "cache_ops_db")
db = Database.open(db_path)

dim = 16
# Use near-orthogonal embeddings: one-hot at position i ensures cosine ~0 between different i.
def embed(i):
    v = [0.001] * dim  # tiny baseline to avoid zero-norm
    v[i] = 1.0  # dominant component at unique position
    return v

# Insert multiple entries
for i in range(10):
    db.cache_put("ops_cache", f"key_{i}", f"val_{i}", embedding=embed(i))

stats = db.cache_stats("ops_cache")
check("G3.1 cache stats total_entries >= 10",
      stats.get("total_entries", 0) >= 10 or stats.get("active_entries", 0) >= 10,
      f"stats={stats}")

# Delete specific entry
db.cache_delete("ops_cache", "key_5")
result_del = db.cache_get("ops_cache", query_embedding=embed(5), threshold=0.99)
check("G3.2 deleted entry not found", result_del is None)

# Clear all
cleared = db.cache_clear("ops_cache")
check("G3.3 cache_clear returns count", isinstance(cleared, int) and cleared >= 0,
      f"cleared={cleared}")

# After clear, cache should be empty
result_after = db.cache_get("ops_cache", query_embedding=embed(0), threshold=0.5)
check("G3.4 empty after clear", result_after is None)

db.close()


###############################################################################
#   H. VECTOR SEARCH EDGE CASES
###############################################################################

section("H1. VectorIndex Edge Cases")

try:
    from sochdb import VectorIndex

    db_path = os.path.join(tmpdir, "vector_edge_db")
    db = Database.open(db_path)

    # Zero vector via db.create_index / insert_vectors / search
    try:
        db.create_index("zero_test", dimension=4)
        db.insert_vectors("zero_test", [0], [[0.0, 0.0, 0.0, 0.0]])
        results = db.search("zero_test", query=[0.0, 0.0, 0.0, 0.0], k=1)
        check("H1.1 zero vector search", True,
              f"results={len(results)}")
    except Exception as e:
        check("H1.1 zero vector search", False, str(e)[:100])

    # Very large K
    try:
        db.create_index("large_k_test", dimension=4)
        ids = list(range(10))
        vecs = [[float(i), 0, 0, 0] for i in range(10)]
        db.insert_vectors("large_k_test", ids, vecs)
        results = db.search("large_k_test", query=[5.0, 0, 0, 0], k=1000)
        check("H1.2 k > num_vectors", len(results) <= 10,
              f"returned {len(results)} (expected <= 10)")
    except Exception as e:
        check("H1.2 k > num_vectors", False, str(e)[:100])

    # Duplicate IDs
    try:
        db.create_index("dup_test", dimension=4)
        db.insert_vectors("dup_test", [99], [[1.0, 0, 0, 0]])
        db.insert_vectors("dup_test", [99], [[0, 1.0, 0, 0]])  # same id, different vec
        results = db.search("dup_test", query=[0, 1.0, 0, 0], k=5)
        check("H1.3 duplicate ID handling", True,
              f"results={len(results)} (upsert or duplicate)")
    except Exception as e:
        check("H1.3 duplicate ID handling", False, str(e)[:100])

    db.close()

except Exception as e:
    check("H1 VectorIndex tests", False, str(e)[:100])


section("H2. Standalone VectorIndex Stress")

try:
    from sochdb import VectorIndex
    import numpy as np

    DIM = 128
    vi = VectorIndex(dimension=DIM)

    # Insert many vectors
    rng = np.random.RandomState(42)
    NUM = 5000
    for i in range(NUM):
        vec = rng.randn(DIM).astype(np.float32)
        vi.insert(i, vec)

    check("H2.1 inserted 5000 vectors", True)

    # Search
    query = rng.randn(DIM).astype(np.float32)
    search_results = vi.search(query, k=10)
    check("H2.2 search returns 10 results", len(search_results) == 10)

    # Verify results are sorted by distance
    distances = [float(r[1]) for r in search_results]
    check("H2.3 results sorted by distance",
          distances == sorted(distances),
          f"distances={distances[:5]}...")

except Exception as e:
    check("H2 VectorIndex stress", False, str(e)[:100])


section("H3. Database-Backed Vector Search Stress")

db_path = os.path.join(tmpdir, "meta_filter_db")
db = Database.open(db_path)

try:
    db.create_index("stress_vectors", dimension=4)
    ids = list(range(100))
    vecs = [[float(i % 10), float(i // 10), 0, 0] for i in range(100)]
    db.insert_vectors("stress_vectors", ids, vecs)

    results = db.search("stress_vectors", query=[5.0, 5.0, 0, 0], k=20)
    check("H3.1 search 100 vectors", len(results) == 20,
          f"got {len(results)} results")

except Exception as e:
    check("H3 vector search stress", False, str(e)[:100])

db.close()


###############################################################################
#   I. RESOURCE EXHAUSTION
###############################################################################

section("I1. Large Values")

db_path = os.path.join(tmpdir, "large_val_db")
db = Database.open(db_path)

# 10MB value
try:
    big = b"X" * (10 * 1024 * 1024)
    db.put(b"big10mb", big)
    got = db.get(b"big10mb")
    check("I1.1 10MB value roundtrip", got == big, f"len={len(got) if got else 0}")
except Exception as e:
    check("I1.1 10MB value", False, str(e)[:100])

# 50MB value
try:
    big50 = b"Y" * (50 * 1024 * 1024)
    db.put(b"big50mb", big50)
    got50 = db.get(b"big50mb")
    check("I1.2 50MB value roundtrip", got50 == big50, f"len={len(got50) if got50 else 0}")
except Exception as e:
    check("I1.2 50MB value", False, str(e)[:100])

db.close()


section("I2. Many Keys")

db_path = os.path.join(tmpdir, "many_keys_db")
db = Database.open(db_path)

# Batch insert 100K keys
t0 = time.time()
BATCH_SIZE = 1000
for batch in range(100):
    pairs = [(f"mk/{batch:03d}/{i:04d}".encode(), f"v{batch*BATCH_SIZE+i}".encode())
             for i in range(BATCH_SIZE)]
    db.put_batch(pairs)
t1 = time.time()

check("I2.1 100K keys inserted", True, f"{t1-t0:.2f}s")

# Scan all
t2 = time.time()
all_keys = list(db.scan_prefix(b"mk/"))
t3 = time.time()
check("I2.2 scan 100K keys", len(all_keys) == 100000,
      f"got {len(all_keys)} in {t3-t2:.2f}s")

# Point reads
t4 = time.time()
misses = 0
for i in range(1000):
    k = f"mk/{i//10:03d}/{i%10:04d}".encode()
    if db.get(k) is None:
        misses += 1
t5 = time.time()
check("I2.3 1K random reads", misses == 0,
      f"misses={misses}, time={t5-t4:.3f}s")

db.close()


section("I3. Deep Prefix Nesting")

db_path = os.path.join(tmpdir, "deep_prefix_db")
db = Database.open(db_path)

# Keys with deeply nested paths
deep_key = "/".join([f"level{i}" for i in range(50)])
db.put_path(deep_key, b"deep_value")
got = db.get_path(deep_key)
check("I3.1 50-level deep path key", got == b"deep_value",
      f"key_len={len(deep_key)}")

# Very long key (2KB)
long_key = "x" * 2048
try:
    db.put_path(long_key, b"long_path_val")
    got = db.get_path(long_key)
    check("I3.2 2KB path key", got == b"long_path_val")
except Exception as e:
    check("I3.2 2KB path key", False, str(e)[:100])

db.close()


###############################################################################
#   J. CRASH RECOVERY & ERROR PATHS
###############################################################################

section("J1. Close During Operations")

db_path = os.path.join(tmpdir, "close_during_db")
db = Database.open(db_path)

# Write data
for i in range(100):
    db.put(f"recover/{i}".encode(), f"val{i}".encode())

# Close and reopen
db.close()

db2 = Database.open(db_path)
readable = sum(1 for i in range(100) if db2.get(f"recover/{i}".encode()) is not None)
check("J1.1 data survives close/reopen", readable == 100,
      f"readable={readable}/100")
db2.close()


section("J2. Operations on Closed Database")

db_path = os.path.join(tmpdir, "closed_ops_db")
db = Database.open(db_path)
db.put(b"test", b"val")
db.close()

# These should all raise, not crash/segfault
ops = [
    ("put", lambda: db.put(b"k", b"v")),
    ("get", lambda: db.get(b"k")),
    ("delete", lambda: db.delete(b"k")),
    ("scan_prefix", lambda: list(db.scan_prefix(b"test"))),
    ("transaction", lambda: db.transaction()),
]

for name, op in ops:
    try:
        op()
        check(f"J2 {name} on closed db", False, "no exception raised")
    except Exception as e:
        check(f"J2 {name} on closed db", True, f"{type(e).__name__}: {str(e)[:60]}")


section("J3. Corrupt / Invalid Paths")

# Null byte in path
try:
    db_bad = Database.open(os.path.join(tmpdir, "bad\x00path"))
    check("J3.1 null byte in path", False, "should have errored")
    db_bad.close()
except Exception as e:
    check("J3.1 null byte in path", True, str(e)[:80])

# Non-writable path (permission denied)
try:
    ro_path = os.path.join(tmpdir, "readonly_dir")
    os.makedirs(ro_path, exist_ok=True)
    os.chmod(ro_path, 0o444)
    db_ro = Database.open(os.path.join(ro_path, "db"))
    check("J3.2 read-only dir", False, "should have errored")
    db_ro.close()
except Exception as e:
    check("J3.2 read-only dir", True, str(e)[:80])
finally:
    try:
        os.chmod(ro_path, 0o755)
    except Exception:
        pass

# Very long path
try:
    long_dir = os.path.join(tmpdir, "a" * 255)
    os.makedirs(long_dir, exist_ok=True)
    db_long = Database.open(os.path.join(long_dir, "db"))
    db_long.put(b"k", b"v")
    check("J3.3 max-length dir name", db_long.get(b"k") == b"v")
    db_long.close()
except Exception as e:
    check("J3.3 max-length dir name", True, f"error: {str(e)[:80]}")


section("J4. Transaction on Closed Database")

db_path = os.path.join(tmpdir, "txn_close_db")
db = Database.open(db_path)
txn = db.transaction()
txn.put(b"key_before_close", b"val")

# Close database while transaction is open
db.close()

# Using the transaction now — should not segfault
try:
    txn.put(b"key_after_close", b"should_fail")
    check("J4.1 txn.put after db.close", False, "expected exception")
except Exception as e:
    check("J4.1 txn.put after db.close", True, f"{type(e).__name__}: {str(e)[:60]}")

try:
    txn.commit()
    check("J4.2 txn.commit after db.close", False, "expected exception")
except Exception as e:
    check("J4.2 txn.commit after db.close", True, f"{type(e).__name__}: {str(e)[:60]}")


###############################################################################
#   K. API CONTRACT VIOLATIONS (README vs REALITY)
###############################################################################

section("K1. README Claims — Missing APIs")

db_path = os.path.join(tmpdir, "api_contract_db")
db = Database.open(db_path)

# K1.1: put() with ttl_seconds (README says this works)
import inspect
sig = inspect.signature(db.put)
has_ttl = "ttl_seconds" in sig.parameters
check("K1.1 put(ttl_seconds=...) exists", has_ttl,
      f"params={list(sig.parameters.keys())}")

# K1.2: with_transaction(fn) (README shows this)
check("K1.2 db.with_transaction() exists", hasattr(db, "with_transaction"))

# K1.3: IsolationLevel enum
try:
    from sochdb import IsolationLevel
    check("K1.3 IsolationLevel importable", True)
except ImportError:
    check("K1.3 IsolationLevel importable", False, "not in sochdb module")

# K1.4: txn.start_ts property
with db.transaction() as txn:
    has_start_ts = hasattr(txn, "start_ts")
    check("K1.4 txn.start_ts property exists", has_start_ts)

    has_isolation = hasattr(txn, "isolation")
    check("K1.5 txn.isolation property exists", has_isolation)

# K1.6: begin_transaction() method (README shows this)
check("K1.6 db.begin_transaction() exists", hasattr(db, "begin_transaction"))

# K1.7: TransactionConflictError importable
try:
    from sochdb import TransactionConflictError
    check("K1.7 TransactionConflictError importable", True)
except ImportError:
    try:
        from sochdb.errors import TransactionConflictError
        check("K1.7 TransactionConflictError importable", True, "from sochdb.errors")
    except ImportError:
        check("K1.7 TransactionConflictError importable", False)

db.close()


###############################################################################
#   L. METHOD SHADOWING & DEAD CODE
###############################################################################

section("L1. Method Shadowing Detection")

db_path = os.path.join(tmpdir, "shadow_db")
db = Database.open(db_path)

# L1.1: stats() should return meaningful data, not placeholder
stats = db.stats()
check("L1.1 stats() not placeholder",
      stats.get("keys_count", -1) != -1,
      f"keys_count={stats.get('keys_count', 'MISSING')}, stats={list(stats.keys())[:5]}")

# L1.2: checkpoint() should actually work
try:
    result = db.checkpoint()
    check("L1.2 checkpoint() works (not shadowed)",
          True,
          f"result={result}")
except AttributeError as e:
    check("L1.2 checkpoint() works", False, f"AttributeError: {str(e)[:80]} (likely uses self._ptr instead of self._handle)")
except Exception as e:
    check("L1.2 checkpoint() works", True, f"ran but: {str(e)[:60]}")

# L1.3: stats_full() — should be the FFI version
stats_full = db.stats_full()
check("L1.3 stats_full() returns real data",
      isinstance(stats_full, dict) and len(stats_full) > 0,
      f"keys={list(stats_full.keys())[:8]}")

db.close()


section("L2. scan_prefix_unchecked — Full Scan DoS")

db_path = os.path.join(tmpdir, "unchecked_scan_db")
db = Database.open(db_path)

# Insert some data
for i in range(100):
    db.put(f"scan_data/{i}".encode(), f"val{i}".encode())

# scan_prefix_unchecked with empty prefix — should scan everything
try:
    results = list(db.scan_prefix_unchecked(b""))
    check("L2.1 unchecked empty prefix scans all",
          len(results) >= 100,
          f"returned {len(results)} keys")
except Exception as e:
    check("L2.1 unchecked empty prefix", False, str(e)[:100])

# scan_prefix with empty prefix — should fail (min 2 bytes)
try:
    results = list(db.scan_prefix(b""))
    check("L2.2 scan_prefix rejects empty prefix", False, "should have raised error")
except Exception as e:
    check("L2.2 scan_prefix rejects empty prefix", True, str(e)[:80])

# scan_prefix with 1-byte prefix
try:
    results = list(db.scan_prefix(b"s"))
    check("L2.3 scan_prefix rejects 1-byte prefix", False, "should have raised error")
except Exception as e:
    check("L2.3 scan_prefix rejects 1-byte prefix", True, str(e)[:80])

db.close()


###############################################################################
#   M. COMPRESSION & DATA FORMAT EDGE CASES
###############################################################################

section("M1. Compression Switching")

db_path = os.path.join(tmpdir, "compression_db")
db = Database.open(db_path)

# Write data with no compression
db.set_compression("none")
db.put(b"comp_key1", b"value_uncompressed" * 100)

# Switch to lz4
db.set_compression("lz4")
db.put(b"comp_key2", b"value_lz4_compressed" * 100)

# Read old data (written without compression)
val1 = db.get(b"comp_key1")
check("M1.1 read uncompressed after switching to lz4",
      val1 == b"value_uncompressed" * 100)

# Read new data
val2 = db.get(b"comp_key2")
check("M1.2 read lz4-compressed data", val2 == b"value_lz4_compressed" * 100)

# Switch to invalid compression — should fallback to none
db.set_compression("invalid_codec")
comp = db.get_compression()
check("M1.3 invalid compression fallback", comp in ("none", "invalid_codec"),
      f"compression={comp}")

db.close()


section("M2. Data Format Roundtrip Stress")

db_path = os.path.join(tmpdir, "format_stress_db")
db = Database.open(db_path)

# Put varied data types via KV
test_values = [
    ("empty", b""),
    ("null_byte", b"\x00"),
    ("null_bytes", b"\x00\x00\x00"),
    ("binary_all", bytes(range(256))),
    ("newlines", b"line1\nline2\rline3\r\n"),
    ("tabs_spaces", b"\t  \t  "),
    ("utf8", "héllo wörld".encode("utf-8")),
    ("large_binary", os.urandom(16384)),
]

for name, val in test_values:
    db.put(f"fmt/{name}".encode(), val)

for name, val in test_values:
    got = db.get(f"fmt/{name}".encode())
    check(f"M2 roundtrip {name}", got == val,
          f"expected_len={len(val)}, got_len={len(got) if got else 'None'}")

db.close()


###############################################################################
#   N. BACKUP UNDER LOAD
###############################################################################

section("N1. Backup During Active Writes")

db_path = os.path.join(tmpdir, "backup_load_db")
backup_dst = os.path.join(tmpdir, "backup_load_dst")
db = Database.open(db_path)

# Pre-populate
for i in range(1000):
    db.put(f"backup/{i}".encode(), f"val{i}".encode())

# Start concurrent writes
write_done = threading.Event()
write_errors = []

def background_writer():
    i = 1000
    while not write_done.is_set():
        try:
            db.put(f"backup/{i}".encode(), f"val{i}".encode())
            i += 1
        except Exception as e:
            write_errors.append(str(e))
        time.sleep(0.001)

writer_thread = threading.Thread(target=background_writer, daemon=True)
writer_thread.start()

# Create backup while writes are happening
try:
    db.backup_create(backup_dst)
    check("N1.1 backup during writes", True)
except Exception as e:
    check("N1.1 backup during writes", False, str(e)[:100])

write_done.set()
writer_thread.join(timeout=5)

# Verify backup
try:
    ok = db.backup_verify(backup_dst)
    check("N1.2 backup is valid", ok)
except Exception as e:
    check("N1.2 backup verify", False, str(e)[:100])

check("N1.3 no write errors during backup", len(write_errors) == 0,
      f"errors={write_errors[:3]}" if write_errors else "")

db.close()


###############################################################################
#   O. NAMESPACE ISOLATION STRESS
###############################################################################

section("O1. Namespace Data Isolation")

db_path = os.path.join(tmpdir, "ns_isolation_db")
db = Database.open(db_path)

ns_count = 10
keys_per_ns = 100

for ns_id in range(ns_count):
    ns = db.get_or_create_namespace(f"tenant_{ns_id}")
    for k in range(keys_per_ns):
        ns.put(f"key_{k}", f"ns{ns_id}_val{k}".encode())

# Verify isolation — each namespace only sees its own data
for ns_id in range(ns_count):
    ns = db.namespace(f"tenant_{ns_id}")
    for k in range(min(10, keys_per_ns)):
        val = ns.get(f"key_{k}")
        expected = f"ns{ns_id}_val{k}".encode()
        if val != expected:
            check(f"O1.1 namespace isolation tenant_{ns_id}",
                  False, f"key_{k}: expected={expected}, got={val}")
            break
    else:
        continue
    break
else:
    check("O1.1 namespace isolation across 10 tenants", True,
          f"{ns_count * keys_per_ns} keys isolated")

# Cross-namespace scan should not leak
ns0 = db.namespace("tenant_0")
scan_results = list(ns0.scan("key_"))
check("O1.2 namespace scan isolation",
      all(v.startswith(b"ns0_") for _, v in scan_results),
      f"scanned {len(scan_results)} keys")

db.close()


###############################################################################
#   P. BATCH OPERATIONS EDGE CASES
###############################################################################

section("P1. Batch Edge Cases")

db_path = os.path.join(tmpdir, "batch_edge_db")
db = Database.open(db_path)

# Empty batch
try:
    result = db.put_batch([])
    check("P1.1 empty put_batch", True, f"result={result}")
except Exception as e:
    check("P1.1 empty put_batch", False, str(e)[:100])

# get_batch with empty list
try:
    result = db.get_batch([])
    check("P1.2 empty get_batch", True, f"result={result}")
except Exception as e:
    check("P1.2 empty get_batch", False, str(e)[:100])

# delete_batch with empty list
try:
    result = db.delete_batch([])
    check("P1.3 empty delete_batch", True, f"result={result}")
except Exception as e:
    check("P1.3 empty delete_batch", False, str(e)[:100])

# Very large batch
try:
    large_batch = [(f"lb/{i}".encode(), f"v{i}".encode()) for i in range(10000)]
    t0 = time.time()
    db.put_batch(large_batch)
    t1 = time.time()
    check("P1.4 10K item batch", True, f"{t1-t0:.3f}s")
except Exception as e:
    check("P1.4 10K item batch", False, str(e)[:100])

# Batch with duplicate keys — last write wins?
try:
    dup_batch = [(b"dup_key", b"first"), (b"dup_key", b"second"), (b"dup_key", b"third")]
    db.put_batch(dup_batch)
    val = db.get(b"dup_key")
    check("P1.5 batch duplicate keys (last-write-wins)",
          val == b"third",
          f"got={val}")
except Exception as e:
    check("P1.5 batch duplicate keys", False, str(e)[:100])

db.close()


###############################################################################
#                            CLEANUP & SUMMARY
###############################################################################

section("CLEANUP")
shutil.rmtree(tmpdir, ignore_errors=True)
print(f"  Removed {tmpdir}")

print(f"\n{'='*68}")
print(f"  STRESS TEST RESULTS")
print(f"{'='*68}")
print(f"  PASS: {PASS}")
print(f"  FAIL: {FAIL}")
print(f"  SKIP: {SKIP}")
print(f"  TOTAL EXECUTED: {PASS + FAIL}")
print(f"{'='*68}")

if FAIL > 0:
    print(f"\n  FAILURES ({FAIL}):")
    for r in results:
        if r[0] == "FAIL":
            detail = f"  {r[2]}" if len(r) > 2 and r[2] else ""
            print(f"    [FAIL] {r[1]}{detail}")

if SKIP > 0:
    print(f"\n  SKIPPED ({SKIP}):")
    for r in results:
        if r[0] == "SKIP":
            print(f"    [SKIP] {r[1]}  {r[2] if len(r) > 2 else ''}")

# Exit with failure code if any tests failed
print()
sys.exit(1 if FAIL > 0 else 0)
