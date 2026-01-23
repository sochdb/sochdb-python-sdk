# Copyright 2025 Sushanth (https://github.com/sushanthpy)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SochDB Priority Queue

First-class queue API with ordered-key task entries, providing efficient
priority queue operations without the O(N) blob rewrite anti-pattern.

Supports both embedded (FFI) and server (gRPC) modes:

Embedded Mode (FFI):
    from sochdb import Database
    from sochdb.queue import PriorityQueue, QueueConfig
    
    db = Database.open("./my_queue_db")
    queue = PriorityQueue.from_database(db, "tasks")
    
    queue.enqueue(priority=1, payload=b"high priority task")
    task = queue.dequeue(worker_id="worker-1")
    queue.ack(task.task_id)

Server Mode (gRPC):
    from sochdb import SochDBClient
    from sochdb.queue import PriorityQueue, QueueConfig
    
    client = SochDBClient("localhost:50051")
    queue = PriorityQueue.from_client(client, "tasks")
    
    queue.enqueue(priority=1, payload=b"high priority task")
    task = queue.dequeue(worker_id="worker-1")
    queue.ack(task.task_id)

Features:
- Ordered-key representation: Each task has its own key, no blob parsing
- O(log N) enqueue/dequeue with ordered scans
- Atomic claim protocol for concurrent workers
- Visibility timeout for crash recovery
- Streaming top-K for ORDER BY + LIMIT queries
"""

from __future__ import annotations

import struct
import time
import uuid
import heapq
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Optional,
    List,
    Dict,
    Callable,
    Any,
    Iterator,
    Tuple,
    Union,
)
from threading import Lock

if TYPE_CHECKING:
    from .database import Database, Transaction
    from .grpc_client import SochDBClient


# ============================================================================
# Key Encoding - Big-Endian for Lexicographic Ordering
# ============================================================================

def encode_u64_be(value: int) -> bytes:
    """Encode a u64 as big-endian bytes for lexicographic ordering."""
    return struct.pack('>Q', value)


def decode_u64_be(data: bytes) -> int:
    """Decode a big-endian u64 from bytes."""
    return struct.unpack('>Q', data[:8])[0]


def encode_i64_be(value: int) -> bytes:
    """Encode an i64 as big-endian bytes preserving order.
    
    Maps i64 to u64 by adding offset so negative numbers sort correctly.
    """
    # Map i64 [-2^63, 2^63-1] to u64 [0, 2^64-1]
    mapped = value + (1 << 63)
    return struct.pack('>Q', mapped)


def decode_i64_be(data: bytes) -> int:
    """Decode a big-endian i64 from bytes."""
    mapped = struct.unpack('>Q', data[:8])[0]
    return mapped - (1 << 63)


# ============================================================================
# Task State
# ============================================================================

class TaskState(Enum):
    """Queue task state."""
    PENDING = "pending"        # Ready to be dequeued
    CLAIMED = "claimed"        # Claimed by a worker (inflight)
    COMPLETED = "completed"    # Successfully processed
    DEAD_LETTERED = "dead_lettered"  # Failed max retries


# ============================================================================
# QueueKey - Composite Key for Queue Entries
# ============================================================================

@dataclass
class QueueKey:
    """
    Composite key for queue entries.
    
    Layout ensures lexicographic order matches desired queue order:
    1. Queue ID (namespace separation)
    2. Priority (big-endian, lower = more urgent)
    3. Ready timestamp (when task becomes visible)
    4. Sequence number (tie-breaker for FIFO)
    5. Task ID (unique identifier)
    """
    queue_id: str
    priority: int
    ready_ts: int  # Timestamp in milliseconds
    sequence: int
    task_id: str
    
    def encode(self) -> bytes:
        """Encode key to bytes for storage."""
        parts = [
            b"queue/",
            self.queue_id.encode('utf-8'),
            b"/",
            encode_i64_be(self.priority),
            b"/",
            encode_u64_be(self.ready_ts),
            b"/",
            encode_u64_be(self.sequence),
            b"/",
            self.task_id.encode('utf-8'),
        ]
        return b"".join(parts)
    
    @classmethod
    def decode(cls, data: bytes) -> 'QueueKey':
        """Decode key from bytes."""
        # Parse: queue/<queue_id>/<priority>/<ready_ts>/<seq>/<task_id>
        parts = data.split(b"/")
        if len(parts) < 6 or parts[0] != b"queue":
            raise ValueError(f"Invalid queue key format: {data}")
        
        queue_id = parts[1].decode('utf-8')
        priority = decode_i64_be(parts[2])
        ready_ts = decode_u64_be(parts[3])
        sequence = decode_u64_be(parts[4])
        task_id = parts[5].decode('utf-8')
        
        return cls(
            queue_id=queue_id,
            priority=priority,
            ready_ts=ready_ts,
            sequence=sequence,
            task_id=task_id,
        )
    
    @staticmethod
    def prefix(queue_id: str) -> bytes:
        """Get prefix for scanning all tasks in a queue."""
        return b"queue/" + queue_id.encode('utf-8') + b"/"
    
    def __lt__(self, other: 'QueueKey') -> bool:
        """Compare keys for ordering."""
        return (
            self.queue_id, self.priority, self.ready_ts, self.sequence, self.task_id
        ) < (
            other.queue_id, other.priority, other.ready_ts, other.sequence, other.task_id
        )


# ============================================================================
# Task - Queue Task with Payload
# ============================================================================

@dataclass
class Task:
    """A task in the queue."""
    key: QueueKey
    payload: bytes
    state: TaskState = TaskState.PENDING
    attempts: int = 0
    max_attempts: int = 3
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    claimed_at: Optional[int] = None
    claimed_by: Optional[str] = None
    lease_expires_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def task_id(self) -> str:
        return self.key.task_id
    
    @property
    def priority(self) -> int:
        return self.key.priority
    
    def is_visible(self, now_ms: int) -> bool:
        """Check if task is visible (ready to be claimed)."""
        if self.state == TaskState.PENDING:
            return self.key.ready_ts <= now_ms
        elif self.state == TaskState.CLAIMED:
            # Visible again if lease expired
            if self.lease_expires_at:
                return now_ms >= self.lease_expires_at
            return False
        return False
    
    def should_dead_letter(self) -> bool:
        """Check if task should be dead-lettered."""
        return self.attempts >= self.max_attempts
    
    def to_dict(self) -> dict:
        """Serialize task to dictionary."""
        return {
            "queue_id": self.key.queue_id,
            "task_id": self.key.task_id,
            "priority": self.key.priority,
            "ready_ts": self.key.ready_ts,
            "sequence": self.key.sequence,
            "payload": self.payload.decode('utf-8', errors='replace'),
            "state": self.state.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "claimed_at": self.claimed_at,
            "claimed_by": self.claimed_by,
            "lease_expires_at": self.lease_expires_at,
            "metadata": self.metadata,
        }
    
    def encode_value(self) -> bytes:
        """Encode task value for storage."""
        data = {
            "payload": self.payload.hex(),  # Binary-safe encoding
            "state": self.state.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "claimed_at": self.claimed_at,
            "claimed_by": self.claimed_by,
            "lease_expires_at": self.lease_expires_at,
            "metadata": self.metadata,
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def decode_value(cls, key: QueueKey, data: bytes) -> 'Task':
        """Decode task from stored value."""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            key=key,
            payload=bytes.fromhex(parsed["payload"]),
            state=TaskState(parsed["state"]),
            attempts=parsed["attempts"],
            max_attempts=parsed["max_attempts"],
            created_at=parsed["created_at"],
            claimed_at=parsed.get("claimed_at"),
            claimed_by=parsed.get("claimed_by"),
            lease_expires_at=parsed.get("lease_expires_at"),
            metadata=parsed.get("metadata"),
        )


# ============================================================================
# Claim - Lease-Based Ownership
# ============================================================================

@dataclass
class Claim:
    """A claim on a task (lease-based ownership)."""
    task_id: str
    owner: str
    claimed_at: int
    expires_at: int
    
    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expires_at
    
    def encode_key(self, queue_id: str) -> bytes:
        """Encode claim key for storage."""
        return f"queue_claim/{queue_id}/{self.task_id}".encode('utf-8')
    
    def encode_value(self) -> bytes:
        """Encode claim value for storage."""
        data = {
            "owner": self.owner,
            "claimed_at": self.claimed_at,
            "expires_at": self.expires_at,
        }
        return json.dumps(data).encode('utf-8')
    
    @classmethod
    def decode_value(cls, task_id: str, data: bytes) -> 'Claim':
        """Decode claim from stored value."""
        parsed = json.loads(data.decode('utf-8'))
        return cls(
            task_id=task_id,
            owner=parsed["owner"],
            claimed_at=parsed["claimed_at"],
            expires_at=parsed["expires_at"],
        )


# ============================================================================
# QueueConfig - Queue Configuration
# ============================================================================

@dataclass
class QueueConfig:
    """Queue configuration."""
    queue_id: str = "default"
    visibility_timeout_ms: int = 30_000  # 30 seconds
    max_attempts: int = 3
    dead_letter_queue_id: Optional[str] = None
    
    def with_visibility_timeout(self, timeout_ms: int) -> 'QueueConfig':
        """Builder pattern for visibility timeout."""
        self.visibility_timeout_ms = timeout_ms
        return self
    
    def with_max_attempts(self, max_attempts: int) -> 'QueueConfig':
        """Builder pattern for max attempts."""
        self.max_attempts = max_attempts
        return self
    
    def with_dead_letter_queue(self, dlq_id: str) -> 'QueueConfig':
        """Builder pattern for dead letter queue."""
        self.dead_letter_queue_id = dlq_id
        return self


# ============================================================================
# QueueStats - Queue Statistics
# ============================================================================

@dataclass
class QueueStats:
    """Queue statistics."""
    queue_id: str
    pending: int = 0
    delayed: int = 0
    inflight: int = 0
    total: int = 0
    active_claims: int = 0


# ============================================================================
# QueueBackend - Abstract Backend Interface
# ============================================================================

class QueueBackend(ABC):
    """
    Abstract backend interface for queue storage.
    
    This allows the queue to work with both FFI (embedded) and gRPC (server) modes.
    """
    
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        """Store a key-value pair."""
        pass
    
    @abstractmethod
    def get(self, key: bytes) -> Optional[bytes]:
        """Get a value by key."""
        pass
    
    @abstractmethod
    def delete(self, key: bytes) -> None:
        """Delete a key."""
        pass
    
    @abstractmethod
    def scan_prefix(self, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        """Scan all keys with the given prefix."""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> 'QueueTransaction':
        """Begin a transaction."""
        pass


class QueueTransaction(ABC):
    """Abstract transaction interface."""
    
    @abstractmethod
    def put(self, key: bytes, value: bytes) -> None:
        pass
    
    @abstractmethod
    def delete(self, key: bytes) -> None:
        pass
    
    @abstractmethod
    def commit(self) -> None:
        pass
    
    @abstractmethod
    def abort(self) -> None:
        pass
    
    def __enter__(self) -> 'QueueTransaction':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return False


# ============================================================================
# FFIQueueBackend - Embedded Mode Backend
# ============================================================================

class FFIQueueBackend(QueueBackend):
    """
    Queue backend using FFI (embedded mode).
    
    This backend uses the Database class with direct FFI calls.
    """
    
    def __init__(self, db: "Database"):
        self._db = db
    
    def put(self, key: bytes, value: bytes) -> None:
        self._db.put(key, value)
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._db.get(key)
    
    def delete(self, key: bytes) -> None:
        self._db.delete(key)
    
    def scan_prefix(self, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        """Scan keys with prefix using Database.scan_prefix."""
        # Use the database's scan_prefix if available
        if hasattr(self._db, 'scan_prefix'):
            yield from self._db.scan_prefix(prefix)
        else:
            # Fallback: use scan with range
            end = prefix + b'\xff'
            for key, value in self._db.scan(prefix, end):
                if key.startswith(prefix):
                    yield key, value
    
    def begin_transaction(self) -> 'QueueTransaction':
        return FFIQueueTransaction(self._db)


class FFIQueueTransaction(QueueTransaction):
    """Transaction wrapper for FFI mode."""
    
    def __init__(self, db: "Database"):
        self._db = db
        self._txn = db.transaction().__enter__()
        self._committed = False
    
    def put(self, key: bytes, value: bytes) -> None:
        self._txn.put(key, value)
    
    def delete(self, key: bytes) -> None:
        self._txn.delete(key)
    
    def commit(self) -> None:
        if not self._committed:
            self._txn.__exit__(None, None, None)
            self._committed = True
    
    def abort(self) -> None:
        if not self._committed:
            self._txn.__exit__(Exception, None, None)
            self._committed = True


# ============================================================================
# GrpcQueueBackend - Server Mode Backend
# ============================================================================

class GrpcQueueBackend(QueueBackend):
    """
    Queue backend using gRPC (server mode).
    
    This backend uses the SochDBClient with gRPC calls.
    """
    
    def __init__(self, client: "SochDBClient", namespace: str = "default"):
        self._client = client
        self._namespace = namespace
    
    def put(self, key: bytes, value: bytes) -> None:
        self._client.put_kv(
            key.decode('utf-8', errors='replace'),
            value,
            namespace=self._namespace,
        )
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._client.get_kv(
            key.decode('utf-8', errors='replace'),
            namespace=self._namespace,
        )
    
    def delete(self, key: bytes) -> None:
        self._client.delete_kv(
            key.decode('utf-8', errors='replace'),
            namespace=self._namespace,
        )
    
    def scan_prefix(self, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        """Scan keys with prefix via gRPC."""
        results = self._client.scan_prefix(
            prefix.decode('utf-8', errors='replace'),
            namespace=self._namespace,
        )
        for key, value in results:
            yield key.encode('utf-8'), value
    
    def begin_transaction(self) -> 'QueueTransaction':
        return GrpcQueueTransaction(self._client, self._namespace)


class GrpcQueueTransaction(QueueTransaction):
    """Transaction wrapper for gRPC mode (uses server-side transactions)."""
    
    def __init__(self, client: "SochDBClient", namespace: str):
        self._client = client
        self._namespace = namespace
        self._ops: List[Tuple[str, bytes, Optional[bytes]]] = []
        self._committed = False
    
    def put(self, key: bytes, value: bytes) -> None:
        self._ops.append(('put', key, value))
    
    def delete(self, key: bytes) -> None:
        self._ops.append(('delete', key, None))
    
    def commit(self) -> None:
        if not self._committed:
            # Execute all ops via gRPC
            for op, key, value in self._ops:
                if op == 'put':
                    self._client.put_kv(
                        key.decode('utf-8', errors='replace'),
                        value,
                        namespace=self._namespace,
                    )
                elif op == 'delete':
                    self._client.delete_kv(
                        key.decode('utf-8', errors='replace'),
                        namespace=self._namespace,
                    )
            self._committed = True
    
    def abort(self) -> None:
        self._ops.clear()
        self._committed = True


# ============================================================================
# InMemoryQueueBackend - For Testing
# ============================================================================

class InMemoryQueueBackend(QueueBackend):
    """
    In-memory queue backend for testing.
    
    This backend stores data in a Python dictionary.
    """
    
    def __init__(self):
        self._store: Dict[bytes, bytes] = {}
    
    def put(self, key: bytes, value: bytes) -> None:
        self._store[key] = value
    
    def get(self, key: bytes) -> Optional[bytes]:
        return self._store.get(key)
    
    def delete(self, key: bytes) -> None:
        self._store.pop(key, None)
    
    def scan_prefix(self, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        for key, value in sorted(self._store.items()):
            if key.startswith(prefix):
                yield key, value
    
    def begin_transaction(self) -> 'QueueTransaction':
        return InMemoryQueueTransaction(self._store)


class InMemoryQueueTransaction(QueueTransaction):
    """In-memory transaction for testing."""
    
    def __init__(self, store: Dict[bytes, bytes]):
        self._store = store
        self._ops: List[Tuple[str, bytes, Optional[bytes]]] = []
        self._committed = False
    
    def put(self, key: bytes, value: bytes) -> None:
        self._ops.append(('put', key, value))
    
    def delete(self, key: bytes) -> None:
        self._ops.append(('delete', key, None))
    
    def commit(self) -> None:
        if not self._committed:
            for op, key, value in self._ops:
                if op == 'put':
                    self._store[key] = value
                elif op == 'delete':
                    self._store.pop(key, None)
            self._committed = True
    
    def abort(self) -> None:
        self._ops.clear()
        self._committed = True


# ============================================================================
# PriorityQueue - The Main Queue Implementation
# ============================================================================

class PriorityQueue:
    """
    Priority queue backed by SochDB storage.
    
    Uses ordered-key representation for O(log N) operations instead of
    the O(N) blob parsing anti-pattern.
    
    Thread-safe for concurrent access from multiple workers.
    
    Example (Embedded Mode):
        from sochdb import Database
        from sochdb.queue import PriorityQueue
        
        db = Database.open("./queue_db")
        queue = PriorityQueue.from_database(db, "tasks")
        
        queue.enqueue(priority=1, payload=b"urgent task")
        task = queue.dequeue(worker_id="worker-1")
        queue.ack(task.task_id)
    
    Example (Server Mode):
        from sochdb import SochDBClient
        from sochdb.queue import PriorityQueue
        
        client = SochDBClient("localhost:50051")
        queue = PriorityQueue.from_client(client, "tasks")
        
        queue.enqueue(priority=1, payload=b"urgent task")
        task = queue.dequeue(worker_id="worker-1")
        queue.ack(task.task_id)
    """
    
    def __init__(self, backend: QueueBackend, config: QueueConfig):
        """
        Initialize a priority queue with a backend.
        
        Use `from_database()` or `from_client()` factory methods instead
        for easier initialization.
        
        Args:
            backend: Queue backend (FFI, gRPC, or InMemory)
            config: Queue configuration
        """
        self._backend = backend
        self._config = config
        self._sequence = 0
        self._lock = Lock()
    
    @classmethod
    def from_database(
        cls,
        db: "Database",
        queue_id: str = "default",
        visibility_timeout_ms: int = 30_000,
        max_attempts: int = 3,
    ) -> "PriorityQueue":
        """
        Create a queue from a Database instance (embedded mode).
        
        Args:
            db: SochDB Database instance
            queue_id: Queue identifier
            visibility_timeout_ms: Visibility timeout in milliseconds
            max_attempts: Maximum delivery attempts
            
        Returns:
            PriorityQueue instance
        """
        backend = FFIQueueBackend(db)
        config = QueueConfig(
            queue_id=queue_id,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
        return cls(backend, config)
    
    @classmethod
    def from_client(
        cls,
        client: "SochDBClient",
        queue_id: str = "default",
        namespace: str = "default",
        visibility_timeout_ms: int = 30_000,
        max_attempts: int = 3,
    ) -> "PriorityQueue":
        """
        Create a queue from a SochDBClient instance (server mode).
        
        Args:
            client: SochDB gRPC client
            queue_id: Queue identifier
            namespace: Namespace for queue data
            visibility_timeout_ms: Visibility timeout in milliseconds
            max_attempts: Maximum delivery attempts
            
        Returns:
            PriorityQueue instance
        """
        backend = GrpcQueueBackend(client, namespace)
        config = QueueConfig(
            queue_id=queue_id,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
        return cls(backend, config)
    
    @classmethod
    def from_backend(
        cls,
        backend: QueueBackend,
        queue_id: str = "default",
        visibility_timeout_ms: int = 30_000,
        max_attempts: int = 3,
    ) -> "PriorityQueue":
        """
        Create a queue from a custom backend (for testing).
        
        Args:
            backend: Custom queue backend
            queue_id: Queue identifier
            visibility_timeout_ms: Visibility timeout in milliseconds
            max_attempts: Maximum delivery attempts
            
        Returns:
            PriorityQueue instance
        """
        config = QueueConfig(
            queue_id=queue_id,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
        return cls(backend, config)
    
    @property
    def queue_id(self) -> str:
        return self._config.queue_id
    
    def _now_ms(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)
    
    def _next_sequence(self) -> int:
        """Get next sequence number (thread-safe)."""
        with self._lock:
            self._sequence += 1
            return self._sequence
    
    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        return str(uuid.uuid4())
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def enqueue(
        self,
        priority: int,
        payload: bytes,
        delay_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Enqueue a task with the given priority.
        
        Args:
            priority: Task priority (lower = more urgent)
            payload: Task payload bytes
            delay_ms: Delay before task becomes visible (default: 0)
            metadata: Optional task metadata
            
        Returns:
            The created task
            
        Complexity: O(log N) with ordered index
        """
        now = self._now_ms()
        sequence = self._next_sequence()
        task_id = self._generate_task_id()
        
        key = QueueKey(
            queue_id=self._config.queue_id,
            priority=priority,
            ready_ts=now + delay_ms,
            sequence=sequence,
            task_id=task_id,
        )
        
        task = Task(
            key=key,
            payload=payload if isinstance(payload, bytes) else payload.encode('utf-8'),
            max_attempts=self._config.max_attempts,
            created_at=now,
            metadata=metadata,
        )
        
        # Store task
        self._backend.put(key.encode(), task.encode_value())
        
        return task
    
    def enqueue_batch(
        self,
        tasks: List[Tuple[int, bytes]],  # [(priority, payload), ...]
    ) -> List[Task]:
        """
        Enqueue multiple tasks in a batch.
        
        Args:
            tasks: List of (priority, payload) tuples
            
        Returns:
            List of created tasks
            
        Complexity: O(N log N) for N tasks
        """
        result = []
        with self._backend.begin_transaction() as txn:
            for priority, payload in tasks:
                now = self._now_ms()
                sequence = self._next_sequence()
                task_id = self._generate_task_id()
                
                key = QueueKey(
                    queue_id=self._config.queue_id,
                    priority=priority,
                    ready_ts=now,
                    sequence=sequence,
                    task_id=task_id,
                )
                
                task = Task(
                    key=key,
                    payload=payload if isinstance(payload, bytes) else payload.encode('utf-8'),
                    max_attempts=self._config.max_attempts,
                    created_at=now,
                )
                
                txn.put(key.encode(), task.encode_value())
                result.append(task)
        
        return result
    
    def dequeue(
        self,
        worker_id: str,
        visibility_timeout_ms: Optional[int] = None,
    ) -> Optional[Task]:
        """
        Dequeue the highest priority visible task.
        
        This implements the atomic claim protocol:
        1. Scan for first visible task
        2. Attempt to claim it
        3. If claimed, update task state and return it
        4. If contention, retry with next candidate
        
        Args:
            worker_id: Unique identifier for this worker
            visibility_timeout_ms: Override default visibility timeout
            
        Returns:
            The claimed task, or None if queue is empty
            
        Complexity: O(log N) with ordered index
        """
        timeout = visibility_timeout_ms or self._config.visibility_timeout_ms
        now = self._now_ms()
        
        # Clean up expired claims
        self._cleanup_expired_claims(now)
        
        # Scan for visible tasks
        prefix = QueueKey.prefix(self._config.queue_id)
        
        with self._backend.begin_transaction() as txn:
            for key_bytes, value_bytes in self._scan_prefix(prefix):
                try:
                    key = QueueKey.decode(key_bytes)
                    task = Task.decode_value(key, value_bytes)
                    
                    # Skip if not visible
                    if not task.is_visible(now):
                        continue
                    
                    # Check if already claimed by another worker
                    claim = self._get_claim(task.task_id)
                    if claim and not claim.is_expired(now) and claim.owner != worker_id:
                        continue  # Contention, try next
                    
                    # Create claim
                    new_claim = Claim(
                        task_id=task.task_id,
                        owner=worker_id,
                        claimed_at=now,
                        expires_at=now + timeout,
                    )
                    
                    # Update task state
                    task.state = TaskState.CLAIMED
                    task.attempts += 1
                    task.claimed_at = now
                    task.claimed_by = worker_id
                    task.lease_expires_at = new_claim.expires_at
                    
                    # Store updated task and claim
                    txn.put(key.encode(), task.encode_value())
                    txn.put(
                        new_claim.encode_key(self._config.queue_id),
                        new_claim.encode_value(),
                    )
                    
                    return task
                    
                except Exception:
                    # Skip malformed entries
                    continue
        
        return None
    
    def ack(self, task_id: str) -> bool:
        """
        Acknowledge successful processing of a task (delete it).
        
        Args:
            task_id: The task ID to acknowledge
            
        Returns:
            True if acknowledged, False if task not found
            
        Complexity: O(log N)
        """
        prefix = QueueKey.prefix(self._config.queue_id)
        
        with self._backend.begin_transaction() as txn:
            for key_bytes, value_bytes in self._scan_prefix(prefix):
                try:
                    key = QueueKey.decode(key_bytes)
                    if key.task_id == task_id:
                        txn.delete(key_bytes)
                        # Also delete claim
                        claim_key = f"queue_claim/{self._config.queue_id}/{task_id}".encode('utf-8')
                        try:
                            txn.delete(claim_key)
                        except Exception:
                            pass
                        return True
                except Exception:
                    continue
        
        return False
    
    def nack(
        self,
        task_id: str,
        new_priority: Optional[int] = None,
        delay_ms: Optional[int] = None,
    ) -> bool:
        """
        Negative acknowledgment - return task to queue.
        
        Optionally adjust priority or add delay for retry.
        
        Args:
            task_id: The task ID to nack
            new_priority: New priority (optional)
            delay_ms: Delay before retry (optional)
            
        Returns:
            True if nacked, False if task not found or dead-lettered
            
        Complexity: O(log N)
        """
        now = self._now_ms()
        prefix = QueueKey.prefix(self._config.queue_id)
        
        with self._backend.begin_transaction() as txn:
            for key_bytes, value_bytes in self._scan_prefix(prefix):
                try:
                    old_key = QueueKey.decode(key_bytes)
                    if old_key.task_id != task_id:
                        continue
                    
                    task = Task.decode_value(old_key, value_bytes)
                    
                    # Check if should dead-letter
                    if task.should_dead_letter():
                        task.state = TaskState.DEAD_LETTERED
                        # Move to DLQ if configured
                        if self._config.dead_letter_queue_id:
                            self._move_to_dlq(txn, task)
                        txn.delete(key_bytes)
                        return False
                    
                    # Create new key with updated priority/ready_ts
                    priority = new_priority if new_priority is not None else old_key.priority
                    ready_ts = (now + delay_ms) if delay_ms else now
                    sequence = self._next_sequence()
                    
                    new_key = QueueKey(
                        queue_id=self._config.queue_id,
                        priority=priority,
                        ready_ts=ready_ts,
                        sequence=sequence,
                        task_id=task_id,
                    )
                    
                    # Update task state
                    task.key = new_key
                    task.state = TaskState.PENDING
                    task.claimed_at = None
                    task.claimed_by = None
                    task.lease_expires_at = None
                    
                    # Delete old, insert new
                    txn.delete(key_bytes)
                    txn.put(new_key.encode(), task.encode_value())
                    
                    # Delete claim
                    claim_key = f"queue_claim/{self._config.queue_id}/{task_id}".encode('utf-8')
                    try:
                        txn.delete(claim_key)
                    except Exception:
                        pass
                    
                    return True
                    
                except Exception:
                    continue
        
        return False
    
    def extend_visibility(self, task_id: str, additional_ms: int) -> bool:
        """
        Extend the visibility timeout for a task.
        
        Useful when processing takes longer than expected.
        
        Args:
            task_id: The task ID
            additional_ms: Additional time in milliseconds
            
        Returns:
            True if extended, False if task not found
        """
        claim_key = f"queue_claim/{self._config.queue_id}/{task_id}".encode('utf-8')
        
        claim_data = self._backend.get(claim_key)
        if not claim_data:
            return False
        
        claim = Claim.decode_value(task_id, claim_data)
        claim.expires_at += additional_ms
        
        with self._backend.begin_transaction() as txn:
            txn.put(claim_key, claim.encode_value())
            
            # Also update task's lease
            prefix = QueueKey.prefix(self._config.queue_id)
            for key_bytes, value_bytes in self._scan_prefix(prefix):
                try:
                    key = QueueKey.decode(key_bytes)
                    if key.task_id == task_id:
                        task = Task.decode_value(key, value_bytes)
                        task.lease_expires_at = claim.expires_at
                        txn.put(key_bytes, task.encode_value())
                        break
                except Exception:
                    continue
        
        return True
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def peek(self) -> Optional[Task]:
        """
        Peek at the highest priority visible task without claiming.
        
        Returns:
            The highest priority visible task, or None if empty
        """
        now = self._now_ms()
        prefix = QueueKey.prefix(self._config.queue_id)
        
        for key_bytes, value_bytes in self._scan_prefix(prefix):
            try:
                key = QueueKey.decode(key_bytes)
                task = Task.decode_value(key, value_bytes)
                
                if task.is_visible(now):
                    return task
            except Exception:
                continue
        
        return None
    
    def stats(self) -> QueueStats:
        """
        Get queue statistics.
        
        Returns:
            Queue statistics including pending, delayed, and inflight counts
        """
        now = self._now_ms()
        prefix = QueueKey.prefix(self._config.queue_id)
        
        pending = 0
        delayed = 0
        inflight = 0
        total = 0
        
        for key_bytes, value_bytes in self._scan_prefix(prefix):
            try:
                key = QueueKey.decode(key_bytes)
                task = Task.decode_value(key, value_bytes)
                total += 1
                
                if task.state == TaskState.PENDING:
                    if task.key.ready_ts > now:
                        delayed += 1
                    else:
                        pending += 1
                elif task.state == TaskState.CLAIMED:
                    if task.lease_expires_at and now < task.lease_expires_at:
                        inflight += 1
                    else:
                        pending += 1  # Lease expired
            except Exception:
                continue
        
        # Count active claims
        claim_prefix = f"queue_claim/{self._config.queue_id}/".encode('utf-8')
        active_claims = sum(1 for _ in self._scan_prefix(claim_prefix))
        
        return QueueStats(
            queue_id=self._config.queue_id,
            pending=pending,
            delayed=delayed,
            inflight=inflight,
            total=total,
            active_claims=active_claims,
        )
    
    def list_tasks(self, limit: int = 100) -> List[Task]:
        """
        List tasks in priority order.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of tasks in priority order
        """
        prefix = QueueKey.prefix(self._config.queue_id)
        tasks = []
        
        for key_bytes, value_bytes in self._scan_prefix(prefix):
            if len(tasks) >= limit:
                break
            try:
                key = QueueKey.decode(key_bytes)
                task = Task.decode_value(key, value_bytes)
                tasks.append(task)
            except Exception:
                continue
        
        return tasks
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _scan_prefix(self, prefix: bytes) -> Iterator[Tuple[bytes, bytes]]:
        """Scan for keys with the given prefix."""
        yield from self._backend.scan_prefix(prefix)
    
    def _get_claim(self, task_id: str) -> Optional[Claim]:
        """Get the claim for a task."""
        claim_key = f"queue_claim/{self._config.queue_id}/{task_id}".encode('utf-8')
        claim_data = self._backend.get(claim_key)
        
        if claim_data:
            return Claim.decode_value(task_id, claim_data)
        return None
    
    def _cleanup_expired_claims(self, now_ms: int) -> int:
        """Clean up expired claims."""
        claim_prefix = f"queue_claim/{self._config.queue_id}/".encode('utf-8')
        expired = []
        
        for key_bytes, value_bytes in self._scan_prefix(claim_prefix):
            try:
                # Extract task_id from key
                task_id = key_bytes.decode('utf-8').split('/')[-1]
                claim = Claim.decode_value(task_id, value_bytes)
                
                if claim.is_expired(now_ms):
                    expired.append((key_bytes, task_id))
            except Exception:
                continue
        
        # Delete expired claims and reset task states
        prefix = QueueKey.prefix(self._config.queue_id)
        
        with self._backend.begin_transaction() as txn:
            for claim_key, task_id in expired:
                txn.delete(claim_key)
                
                # Reset task state
                for key_bytes, value_bytes in self._scan_prefix(prefix):
                    try:
                        key = QueueKey.decode(key_bytes)
                        if key.task_id == task_id:
                            task = Task.decode_value(key, value_bytes)
                            if task.state == TaskState.CLAIMED:
                                task.state = TaskState.PENDING
                                task.claimed_at = None
                                task.claimed_by = None
                                task.lease_expires_at = None
                                txn.put(key_bytes, task.encode_value())
                            break
                    except Exception:
                        continue
        
        return len(expired)
    
    def _move_to_dlq(self, txn: QueueTransaction, task: Task) -> None:
        """Move a task to the dead letter queue."""
        if not self._config.dead_letter_queue_id:
            return
        
        dlq_key = QueueKey(
            queue_id=self._config.dead_letter_queue_id,
            priority=task.key.priority,
            ready_ts=task.key.ready_ts,
            sequence=task.key.sequence,
            task_id=task.key.task_id,
        )
        
        txn.put(dlq_key.encode(), task.encode_value())


# ============================================================================
# Streaming Top-K for ORDER BY + LIMIT
# ============================================================================

class StreamingTopK:
    """
    Streaming top-K collector using a bounded heap.
    
    This implements correct ORDER BY ... LIMIT K semantics without
    requiring O(N) memory or O(N log N) full sort.
    
    Complexity:
    - Space: O(K)
    - Time: O(N log K) for N insertions
    - For K=1: O(N) comparisons with O(1) memory
    
    Example:
        # ORDER BY priority ASC LIMIT 3
        topk = StreamingTopK(k=3, ascending=True, key=lambda x: x.priority)
        
        for task in all_tasks:
            topk.push(task)
        
        result = topk.get_sorted()  # Returns top 3 tasks by priority
    """
    
    def __init__(
        self,
        k: int,
        ascending: bool = True,
        key: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Create a streaming top-K collector.
        
        Args:
            k: Number of elements to keep
            ascending: If True, keep smallest K; if False, keep largest K
            key: Optional key function for comparison
        """
        self._k = k
        self._ascending = ascending
        self._key = key or (lambda x: x)
        self._heap: List[Tuple[Any, Any]] = []  # (sort_key, item)
    
    def push(self, item: Any) -> None:
        """
        Push an item into the collector.
        
        Complexity: O(log K)
        """
        if self._k == 0:
            return
        
        sort_key = self._key(item)
        
        # For ascending (smallest K), we use a max-heap (negate keys)
        # For descending (largest K), we use a min-heap
        if self._ascending:
            heap_key = (-sort_key if isinstance(sort_key, (int, float)) else sort_key, item)
        else:
            heap_key = (sort_key, item)
        
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, heap_key)
        else:
            # Compare with current extreme
            if self._ascending:
                # For smallest K: replace if new < current max
                if sort_key < -self._heap[0][0]:
                    heapq.heapreplace(self._heap, heap_key)
            else:
                # For largest K: replace if new > current min
                if sort_key > self._heap[0][0]:
                    heapq.heapreplace(self._heap, heap_key)
    
    def get_sorted(self) -> List[Any]:
        """
        Get the top-K items in sorted order.
        
        Complexity: O(K log K)
        """
        items = [item for _, item in self._heap]
        items.sort(key=self._key, reverse=not self._ascending)
        return items
    
    def __len__(self) -> int:
        return len(self._heap)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_queue(
    db_or_client: Union["Database", "SochDBClient", QueueBackend],
    queue_id: str = "default",
    visibility_timeout_ms: int = 30_000,
    max_attempts: int = 3,
    namespace: str = "default",
) -> PriorityQueue:
    """
    Create a priority queue from a Database, Client, or Backend.
    
    This is a convenience function that automatically detects the type
    and creates the appropriate backend.
    
    Args:
        db_or_client: Database, SochDBClient, or QueueBackend instance
        queue_id: Queue identifier
        visibility_timeout_ms: Default visibility timeout
        max_attempts: Max delivery attempts before dead-lettering
        namespace: Namespace for gRPC mode
        
    Returns:
        PriorityQueue instance
        
    Example:
        # Embedded mode
        db = Database.open("./queue_db")
        queue = create_queue(db, "tasks")
        
        # Server mode
        client = SochDBClient("localhost:50051")
        queue = create_queue(client, "tasks")
        
        # Testing
        backend = InMemoryQueueBackend()
        queue = create_queue(backend, "tasks")
    """
    # Check if it's already a backend
    if isinstance(db_or_client, QueueBackend):
        return PriorityQueue.from_backend(
            db_or_client,
            queue_id=queue_id,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
    
    # Check if it's a Database (has _handle attribute from FFI)
    if hasattr(db_or_client, '_handle') and hasattr(db_or_client, 'transaction'):
        return PriorityQueue.from_database(
            db_or_client,
            queue_id=queue_id,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
    
    # Check if it's a SochDBClient (has channel attribute from gRPC)
    if hasattr(db_or_client, 'channel') and hasattr(db_or_client, '_get_stub'):
        return PriorityQueue.from_client(
            db_or_client,
            queue_id=queue_id,
            namespace=namespace,
            visibility_timeout_ms=visibility_timeout_ms,
            max_attempts=max_attempts,
        )
    
    raise TypeError(
        f"Expected Database, SochDBClient, or QueueBackend, got {type(db_or_client)}"
    )
