#!/usr/bin/env python3
"""
Tests for SochDB Queue Module

Tests the queue implementation with both FFI and gRPC backends.
"""

import pytest
import time
from dataclasses import dataclass
from typing import List

from sochdb.queue import (
    PriorityQueue,
    QueueConfig,
    QueueKey,
    Task,
    TaskState,
    QueueStats,
    StreamingTopK,
    Claim,
    encode_u64_be,
    decode_u64_be,
    encode_i64_be,
    decode_i64_be,
    create_queue,
    InMemoryQueueBackend,
)


# ============================================================================
# Helper for creating test queues
# ============================================================================

def create_test_queue(queue_id: str = "test") -> PriorityQueue:
    """Create a queue with InMemoryQueueBackend for testing."""
    backend = InMemoryQueueBackend()
    return PriorityQueue.from_backend(backend, queue_id=queue_id)


# ============================================================================
# Key Encoding Tests
# ============================================================================

class TestKeyEncoding:
    """Test big-endian key encoding for lexicographic ordering."""
    
    def test_u64_encode_decode(self):
        """Test u64 encoding roundtrip."""
        values = [0, 1, 100, 1000, 2**32, 2**63-1, 2**64-1]
        for value in values:
            encoded = encode_u64_be(value)
            decoded = decode_u64_be(encoded)
            assert decoded == value, f"Failed for {value}"
    
    def test_u64_ordering(self):
        """Test that encoded u64 preserves lexicographic order."""
        values = [0, 100, 200, 1000, 10000, 2**32]
        encoded = [encode_u64_be(v) for v in values]
        assert encoded == sorted(encoded), "Ordering not preserved"
    
    def test_i64_encode_decode(self):
        """Test i64 encoding roundtrip."""
        values = [-(2**63), -1000, -1, 0, 1, 1000, 2**63-1]
        for value in values:
            encoded = encode_i64_be(value)
            decoded = decode_i64_be(encoded)
            assert decoded == value, f"Failed for {value}"
    
    def test_i64_ordering(self):
        """Test that encoded i64 preserves lexicographic order."""
        values = [-1000, -100, -1, 0, 1, 100, 1000]
        encoded = [encode_i64_be(v) for v in values]
        assert encoded == sorted(encoded), "Ordering not preserved"


# ============================================================================
# QueueKey Tests
# ============================================================================

class TestQueueKey:
    """Test QueueKey encoding and comparison."""
    
    def test_encode_decode_roundtrip(self):
        """Test QueueKey encoding and decoding."""
        key = QueueKey(
            queue_id="test_queue",
            priority=100,
            ready_ts=int(time.time() * 1000),
            sequence=12345,
            task_id="task-uuid-123",
        )
        
        encoded = key.encode()
        decoded = QueueKey.decode(encoded)
        
        assert decoded.queue_id == key.queue_id
        assert decoded.priority == key.priority
        assert decoded.ready_ts == key.ready_ts
        assert decoded.sequence == key.sequence
        assert decoded.task_id == key.task_id
    
    def test_prefix(self):
        """Test queue prefix generation."""
        prefix = QueueKey.prefix("my_queue")
        assert prefix == b"queue/my_queue/"
    
    def test_ordering_by_priority(self):
        """Test that keys with lower priority come first."""
        key1 = QueueKey("q", priority=1, ready_ts=1000, sequence=1, task_id="t1")
        key2 = QueueKey("q", priority=5, ready_ts=1000, sequence=1, task_id="t2")
        key3 = QueueKey("q", priority=10, ready_ts=1000, sequence=1, task_id="t3")
        
        assert key1 < key2 < key3
        
        # Also check encoded bytes
        assert key1.encode() < key2.encode() < key3.encode()
    
    def test_ordering_by_ready_ts(self):
        """Test FIFO within same priority (by ready_ts then sequence)."""
        key1 = QueueKey("q", priority=5, ready_ts=1000, sequence=1, task_id="t1")
        key2 = QueueKey("q", priority=5, ready_ts=2000, sequence=1, task_id="t2")
        
        assert key1 < key2
    
    def test_ordering_by_sequence(self):
        """Test FIFO by sequence number."""
        key1 = QueueKey("q", priority=5, ready_ts=1000, sequence=1, task_id="t1")
        key2 = QueueKey("q", priority=5, ready_ts=1000, sequence=2, task_id="t2")
        
        assert key1 < key2


# ============================================================================
# Task Tests
# ============================================================================

class TestTask:
    """Test Task serialization and lifecycle."""
    
    def test_encode_decode_roundtrip(self):
        """Test Task encoding and decoding."""
        key = QueueKey("q", 1, 1000, 1, "task1")
        task = Task(
            key=key,
            payload=b"test payload",
            state=TaskState.PENDING,
            attempts=0,
            max_attempts=3,
            metadata={"key": "value"},
        )
        
        encoded = task.encode_value()
        decoded = Task.decode_value(key, encoded)
        
        assert decoded.payload == task.payload
        assert decoded.state == task.state
        assert decoded.attempts == task.attempts
        assert decoded.max_attempts == task.max_attempts
        assert decoded.metadata == task.metadata
    
    def test_is_visible_pending(self):
        """Test visibility for pending tasks."""
        now = int(time.time() * 1000)
        
        # Ready task
        key = QueueKey("q", 1, now - 1000, 1, "task1")
        task = Task(key=key, payload=b"")
        assert task.is_visible(now) is True
        
        # Delayed task
        key = QueueKey("q", 1, now + 10000, 1, "task2")
        task = Task(key=key, payload=b"")
        assert task.is_visible(now) is False
    
    def test_is_visible_claimed(self):
        """Test visibility for claimed tasks."""
        now = int(time.time() * 1000)
        key = QueueKey("q", 1, now, 1, "task1")
        
        # Claimed with valid lease
        task = Task(
            key=key,
            payload=b"",
            state=TaskState.CLAIMED,
            lease_expires_at=now + 10000,
        )
        assert task.is_visible(now) is False
        
        # Claimed with expired lease
        task.lease_expires_at = now - 1000
        assert task.is_visible(now) is True
    
    def test_should_dead_letter(self):
        """Test dead letter detection."""
        key = QueueKey("q", 1, 1000, 1, "task1")
        task = Task(key=key, payload=b"", max_attempts=3)
        
        task.attempts = 2
        assert task.should_dead_letter() is False
        
        task.attempts = 3
        assert task.should_dead_letter() is True


# ============================================================================
# PriorityQueue Tests
# ============================================================================

class TestPriorityQueue:
    """Test PriorityQueue operations."""
    
    def test_enqueue(self):
        """Test basic enqueue."""
        queue = create_test_queue("test")
        
        task = queue.enqueue(priority=5, payload=b"test task")
        
        assert task.priority == 5
        assert task.payload == b"test task"
        assert task.state == TaskState.PENDING
    
    def test_dequeue_priority_order(self):
        """Test that dequeue returns highest priority first."""
        queue = create_test_queue("test")
        
        # Enqueue in random order
        queue.enqueue(priority=5, payload=b"low")
        queue.enqueue(priority=1, payload=b"high")
        queue.enqueue(priority=3, payload=b"medium")
        
        # Dequeue should return in priority order
        task1 = queue.dequeue(worker_id="w1")
        assert task1.priority == 1
        assert task1.payload == b"high"
        queue.ack(task1.task_id)
        
        task2 = queue.dequeue(worker_id="w1")
        assert task2.priority == 3
        queue.ack(task2.task_id)
        
        task3 = queue.dequeue(worker_id="w1")
        assert task3.priority == 5
        queue.ack(task3.task_id)
    
    def test_dequeue_empty_queue(self):
        """Test dequeue on empty queue."""
        queue = create_test_queue("test")
        
        task = queue.dequeue(worker_id="w1")
        assert task is None
    
    def test_ack(self):
        """Test task acknowledgment."""
        queue = create_test_queue("test")
        
        task = queue.enqueue(priority=1, payload=b"test")
        claimed = queue.dequeue(worker_id="w1")
        
        assert queue.ack(claimed.task_id) is True
        
        # Task should be removed
        assert queue.peek() is None
    
    def test_nack(self):
        """Test negative acknowledgment."""
        queue = create_test_queue("test")
        
        task = queue.enqueue(priority=1, payload=b"test")
        claimed = queue.dequeue(worker_id="w1")
        
        # Nack should return task to queue
        assert queue.nack(claimed.task_id) is True
        
        # Task should be available again
        reclaimed = queue.dequeue(worker_id="w2")
        assert reclaimed is not None
        assert reclaimed.payload == b"test"
    
    def test_nack_with_new_priority(self):
        """Test nack with priority change."""
        queue = create_test_queue("test")
        
        queue.enqueue(priority=1, payload=b"high")
        queue.enqueue(priority=5, payload=b"low")
        
        # Dequeue high priority
        task = queue.dequeue(worker_id="w1")
        assert task.priority == 1
        
        # Nack with lower priority
        queue.nack(task.task_id, new_priority=10)
        
        # Next dequeue should get the "low" task
        task = queue.dequeue(worker_id="w1")
        assert task.priority == 5
    
    def test_delayed_enqueue(self):
        """Test delayed visibility."""
        queue = create_test_queue("test")
        
        # Enqueue with delay
        task = queue.enqueue(priority=1, payload=b"delayed", delay_ms=10000)
        
        # Should not be visible
        result = queue.dequeue(worker_id="w1")
        assert result is None
    
    def test_peek(self):
        """Test peek operation."""
        queue = create_test_queue("test")
        
        queue.enqueue(priority=1, payload=b"task1")
        
        # Peek should not remove
        task = queue.peek()
        assert task is not None
        assert task.payload == b"task1"
        
        # Peek again should return same
        task2 = queue.peek()
        assert task2.task_id == task.task_id
    
    def test_stats(self):
        """Test queue statistics."""
        queue = create_test_queue("test")
        
        queue.enqueue(priority=1, payload=b"task1")
        queue.enqueue(priority=2, payload=b"task2")
        
        stats = queue.stats()
        assert stats.queue_id == "test"
        assert stats.pending == 2
        assert stats.total == 2


# ============================================================================
# StreamingTopK Tests
# ============================================================================

class TestStreamingTopK:
    """Test StreamingTopK heap-based selection."""
    
    def test_ascending_simple(self):
        """Test smallest K elements."""
        data = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
        
        topk = StreamingTopK(k=3, ascending=True)
        for x in data:
            topk.push(x)
        
        result = topk.get_sorted()
        assert result == [0, 1, 2]
    
    def test_descending_simple(self):
        """Test largest K elements."""
        data = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
        
        topk = StreamingTopK(k=3, ascending=False)
        for x in data:
            topk.push(x)
        
        result = topk.get_sorted()
        assert result == [9, 8, 7]
    
    def test_with_key_function(self):
        """Test with custom key function."""
        @dataclass
        class Item:
            name: str
            priority: int
        
        items = [
            Item("a", 5),
            Item("b", 2),
            Item("c", 8),
            Item("d", 1),
        ]
        
        topk = StreamingTopK(k=2, ascending=True, key=lambda x: x.priority)
        for item in items:
            topk.push(item)
        
        result = topk.get_sorted()
        names = [x.name for x in result]
        assert names == ["d", "b"]  # priorities 1, 2
    
    def test_k_greater_than_n(self):
        """Test when k > number of elements."""
        topk = StreamingTopK(k=100, ascending=True)
        for x in [3, 1, 2]:
            topk.push(x)
        
        result = topk.get_sorted()
        assert result == [1, 2, 3]
    
    def test_k_zero(self):
        """Test k=0 returns empty."""
        topk = StreamingTopK(k=0, ascending=True)
        for x in [1, 2, 3]:
            topk.push(x)
        
        result = topk.get_sorted()
        assert result == []
    
    def test_large_dataset(self):
        """Test with large dataset."""
        import random
        random.seed(42)
        
        data = list(range(10000))
        random.shuffle(data)
        
        topk = StreamingTopK(k=10, ascending=True)
        for x in data:
            topk.push(x)
        
        result = topk.get_sorted()
        assert result == list(range(10))


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for queue module."""
    
    def test_batch_enqueue(self):
        """Test batch enqueue operation."""
        queue = create_test_queue("test")
        
        tasks = [(i % 3, f"task-{i}".encode()) for i in range(10)]
        result = queue.enqueue_batch(tasks)
        
        assert len(result) == 10
        
        # Verify ordering
        for i in range(10):
            task = queue.dequeue(worker_id="w1")
            assert task is not None
            queue.ack(task.task_id)
    
    def test_list_tasks(self):
        """Test listing tasks."""
        queue = create_test_queue("test")
        
        for i in range(5):
            queue.enqueue(priority=i, payload=f"task-{i}".encode())
        
        tasks = queue.list_tasks(limit=3)
        assert len(tasks) == 3
        
        # Should be in priority order
        assert tasks[0].priority == 0
        assert tasks[1].priority == 1
        assert tasks[2].priority == 2
    
    def test_create_queue_with_backend(self):
        """Test create_queue with different backend types."""
        # Test with InMemoryQueueBackend
        backend = InMemoryQueueBackend()
        queue = create_queue(backend, "test")
        
        task = queue.enqueue(priority=1, payload=b"test")
        assert task.priority == 1
        
        dequeued = queue.dequeue(worker_id="w1")
        assert dequeued.task_id == task.task_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
