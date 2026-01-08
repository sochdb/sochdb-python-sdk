#!/usr/bin/env python3
"""
ToonDB Python SDK - Example: Batch Operations (Embedded FFI)

Atomic batch operations ensure all-or-nothing semantics.
If any operation fails, the entire batch is rolled back.

No server required - uses embedded FFI.
"""

from toondb import Database

def main():
    print("=" * 60)
    print("ToonDB - Batch Operations Example (Embedded FFI)")
    print("=" * 60)
    print("Note: This uses embedded Database - no server required!\n")
    
    # Open embedded database
    db = Database.open("./example_batch_db")
    
    print("--- Single Batch Put ---")
    
    # Atomic multi-key write using transaction
    with db.transaction() as txn:
        # All these writes happen atomically
        txn.put(b"user:alice:email", b"alice@example.com")
        txn.put(b"user:alice:age", b"30")
        txn.put(b"user:alice:created", b"2026-01-07")
        txn.put(b"user:alice:verified", b"true")
    
    print("✓ Atomically stored 4 keys for user:alice")
    
    # Verify
    email = db.get(b"user:alice:email")
    print(f"  Verification: email = {email.decode()}")
    
    print("\n--- Batch Put with Rollback ---")
    
    try:
        with db.transaction() as txn:
            txn.put(b"user:bob:email", b"bob@example.com")
            txn.put(b"user:bob:age", b"25")
            
            # Simulate an error - none of the writes should persist
            raise ValueError("Simulated failure!")
            
            txn.put(b"user:bob:verified", b"true")  # Never reached
    except ValueError as e:
        print(f"✓ Transaction aborted: {e}")
    
    # Verify rollback - bob's data should NOT exist
    bob_email = db.get(b"user:bob:email")
    print(f"  Verification: bob's email = {bob_email}")  # Should be None
    
    print("\n--- Batch Scan and Update ---")
    
    # Add some products
    products = [
        (b"product:001:price", b"99.99"),
        (b"product:001:stock", b"50"),
        (b"product:002:price", b"149.99"),
        (b"product:002:stock", b"30"),
        (b"product:003:price", b"79.99"),
        (b"product:003:stock", b"100"),
    ]
    
    with db.transaction() as txn:
        for key, value in products:
            txn.put(key, value)
    
    print(f"✓ Added {len(products)} product entries")
    
    # Atomic inventory update - decrease all stock by 10%
    with db.transaction() as txn:
        stock_updates = []
        for key, value in txn.scan_prefix(b"product:"):
            if b":stock" in key:
                current = int(value.decode())
                new_stock = int(current * 0.9)  # 10% decrease
                stock_updates.append((key, str(new_stock).encode()))
        
        for key, value in stock_updates:
            txn.put(key, value)
            print(f"  Updated {key.decode()}: {value.decode()}")
    
    print("\n--- Conditional Batch Updates ---")
    
    # Only update if condition is met
    with db.transaction() as txn:
        # Check current stock before update
        stock = txn.get(b"product:003:stock")
        if stock and int(stock.decode()) > 80:
            txn.put(b"product:003:status", b"high_stock")
            txn.put(b"product:003:reorder", b"false")
            print("✓ Product 003 marked as high_stock")
        else:
            txn.put(b"product:003:status", b"low_stock")
            txn.put(b"product:003:reorder", b"true")
            print("✓ Product 003 marked as low_stock")
    
    print("\n--- Benefits of Atomic Batches ---")
    print("""
    1. Consistency: All writes succeed or all fail
    2. Isolation: Other readers see old or new state, never partial
    3. Durability: Committed batches survive crashes
    4. Performance: Single disk sync for many operations
    
    Use cases:
    - User registration (email + profile + settings)
    - E-commerce orders (inventory + order + payment)
    - Agent memory updates (context + facts + embeddings)
    """)
    
    # Cleanup
    db.close()
    import shutil
    shutil.rmtree("./example_batch_db", ignore_errors=True)
    print("✅ Batch operations example complete!")


if __name__ == "__main__":
    main()
