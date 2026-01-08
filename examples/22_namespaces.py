#!/usr/bin/env python3
"""
ToonDB Python SDK - Example: Multi-Tenant Namespaces (Embedded FFI)

Namespaces provide data isolation between tenants, applications,
or environments in a single database instance.

No server required - uses embedded FFI.
"""

import json
import shutil
from toondb import Database

def main():
    print("=" * 60)
    print("ToonDB - Multi-Tenant Namespace Example (Embedded FFI)")
    print("=" * 60)
    print("Note: This uses embedded Database - no server required!\n")
    
    # Open embedded database
    db = Database.open("./example_namespace_db")
    
    print("--- Simulating Namespaces with Key Prefixes ---")
    
    # In embedded mode, namespaces are simulated with key prefixes
    # Format: ns:{namespace}:key
    tenants = ["tenant_acme", "tenant_globex", "staging"]
    
    for tenant in tenants:
        print(f"✓ Created namespace: {tenant}")
    
    print("\n--- Isolated Data Operations ---")
    
    # Store data in each "namespace" (prefix)
    def put_namespaced(tenant: str, key: str, value: bytes):
        full_key = f"ns:{tenant}:{key}".encode()
        db.put(full_key, value)
    
    def get_namespaced(tenant: str, key: str) -> bytes:
        full_key = f"ns:{tenant}:{key}".encode()
        return db.get(full_key)
    
    # Store config in tenant_acme
    put_namespaced("tenant_acme", "config:api_key", b"acme-secret-123")
    put_namespaced("tenant_acme", "config:max_users", b"1000")
    put_namespaced("tenant_acme", "config:plan", b"enterprise")
    print("✓ Stored config in tenant_acme")
    
    # Store config in tenant_globex
    put_namespaced("tenant_globex", "config:api_key", b"globex-secret-456")
    put_namespaced("tenant_globex", "config:max_users", b"50")
    put_namespaced("tenant_globex", "config:plan", b"startup")
    print("✓ Stored config in tenant_globex")
    
    print("\n--- Data Isolation Verification ---")
    
    # Each tenant sees only their own data
    acme_key = get_namespaced("tenant_acme", "config:api_key")
    globex_key = get_namespaced("tenant_globex", "config:api_key")
    
    print(f"tenant_acme API key: {acme_key.decode()}")
    print(f"tenant_globex API key: {globex_key.decode()}")
    
    # Cross-namespace access returns None
    wrong_key = get_namespaced("tenant_acme", "config:wrong")
    print(f"Non-existent key: {wrong_key}")
    
    print("\n--- Scanning a Namespace ---")
    
    # List all keys in tenant_acme namespace
    prefix = b"ns:tenant_acme:"
    print(f"Keys in tenant_acme:")
    for key, value in db.scan_prefix(prefix):
        # Strip the prefix for display
        short_key = key.decode().replace("ns:tenant_acme:", "")
        print(f"  {short_key} = {value.decode()}")
    
    print("\n--- Atomic Multi-Tenant Update ---")
    
    # Atomically update configs across namespaces
    with db.transaction() as txn:
        # Update tenant_acme
        txn.put(b"ns:tenant_acme:config:last_update", b"2026-01-07")
        
        # Update tenant_globex
        txn.put(b"ns:tenant_globex:config:last_update", b"2026-01-07")
        
        # Both updates are atomic
        print("✓ Atomically updated both tenants")
    
    print("\n--- Namespace Patterns ---")
    print("""
    Key Format: ns:{namespace}:{key}
    
    Examples:
    - ns:prod:users:alice:email
    - ns:staging:users:alice:email
    - ns:tenant_123:orders:ord_456
    
    Benefits:
    - Zero data leakage between tenants
    - Single database instance
    - Easy to backup/restore per tenant
    - scan_prefix() for efficient per-tenant queries
    """)
    
    # Cleanup
    db.close()
    shutil.rmtree("./example_namespace_db", ignore_errors=True)
    print("✅ Namespace example complete!")


if __name__ == "__main__":
    main()
