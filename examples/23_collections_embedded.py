#!/usr/bin/env python3
"""
ToonDB Python SDK - Example: Collections (Embedded FFI)

This example demonstrates collection operations using the embedded
Database class (FFI). No server required!

Collections provide structured document storage with vector embeddings
for semantic search.
"""

import json
from toondb import Database

def main():
    print("=" * 60)
    print("ToonDB - Collections Example (Embedded FFI)")
    print("=" * 60)
    print("Note: This uses embedded Database - no server required!\n")
    
    # Open embedded database
    db = Database.open("./example_collections_db")
    
    print("--- Creating a Collection ---")
    
    # Collections are stored using a naming convention in the KV store
    # Format: collection:{name}:doc:{id}
    collection_name = "articles"
    
    # Simulate documents with embeddings
    documents = [
        {
            "id": "article_1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of AI...",
            "category": "AI",
            "embedding": [0.1, 0.2, 0.3, 0.4]  # Simplified 4D embedding
        },
        {
            "id": "article_2", 
            "title": "Python Best Practices",
            "content": "Writing clean Python code requires...",
            "category": "Programming",
            "embedding": [0.5, 0.6, 0.7, 0.8]
        },
        {
            "id": "article_3",
            "title": "Deep Learning Fundamentals",
            "content": "Neural networks are the foundation...",
            "category": "AI",
            "embedding": [0.15, 0.25, 0.35, 0.45]
        }
    ]
    
    print(f"✓ Adding {len(documents)} documents to '{collection_name}'")
    
    # Store documents
    for doc in documents:
        key = f"collection:{collection_name}:doc:{doc['id']}".encode()
        db.put(key, json.dumps(doc).encode())
        print(f"  → Added: {doc['title']}")
    
    print("\n--- Querying Documents ---")
    
    # Get a specific document
    doc_key = f"collection:{collection_name}:doc:article_1".encode()
    raw = db.get(doc_key)
    if raw:
        doc = json.loads(raw.decode())
        print(f"Retrieved: {doc['title']}")
        print(f"  Category: {doc['category']}")
    
    print("\n--- Scanning Collection ---")
    
    # Scan all documents in collection
    prefix = f"collection:{collection_name}:doc:".encode()
    ai_articles = []
    
    for key, value in db.scan_prefix(prefix):
        doc = json.loads(value.decode())
        print(f"  • {doc['title']} [{doc['category']}]")
        if doc['category'] == "AI":
            ai_articles.append(doc)
    
    print(f"\n  Found {len(ai_articles)} AI articles")
    
    print("\n--- Using Transactions ---")
    
    # Atomic update of multiple documents
    with db.transaction() as txn:
        # Update article category
        doc_key = f"collection:{collection_name}:doc:article_2".encode()
        raw = txn.get(doc_key)
        if raw:
            doc = json.loads(raw.decode())
            doc['category'] = "Python"
            doc['updated'] = True
            txn.put(doc_key, json.dumps(doc).encode())
            print(f"✓ Updated category for: {doc['title']}")
        
        # Add a new document atomically
        new_doc = {
            "id": "article_4",
            "title": "Async Python Patterns",
            "content": "Asynchronous programming in Python...",
            "category": "Python",
            "embedding": [0.6, 0.7, 0.8, 0.9]
        }
        new_key = f"collection:{collection_name}:doc:article_4".encode()
        txn.put(new_key, json.dumps(new_doc).encode())
        print(f"✓ Added new document: {new_doc['title']}")
    
    print("\n--- Embedded vs gRPC ---")
    print("""
    Embedded (Database class):
    ✓ No server needed - runs in-process
    ✓ Zero network latency
    ✓ Simple deployment
    ✗ Business logic in Python (maintenance overhead)
    
    gRPC (ToonDBClient):
    ✓ All logic in Rust server (single source of truth)
    ✓ Language-agnostic API
    ✓ Temporal graphs, policies, and advanced features
    ✗ Requires server process
    
    Choose embedded for simple apps, gRPC for production.
    """)
    
    # Cleanup
    db.close()
    import shutil
    shutil.rmtree("./example_collections_db", ignore_errors=True)
    print("✅ Collection example complete!")


if __name__ == "__main__":
    main()
