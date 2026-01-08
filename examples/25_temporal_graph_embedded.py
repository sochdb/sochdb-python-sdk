#!/usr/bin/env python3
"""
ToonDB Python SDK - Example: Temporal Graph Operations (Embedded FFI)

This demonstrates temporal graph operations using the EMBEDDED FFI mode.
NO SERVER REQUIRED - runs directly with local database files.

Temporal graphs allow you to query "What did the system know at time T?"
This is essential for agent memory systems that need to reason about
state changes over time.

Comparison:
- 21_temporal_graph.py: Uses gRPC client (requires server)
- 25_temporal_graph_embedded.py: Uses embedded FFI (this file, no server)
"""

import time
from toondb import Database

def main():
    print("=" * 60)
    print("ToonDB - Temporal Graph Example (Embedded FFI)")
    print("=" * 60)
    
    # Open database with embedded FFI - NO SERVER NEEDED
    with Database.open("./test_temporal_db") as db:
        namespace = "smart_home"
        
        # Current time in milliseconds
        now = int(time.time() * 1000)
        one_hour = 60 * 60 * 1000
        
        print("\n--- Creating Temporal Edges (Embedded FFI) ---")
        
        # Record: Door was opened at 10:00, closed at 11:00
        db.add_temporal_edge(
            namespace=namespace,
            from_id="door_front",
            edge_type="STATE",
            to_id="open",
            valid_from=now - (2 * one_hour),  # 2 hours ago
            valid_until=now - one_hour,        # 1 hour ago
            properties={"sensor": "motion_1"}
        )
        print("✓ Added: door_front was OPEN from 2 hours ago to 1 hour ago")
        
        # Record: Door was closed at 11:00 (still closed)
        db.add_temporal_edge(
            namespace=namespace,
            from_id="door_front",
            edge_type="STATE",
            to_id="closed",
            valid_from=now - one_hour,  # 1 hour ago
            valid_until=0,               # No expiry (still valid)
            properties={"sensor": "motion_1"}
        )
        print("✓ Added: door_front is CLOSED since 1 hour ago")
        
        # Record: Light was on from 10:30 to 11:30
        db.add_temporal_edge(
            namespace=namespace,
            from_id="light_living",
            edge_type="STATE",
            to_id="on",
            valid_from=now - int(1.5 * one_hour),
            valid_until=now - int(0.5 * one_hour),
            properties={"brightness": "100"}
        )
        print("✓ Added: light_living was ON from 1.5 hours ago to 0.5 hours ago")
        
        print("\n--- Querying: POINT_IN_TIME (Embedded FFI) ---")
        print('Query: "Was the door open 1.5 hours ago?"')
        
        # Query door state 1.5 hours ago
        edges = db.query_temporal_graph(
            namespace=namespace,
            node_id="door_front",
            mode="POINT_IN_TIME",
            timestamp=now - int(1.5 * one_hour)
        )
        
        for edge in edges:
            print(f"  → {edge['from_id']} --[{edge['edge_type']}]--> {edge['to_id']}")
        
        if any(e["to_id"] == "open" for e in edges):
            print("  Answer: Yes, the door was OPEN")
        else:
            print("  Answer: No, the door was CLOSED")
        
        print("\n--- Querying: CURRENT (Embedded FFI) ---")
        print('Query: "What is the current door state?"')
        
        edges = db.query_temporal_graph(
            namespace=namespace,
            node_id="door_front",
            mode="CURRENT"
        )
        
        for edge in edges:
            print(f"  → {edge['from_id']} --[{edge['edge_type']}]--> {edge['to_id']}")
        
        print("\n--- Agent Memory Use Case ---")
        print("Agent reasoning: 'I need to check if the door was secure during the incident'")
        
        # Query historical state during incident time
        incident_time = now - int(1.75 * one_hour)
        edges_during_incident = db.query_temporal_graph(
            namespace=namespace,
            node_id="door_front",
            mode="POINT_IN_TIME",
            timestamp=incident_time
        )
        
        door_state = edges_during_incident[0]["to_id"] if edges_during_incident else "unknown"
        print(f"  At incident time (1.75 hours ago): door was {door_state.upper()}")
        
        print("\n--- Benefits of Temporal Graphs ---")
        print("  • Time-travel queries: 'What did the system know at time T?'")
        print("  • Agent memory: Track beliefs and state changes over time")
        print("  • Audit trail: Full history of all state transitions")
        print("  • No data loss: Old states preserved, not overwritten")
        print("  • Efficient: O(log N) queries with proper indexing")
        
        print("\n--- Embedded FFI vs gRPC ---")
        print("  Embedded FFI (this example):")
        print("    ✅ No server required")
        print("    ✅ Direct database access")
        print("    ✅ Best for: Local dev, notebooks, single-process apps")
        print("  gRPC (21_temporal_graph.py):")
        print("    ✅ Centralized server")
        print("    ✅ Multi-language support")
        print("    ✅ Best for: Production, distributed systems")

if __name__ == "__main__":
    main()
