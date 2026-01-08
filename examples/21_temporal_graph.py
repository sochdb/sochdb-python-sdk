#!/usr/bin/env python3
"""
ToonDB Python SDK - Example: Temporal Graph Operations (PREVIEW)

NOTE: This feature requires ToonDB Server 0.3.5+
The temporal graph RPC methods are defined in proto but may not be 
fully implemented in earlier server versions.

Temporal graphs allow you to query "What did the system know at time T?"
This is essential for agent memory systems that need to reason about
state changes over time.

Requires: ToonDB gRPC server running on localhost:50051
Start with: cargo run -p toondb-grpc --release
"""

import time
from toondb import ToonDBClient

def main():
    print("=" * 60)
    print("ToonDB - Temporal Graph Example")
    print("=" * 60)
    
    # Connect to server
    client = ToonDBClient("localhost:50051")
    namespace = "smart_home"
    
    # Current time in milliseconds
    now = int(time.time() * 1000)
    one_hour = 60 * 60 * 1000
    
    print("\n--- Creating Temporal Edges ---")
    
    # Record: Door was opened at 10:00, closed at 11:00
    client.add_temporal_edge(
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
    client.add_temporal_edge(
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
    client.add_temporal_edge(
        namespace=namespace,
        from_id="light_living",
        edge_type="STATE",
        to_id="on",
        valid_from=now - int(1.5 * one_hour),
        valid_until=now - int(0.5 * one_hour),
        properties={"brightness": "100"}
    )
    print("✓ Added: light_living was ON from 1.5 hours ago to 0.5 hours ago")
    
    print("\n--- Querying: POINT_IN_TIME ---")
    print('Query: "Was the door open 1.5 hours ago?"')
    
    # Query door state 1.5 hours ago
    edges = client.query_temporal_graph(
        namespace=namespace,
        node_id="door_front",
        mode="POINT_IN_TIME",
        timestamp=now - int(1.5 * one_hour)
    )
    
    for edge in edges:
        print(f"  → {edge.from_id} --[{edge.edge_type}]--> {edge.to_id}")
    
    if any(e.to_id == "open" for e in edges):
        print("  Answer: Yes, the door was OPEN")
    else:
        print("  Answer: No, the door was CLOSED")
    
    print("\n--- Querying: CURRENT ---")
    print('Query: "What is the current door state?"')
    
    edges = client.query_temporal_graph(
        namespace=namespace,
        node_id="door_front",
        mode="CURRENT"
    )
    
    for edge in edges:
        print(f"  → {edge.from_id} --[{edge.edge_type}]--> {edge.to_id}")
    
    print("\n--- Querying: RANGE ---")
    print('Query: "What changed in the last 3 hours?"')
    
    edges = client.query_temporal_graph(
        namespace=namespace,
        node_id="door_front",
        mode="RANGE",
        start_time=now - (3 * one_hour),
        end_time=now
    )
    
    print(f"  Found {len(edges)} state changes:")
    for edge in edges:
        valid_from_str = time.strftime('%H:%M', time.localtime(edge.valid_from / 1000))
        valid_until_str = "now" if edge.valid_until == 0 else time.strftime('%H:%M', time.localtime(edge.valid_until / 1000))
        print(f"  → {edge.to_id.upper()} from {valid_from_str} to {valid_until_str}")
    
    print("\n--- Use Case: Agent Time-Travel Memory ---")
    print("""
    Temporal graphs enable agents to answer questions like:
    - "Was the door open when the alarm went off?"
    - "Who was in the room between 2pm and 3pm?"
    - "What sensors triggered before the incident?"
    
    This is critical for:
    - Debugging agent behavior
    - Auditing decisions
    - Explaining reasoning at a specific point in time
    """)
    
    client.close()
    print("✅ Temporal graph example complete!")


if __name__ == "__main__":
    main()
