#!/usr/bin/env bash
# Regenerate gRPC Python stubs from the SochDB proto (synced from sochdb-grpc).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_SRC="${SOCHDB_PROTO:-$(cd "$ROOT/../../sochdb/sochdb-grpc/proto" 2>/dev/null && pwd)/sochdb.proto}"

if [[ ! -f "$PROTO_SRC" ]]; then
  PROTO_SRC="$ROOT/proto/sochdb.proto"
fi

if [[ ! -f "$PROTO_SRC" ]]; then
  echo "ERROR: sochdb.proto not found" >&2
  exit 1
fi

mkdir -p "$ROOT/proto"
cp "$PROTO_SRC" "$ROOT/proto/sochdb.proto"

python3 -m grpc_tools.protoc \
  -I"$ROOT/proto" \
  --python_out="$ROOT/src/sochdb/proto" \
  --grpc_python_out="$ROOT/src/sochdb/proto" \
  "$ROOT/proto/sochdb.proto"

# Package-relative imports + re-export at package root.
sed -i '' 's/^import sochdb_pb2 as sochdb__pb2$/from . import sochdb_pb2 as sochdb__pb2/' \
  "$ROOT/src/sochdb/proto/sochdb_pb2_grpc.py" 2>/dev/null || \
  sed -i 's/^import sochdb_pb2 as sochdb__pb2$/from . import sochdb_pb2 as sochdb__pb2/' \
  "$ROOT/src/sochdb/proto/sochdb_pb2_grpc.py"

cp "$ROOT/src/sochdb/proto/sochdb_pb2.py" "$ROOT/src/sochdb/sochdb_pb2.py"
cp "$ROOT/src/sochdb/proto/sochdb_pb2_grpc.py" "$ROOT/src/sochdb/sochdb_pb2_grpc.py"
sed -i '' 's/^import sochdb_pb2 as sochdb__pb2$/from . import sochdb_pb2 as sochdb__pb2/' \
  "$ROOT/src/sochdb/sochdb_pb2_grpc.py" 2>/dev/null || \
  sed -i 's/^import sochdb_pb2 as sochdb__pb2$/from . import sochdb_pb2 as sochdb__pb2/' \
  "$ROOT/src/sochdb/sochdb_pb2_grpc.py"

echo "Generated stubs from $PROTO_SRC"