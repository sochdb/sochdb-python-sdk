#!/usr/bin/env bash
# =============================================================================
# Build Rust binary for the current platform
# =============================================================================
#
# This script is called by cibuildwheel before building the Python wheel.
# It compiles SochDB CLI binaries and places them in the correct _bin directory.
#
# Usage:
#   ./scripts/build_rust_binary.sh
#
# Environment:
#   CARGO_BUILD_TARGET - Optional: Override target triple
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SDK_DIR="$PROJECT_DIR"
find_workspace_root() {
    local current="$SDK_DIR"
    while [[ "$current" != "/" ]]; do
        if [[ -f "$current/Cargo.toml" ]] && grep -q "\[workspace\]" "$current/Cargo.toml"; then
            echo "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done
    return 1
}

WORKSPACE_ROOT="$(find_workspace_root || true)"
if [[ -z "$WORKSPACE_ROOT" ]]; then
    if [[ -f "$(dirname "$SDK_DIR")/sochdb/Cargo.toml" ]] && grep -q "\[workspace\]" "$(dirname "$SDK_DIR")/sochdb/Cargo.toml"; then
        WORKSPACE_ROOT="$(dirname "$SDK_DIR")/sochdb"
    fi
fi

echo "=== SochDB Rust Binary Build ==="
echo "Project: $PROJECT_DIR"
echo "Workspace: ${WORKSPACE_ROOT:-unknown}"

# Detect platform and architecture
detect_platform() {
    local os arch
    
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    
    # Normalize OS name
    case "$os" in
        linux*)   os="linux" ;;
        darwin*)  os="darwin" ;;
        mingw*|msys*|cygwin*) os="windows" ;;
    esac
    
    # Normalize architecture
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        i686|i386) arch="i686" ;;
    esac
    
    echo "${os}-${arch}"
}

# Get the binary name for the platform
get_binary_names() {
    local platform="$1"
    if [[ "$platform" == windows-* ]]; then
        echo "sochdb-bulk.exe sochdb-server.exe sochdb-grpc-server.exe"
    else
        echo "sochdb-bulk sochdb-server sochdb-grpc-server"
    fi
}

# Get the Rust target triple
get_rust_target() {
    local platform="$1"
    
    case "$platform" in
        linux-x86_64)   echo "x86_64-unknown-linux-gnu" ;;
        linux-aarch64)  echo "aarch64-unknown-linux-gnu" ;;
        darwin-x86_64)  echo "x86_64-apple-darwin" ;;
        darwin-aarch64) echo "aarch64-apple-darwin" ;;
        windows-x86_64) echo "x86_64-pc-windows-msvc" ;;
        *)
            echo "Unknown platform: $platform" >&2
            exit 1
            ;;
    esac
}

# Main build logic
main() {
    local platform target bin_dir
    
    platform="${PLATFORM:-$(detect_platform)}"
    echo "Platform: $platform"
    
    bin_dir="$SDK_DIR/src/sochdb/_bin/$platform"
    
    # Create bin directory
    mkdir -p "$bin_dir"
    
    # Check if we should use a specific target
    if [[ -n "${CARGO_BUILD_TARGET:-}" ]]; then
        target="$CARGO_BUILD_TARGET"
    else
        target="$(get_rust_target "$platform")"
    fi
    
    echo "Target: $target"
    echo "Output dir: $bin_dir"
    
    # Ensure Rust is available
    if ! command -v cargo &> /dev/null; then
        echo "Error: cargo not found. Install Rust first." >&2
        exit 1
    fi
    
    # Build the binary
    echo ""
    echo "Building SochDB binaries..."
    if [[ -z "$WORKSPACE_ROOT" ]]; then
        echo "Error: Could not locate Cargo workspace root." >&2
        exit 1
    fi
    cd "$WORKSPACE_ROOT"
    
    if [[ "$target" != "$(rustc -vV | grep host | cut -d' ' -f2)" ]]; then
        # Cross-compilation: need explicit target
        cargo build --release -p sochdb-tools --target "$target"
        cargo build --release -p sochdb-grpc --target "$target"
        for binary_name in $(get_binary_names "$platform"); do
            cp "target/$target/release/$binary_name" "$bin_dir/" 2>/dev/null || true
        done
    else
        # Native build
        cargo build --release -p sochdb-tools
        cargo build --release -p sochdb-grpc
        for binary_name in $(get_binary_names "$platform"); do
            cp "target/release/$binary_name" "$bin_dir/" 2>/dev/null || true
        done
    fi
    
    # Make executable
    chmod +x "$bin_dir"/* 2>/dev/null || true
    
    echo ""
    echo "✓ Binaries installed in: $bin_dir"
    
    # Verify
    if [[ "$platform" != windows-* ]]; then
        for binary_name in $(get_binary_names "$platform"); do
            if [[ -x "$bin_dir/$binary_name" ]]; then
                echo "✓ $binary_name is executable"
                "$bin_dir/$binary_name" --version 2>/dev/null || true
            fi
        done
    fi
}

main "$@"
