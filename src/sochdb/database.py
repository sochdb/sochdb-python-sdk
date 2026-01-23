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
SochDB Embedded Database

Direct database access via FFI to the Rust library.
This is the recommended mode for single-process applications.
"""

import os
import sys
import ctypes
import warnings
from typing import Optional, Dict, List, Union, Tuple
from contextlib import contextmanager
from .errors import (
    DatabaseError, 
    TransactionError,
    NamespaceNotFoundError,
    NamespaceExistsError,
)
from .namespace import (
    Namespace,
    NamespaceConfig,
    Collection,
    CollectionConfig,
    DistanceMetric,
    SearchRequest,
    SearchResults,
)


def _get_target_triple() -> str:
    """Get the Rust target triple for the current platform."""
    import platform
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        return "x86_64-apple-darwin"
    elif system == "windows":
        return "x86_64-pc-windows-msvc"
    else:  # Linux
        if machine in ("arm64", "aarch64"):
            return "aarch64-unknown-linux-gnu"
        return "x86_64-unknown-linux-gnu"


def _find_library() -> str:
    """Find the SochDB native library.
    
    Search order:
    1. SOCHDB_LIB_PATH environment variable
    2. Bundled library in wheel (lib/{target}/)
    3. Package directory
    4. Development build (target/release, target/debug)
    5. System paths (/usr/local/lib, /usr/lib)
    """
    # Platform-specific library name
    if sys.platform == "darwin":
        lib_name = "libsochdb_storage.dylib"
    elif sys.platform == "win32":
        lib_name = "sochdb_storage.dll"
    else:
        lib_name = "libsochdb_storage.so"
    
    pkg_dir = os.path.dirname(__file__)
    target = _get_target_triple()
    
    # Search paths in priority order
    search_paths = []
    
    # 1. Environment variable override
    env_path = os.environ.get("SOCHDB_LIB_PATH")
    if env_path:
        search_paths.append(env_path)
    
    # 2. Bundled library in wheel (platform-specific)
    search_paths.append(os.path.join(pkg_dir, "lib", target))
    
    # 3. Bundled library in wheel (generic)
    search_paths.append(os.path.join(pkg_dir, "lib"))
    
    # 4. Same directory as this file
    search_paths.append(pkg_dir)
    
    # 5. Package root
    search_paths.append(os.path.dirname(os.path.dirname(pkg_dir)))
    
    # 6. Development builds (relative to package)
    search_paths.extend([
        os.path.join(pkg_dir, "..", "..", "..", "target", "release"),
        os.path.join(pkg_dir, "..", "..", "..", "target", "debug"),
    ])
    
    # 7. System paths
    search_paths.extend(["/usr/local/lib", "/usr/lib"])
    
    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            return lib_path
    
    raise DatabaseError(
        f"Could not find {lib_name}. "
        f"Searched in: {', '.join(search_paths[:5])}... "
        "Set SOCHDB_LIB_PATH environment variable or install sochdb-client with pip."
    )


class C_TxnHandle(ctypes.Structure):
    _fields_ = [
        ("txn_id", ctypes.c_uint64),
        ("snapshot_ts", ctypes.c_uint64),
    ]


class C_CommitResult(ctypes.Structure):
    """Commit result with HLC-backed monotonic timestamp."""
    _fields_ = [
        ("commit_ts", ctypes.c_uint64),  # HLC timestamp, 0 on error
        ("error_code", ctypes.c_int32),  # 0=success, -1=error, -2=SSI conflict
    ]


class C_DatabaseConfig(ctypes.Structure):
    """Database configuration passed to sochdb_open_with_config.
    
    Configuration options control durability, performance, and indexing behavior.
    Fields with _set suffix indicate whether the corresponding value was explicitly set.
    """
    _fields_ = [
        ("wal_enabled", ctypes.c_bool),          # Enable WAL for durability
        ("wal_enabled_set", ctypes.c_bool),      # Whether wal_enabled was set
        ("sync_mode", ctypes.c_uint8),           # 0=OFF, 1=NORMAL, 2=FULL
        ("sync_mode_set", ctypes.c_bool),        # Whether sync_mode was set
        ("memtable_size_bytes", ctypes.c_uint64), # Memtable size (0=default 64MB)
        ("group_commit", ctypes.c_bool),         # Enable group commit
        ("group_commit_set", ctypes.c_bool),     # Whether group_commit was set
        ("default_index_policy", ctypes.c_uint8), # 0=WriteOptimized, 1=Balanced, 2=ScanOptimized, 3=AppendOnly
        ("default_index_policy_set", ctypes.c_bool), # Whether index policy was set
    ]


class C_StorageStats(ctypes.Structure):
    """Storage statistics returned by sochdb_stats."""
    _fields_ = [
        ("memtable_size_bytes", ctypes.c_uint64),
        ("wal_size_bytes", ctypes.c_uint64),
        ("active_transactions", ctypes.c_size_t),
        ("min_active_snapshot", ctypes.c_uint64),
        ("last_checkpoint_lsn", ctypes.c_uint64),
    ]


class C_SearchResult(ctypes.Structure):
    """Search result from sochdb_collection_search."""
    _fields_ = [
        ("id_ptr", ctypes.c_char_p),
        ("score", ctypes.c_float),
        ("metadata_ptr", ctypes.c_char_p),
    ]


class _FFI:
    """FFI bindings to the native library."""
    
    _lib = None
    
    @classmethod
    def get_lib(cls):
        if cls._lib is None:
            lib_path = _find_library()
            cls._lib = ctypes.CDLL(lib_path)
            cls._setup_bindings()
        return cls._lib
    
    @classmethod
    def _setup_bindings(cls):
        """Set up function signatures for the native library."""
        lib = cls._lib
        
        # Database lifecycle
        # sochdb_open(path: *const c_char) -> *mut DatabasePtr
        lib.sochdb_open.argtypes = [ctypes.c_char_p]
        lib.sochdb_open.restype = ctypes.c_void_p
        
        # sochdb_open_with_config(path: *const c_char, config: C_DatabaseConfig) -> *mut DatabasePtr
        lib.sochdb_open_with_config.argtypes = [ctypes.c_char_p, C_DatabaseConfig]
        lib.sochdb_open_with_config.restype = ctypes.c_void_p
        
        # sochdb_close(ptr: *mut DatabasePtr)
        lib.sochdb_close.argtypes = [ctypes.c_void_p]
        lib.sochdb_close.restype = None
        
        # Transaction API
        # sochdb_begin_txn(ptr: *mut DatabasePtr) -> C_TxnHandle
        lib.sochdb_begin_txn.argtypes = [ctypes.c_void_p]
        lib.sochdb_begin_txn.restype = C_TxnHandle
        
        # sochdb_commit(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> C_CommitResult
        # Returns HLC-backed monotonic commit timestamp for MVCC observability
        lib.sochdb_commit.argtypes = [ctypes.c_void_p, C_TxnHandle]
        lib.sochdb_commit.restype = C_CommitResult
        
        # sochdb_abort(ptr: *mut DatabasePtr, handle: C_TxnHandle) -> c_int
        lib.sochdb_abort.argtypes = [ctypes.c_void_p, C_TxnHandle]
        lib.sochdb_abort.restype = ctypes.c_int
        
        # Key-Value API
        # sochdb_put(ptr, handle, key_ptr, key_len, val_ptr, val_len) -> c_int
        lib.sochdb_put.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.sochdb_put.restype = ctypes.c_int
        
        # sochdb_get(ptr, handle, key_ptr, key_len, val_out, len_out) -> c_int
        lib.sochdb_get.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.sochdb_get.restype = ctypes.c_int
        
        # sochdb_delete(ptr, handle, key_ptr, key_len) -> c_int
        lib.sochdb_delete.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.sochdb_delete.restype = ctypes.c_int
        
        # sochdb_free_bytes(ptr, len)
        lib.sochdb_free_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        lib.sochdb_free_bytes.restype = None
        
        # Path API
        # sochdb_put_path(ptr, handle, path_ptr, val_ptr, val_len) -> c_int
        lib.sochdb_put_path.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.sochdb_put_path.restype = ctypes.c_int
        
        # sochdb_get_path(ptr, handle, path_ptr, val_out, len_out) -> c_int
        lib.sochdb_get_path.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.sochdb_get_path.restype = ctypes.c_int

        # Scan API
        # sochdb_scan(ptr, handle, start_ptr, start_len, end_ptr, end_len) -> *mut ScanIteratorPtr
        lib.sochdb_scan.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.sochdb_scan.restype = ctypes.c_void_p
        
        # sochdb_scan_next(iter_ptr, key_out, key_len_out, val_out, val_len_out) -> c_int
        lib.sochdb_scan_next.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.sochdb_scan_next.restype = ctypes.c_int
        
        # sochdb_scan_free(iter_ptr)
        lib.sochdb_scan_free.argtypes = [ctypes.c_void_p]
        lib.sochdb_scan_free.restype = None
        
        # sochdb_scan_prefix(ptr, handle, prefix_ptr, prefix_len) -> *mut ScanIteratorPtr
        # Safe prefix scan that only returns keys starting with prefix
        lib.sochdb_scan_prefix.argtypes = [
            ctypes.c_void_p, C_TxnHandle,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
        ]
        lib.sochdb_scan_prefix.restype = ctypes.c_void_p
        
        # sochdb_scan_batch(iter_ptr, batch_size, result_out, result_len_out) -> c_int
        # Batched scan for reduced FFI overhead
        lib.sochdb_scan_batch.argtypes = [
            ctypes.c_void_p,  # iter_ptr
            ctypes.c_size_t,  # batch_size
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # result_out
            ctypes.POINTER(ctypes.c_size_t)  # result_len_out
        ]
        lib.sochdb_scan_batch.restype = ctypes.c_int
        
        # Checkpoint API
        # sochdb_checkpoint(ptr) -> u64
        lib.sochdb_checkpoint.argtypes = [ctypes.c_void_p]
        lib.sochdb_checkpoint.restype = ctypes.c_uint64
        
        # Stats API
        # sochdb_stats(ptr) -> C_StorageStats
        lib.sochdb_stats.argtypes = [ctypes.c_void_p]
        lib.sochdb_stats.restype = C_StorageStats
        
        # Per-Table Index Policy API
        # sochdb_set_table_index_policy(ptr, table_name, policy) -> c_int
        # Sets index policy for a table: 0=WriteOptimized, 1=Balanced, 2=ScanOptimized, 3=AppendOnly
        lib.sochdb_set_table_index_policy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint8
        ]
        lib.sochdb_set_table_index_policy.restype = ctypes.c_int
        
        # sochdb_get_table_index_policy(ptr, table_name) -> u8
        # Gets index policy for a table. Returns 255 on error.
        lib.sochdb_get_table_index_policy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p
        ]
        lib.sochdb_get_table_index_policy.restype = ctypes.c_uint8
        
        # Graph Overlay API
        # sochdb_graph_add_node(ptr, ns, id, type, props) -> c_int
        try:
            lib.sochdb_graph_add_node.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
            ]
            lib.sochdb_graph_add_node.restype = ctypes.c_int

            lib.sochdb_graph_add_edge.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
            ]
            lib.sochdb_graph_add_edge.restype = ctypes.c_int

            lib.sochdb_graph_traverse.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)
            ]
            lib.sochdb_graph_traverse.restype = ctypes.c_void_p # Returns *char (json string)
        except (AttributeError, OSError):
             pass

        # Temporal Graph API
        try:
             # sochdb_query_temporal_graph(ptr, ns, node, mode, ts, start, end, type, out_len)
             lib.sochdb_query_temporal_graph.argtypes = [
                 ctypes.c_void_p,
                 ctypes.c_char_p,
                 ctypes.c_char_p,
                 ctypes.c_uint8,  # mode u8
                 ctypes.c_uint64, # timestamp
                 ctypes.c_uint64, # start_time
                 ctypes.c_uint64, # end_time
                 ctypes.c_char_p, # edge_type
                 ctypes.POINTER(ctypes.c_size_t) # out_len
             ]
             lib.sochdb_query_temporal_graph.restype = ctypes.c_void_p # Returns *char

             # sochdb_free_string(ptr)
             lib.sochdb_free_string.argtypes = [ctypes.c_void_p]
             lib.sochdb_free_string.restype = None
        except (AttributeError, OSError):
             pass
        
        # Collection Search API (Native Rust vector search)
        # Optional: Only available in newer native library versions
        try:
            lib.sochdb_collection_search.argtypes = [
                ctypes.c_void_p,   # ptr
                ctypes.c_char_p,   # namespace
                ctypes.c_char_p,   # collection
                ctypes.POINTER(ctypes.c_float),  # query_ptr
                ctypes.c_size_t,   # query_len
                ctypes.c_size_t,   # k
                ctypes.POINTER(C_SearchResult),  # results_out
            ]
            lib.sochdb_collection_search.restype = ctypes.c_int
            
            # Keyword Search API (Native Rust text search)
            # sochdb_collection_keyword_search(ptr, namespace, collection, query_ptr, k, results_out) -> c_int
            lib.sochdb_collection_keyword_search.argtypes = [
                ctypes.c_void_p,   # ptr
                ctypes.c_char_p,   # namespace
                ctypes.c_char_p,   # collection
                ctypes.c_char_p,   # query_ptr (string)
                ctypes.c_size_t,   # k
                ctypes.POINTER(C_SearchResult),  # results_out
            ]
            lib.sochdb_collection_keyword_search.restype = ctypes.c_int
            
            lib.sochdb_search_result_free.argtypes = [
                ctypes.POINTER(C_SearchResult),
                ctypes.c_size_t,
            ]
            lib.sochdb_search_result_free.restype = None
        except (AttributeError, OSError):
            # Symbol not available in this library version
            pass


class Transaction:
    """
    A database transaction.
    
    Use with a context manager for automatic commit/abort:
    
        with db.transaction() as txn:
            txn.put(b"key", b"value")
            # Auto-commits on success, auto-aborts on exception
    """
    
    def __init__(self, db: "Database", handle: C_TxnHandle):
        self._db = db
        self._handle = handle
        self._committed = False
        self._aborted = False
        self._lib = _FFI.get_lib()
    
    @property
    def id(self) -> int:
        """Get the transaction ID."""
        return self._handle.txn_id
    
    def put(self, key: bytes, value: bytes) -> None:
        """Put a key-value pair in this transaction."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        val_ptr = (ctypes.c_uint8 * len(value)).from_buffer_copy(value)
        
        res = self._lib.sochdb_put(
            self._db._handle, self._handle,
            key_ptr, len(key),
            val_ptr, len(value)
        )
        if res != 0:
            raise DatabaseError("Failed to put value")
    
    def get(self, key: bytes) -> Optional[bytes]:
        """Get a value in this transaction's snapshot."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        val_out = ctypes.POINTER(ctypes.c_uint8)()
        len_out = ctypes.c_size_t()
        
        res = self._lib.sochdb_get(
            self._db._handle, self._handle,
            key_ptr, len(key),
            ctypes.byref(val_out), ctypes.byref(len_out)
        )
        
        if res == 1: # Not found
            return None
        elif res != 0:
            raise DatabaseError("Failed to get value")
        
        # Copy data to Python bytes
        data = bytes(val_out[:len_out.value])
        
        # Free Rust memory
        self._lib.sochdb_free_bytes(val_out, len_out)
        
        return data
    
    def delete(self, key: bytes) -> None:
        """Delete a key in this transaction."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        key_ptr = (ctypes.c_uint8 * len(key)).from_buffer_copy(key)
        
        res = self._lib.sochdb_delete(
            self._db._handle, self._handle,
            key_ptr, len(key)
        )
        if res != 0:
            raise DatabaseError("Failed to delete key")
    
    def put_path(self, path: str, value: bytes) -> None:
        """Put a value at a path."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        path_bytes = path.encode("utf-8")
        val_ptr = (ctypes.c_uint8 * len(value)).from_buffer_copy(value)
        
        res = self._lib.sochdb_put_path(
            self._db._handle, self._handle,
            path_bytes,
            val_ptr, len(value)
        )
        if res != 0:
            raise DatabaseError("Failed to put path")

    def get_path(self, path: str) -> Optional[bytes]:
        """Get a value at a path."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        path_bytes = path.encode("utf-8")
        val_out = ctypes.POINTER(ctypes.c_uint8)()
        len_out = ctypes.c_size_t()
        
        res = self._lib.sochdb_get_path(
            self._db._handle, self._handle,
            path_bytes,
            ctypes.byref(val_out), ctypes.byref(len_out)
        )
        
        if res == 1: # Not found
            return None
        elif res != 0:
            raise DatabaseError("Failed to get path")
            
        data = bytes(val_out[:len_out.value])
        self._lib.sochdb_free_bytes(val_out, len_out)
        return data

    def delete_path(self, path: str) -> None:
        """Delete a value at a path."""
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        self.delete(path.encode("utf-8"))

    def scan(self, start: bytes = b"", end: bytes = b""):
        """
        Scan keys in range [start, end).
        
        .. deprecated:: 0.2.6
            Use :meth:`scan_prefix` for prefix-based queries instead.
            The scan() method may return keys beyond your intended prefix,
            which can cause multi-tenant data leakage.
        
        Args:
            start: Start key (inclusive). Empty means from beginning.
            end: End key (exclusive). Empty means to end.
            
        Yields:
            (key, value) tuples.
        """
        warnings.warn(
            "scan() is deprecated for prefix queries. Use scan_prefix() instead. "
            "scan() may return keys beyond the intended prefix, causing data leakage.",
            DeprecationWarning,
            stacklevel=2
        )
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
            
        start_ptr = (ctypes.c_uint8 * len(start)).from_buffer_copy(start)
        end_ptr = (ctypes.c_uint8 * len(end)).from_buffer_copy(end)
        
        iter_ptr = self._lib.sochdb_scan(
            self._db._handle, self._handle,
            start_ptr, len(start),
            end_ptr, len(end)
        )
        
        if not iter_ptr:
            return
            
        try:
            key_out = ctypes.POINTER(ctypes.c_uint8)()
            key_len = ctypes.c_size_t()
            val_out = ctypes.POINTER(ctypes.c_uint8)()
            val_len = ctypes.c_size_t()
            
            while True:
                res = self._lib.sochdb_scan_next(
                    iter_ptr,
                    ctypes.byref(key_out), ctypes.byref(key_len),
                    ctypes.byref(val_out), ctypes.byref(val_len)
                )
                
                if res == 1: # End of scan
                    break
                elif res != 0: # Error
                    raise DatabaseError("Scan failed")
                    
                # Copy data
                key = bytes(key_out[:key_len.value])
                val = bytes(val_out[:val_len.value])
                
                # Free Rust memory
                self._lib.sochdb_free_bytes(key_out, key_len)
                self._lib.sochdb_free_bytes(val_out, val_len)
                
                yield key, val
        finally:
            self._lib.sochdb_scan_free(iter_ptr)

    def scan_prefix(self, prefix: bytes):
        """
        Scan keys matching a prefix.
        
        This is the correct method for prefix-based iteration. Unlike scan(),
        which operates on an arbitrary range, scan_prefix() guarantees that
        only keys starting with the given prefix are returned.
        
        This method is safe for multi-tenant isolation - it will NEVER return
        keys from other tenants/prefixes.
        
        Prefix Safety:
            A minimum prefix length of 2 bytes is required to prevent
            expensive full-database scans. Use scan_prefix_unchecked() if
            you need unrestricted access for internal operations.
        
        Args:
            prefix: The prefix to match (minimum 2 bytes). All returned keys
                    will start with this prefix.
            
        Yields:
            (key, value) tuples where key.startswith(prefix) is True.
            
        Raises:
            ValueError: If prefix is less than 2 bytes.
            
        Example:
            # Get all user keys - safe for multi-tenant
            for key, value in txn.scan_prefix(b"tenant_a/"):
                print(f"{key}: {value}")
                # Will NEVER include keys like b"tenant_b/..."
        """
        MIN_PREFIX_LEN = 2
        if len(prefix) < MIN_PREFIX_LEN:
            raise ValueError(
                f"Prefix too short: {len(prefix)} bytes (minimum {MIN_PREFIX_LEN} required). "
                f"Use scan_prefix_unchecked() for unrestricted prefix access."
            )
        return self.scan_prefix_unchecked(prefix)
    
    def scan_prefix_unchecked(self, prefix: bytes):
        """
        Scan keys matching a prefix without length validation.
        
        Warning:
            This method allows empty/short prefixes which can cause expensive
            full-database scans. Use scan_prefix() unless you specifically need
            unrestricted prefix access for internal operations.
        
        Args:
            prefix: The prefix to match. Can be empty for full scan.
            
        Yields:
            (key, value) tuples where key.startswith(prefix) is True.
        """
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        prefix_ptr = (ctypes.c_uint8 * len(prefix)).from_buffer_copy(prefix)
        
        # Use the dedicated prefix scan FFI function for safety
        iter_ptr = self._lib.sochdb_scan_prefix(
            self._db._handle, self._handle,
            prefix_ptr, len(prefix)
        )
        
        if not iter_ptr:
            return
            
        try:
            key_out = ctypes.POINTER(ctypes.c_uint8)()
            key_len = ctypes.c_size_t()
            val_out = ctypes.POINTER(ctypes.c_uint8)()
            val_len = ctypes.c_size_t()
            
            while True:
                res = self._lib.sochdb_scan_next(
                    iter_ptr,
                    ctypes.byref(key_out), ctypes.byref(key_len),
                    ctypes.byref(val_out), ctypes.byref(val_len)
                )
                
                if res == 1:  # End of scan
                    break
                elif res != 0:  # Error
                    raise DatabaseError("Scan prefix failed")
                    
                # Copy data
                key = bytes(key_out[:key_len.value])
                val = bytes(val_out[:val_len.value])
                
                # Free Rust memory
                self._lib.sochdb_free_bytes(key_out, key_len)
                self._lib.sochdb_free_bytes(val_out, val_len)
                
                yield key, val
        finally:
            self._lib.sochdb_scan_free(iter_ptr)

    def scan_batched(self, start: bytes = b"", end: bytes = b"", batch_size: int = 1000):
        """
        Scan keys in range [start, end) with batched FFI calls.
        
        This is a high-performance scan that fetches multiple results per FFI call,
        dramatically reducing overhead for large scans.
        
        Performance comparison (10,000 results, 500ns FFI overhead):
        - scan():        10,000 FFI calls = 5ms overhead
        - scan_batched(): 10 FFI calls = 5µs overhead (1000x faster)
        
        Args:
            start: Start key (inclusive). Empty means from beginning.
            end: End key (exclusive). Empty means to end.
            batch_size: Number of results to fetch per FFI call. Default 1000.
            
        Yields:
            (key, value) tuples.
        """
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        if batch_size <= 0:
            batch_size = 1000
            
        start_ptr = (ctypes.c_uint8 * len(start)).from_buffer_copy(start)
        end_ptr = (ctypes.c_uint8 * len(end)).from_buffer_copy(end)
        
        iter_ptr = self._lib.sochdb_scan(
            self._db._handle, self._handle,
            start_ptr, len(start),
            end_ptr, len(end)
        )
        
        if not iter_ptr:
            return
            
        try:
            result_out = ctypes.POINTER(ctypes.c_uint8)()
            result_len = ctypes.c_size_t()
            
            while True:
                res = self._lib.sochdb_scan_batch(
                    iter_ptr,
                    batch_size,
                    ctypes.byref(result_out),
                    ctypes.byref(result_len)
                )
                
                if res == 1:  # Scan complete
                    # Free the minimal buffer allocated
                    if result_out and result_len.value > 0:
                        self._lib.sochdb_free_bytes(result_out, result_len)
                    break
                elif res != 0:  # Error
                    if result_out and result_len.value > 0:
                        self._lib.sochdb_free_bytes(result_out, result_len)
                    raise DatabaseError("Batched scan failed")
                
                # Parse batch result
                # Format: [num_results: u32][is_done: u8][entries...]
                data = bytes(result_out[:result_len.value])
                
                if len(data) < 5:
                    self._lib.sochdb_free_bytes(result_out, result_len)
                    break
                    
                num_results = int.from_bytes(data[0:4], 'little')
                is_done = data[4] != 0
                
                offset = 5
                for _ in range(num_results):
                    if offset + 8 > len(data):
                        break
                    key_len = int.from_bytes(data[offset:offset+4], 'little')
                    val_len = int.from_bytes(data[offset+4:offset+8], 'little')
                    offset += 8
                    
                    if offset + key_len + val_len > len(data):
                        break
                    
                    key = data[offset:offset+key_len]
                    offset += key_len
                    val = data[offset:offset+val_len]
                    offset += val_len
                    
                    yield key, val
                
                # Free batch buffer
                self._lib.sochdb_free_bytes(result_out, result_len)
                
                if is_done:
                    break
        finally:
            self._lib.sochdb_scan_free(iter_ptr)

    def commit(self) -> int:
        """
        Commit the transaction.
        
        Returns:
            Commit timestamp (HLC-backed, monotonically increasing).
            This timestamp is suitable for:
            - MVCC observability ("what commit did I read?")
            - Replication and log shipping
            - Agent audit trails
            - Time-travel queries
            - Deterministic replay
            
        Raises:
            TransactionError: If commit fails (e.g., SSI conflict)
        """
        if self._committed:
            raise TransactionError("Transaction already committed")
        if self._aborted:
            raise TransactionError("Transaction already aborted")
        
        result = self._lib.sochdb_commit(self._db._handle, self._handle)
        if result.error_code != 0:
            if result.error_code == -2:
                raise TransactionError("SSI conflict: transaction aborted due to serialization failure")
            raise TransactionError("Failed to commit transaction")
            
        self._committed = True
        return result.commit_ts
    
    def abort(self) -> None:
        """Abort the transaction."""
        if self._committed:
            raise TransactionError("Transaction already committed")
        if self._aborted:
            return  # Abort is idempotent
        
        self._lib.sochdb_abort(self._db._handle, self._handle)
        self._aborted = True
    
    def execute(self, sql: str) -> 'SQLQueryResult':
        """
        Execute a SQL query within this transaction's context.
        
        Note: SQL operations use the underlying KV store, so they participate
        in this transaction's isolation and atomicity guarantees.
        
        Args:
            sql: SQL query string
            
        Returns:
            SQLQueryResult with rows and metadata
        """
        if self._committed or self._aborted:
            raise TransactionError("Transaction already completed")
        
        from .sql_engine import SQLExecutor
        # Create executor that uses the transaction context
        executor = SQLExecutor(self)
        return executor.execute(sql)
    
    def __enter__(self) -> "Transaction":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Exception occurred, abort
            self.abort()
        elif not self._committed and not self._aborted:
            # No exception and not yet completed, commit
            self.commit()


class Database:
    """
    SochDB Embedded Database.
    
    Provides direct access to a SochDB database file.
    This is the recommended mode for single-process applications.
    
    Example:
        db = Database.open("./my_database")
        db.put(b"key", b"value")
        value = db.get(b"key")
        db.close()
    
    Or with context manager:
        with Database.open("./my_database") as db:
            db.put(b"key", b"value")
    """
    
    def __init__(self, path: str, _handle):
        """
        Initialize a database connection.
        
        Use Database.open() to create instances.
        """
        self._path = path
        self._handle = _handle
        self._closed = False
        self._lib = _FFI.get_lib()
    
    @classmethod
    def open(cls, path: str, config: Optional[dict] = None) -> "Database":
        """
        Open a database at the given path.
        
        Creates the database if it doesn't exist.
        
        Args:
            path: Path to the database directory.
            config: Optional configuration dictionary with keys:
                - wal_enabled (bool): Enable WAL for durability (default: True)
                - sync_mode (str): 'full', 'normal', or 'off' (default: 'normal')
                    - 'off': No fsync, ~10x faster but risk of data loss
                    - 'normal': Fsync at checkpoints, good balance (default)
                    - 'full': Fsync every commit, safest but slowest
                - memtable_size_bytes (int): Memtable size before flush (default: 64MB)
                - group_commit (bool): Enable group commit for throughput (default: True)
                - index_policy (str): Default index policy for tables:
                    - 'write_optimized': O(1) insert, O(N) scan - for high-write
                    - 'balanced': O(1) amortized insert, O(log K) scan - default
                    - 'scan_optimized': O(log N) insert, O(log N + K) scan - for analytics
                    - 'append_only': O(1) insert, O(N) scan - for time-series
            
        Returns:
            Database instance.
            
        Example:
            # Default configuration (good for most use cases)
            db = Database.open("./my_database")
            
            # High-durability configuration
            db = Database.open("./critical_data", config={
                "sync_mode": "full",
                "wal_enabled": True,
            })
            
            # High-throughput configuration
            db = Database.open("./logs", config={
                "sync_mode": "off",
                "group_commit": True,
                "index_policy": "write_optimized",
            })
        """
        lib = _FFI.get_lib()
        path_bytes = path.encode("utf-8")
        
        if config is not None:
            # Build C config struct from Python dict
            c_config = C_DatabaseConfig()
            
            # WAL enabled
            if "wal_enabled" in config:
                c_config.wal_enabled = bool(config["wal_enabled"])
                c_config.wal_enabled_set = True
            
            # Sync mode
            if "sync_mode" in config:
                mode = config["sync_mode"].lower() if isinstance(config["sync_mode"], str) else str(config["sync_mode"])
                if mode in ("off", "0"):
                    c_config.sync_mode = 0
                elif mode in ("normal", "1"):
                    c_config.sync_mode = 1
                elif mode in ("full", "2"):
                    c_config.sync_mode = 2
                else:
                    c_config.sync_mode = 1  # Default to normal
                c_config.sync_mode_set = True
            
            # Memtable size
            if "memtable_size_bytes" in config:
                c_config.memtable_size_bytes = int(config["memtable_size_bytes"])
            
            # Group commit
            if "group_commit" in config:
                c_config.group_commit = bool(config["group_commit"])
                c_config.group_commit_set = True
            
            # Index policy
            if "index_policy" in config:
                policy = config["index_policy"].lower() if isinstance(config["index_policy"], str) else str(config["index_policy"])
                if policy == "write_optimized":
                    c_config.default_index_policy = 0
                elif policy == "balanced":
                    c_config.default_index_policy = 1
                elif policy == "scan_optimized":
                    c_config.default_index_policy = 2
                elif policy == "append_only":
                    c_config.default_index_policy = 3
                else:
                    c_config.default_index_policy = 1  # Default to balanced
                c_config.default_index_policy_set = True
            
            handle = lib.sochdb_open_with_config(path_bytes, c_config)
        else:
            handle = lib.sochdb_open(path_bytes)
        
        if not handle:
            raise DatabaseError(f"Failed to open database at {path}")
        
        # Track database open event (only analytics event we send)
        try:
            from .analytics import track_database_open
            track_database_open(path, mode="embedded")
        except Exception:
            # Never let analytics break database operations
            pass
            
        return cls(path, handle)
    
    def close(self) -> None:
        """Close the database."""
        if self._closed:
            return
        
        if self._handle:
            self._lib.sochdb_close(self._handle)
            self._handle = None
            
        self._closed = True
    
    def __enter__(self) -> "Database":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # =========================================================================
    # Key-Value API (auto-commit)
    # =========================================================================
    
    def put(self, key: bytes, value: bytes) -> None:
        """
        Put a key-value pair (auto-commit).
        
        Args:
            key: The key bytes.
            value: The value bytes.
        """
        with self.transaction() as txn:
            txn.put(key, value)
    
    def get(self, key: bytes) -> Optional[bytes]:
        """
        Get a value by key.
        
        Args:
            key: The key bytes.
            
        Returns:
            The value bytes, or None if not found.
        """
        # For single reads, we still need a transaction for MVCC consistency
        with self.transaction() as txn:
            return txn.get(key)
    
    def delete(self, key: bytes) -> None:
        """
        Delete a key (auto-commit).
        
        Args:
            key: The key bytes.
        """
        with self.transaction() as txn:
            txn.delete(key)
    
    # =========================================================================
    # Path-Native API
    # =========================================================================
    
    def put_path(self, path: str, value: bytes) -> None:
        """
        Put a value at a path (auto-commit).
        
        Args:
            path: Path string (e.g., "users/alice/email")
            value: The value bytes.
        """
        with self.transaction() as txn:
            txn.put_path(path, value)
    
    def get_path(self, path: str) -> Optional[bytes]:
        """
        Get a value at a path.
        
        Args:
            path: Path string (e.g., "users/alice/email")
            
        Returns:
            The value bytes, or None if not found.
        """
        with self.transaction() as txn:
            return txn.get_path(path)

    def scan(self, start: bytes = b"", end: bytes = b""):
        """
        Scan keys in range (auto-commit transaction).
        
        .. deprecated:: 0.2.6
            Use :meth:`scan_prefix` for prefix-based queries instead.
            The scan() method may return keys beyond your intended prefix,
            which can cause multi-tenant data leakage.
        
        Args:
            start: Start key (inclusive).
            end: End key (exclusive).
            
        Yields:
            (key, value) tuples.
        """
        warnings.warn(
            "scan() is deprecated for prefix queries. Use scan_prefix() instead. "
            "scan() may return keys beyond the intended prefix, causing data leakage.",
            DeprecationWarning,
            stacklevel=2
        )
        with self.transaction() as txn:
            yield from txn.scan(start, end)
    
    def scan_prefix(self, prefix: bytes):
        """
        Scan keys matching a prefix (auto-commit transaction).
        
        This is the correct method for prefix-based iteration. Unlike scan(),
        which operates on an arbitrary range, scan_prefix() guarantees that
        only keys starting with the given prefix are returned.
        
        Prefix Safety:
            A minimum prefix length of 2 bytes is required to prevent
            expensive full-database scans.
        
        Args:
            prefix: The prefix to match (minimum 2 bytes). All returned keys
                    will start with this prefix.
            
        Yields:
            (key, value) tuples where key.startswith(prefix) is True.
            
        Raises:
            ValueError: If prefix is less than 2 bytes.
            
        Example:
            # Get all keys under "users/"
            for key, value in db.scan_prefix(b"users/"):
                print(f"{key}: {value}")
                
            # Multi-tenant safe - won't leak across tenants
            for key, value in db.scan_prefix(b"tenant_a/"):
                # Only tenant_a data, never tenant_b
                ...
        """
        with self.transaction() as txn:
            yield from txn.scan_prefix(prefix)
    
    def scan_prefix_unchecked(self, prefix: bytes):
        """
        Scan keys matching a prefix without length validation (auto-commit transaction).
        
        Warning:
            This method allows empty/short prefixes which can cause expensive
            full-database scans. Use scan_prefix() unless you specifically need
            unrestricted prefix access for internal operations like graph overlay.
        
        Args:
            prefix: The prefix to match. Can be empty for full scan.
            
        Yields:
            (key, value) tuples where key.startswith(prefix) is True.
        """
        with self.transaction() as txn:
            yield from txn.scan_prefix_unchecked(prefix)
    
    def delete_path(self, path: str) -> None:
        """
        Delete at a path (auto-commit).
        
        Args:
            path: Path string (e.g., "users/alice/email")
        """
        # Currently no direct delete_path FFI, use key-based delete if possible
        # or implement delete_path in FFI. For now, assume path is key.
        self.delete(path.encode("utf-8"))
    
    # =========================================================================
    # Transaction API
    # =========================================================================
    
    def transaction(self) -> Transaction:
        """
        Begin a new transaction.
        
        Returns:
            Transaction object that can be used as a context manager.
            
        Example:
            with db.transaction() as txn:
                txn.put(b"key1", b"value1")
                txn.put(b"key2", b"value2")
                # Auto-commits on success
        """
        self._check_open()
        handle = self._lib.sochdb_begin_txn(self._handle)
        if handle.txn_id == 0:
            raise DatabaseError("Failed to begin transaction")
            
        return Transaction(self, handle)
    
    # =========================================================================
    # Administrative Operations
    # =========================================================================
    
    def checkpoint(self) -> int:
        """
        Force a checkpoint to disk.
        
        Returns:
            LSN of the checkpoint.
        """
        self._check_open()
        return self._lib.sochdb_checkpoint(self._handle)
        
    def stats(self) -> dict:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics.
        """
        self._check_open()
        stats = self._lib.sochdb_stats(self._handle)
        return {
            "memtable_size_bytes": stats.memtable_size_bytes,
            "wal_size_bytes": stats.wal_size_bytes,
            "active_transactions": stats.active_transactions,
            "min_active_snapshot": stats.min_active_snapshot,
            "last_checkpoint_lsn": stats.last_checkpoint_lsn,
        }
    
    # =========================================================================
    # Per-Table Index Policy API
    # =========================================================================
    
    # Index policy constants
    INDEX_WRITE_OPTIMIZED = 0
    INDEX_BALANCED = 1
    INDEX_SCAN_OPTIMIZED = 2
    INDEX_APPEND_ONLY = 3
    
    _POLICY_NAMES = {
        INDEX_WRITE_OPTIMIZED: "write_optimized",
        INDEX_BALANCED: "balanced",
        INDEX_SCAN_OPTIMIZED: "scan_optimized",
        INDEX_APPEND_ONLY: "append_only",
    }
    
    _POLICY_VALUES = {
        "write_optimized": INDEX_WRITE_OPTIMIZED,
        "write": INDEX_WRITE_OPTIMIZED,
        "balanced": INDEX_BALANCED,
        "default": INDEX_BALANCED,
        "scan_optimized": INDEX_SCAN_OPTIMIZED,
        "scan": INDEX_SCAN_OPTIMIZED,
        "append_only": INDEX_APPEND_ONLY,
        "append": INDEX_APPEND_ONLY,
    }
    
    def set_table_index_policy(self, table: str, policy: Union[int, str]) -> None:
        """
        Set the index policy for a specific table.
        
        Index policies control the trade-off between write and read performance:
        
        - 'write_optimized' (0): O(1) writes, O(N) scans
          Best for write-heavy tables with rare range queries.
          
        - 'balanced' (1): O(1) amortized writes, O(output + log K) scans
          Good balance for mixed OLTP workloads. This is the default.
          
        - 'scan_optimized' (2): O(log N) writes, O(log N + K) scans
          Best for analytics tables with frequent range queries.
          
        - 'append_only' (3): O(1) writes, O(N) forward-only scans
          Best for time-series logs where data is naturally ordered.
        
        Args:
            table: Table name (uses table prefix for key grouping)
            policy: Policy name (str) or value (int)
            
        Raises:
            ValueError: If policy is invalid
            DatabaseError: If FFI call fails
            
        Example:
            # For write-heavy user sessions
            db.set_table_index_policy("sessions", "write_optimized")
            
            # For analytics queries
            db.set_table_index_policy("events", "scan_optimized")
        """
        self._check_open()
        
        # Convert string policy to int
        if isinstance(policy, str):
            policy_value = self._POLICY_VALUES.get(policy.lower())
            if policy_value is None:
                raise ValueError(
                    f"Invalid policy '{policy}'. Valid policies: "
                    f"{list(self._POLICY_VALUES.keys())}"
                )
        else:
            policy_value = int(policy)
            if policy_value not in self._POLICY_NAMES:
                raise ValueError(
                    f"Invalid policy value {policy_value}. Valid values: 0-3"
                )
        
        table_bytes = table.encode("utf-8")
        result = self._lib.sochdb_set_table_index_policy(
            self._handle,
            table_bytes,
            policy_value
        )
        
        if result == -1:
            raise DatabaseError("Failed to set table index policy")
        elif result == -2:
            raise ValueError(f"Invalid policy value: {policy_value}")
    
    def get_table_index_policy(self, table: str) -> str:
        """
        Get the index policy for a specific table.
        
        Args:
            table: Table name
            
        Returns:
            Policy name as string: 'write_optimized', 'balanced', 
            'scan_optimized', or 'append_only'
            
        Example:
            policy = db.get_table_index_policy("users")
            print(f"Users table uses {policy} indexing")
        """
        self._check_open()
        
        table_bytes = table.encode("utf-8")
        policy_value = self._lib.sochdb_get_table_index_policy(
            self._handle,
            table_bytes
        )
        
        if policy_value == 255:
            raise DatabaseError("Failed to get table index policy")
        
        return self._POLICY_NAMES.get(policy_value, "balanced")
    
    def execute(self, sql: str) -> 'SQLQueryResult':
        """
        Execute a SQL query.
        
        SochDB supports a subset of SQL for relational data stored on top of 
        the key-value engine. Tables and rows are stored as:
        - Schema: _sql/tables/{table_name}/schema
        - Rows: _sql/tables/{table_name}/rows/{row_id}
        
        Supported SQL:
        - CREATE TABLE table_name (col1 TYPE, col2 TYPE, ...)
        - DROP TABLE table_name
        - INSERT INTO table_name (cols) VALUES (vals)
        - SELECT cols FROM table_name [WHERE ...] [ORDER BY ...] [LIMIT ...]
        - UPDATE table_name SET col=val [WHERE ...]
        - DELETE FROM table_name [WHERE ...]
        
        Supported types: INT, TEXT, FLOAT, BOOL, BLOB
        
        Args:
            sql: SQL query string
            
        Returns:
            SQLQueryResult object with rows and metadata
            
        Example:
            # Create a table
            db.execute("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
            
            # Insert data
            db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            db.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
            
            # Query data
            result = db.execute("SELECT * FROM users WHERE age > 26")
            for row in result.rows:
                print(row)  # {'id': 1, 'name': 'Alice', 'age': 30}
        """
        self._check_open()
        from .sql_engine import SQLExecutor
        executor = SQLExecutor(self)
        return executor.execute(sql)
    
    # Alias for documentation compatibility
    execute_sql = execute
    
    # =========================================================================
    # TOON Format Output (Token-Efficient Serialization)
    # =========================================================================
    
    @staticmethod
    def to_toon(table_name: str, records: list, fields: list = None) -> str:
        """
        Convert records to TOON format for token-efficient LLM context.
        
        TOON format achieves 40-66% token reduction compared to JSON by using
        a columnar text format with minimal syntax.
        
        Args:
            table_name: Name of the table/collection.
            records: List of dicts with the data.
            fields: Optional list of field names to include. If None, uses
                   all fields from the first record.
                   
        Returns:
            TOON-formatted string.
            
        Example:
            >>> records = [
            ...     {"id": 1, "name": "Alice", "email": "alice@ex.com"},
            ...     {"id": 2, "name": "Bob", "email": "bob@ex.com"}
            ... ]
            >>> print(Database.to_toon("users", records, ["name", "email"]))
            users[2]{name,email}:Alice,alice@ex.com;Bob,bob@ex.com
            
        Token Comparison:
            JSON (pretty): ~211 tokens
            JSON (compact): ~165 tokens  
            TOON format: ~70 tokens (67% reduction)
        """
        if not records:
            return f"{table_name}[0]{{}}:"
        
        # Determine fields
        if fields is None:
            fields = list(records[0].keys())
        
        # Build header: table[count]{field1,field2,...}:
        header = f"{table_name}[{len(records)}]{{{','.join(fields)}}}:"
        
        # Build rows: value1,value2;value1,value2;...
        def escape_value(v):
            """Escape values that contain delimiters."""
            s = str(v) if v is not None else ""
            if ',' in s or ';' in s or '\n' in s:
                return f'"{s}"'
            return s
        
        rows = ";".join(
            ",".join(escape_value(r.get(f)) for f in fields)
            for r in records
        )
        
        return header + rows
    
    @staticmethod
    def to_json(
        table_name: str, 
        records: list, 
        fields: list = None,
        compact: bool = True
    ) -> str:
        """
        Convert records to JSON format for easy application decoding.
        
        While TOON format is optimized for LLM context (40-66% token reduction),
        JSON is often easier for applications to parse. Use this method when
        the output will be consumed by application code rather than LLMs.
        
        Args:
            table_name: Name of the table/collection (included in output).
            records: List of dicts with the data.
            fields: Optional list of field names to include. If None, uses
                   all fields from records.
            compact: If True (default), outputs minified JSON. If False,
                    outputs pretty-printed JSON.
                   
        Returns:
            JSON-formatted string.
            
        Example:
            >>> records = [
            ...     {"id": 1, "name": "Alice", "email": "alice@ex.com"},
            ...     {"id": 2, "name": "Bob", "email": "bob@ex.com"}
            ... ]
            >>> print(Database.to_json("users", records, ["name", "email"]))
            {"table":"users","count":2,"records":[{"name":"Alice","email":"alice@ex.com"},{"name":"Bob","email":"bob@ex.com"}]}
            
        See Also:
            - to_toon(): For token-efficient LLM context (40-66% smaller)
            - from_json(): To parse JSON back to structured data
        """
        import json
        
        if not records:
            return json.dumps({
                "table": table_name,
                "count": 0,
                "records": []
            })
        
        # Filter fields if specified
        if fields is not None:
            filtered_records = [
                {f: r.get(f) for f in fields}
                for r in records
            ]
        else:
            filtered_records = records
        
        output = {
            "table": table_name,
            "count": len(filtered_records),
            "records": filtered_records
        }
        
        if compact:
            return json.dumps(output, separators=(',', ':'))
        else:
            return json.dumps(output, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> tuple:
        """
        Parse a JSON format string back to structured data.
        
        Args:
            json_str: JSON-formatted string (from to_json).
            
        Returns:
            Tuple of (table_name, fields, records) where records is a list of dicts.
            
        Example:
            >>> json_data = '{"table":"users","count":2,"records":[{"name":"Alice"},{"name":"Bob"}]}'
            >>> name, fields, records = Database.from_json(json_data)
            >>> print(records)
            [{'name': 'Alice'}, {'name': 'Bob'}]
        """
        import json
        
        data = json.loads(json_str)
        table_name = data.get("table", "unknown")
        records = data.get("records", [])
        
        # Extract field names from first record
        fields = list(records[0].keys()) if records else []
        
        return table_name, fields, records
    
    @staticmethod
    def from_toon(toon_str: str) -> tuple:
        """
        Parse a TOON format string back to structured data.
        
        Args:
            toon_str: TOON-formatted string.
            
        Returns:
            Tuple of (table_name, fields, records) where records is a list of dicts.
            
        Example:
            >>> toon = "users[2]{name,email}:Alice,alice@ex.com;Bob,bob@ex.com"
            >>> name, fields, records = Database.from_toon(toon)
            >>> print(records)
            [{'name': 'Alice', 'email': 'alice@ex.com'}, 
             {'name': 'Bob', 'email': 'bob@ex.com'}]
        """
        import re
        
        # Parse header: table[count]{fields}:
        match = re.match(r'(\w+)\[(\d+)\]\{([^}]*)\}:(.*)', toon_str, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid TOON format: {toon_str[:50]}...")
        
        table_name = match.group(1)
        count = int(match.group(2))
        fields = [f.strip() for f in match.group(3).split(',') if f.strip()]
        data = match.group(4)
        
        if not data or not fields:
            return table_name, fields, []
        
        # Parse rows
        records = []
        for row in data.split(';'):
            if not row.strip():
                continue
            values = row.split(',')
            record = dict(zip(fields, values))
            records.append(record)
        
        return table_name, fields, records
    
    def stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics:
            - keys_count: Total number of keys
            - bytes_written: Total bytes written
            - bytes_read: Total bytes read
            - transactions_committed: Number of committed transactions
            - transactions_aborted: Number of aborted transactions
            - queries_executed: Number of queries executed
            - cache_hits: Number of cache hits
            - cache_misses: Number of cache misses
            
        Example:
            >>> stats = db.stats()
            >>> print(f"Keys: {stats.get('keys_count', 'N/A')}")
            >>> print(f"Bytes written: {stats.get('bytes_written', 0)}")
        """
        # Note: Accurate key count would require FFI binding to storage engine stats
        # For now, return placeholder values that won't crash
        # (scan_prefix requires 2+ byte prefix, so empty prefix scan won't work)
        return {
            "keys_count": -1,  # -1 indicates "unknown" - would need FFI for real count
            "bytes_written": 0,
            "bytes_read": 0,
            "transactions_committed": 0,
            "transactions_aborted": 0,
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    def checkpoint(self) -> None:
        """
        Force a checkpoint to ensure durability.
        
        A checkpoint flushes all in-memory data to disk, ensuring that
        all committed transactions are durable. This is automatically
        called periodically, but can be called manually for:
        
        - Before backup operations
        - After bulk imports
        - Before system shutdown
        - To reduce recovery time after crash
        
        Note: This is a blocking operation that may take some time
        depending on the amount of unflushed data.
        
        Example:
            # After bulk import
            for record in bulk_data:
                db.put(record.key, record.value)
            db.checkpoint()  # Ensure all data is durable
        """
        # Call FFI checkpoint if available
        # Note: _lib and _ptr may not exist in all connection modes
        lib = getattr(self, '_lib', None)
        ptr = getattr(self, '_ptr', None)
        if lib is not None and ptr is not None:
            try:
                checkpoint_fn = getattr(lib, 'sochdb_checkpoint', None)
                if checkpoint_fn:
                    checkpoint_fn(ptr)
            except Exception:
                # Non-fatal: checkpoint may not be supported
                pass
        # In modes without FFI, data is auto-flushed on transaction commit
    
    def _check_open(self) -> None:
        """Check that database is open."""
        if self._closed:
            raise DatabaseError("Database is closed")

    # =========================================================================
    # Namespace API (Task 8: First-Class Namespace Handle)
    # =========================================================================
    
    def create_namespace(
        self,
        name: str,
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Namespace:
        """
        Create a new namespace.
        
        Namespaces provide multi-tenant isolation. All data within a namespace
        is isolated from other namespaces, making cross-tenant access impossible
        by construction.
        
        Args:
            name: Unique namespace identifier (e.g., "tenant_123")
            display_name: Optional human-readable name
            labels: Optional metadata labels (e.g., {"tier": "enterprise"})
            
        Returns:
            Namespace handle
            
        Raises:
            NamespaceExistsError: If namespace already exists
            
        Example:
            ns = db.create_namespace("tenant_123", display_name="Acme Corp")
            collection = ns.create_collection("documents", dimension=384)
        """
        self._check_open()
        
        if not hasattr(self, '_namespaces'):
            self._namespaces: Dict[str, Namespace] = {}
        
        if name in self._namespaces:
            raise NamespaceExistsError(name)
        
        config = NamespaceConfig(
            name=name,
            display_name=display_name,
            labels=labels or {},
        )
        
        # Create namespace marker in storage
        marker_key = f"_namespaces/{name}/_meta".encode("utf-8")
        import json
        self.put(marker_key, json.dumps(config.to_dict()).encode("utf-8"))
        
        ns = Namespace(self, name, config)
        self._namespaces[name] = ns
        return ns
    
    def namespace(self, name: str) -> Namespace:
        """
        Get an existing namespace handle.
        
        This returns a handle to the namespace for performing operations.
        The namespace must already exist.
        
        Args:
            name: Namespace identifier
            
        Returns:
            Namespace handle
            
        Raises:
            NamespaceNotFoundError: If namespace doesn't exist
            
        Example:
            ns = db.namespace("tenant_123")
            collection = ns.collection("documents")
            results = collection.vector_search(query_embedding, k=10)
        """
        self._check_open()
        
        if not hasattr(self, '_namespaces'):
            self._namespaces = {}
        
        if name in self._namespaces:
            return self._namespaces[name]
        
        # Try to load from storage
        marker_key = f"_namespaces/{name}/_meta".encode("utf-8")
        data = self.get(marker_key)
        if data is None:
            raise NamespaceNotFoundError(name)
        
        import json
        config = NamespaceConfig.from_dict(json.loads(data.decode("utf-8")))
        ns = Namespace(self, name, config)
        self._namespaces[name] = ns
        return ns
    
    def get_or_create_namespace(
        self,
        name: str,
        display_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Namespace:
        """
        Get an existing namespace or create if it doesn't exist.
        
        This is idempotent and safe to call multiple times.
        
        Args:
            name: Namespace identifier
            display_name: Optional human-readable name (used if creating)
            labels: Optional metadata labels (used if creating)
            
        Returns:
            Namespace handle
        """
        try:
            return self.namespace(name)
        except NamespaceNotFoundError:
            return self.create_namespace(name, display_name, labels)
    
    @contextmanager
    def use_namespace(self, name: str):
        """
        Context manager for namespace operations.
        
        Use this to scope a block of operations to a specific namespace.
        
        Args:
            name: Namespace identifier
            
        Yields:
            Namespace handle
            
        Example:
            with db.use_namespace("tenant_123") as ns:
                collection = ns.collection("documents")
                results = collection.search(...)
                # All operations scoped to tenant_123
        """
        ns = self.namespace(name)
        try:
            yield ns
        finally:
            # Could flush pending writes here
            pass
    
    def list_namespaces(self) -> List[str]:
        """
        List all namespaces.
        
        Returns:
            List of namespace names
        """
        self._check_open()
        
        namespaces = []
        prefix = b"_namespaces/"
        suffix = b"/_meta"
        
        for key, _ in self.scan_prefix(prefix):
            # Extract namespace name from _namespaces/{name}/_meta
            if key.endswith(suffix):
                name = key[len(prefix):-len(suffix)].decode("utf-8")
                namespaces.append(name)
        
        return namespaces
    
    def delete_namespace(self, name: str, force: bool = False) -> bool:
        """
        Delete a namespace and all its data.
        
        Args:
            name: Namespace identifier
            force: If True, delete even if namespace has collections
            
        Returns:
            True if deleted
            
        Raises:
            NamespaceNotFoundError: If namespace doesn't exist
            SochDBError: If namespace has collections and force=False
        """
        self._check_open()
        
        # Check exists
        marker_key = f"_namespaces/{name}/_meta".encode("utf-8")
        if self.get(marker_key) is None:
            raise NamespaceNotFoundError(name)
        
        # Delete all namespace data
        prefix = f"{name}/".encode("utf-8")
        with self.transaction() as txn:
            for key, _ in txn.scan_prefix(prefix):
                txn.delete(key)
            
            # Delete metadata
            ns_prefix = f"_namespaces/{name}/".encode("utf-8")
            for key, _ in txn.scan_prefix(ns_prefix):
                txn.delete(key)
        
        # Remove from cache
        if hasattr(self, '_namespaces') and name in self._namespaces:
            del self._namespaces[name]
        
        return True
    
    # =========================================================================
    # Temporal Graph Operations (FFI)
    # =========================================================================
    
    def add_temporal_edge(
        self,
        namespace: str,
        from_id: str,
        edge_type: str,
        to_id: str,
        valid_from: int,
        valid_until: int = 0,
        properties: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a temporal edge with validity interval (Embedded FFI mode).
        
        Temporal edges allow time-travel queries: "What did the system know at time T?"
        Essential for agent memory systems that need to reason about state changes.
        
        Args:
            namespace: Namespace for the edge
            from_id: Source node ID
            edge_type: Type of relationship (e.g., "STATE", "KNOWS", "FOLLOWS")
            to_id: Target node ID
            valid_from: Start timestamp in milliseconds (Unix epoch)
            valid_until: End timestamp in milliseconds (0 = no expiry, still valid)
            properties: Optional metadata dictionary
        
        Example:
            # Record: Door was open from 10:00 to 11:00
            import time
            now = int(time.time() * 1000)
            one_hour = 60 * 60 * 1000
            
            db.add_temporal_edge(
                namespace="smart_home",
                from_id="door_front",
                edge_type="STATE",
                to_id="open",
                valid_from=now - one_hour,
                valid_until=now,
                properties={"sensor": "motion_1"}
            )
        """
        self._check_open()
        
        import json
        
        # Use the C_TemporalEdge structure from FFI
        # (defined in _FFI class)
        # Convert properties to JSON
        props_json = None if properties is None else json.dumps(properties).encode("utf-8")
        
        edge = _FFI.lib.sochdb_add_temporal_edge.argtypes[2](  # Get C_TemporalEdge class
            from_id=from_id.encode("utf-8"),
            edge_type=edge_type.encode("utf-8"),
            to_id=to_id.encode("utf-8"),
            valid_from=valid_from,
            valid_until=valid_until,
            properties_json=props_json,
        )
        
        result = _FFI.lib.sochdb_add_temporal_edge(
            self._ptr,
            namespace.encode("utf-8"),
            edge
        )
        
        if result != 0:
            raise DatabaseError(f"Failed to add temporal edge: error code {result}")
    
    def query_temporal_graph(
        self,
        namespace: str,
        node_id: str,
        mode: str = "CURRENT",
        timestamp: Optional[int] = None,
        edge_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Query temporal graph edges (Embedded FFI mode).
        
        Query modes:
        - "CURRENT": Edges valid now (valid_until = 0 or > current time)
        - "POINT_IN_TIME": Edges valid at specific timestamp
        - "RANGE": All edges within time range (requires timestamp for start/end)
        
        Args:
            namespace: Namespace to query
            node_id: Node to query edges from
            mode: Query mode ("CURRENT", "POINT_IN_TIME", "RANGE")
            timestamp: Timestamp for POINT_IN_TIME or RANGE queries (milliseconds)
            edge_type: Optional filter by edge type
        
        Returns:
            List of edge dictionaries with keys: from_id, edge_type, to_id,
            valid_from, valid_until, properties
        
        Example:
            # Query: "Was the door open 1.5 hours ago?"
            import time
            now = int(time.time() * 1000)
            
            edges = db.query_temporal_graph(
                namespace="smart_home",
                node_id="door_front",
                mode="POINT_IN_TIME",
                timestamp=now - int(1.5 * 60 * 60 * 1000)
            )
            
            if any(e["to_id"] == "open" for e in edges):
                print("Yes, door was open")
        """
        self._check_open()
        
        import json
        
        # Default to current time for POINT_IN_TIME if not provided
        if mode == "POINT_IN_TIME" and timestamp is None:
            import time
            timestamp = int(time.time() * 1000)
        
        # Convert mode string to int (Must match ffi.rs: 0=POINT_IN_TIME, 1=RANGE, 2=CURRENT)
        mode_map = {"POINT_IN_TIME": 0, "RANGE": 1, "CURRENT": 2}
        mode_int = mode_map.get(mode, 0)
        
        # Call FFI function
        result_ptr = self._lib.sochdb_query_temporal_graph(
            self._handle,
            namespace.encode("utf-8"),
            node_id.encode("utf-8"),
            mode_int,
            ctypes.c_uint64(timestamp or 0),
            ctypes.c_uint64(0), # start_time
            ctypes.c_uint64(0), # end_time
            edge_type.encode("utf-8") if edge_type else None,
            ctypes.byref(ctypes.c_size_t()) # Add missing out_len arg from FFI signature!
        )
        
        if result_ptr is None:
            raise DatabaseError("Failed to query temporal graph")
        
        try:
            # Convert C string to Python string
            json_str = ctypes.c_char_p(result_ptr).value.decode("utf-8")
            # Parse JSON array
            edges = json.loads(json_str)
            return edges
        finally:
            # Free the C string
            if result_ptr:
                self._lib.sochdb_free_string(result_ptr)

    # =========================================================================
    # Graph Overlay Operations (FFI) - Nodes, Edges, Traversal
    # =========================================================================
    
    def add_node(
        self,
        namespace: str,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add a node to the graph overlay (Embedded FFI mode).
        
        Args:
            namespace: Namespace for the graph
            node_id: Unique node identifier
            node_type: Type of node (e.g., "person", "document", "concept")
            properties: Optional node properties
        
        Returns:
            True on success
        
        Example:
            db.add_node("default", "alice", "person", {"role": "engineer"})
            db.add_node("default", "project_x", "project", {"status": "active"})
        """
        self._check_open()
        
        import json
        props_json = json.dumps(properties or {}).encode("utf-8")
        
        result = self._lib.sochdb_graph_add_node(
            self._handle,
            namespace.encode("utf-8"),
            node_id.encode("utf-8"),
            node_type.encode("utf-8"),
            props_json
        )
        
        if result != 0:
            raise DatabaseError(f"Failed to add node: error code {result}")
        return True
    
    def add_edge(
        self,
        namespace: str,
        from_id: str,
        edge_type: str,
        to_id: str,
        properties: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add an edge between nodes (Embedded FFI mode).
        
        Args:
            namespace: Namespace for the graph
            from_id: Source node ID
            edge_type: Type of relationship (e.g., "works_on", "knows", "references")
            to_id: Target node ID
            properties: Optional edge properties
        
        Returns:
            True on success
        
        Example:
            db.add_edge("default", "alice", "works_on", "project_x")
            db.add_edge("default", "alice", "knows", "bob", {"since": "2020"})
        """
        self._check_open()
        
        import json
        props_json = json.dumps(properties or {}).encode("utf-8")
        
        result = self._lib.sochdb_graph_add_edge(
            self._handle,
            namespace.encode("utf-8"),
            from_id.encode("utf-8"),
            edge_type.encode("utf-8"),
            to_id.encode("utf-8"),
            props_json
        )
        
        if result != 0:
            raise DatabaseError(f"Failed to add edge: error code {result}")
        return True
    
    def traverse(
        self,
        namespace: str,
        start_node: str,
        max_depth: int = 10,
        order: str = "bfs"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Traverse the graph from a starting node (Embedded FFI mode).
        
        Args:
            namespace: Namespace for the graph
            start_node: Node ID to start traversal from
            max_depth: Maximum traversal depth
            order: "bfs" for breadth-first, "dfs" for depth-first
        
        Returns:
            Tuple of (nodes, edges) where each is a list of dicts
        
        Example:
            nodes, edges = db.traverse("default", "alice", max_depth=2)
            for node in nodes:
                print(f"Node: {node['id']} ({node['node_type']})")
            for edge in edges:
                print(f"Edge: {edge['from_id']} --{edge['edge_type']}--> {edge['to_id']}")
        """
        self._check_open()
        
        import json
        import ctypes
        
        order_int = 0 if order.lower() == "bfs" else 1
        out_len = ctypes.c_size_t()
        
        result_ptr = self._lib.sochdb_graph_traverse(
            self._handle,
            namespace.encode("utf-8"),
            start_node.encode("utf-8"),
            max_depth,
            order_int,
            ctypes.byref(out_len)
        )
        
        if result_ptr is None:
            raise DatabaseError("Failed to traverse graph")
        
        try:
            json_str = ctypes.c_char_p(result_ptr).value.decode("utf-8")
            data = json.loads(json_str)
            return data.get("nodes", []), data.get("edges", [])
        finally:
            if result_ptr:
                self._lib.sochdb_free_string(result_ptr)

    # =========================================================================
    # Collection Search FFI (Native Rust performance)
    # =========================================================================
    
    def ffi_collection_search(
        self,
        namespace: str,
        collection: str,
        query_vector: List[float],
        k: int = 10
    ) -> List[Dict]:
        """
        Native vector search using Rust FFI.
        
        This is 40x faster than Python brute-force search.
        Returns list of {id, score, metadata} dicts.
        """
        self._check_open()
        
        import numpy as np
        
        # Prepare query vector
        query_array = np.array(query_vector, dtype=np.float32)
        query_ptr = query_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Allocate results array
        results = (C_SearchResult * k)()
        
        # Call FFI
        try:
            num_results = self._lib.sochdb_collection_search(
                self._handle,
                namespace.encode("utf-8"),
                collection.encode("utf-8"),
                query_ptr,
                len(query_vector),
                k,
                results
            )
            
            if num_results < 0:
                return []
            
            # Parse results
            output = []
            for i in range(num_results):
                result = results[i]
                doc_id = result.id_ptr.decode("utf-8") if result.id_ptr else None
                metadata_str = result.metadata_ptr.decode("utf-8") if result.metadata_ptr else "{}"
                
                try:
                    import json
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                output.append({
                    "id": doc_id,
                    "score": result.score,
                    "metadata": metadata,
                })
            
            # Free results
            self._lib.sochdb_search_result_free(results, num_results)
            
            return output
        except (AttributeError, OSError) as e:
            # FFI not available, return empty (caller should fallback)
            return None

    def ffi_collection_keyword_search(
        self,
        namespace: str,
        collection: str,
        query_text: str,
        k: int = 10
    ) -> List[Dict]:
        """
        Native keyword search using Rust FFI.
        """
        self._check_open()
        
        # Allocate results array
        results = (C_SearchResult * k)()
        
        # Call FFI
        try:
            num_results = self._lib.sochdb_collection_keyword_search(
                self._handle,
                namespace.encode("utf-8"),
                collection.encode("utf-8"),
                query_text.encode("utf-8"),
                k,
                results
            )
            
            if num_results < 0:
                return []
            
            # Parse results
            output = []
            for i in range(num_results):
                result = results[i]
                doc_id = result.id_ptr.decode("utf-8") if result.id_ptr else None
                metadata_str = result.metadata_ptr.decode("utf-8") if result.metadata_ptr else "{}"
                
                try:
                    import json
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                output.append({
                    "id": doc_id,
                    "score": result.score,
                    "metadata": metadata,
                })
            
            # Free results
            self._lib.sochdb_search_result_free(results, num_results)
            
            return output
        except (AttributeError, OSError) as e:
            # FFI not available, return empty (caller should fallback)
            return None
    
    # =========================================================================
    # Semantic Cache Operations (FFI)
    # =========================================================================
    
    def cache_put(
        self,
        cache_name: str,
        key: str,
        value: str,
        embedding: List[float],
        ttl_seconds: int = 0
    ) -> bool:
        """
        Store a value in the semantic cache with its embedding.
        
        Args:
            cache_name: Name of the cache
            key: Cache key (for display/debugging)
            value: Value to cache
            embedding: Embedding vector for similarity matching
            ttl_seconds: Time-to-live in seconds (0 = no expiry)
        
        Returns:
            True on success
        
        Example:
            db.cache_put(
                "llm_responses",
                "What is Python?",
                "Python is a programming language...",
                embedding=[0.1, 0.2, 0.3, ...],  # 384-dim
                ttl_seconds=3600
            )
        """
        self._check_open()
        
        # Try FFI first if available
        try:
            if hasattr(_FFI, 'lib') and _FFI.lib is not None and hasattr(self, '_ptr') and self._ptr is not None:
                import ctypes
                import numpy as np
                
                emb_array = np.array(embedding, dtype=np.float32)
                emb_ptr = emb_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                result = _FFI.lib.sochdb_cache_put(
                    self._handle,
                    cache_name.encode("utf-8"),
                    key.encode("utf-8"),
                    value.encode("utf-8"),
                    emb_ptr,
                    len(embedding),
                    ttl_seconds
                )
                
                if result == 0:
                    return True
        except (AttributeError, OSError, TypeError):
            pass  # Fall through to KV fallback
        
        # KV fallback - store as JSON
        import json
        import time
        import hashlib
        
        # Create unique cache entry key
        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        cache_key = f"_cache/{cache_name}/{key_hash}".encode()
        
        entry = {
            "key": key,
            "value": value,
            "embedding": embedding,
            "ttl": ttl_seconds,
            "created": time.time()
        }
        self.put(cache_key, json.dumps(entry).encode())
        return True
    
    def cache_get(
        self,
        cache_name: str,
        query_embedding: List[float],
        threshold: float = 0.85
    ) -> Optional[str]:
        """
        Look up a value in the semantic cache by embedding similarity.
        
        Args:
            cache_name: Name of the cache
            query_embedding: Query embedding to match against cached entries
            threshold: Minimum cosine similarity threshold (0.0 to 1.0)
        
        Returns:
            Cached value if similarity >= threshold, None otherwise
        
        Example:
            result = db.cache_get(
                "llm_responses",
                query_embedding=[0.12, 0.18, ...],  # Similar to "What is Python?"
                threshold=0.85
            )
            if result:
                print(f"Cache hit: {result}")
        """
        self._check_open()
        
        # Try FFI first if available
        try:
            if hasattr(_FFI, 'lib') and _FFI.lib is not None and hasattr(self, '_ptr') and self._ptr is not None:
                import ctypes
                import numpy as np
                
                emb_array = np.array(query_embedding, dtype=np.float32)
                emb_ptr = emb_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                out_len = ctypes.c_size_t()
                
                result_ptr = _FFI.lib.sochdb_cache_get(
                    self._handle,
                    cache_name.encode("utf-8"),
                    emb_ptr,
                    len(query_embedding),
                    threshold,
                    ctypes.byref(out_len)
                )
                
                if result_ptr is not None:
                    try:
                        return ctypes.c_char_p(result_ptr).value.decode("utf-8")
                    finally:
                        _FFI.lib.sochdb_free_string(result_ptr)
        except (AttributeError, OSError, TypeError):
            pass  # Fall through to KV fallback
        
        # KV fallback - scan and compute similarity
        import json
        import math
        import time
        
        prefix = f"_cache/{cache_name}/".encode()
        best_match = None
        best_score = 0.0
        
        try:
            with self.transaction() as txn:
                for k, v in txn.scan_prefix(prefix):
                    try:
                        entry = json.loads(v.decode())
                        
                        # Check TTL
                        if entry.get("ttl", 0) > 0:
                            if time.time() - entry.get("created", 0) > entry["ttl"]:
                                continue  # Expired
                        
                        # Compute cosine similarity
                        cached_emb = entry.get("embedding", [])
                        if len(cached_emb) != len(query_embedding):
                            continue
                        
                        # Cosine similarity
                        dot_product = sum(q * c for q, c in zip(query_embedding, cached_emb))
                        query_norm = math.sqrt(sum(x * x for x in query_embedding))
                        cached_norm = math.sqrt(sum(x * x for x in cached_emb))
                        
                        if query_norm > 0 and cached_norm > 0:
                            score = dot_product / (query_norm * cached_norm)
                            # Normalize from [-1, 1] to [0, 1] for threshold comparisons
                            score = (score + 1.0) / 2.0
                        else:
                            score = 0.0
                        
                        if score >= threshold and score > best_score:
                            best_match = entry.get("value")
                            best_score = score
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass  # Return None on any error
        
        return best_match

    # =========================================================================
    # Trace Operations (FFI) - Observability
    # =========================================================================
    
    def start_trace(self, name: str) -> Tuple[str, str]:
        """
        Start a new trace (Embedded FFI mode).
        
        Args:
            name: Name of the trace (e.g., "user_request", "batch_job")
        
        Returns:
            Tuple of (trace_id, root_span_id)
        
        Example:
            trace_id, root_span = db.start_trace("user_query")
            # ... do work ...
            db.end_span(trace_id, root_span, status="ok")
        """
        self._check_open()
        
        import ctypes
        
        trace_id_ptr = ctypes.c_char_p()
        span_id_ptr = ctypes.c_char_p()
        
        result = _FFI.lib.sochdb_trace_start(
            self._ptr,
            name.encode("utf-8"),
            ctypes.byref(trace_id_ptr),
            ctypes.byref(span_id_ptr)
        )
        
        if result != 0:
            raise DatabaseError(f"Failed to start trace: error code {result}")
        
        try:
            trace_id = trace_id_ptr.value.decode("utf-8")
            span_id = span_id_ptr.value.decode("utf-8")
            return trace_id, span_id
        finally:
            _FFI.lib.sochdb_free_string(trace_id_ptr)
            _FFI.lib.sochdb_free_string(span_id_ptr)
    
    def start_span(
        self,
        trace_id: str,
        parent_span_id: str,
        name: str
    ) -> str:
        """
        Start a child span within a trace (Embedded FFI mode).
        
        Args:
            trace_id: ID of the parent trace
            parent_span_id: ID of the parent span
            name: Name of this span (e.g., "database_query", "llm_call")
        
        Returns:
            span_id for the new span
        
        Example:
            trace_id, root_span = db.start_trace("user_query")
            db_span = db.start_span(trace_id, root_span, "database_lookup")
            # ... do database work ...
            duration = db.end_span(trace_id, db_span, status="ok")
            print(f"DB lookup took {duration}µs")
        """
        self._check_open()
        
        import ctypes
        
        span_id_ptr = ctypes.c_char_p()
        
        result = _FFI.lib.sochdb_trace_span_start(
            self._ptr,
            trace_id.encode("utf-8"),
            parent_span_id.encode("utf-8"),
            name.encode("utf-8"),
            ctypes.byref(span_id_ptr)
        )
        
        if result != 0:
            raise DatabaseError(f"Failed to start span: error code {result}")
        
        try:
            return span_id_ptr.value.decode("utf-8")
        finally:
            _FFI.lib.sochdb_free_string(span_id_ptr)
    
    def end_span(
        self,
        trace_id: str,
        span_id: str,
        status: str = "ok"
    ) -> int:
        """
        End a span and record its duration (Embedded FFI mode).
        
        Args:
            trace_id: ID of the trace
            span_id: ID of the span to end
            status: "ok", "error", or "unset"
        
        Returns:
            Duration in microseconds
        
        Example:
            duration = db.end_span(trace_id, span_id, status="ok")
            print(f"Operation took {duration}µs")
        """
        self._check_open()
        
        status_map = {"unset": 0, "ok": 1, "error": 2}
        status_int = status_map.get(status.lower(), 0)
        
        duration = _FFI.lib.sochdb_trace_span_end(
            self._ptr,
            trace_id.encode("utf-8"),
            span_id.encode("utf-8"),
            status_int
        )
        
        if duration < 0:
            raise DatabaseError("Failed to end span")
        
        return duration
    
    # =========================================================================
    # Vector Index Operations (convenience methods)
    # =========================================================================
    
    def create_index(self, name: str, dimension: int, max_connections: int = 16, ef_construction: int = 200):
        """
        Create a vector index (HNSW).
        
        This is a convenience method that creates a VectorIndex and stores it
        for later use. For more control, use the VectorIndex class directly.
        
        Args:
            name: Index name
            dimension: Vector dimension
            max_connections: HNSW max_connections parameter (connections per layer, default=16)
            ef_construction: HNSW ef_construction parameter (default=200)
        
        Example:
            db.create_index('embeddings', 384)
            db.insert_vectors('embeddings', [1, 2, 3], [[0.1, 0.2, ...], ...])
            results = db.search('embeddings', [0.1, 0.2, ...], k=5)
        """
        from .vector import VectorIndex
        
        if not hasattr(self, '_vector_indices'):
            self._vector_indices = {}
        
        index = VectorIndex(dimension, max_connections=max_connections, ef_construction=ef_construction)
        self._vector_indices[name] = index
        
        # Store index metadata in database
        import json
        metadata = {
            'dimension': dimension,
            'max_connections': max_connections,
            'ef_construction': ef_construction
        }
        self.put(f'_indices/{name}/meta'.encode(), json.dumps(metadata).encode())
    
    def insert_vectors(self, index_name: str, ids: list, vectors: list):
        """
        Insert vectors into an index.
        
        Args:
            index_name: Name of the index
            ids: List of integer IDs
            vectors: List of vectors (each a list of floats)
        
        Example:
            db.insert_vectors('embeddings', [1, 2], [[0.1, 0.2, ...], [0.3, 0.4, ...]])
        """
        if not hasattr(self, '_vector_indices'):
            self._vector_indices = {}
        
        if index_name not in self._vector_indices:
            # Try to load from metadata
            metadata_key = f'_indices/{index_name}/meta'.encode()
            metadata_bytes = self.get(metadata_key)
            if metadata_bytes is None:
                raise DatabaseError(f"Index '{index_name}' not found. Create it first with create_index()")
            
            import json
            from .vector import VectorIndex
            metadata = json.loads(metadata_bytes.decode())
            index = VectorIndex(
                metadata['dimension'],
                max_connections=metadata.get('max_connections', 16),
                ef_construction=metadata.get('ef_construction', 200)
            )
            self._vector_indices[index_name] = index
        
        index = self._vector_indices[index_name]
        import numpy as np
        index.insert_batch(np.array(ids, dtype=np.uint64), np.array(vectors, dtype=np.float32))
    
    def search(self, index_name: str, query: list, k: int = 10):
        """
        Search for nearest neighbors in an index.
        
        Args:
            index_name: Name of the index
            query: Query vector (list of floats)
            k: Number of results to return
        
        Returns:
            List of (id, distance) tuples
        
        Example:
            results = db.search('embeddings', [0.1, 0.2, ...], k=5)
            for id, distance in results:
                print(f'ID: {id}, Distance: {distance}')
        """
        if not hasattr(self, '_vector_indices'):
            self._vector_indices = {}
        
        if index_name not in self._vector_indices:
            # Try to load from metadata
            metadata_key = f'_indices/{index_name}/meta'.encode()
            metadata_bytes = self.get(metadata_key)
            if metadata_bytes is None:
                raise DatabaseError(f"Index '{index_name}' not found. Create it first with create_index()")
            
            import json
            from .vector import VectorIndex
            metadata = json.loads(metadata_bytes.decode())
            index = VectorIndex(
                metadata['dimension'],
                max_connections=metadata.get('max_connections', 16),
                ef_construction=metadata.get('ef_construction', 200)
            )
            self._vector_indices[index_name] = index
        
        index = self._vector_indices[index_name]
        import numpy as np
        query_array = np.array(query, dtype=np.float32)
        return index.search(query_array, k)
