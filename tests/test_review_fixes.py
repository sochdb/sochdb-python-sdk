"""Regression tests for the architecture-review SDK fixes.

Covers:
  Task 1 — stored embeddings surfaced in the hot search path (no re-embed)
  Task 2 — include_vectors parity on the convenience search methods
  Task 4 — persist_vector_in_kv config gate
  Task 6 — __core_version__ reports the bundled engine version
  Task 7 — Collection.insert_batch routes through the fast flat FFI path

Task 5 was intentionally reverted (the in-memory HNSW is metric-blind via
hnsw_new); a placeholder test documents that non-cosine metrics are not honored.
"""
import shutil
import tempfile

import pytest

import sochdb
from sochdb import Database
from sochdb.namespace import DistanceMetric


@pytest.fixture()
def db():
    path = tempfile.mkdtemp(prefix="soch-review-")
    database = Database.open(path)
    try:
        yield database
    finally:
        database.close()
        shutil.rmtree(path, ignore_errors=True)


def test_core_version_reports_engine():
    # Task 6: package version stays its own SemVer; engine version is separate.
    assert sochdb.__core_version__ == "2.0.9"
    assert sochdb.__version__ == "0.6.3"


def test_include_vectors_surfaces_stored_embedding(db):
    # Tasks 1 + 2 + 7
    ns = db.create_namespace("t")
    col = ns.create_collection("docs", dimension=4, metric=DistanceMetric.COSINE)
    col.insert_batch(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        metadatas=[{"t": "a"}, {"t": "b"}, {"t": "c"}],
    )
    base = [1.0, 0.0, 0.0, 0.0]

    # Default: vectors withheld.
    assert all(r.vector is None for r in col.vector_search(base, k=3).results)

    # include_vectors=True surfaces the *stored* embedding (no re-embedding).
    res = col.vector_search(base, k=3, include_vectors=True)
    assert all(r.vector is not None for r in res.results)
    top = res.results[0]
    assert top.id == "a"
    assert list(top.vector)[:4] == [1.0, 0.0, 0.0, 0.0]
    assert abs(top.score - 1.0) < 1e-3  # exact cosine match


def test_persist_vector_in_kv_gate(db):
    # Task 4
    ns = db.create_namespace("t")

    on = ns.create_collection("on", dimension=3, metric=DistanceMetric.COSINE)
    on.insert("x", [1.0, 0.0, 0.0], {"m": 1})
    assert "vector" in on.get("x")  # default persists the vector

    off = ns.create_collection(
        "off", dimension=3, metric=DistanceMetric.COSINE, persist_vector_in_kv=False
    )
    off.insert("y", [1.0, 0.0, 0.0], {"m": 2})
    assert "vector" not in off.get("y")  # gated out of the KV doc

    # In-session search still works (HNSW graph + _raw_vectors retain the vector).
    res = off.vector_search([1.0, 0.0, 0.0], k=1, include_vectors=True)
    assert len(res.results) == 1
    assert res.results[0].vector is not None


def test_include_vectors_on_exact_search(db):
    # Parity follow-up: vector_search_exact / _f64 honor include_vectors.
    ns = db.create_namespace("t")
    col = ns.create_collection("docs", dimension=4, metric=DistanceMetric.COSINE)
    col.insert_batch(
        ids=["a", "b"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadatas=[{}, {}],
    )
    base = [1.0, 0.0, 0.0, 0.0]

    assert all(r.vector is None for r in col.vector_search_exact(base, k=2).results)
    res = col.vector_search_exact(base, k=2, include_vectors=True)
    assert all(r.vector is not None for r in res.results)
    assert list(res.results[0].vector)[:4] == [1.0, 0.0, 0.0, 0.0]

    res64 = col.vector_search_exact_f64(base, k=2, include_vectors=True)
    assert all(r.vector is not None for r in res64.results)


def test_cosine_ranking_intact_after_task5_revert(db):
    # Task 5 reverted: scoring is cosine-only; verify ranking is unaffected.
    ns = db.create_namespace("t")
    col = ns.create_collection("docs", dimension=4, metric=DistanceMetric.COSINE)
    col.insert_batch(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        metadatas=[{}, {}, {}],
    )
    order = [r.id for r in col.vector_search([0.95, 0.05, 0.0, 0.0], k=3).results]
    assert order.index("a") < order.index("b")
    assert order.index("c") < order.index("b")
