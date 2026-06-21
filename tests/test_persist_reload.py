"""Regression tests: persisted collections must stay searchable after reopen.

Bug (fixed in 0.7.1): the in-memory HNSW was only rebuilt from a numpy snapshot
that was never written -- ``_persist_vectors_snapshot`` had zero call sites -- so
``vector_search`` returned 0 results after reopening a ``persist_directory``
collection, even though the data survived in the KV store (``count`` and
``get_by_ids`` worked). The reload path now falls back to rebuilding the HNSW
from the KV-persisted vectors, which are written transactionally on every insert.
"""

import numpy as np
import pytest

from sochdb import CollectionConfig, Database, DistanceMetric


def _vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _insert(path, dim, n, seed):
    vecs = _vectors(n, dim, seed)
    ids = [f"d{i}" for i in range(n)]
    metas = [{"n": i, "grp": i % 5} for i in range(n)]
    db = Database.open(path)
    ns = db.get_or_create_namespace("default")
    col = ns.create_collection(
        CollectionConfig(name="docs", dimension=dim, metric=DistanceMetric.COSINE)
    )
    col.insert_batch(ids=ids, vectors=[v.tolist() for v in vecs], metadatas=metas)
    # sanity: same-session search works
    assert len(col.vector_search(vecs[0].tolist(), 5)) > 0
    db.close()
    return vecs


def test_vector_search_survives_reopen(tmp_path):
    path = str(tmp_path / "persist_db")
    dim, n = 32, 50
    vecs = _insert(path, dim, n, seed=0)

    # Reopen in a fresh session and search -- must NOT be empty (the regression).
    db = Database.open(path)
    col = db.get_or_create_namespace("default").collection("docs")
    assert col.count() == n, "data must persist in the KV store"

    res = col.vector_search(vecs[0].tolist(), 5)
    assert len(res) > 0, "vector search must return results after reopen"
    # nearest neighbour of vecs[0] is itself
    assert str(res[0].id) == "d0"
    db.close()


def test_filtered_search_survives_reopen(tmp_path):
    path = str(tmp_path / "persist_db_filtered")
    dim, n = 32, 50
    vecs = _insert(path, dim, n, seed=1)

    db = Database.open(path)
    col = db.get_or_create_namespace("default").collection("docs")
    res = col.vector_search(vecs[0].tolist(), 10, filter={"grp": 0})
    assert len(res) > 0, "filtered vector search must return results after reopen"
    assert all(r.metadata.get("grp") == 0 for r in res), "filter must be honoured"
    db.close()
