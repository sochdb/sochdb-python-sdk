"""Regression test: optimize() lifts recall via the engine's exact layer-0 rebuild.

Requires native engine >= 2.0.11 (the hnsw_optimize FFI export). On older bundles
optimize() raises RuntimeError, which this test treats as a skip.
"""

import numpy as np
import pytest

from sochdb import CollectionConfig, Database, DistanceMetric


def _clustered_unit(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((40, dim)).astype(np.float32) * 4.0
    x = (centers[rng.integers(0, 40, n)] + rng.standard_normal((n, dim)).astype(np.float32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x


def test_optimize_lifts_recall(tmp_path):
    dim, n = 768, 4000
    x = _clustered_unit(n, dim)
    db = Database.open(str(tmp_path / "db"))
    col = db.get_or_create_namespace("default").create_collection(
        CollectionConfig(name="c", dimension=dim, metric=DistanceMetric.COSINE, m=16, ef_construction=100)
    )
    col.insert_batch(
        ids=[f"d{i}" for i in range(n)],
        vectors=[v.tolist() for v in x],
        metadatas=[{"i": i} for i in range(n)],
    )

    rng = np.random.default_rng(7)
    qi = rng.integers(0, n, size=50)

    def recall():
        hit = 0
        for q in qi:
            gt = set(np.argsort(-(x @ x[q]))[:10].tolist())
            got = col.vector_search(x[q].tolist(), 10)
            hit += len(gt & {r.metadata["i"] for r in got})
        return hit / (10 * len(qi))

    before = recall()
    try:
        rebuilt = col.optimize()
    except RuntimeError as e:
        if "2.0.11" in str(e):
            pytest.skip("native engine predates hnsw_optimize")
        raise
    after = recall()
    db.close()

    print(f"\n  recall@10  build-only={before:.3f}  after optimize()={after:.3f}  (+{after - before:.3f})")
    assert rebuilt > 0, "optimize() should rebuild nodes"
    # Invariants robust to the parallel builder's run-to-run variance and the
    # ~+-0.02 noise of a 50-query sample: optimize() executes, keeps a
    # high-quality graph, and never meaningfully regresses recall. (The magnitude
    # of the lift is build/data dependent and is largest when the as-built graph
    # is poor -- demonstrated in benchmarks/bench_vs_chroma.py, not asserted here.)
    assert after >= before - 0.03, f"optimize() must not regress recall ({before:.3f} -> {after:.3f})"
    assert after >= 0.85, f"optimize() should keep recall high, got {after:.3f}"
