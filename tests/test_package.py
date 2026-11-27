"""Checks of the Python package on small random data. Run after tools/build_py.sh:
    .venv/bin/python -I -m pytest tests/
"""
import numpy as np
import pytest

import vindex

D, N = 24, 8_000


def exact_top10(x, q, metric):
    if metric == "cos":
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if metric == "l2":
        d = (q * q).sum(1)[:, None] - 2 * q @ x.T + (x * x).sum(1)[None, :]
    else:
        d = -(q @ x.T)
    return np.argsort(d, axis=1, kind="stable")[:, :10]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D), dtype=np.float32)
    train = rng.standard_normal((2_000, D), dtype=np.float32)
    q = rng.standard_normal((100, D), dtype=np.float32)
    return x, train, q


def built(x, train, metric="cos", **settings):
    idx = vindex.Index(D, metric=metric, tune="off", **settings)
    idx.build(x, np.arange(len(x)), train_questions=train)
    return idx


@pytest.mark.parametrize("metric", ["cos", "ip", "l2"])
def test_finds_the_true_top_10(data, metric):
    x, train, q = data
    found, dists, checks = built(x, train, metric).search(q, k=10, target=0.95)
    assert found.shape == (100, 10) and dists.shape == (100, 10) and checks.shape == (100,)
    assert (np.diff(dists, axis=1) >= 0).all(), "closest first"
    truth = exact_top10(x, q, metric)
    recall = np.mean([len(set(f) & set(t)) / 10 for f, t in zip(found, truth)])
    assert recall >= 0.9, recall


def test_refuses_bad_input(data):
    x, train, q = data
    idx = built(x, train)
    with pytest.raises(ValueError):
        idx.search(q, k=0)
    with pytest.raises(TypeError):
        idx.search(q.astype(np.float64), k=10)
    with pytest.raises(ValueError):
        idx.search(np.full((1, D), np.nan, dtype=np.float32), k=10)
    with pytest.raises(ValueError):
        vindex.Index(D, renumber="maybe")
    with pytest.raises(ValueError):
        idx.build(x, np.arange(len(x)), train_questions=train)   # already built


def test_changes(data):
    x, train, q = data
    idx = built(x, train)
    first = idx.search(q[:1], k=1, list_size=100)[0][0, 0]
    idx.delete(int(first))
    assert first not in idx.search(q[:1], k=10, list_size=100)[0]
    idx.update(5, q[1])
    assert idx.search(q[1:2], k=1, list_size=100)[0][0, 0] == 5, "an updated item is found by its new numbers"
    idx.add(10**9, q[2])
    assert idx.search(q[2:3], k=1, list_size=100)[0][0, 0] == 10**9


def test_save_and_load_give_the_same_answers(data, tmp_path):
    x, train, q = data
    idx = built(x, train)
    idx.save(str(tmp_path / "i"))
    for numbers in ("ram", "ssd"):
        back = vindex.Index.load(str(tmp_path / "i"), numbers=numbers)
        a, b = idx.search(q, k=10, target=0.95), back.search(q, k=10, target=0.95)
        assert (a[0] == b[0]).all() and (a[2] == b[2]).all()
    assert not list((tmp_path / "i").glob("*.tmp"))


def test_reports(data):
    x, train, q = data
    idx = built(x, train)
    idx.search(q, k=10, target=0.95)
    p = idx.build_progress()
    assert set(p) == {"phase", "step", "done", "total"}
    assert idx.usage()["questions"] == 100
    assert idx.memory()["total_bytes"] > 0
