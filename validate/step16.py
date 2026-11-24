"""Step 16: spot checks.
  - with every question spot-checked, the estimate equals the accuracy worked out from the true-answer files;
  - with about 1 in 20 spot-checked, over 100 different samples per dataset, how often the real accuracy of all
    1,000 questions falls inside the stated margin (pooled over the five: 92% to 98%, no dataset under 85%);
  - the cost at the default budget, one core: spot-check time over search time, under 2%;
  - results unchanged (step 12's recording), searches running while a spot check runs, and the counts kept
    through save and load.

.venv/bin/python -I validate/step16.py
"""
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, Report, accuracy, build_index, correct_counts, load, truth
from step11 import FINAL
from step12 import BEFORE, answers

TARGET = 0.95
# these checks test the estimates on a fixed table: learning it again from use (step 17) stays off
BASE = dict(FINAL, relearn_every=0)
EVERY = dict(BASE, audit_budget=1e12)   # a budget this large keeps every question


def exact(rep, name):
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    idx, _ = build_index(name, EVERY, train=True)
    ids_t, _, _ = idx.search(Q, k=10, target=TARGET, threads=16)
    ids_l, _, _ = idx.search(Q, k=10, list_size=100, threads=16)
    got = idx.audit()["accuracy"]
    want = {f"target {TARGET}": accuracy(correct_counts(ids_t, Q, X, td, metric)),
            "list 100": accuracy(correct_counts(ids_l, Q, X, td, metric))}
    ok = all(got[k]["questions"] == len(Q) and abs(got[k]["accuracy"] - v) < 1e-12 for k, v in want.items())
    rep.check(f"{name}: every question spot-checked gives the true-answer files' accuracy", ok,
              "; ".join(f"{k}: {got[k]['accuracy']:.2%} (files {v:.2%}, {got[k]['questions']} checked)"
                        for k, v in want.items()))


def coverage(name, runs=100, per_run=50):
    """Each run searches the 1,000 tuning questions in a new order, spot-checks about 50 of them, and asks
    whether the real accuracy of all 1,000 lies inside the estimate's margin."""
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    idx, _ = build_index(name, dict(BASE, audit_budget=0), train=True)
    ids, _, checks = idx.search(Q, k=10, target=TARGET, threads=16)
    real = accuracy(correct_counts(ids, Q, X, td, metric))
    budget = per_run * len(X) / float(checks.sum())     # one call's credit pays for about 50 spot checks
    idx, _ = build_index(name, dict(BASE, audit_budget=budget), train=True)
    inside, widths, sizes = 0, [], []
    for r in range(runs):
        order = np.random.default_rng(r).permutation(len(Q))
        idx.reset_usage()
        idx.search(Q[order], k=10, target=TARGET, threads=16)
        e = idx.audit()["accuracy"][f"target {TARGET}"]
        inside += abs(e["accuracy"] - real) <= e["margin"]
        widths.append(e["margin"])
        sizes.append(e["questions"])
    return inside, real, float(np.median(widths)), float(np.median(sizes))


def cost(name, repeats=20):
    """One core: search the tuning questions 20 times over with the default budget, then run the spot checks
    on one thread; their time against the searches' time."""
    Q = load(name, "q_tune")
    idx, _ = build_index(name, BASE, train=True)
    idx.search(Q, k=10, target=TARGET, threads=1)
    idx.reset_usage()
    t = time.perf_counter()
    for _ in range(repeats):
        idx.search(Q, k=10, target=TARGET, threads=1)
    search_s = time.perf_counter() - t
    pending = idx.usage()["spot_checks"]["pending"]
    t = time.perf_counter()
    r = idx.audit(threads=1)
    audit_s = time.perf_counter() - t
    return audit_s / search_s, r["share_of_search_checks"], pending, search_s


def identical(rep, name):
    ref = np.load(BEFORE / f"{name}.npz")
    before = json.loads((BEFORE / f"{name}.json").read_text())
    idx, _ = build_index(name, FINAL, train=True)
    now = answers(idx, load(name, "q_test"))
    diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
    same_build = idx.stats()["build_checks"] == before["build_checks"]
    rep.check(f"{name}: build and answers identical to step 12's recording", not diff and same_build,
              "identical" if not diff and same_build else f"differ: {diff[:4]}, build checks same: {same_build}")


def side_by_side_and_saved(rep, name="glove"):
    """Searches from 4 threads while spot checks run give the answers they give alone; the counts come back
    after save and load."""
    Q = load(name, "q_tune")
    idx, _ = build_index(name, dict(BASE, audit_budget=0.5), train=True)
    alone = idx.search(Q, k=10, target=TARGET, threads=1)[0]
    stop, errors, rounds = threading.Event(), [], [0]

    def searcher():
        try:
            while not stop.is_set():
                if not np.array_equal(idx.search(Q, k=10, target=TARGET, threads=1)[0], alone):
                    errors.append("different answers")
                rounds[0] += 1
        except Exception as e:   # noqa: BLE001 - any failure is the finding
            errors.append(repr(e))

    ths = [threading.Thread(target=searcher, daemon=True) for _ in range(4)]
    try:
        [th.start() for th in ths]
        audits = 0
        t = time.time()
        while time.time() - t < 3:
            idx.audit(threads=2)
            audits += 1
    finally:
        stop.set()
        [th.join() for th in ths]
    checked = idx.usage()["spot_checks"]["accuracy"][f"target {TARGET}"]["questions"]
    rep.check(f"{name}: searches run while spot checks run, with the same answers", not errors and checked > 0,
              f"{rounds[0]} rounds of 1,000 searches from 4 threads during {audits} spot-check runs "
              f"({checked:,} questions checked); errors: {errors[:2] or 'none'}")
    d = Path(tempfile.mkdtemp(prefix="vindex-step16-"))
    try:
        before = idx.usage()["spot_checks"]
        idx.save(str(d / "ix"))
        after = vindex.Index.load(str(d / "ix")).usage()["spot_checks"]
        same = before["accuracy"] == after["accuracy"] and before["checks"] == after["checks"]
        rep.check(f"{name}: spot-check counts come back after save and load", same,
                  f"{len(after['accuracy'])} kinds of search, {after['checks']:,} checks spent")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    rep = Report("step16")
    for name in DATASETS:
        exact(rep, name)
    total = 0
    for name in DATASETS:
        inside, real, width, size = coverage(name)
        total += inside
        rep.check(f"{name}: real accuracy inside the margin in {inside} of 100 samples", inside >= 85,
                  f"real {real:.2%}; about {size:.0f} questions checked per sample; typical margin ±{width:.2%}")
    rep.check("all five: real accuracy inside the margin in 92% to 98% of the 500 samples", 460 <= total <= 490,
              f"{total} of 500 ({total / 5:.1f}%)")
    for name in DATASETS:
        share, check_share, pending, search_s = cost(name)
        rep.check(f"{name}: spot checks at the default budget cost under 2% of search time", share < 0.02,
                  f"{share:.2%} of {search_s:.2f} s of searching ({pending} questions checked; "
                  f"{check_share:.2%} of the checks)")
    for name in DATASETS:
        identical(rep, name)
    side_by_side_and_saved(rep)
    rep.finish()


if __name__ == "__main__":
    main()
