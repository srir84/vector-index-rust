"""Step 39: buckets and their fast codes (part 5).
  .venv/bin/python -I validate/step39.py [dataset ...]      # default: glove nytimes
On each dataset's sample, part 3's index (every default, the 20,000 training questions) is built, then its items are
grouped into buckets (validate: Index.build_buckets). On the tuning questions:
  - exact when nothing is left out: scanning every bucket and re-checking every item gives the true top 10;
  - speed against accuracy: buckets scanned (n_probe) from 1 up, for several re-check pools; one core, the questions
    repeated to fill 0.3 s, the fastest of 3 runs; read off at exactly 95% as everything else;
  - the link walk on the same index, timed the same way, at its learned targets.
Pass: exact when nothing is left out. The speeds are recorded for step 40.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import (DATASETS, HELD_BACK_3, TARGETS, Report, accuracy, build_index, correct_counts, cost_at_target,
                    load, metric_of, truth)
from step27 import PART3

PROBES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
POOLS = [50, 100, 200, 400]
MIN_RUN_S, RUNS = 0.3, 3


def timed(fn, nq):
    fn()
    best = float("inf")
    for _ in range(RUNS):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < MIN_RUN_S:
            fn()
            n += 1
        best = min(best, (time.perf_counter() - t0) / n)
    return best / nq


def curve_at_95(points):
    acc, sec = [p["accuracy"] for p in points], [p["sec"] for p in points]
    s, status = cost_at_target(acc, sec)
    c, _ = cost_at_target(acc, [p["checks"] for p in points])
    return {"qps95": 1 / s if s else None, "checks95": c, "status": status}


def main():
    common.unlock_held_back()
    rep = Report("step39")
    for name in sys.argv[1:] or ["glove", "nytimes"]:
        assert name in DATASETS or name in HELD_BACK_3
        metric = metric_of(name)
        X, Qt = load(name, "slice"), load(name, "q_tune")
        tdt = truth(name, "slice_tune")
        idx, build_s = build_index(name, PART3, train=True)
        t0 = time.time()
        idx.build_buckets(300)
        bucket_s = time.time() - t0
        k_buckets = len(X) // 300
        # exact when nothing is left out
        ids, _, _ = idx.search_buckets(Qt[:200], k=10, n_probe=k_buckets, pool=len(X), threads=16)
        exact_ids = tdt["ids"][:200, :10]
        acc_all = accuracy(correct_counts(ids, Qt[:200], X, tdt["dists"][:200], metric))
        rep.check(f"{name}: scanning every bucket and re-checking every item finds the true top 10", acc_all == 1.0,
                  f"{acc_all:.2%} of the true top 10 on 200 tuning questions ({int((ids == exact_ids).all(1).sum())} with "
                  f"the same ids in the same order; ties may swap); {k_buckets} buckets built in {bucket_s:.1f} s")
        # speed against accuracy
        links = []
        for T in TARGETS:
            ids, _, checks = idx.search(Qt, k=10, target=T, threads=1)
            links.append({"target": T, "accuracy": accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric)),
                          "checks": float(checks.mean()), "sec": timed(lambda: idx.search(Qt, k=10, target=T, threads=1), len(Qt))})
        lw = curve_at_95(links)
        res = {"links": links, "links95": lw, "buckets": {}, "bucket_seconds": bucket_s, "build_seconds": build_s}
        best = None
        for pool in POOLS:
            pts = []
            for p in PROBES:
                if p > k_buckets:
                    break
                ids, _, checks = idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1)
                a = accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric))
                pts.append({"n_probe": p, "accuracy": a, "checks": float(checks.mean()),
                            "sec": timed(lambda: idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1), len(Qt))})
                if a >= 0.995:
                    break
            r = curve_at_95(pts)
            res["buckets"][pool] = {"points": pts, **r}
            if r["qps95"] and (best is None or r["qps95"] > best[1]["qps95"]):
                best = (pool, r)
        detail = (f"links: {lw['qps95'] or 0:,.0f}/s at 95% ({lw['checks95'] or 0:,.0f} checks); buckets: "
                  + "; ".join(f"pool {p}: {v['qps95'] or 0:,.0f}/s ({v['checks95'] or 0:,.0f} scored)" for p, v in res["buckets"].items())
                  + (f"; best {best[1]['qps95'] / lw['qps95']:.2f}x the links" if best and lw["qps95"] else ""))
        rep.note(f"{name}: {detail}")
        rep.check(f"{name}: speeds recorded", True, detail, res)
        del idx
    rep.finish()


if __name__ == "__main__":
    main()
