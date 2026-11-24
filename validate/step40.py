"""Step 40: buckets, tuned (part 5).
  .venv/bin/python -I validate/step40.py [dataset ...]      # default: glove nytimes
  .venv/bin/python -I validate/step40.py full               # GloVe at full size: is it worth building in?
  .venv/bin/python -I validate/step40.py grouping           # GloVe at full size: buckets trained harder
On each dataset's sample, part 3's index (every default, the 20,000 training questions), its buckets built several
ways: about 100 or 300 items per bucket, and the codes plain (each pair coded to its nearest point) or score-aware
(ScaNN's: errors along the item count more). For each, on the tuning questions: buckets scanned from 1 up, for
re-check pools of 50, 100 and 200; items scored and time per question at exactly 95% (one core, the questions
repeated to fill 0.3 s, the fastest of 3 runs), against the link walk on the same index, timed the same way.
Pass: on GloVe, the best way at least as fast as the link walk at 95% (tuning questions).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import TARGETS, Report, accuracy, build_index, correct_counts, load, metric_of, truth
from step27 import PART3
from step39 import PROBES, curve_at_95, timed

WAYS = [(300, 1.0), (300, None), (100, 1.0), (100, None)]   # items per bucket, weight along the item (None: default)
POOLS = [50, 100, 200]
# at full size, every question scores every centre: with 1.17 million items, 100 per bucket is 11,700 centres
FULL_WAYS = [(300, None), (1000, None), (100, None)]
FULL_POOLS = [100, 200]
FULL_PROBES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
# `full` found the codes lose almost nothing (the best 100 by their codes hold what re-checking every item finds), so
# the answers are lost to the buckets themselves; `grouping` trains them harder: (rounds, sample rows per centre)
GROUPINGS = [(10, 64), (20, 256), (30, 1000)]
SHARES = [0.02, 0.04, 0.07, 0.10]   # of the buckets opened, for "found in the opened buckets"


def grouping():
    rep = Report("step40-grouping")
    metric = metric_of("glove")
    X, Qt = load("glove", "items_full"), load("glove", "q_tune")
    tdt = truth("glove", "full_tune")
    idx, _ = build_index("glove", PART3, train=True, size="full")
    per = 300
    k = len(X) // per
    for rounds, per_centre in GROUPINGS:
        t0 = time.time()
        idx.build_buckets(per, None, rounds=rounds, per_centre=per_centre)
        built = time.time() - t0
        inside = {}
        for share in SHARES:
            ids, _, _ = idx.search_buckets(Qt, k=10, n_probe=max(1, round(share * k)), pool=10**7, threads=16)
            inside[share] = accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric))
        pools = {}
        for pool in FULL_POOLS:
            pts = []
            for p in FULL_PROBES:
                ids, _, checks = idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1)
                a = accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric))
                pts.append({"n_probe": p, "accuracy": a, "checks": float(checks.mean()),
                            "sec": timed(lambda: idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1), len(Qt))})
                if a >= 0.99:
                    break
            pools[pool] = {"points": pts, **curve_at_95(pts)}
        best = max((v for v in pools.values() if v["qps95"]), key=lambda v: v["qps95"], default=None)
        rep.check(f"glove at full size: {rounds} rounds, {per_centre} rows per centre",
                  True, f"built in {built:.0f} s; found in the opened buckets: "
                  + ", ".join(f"{s:.0%} of them {a:.2%}" for s, a in inside.items())
                  + (f"; {best['qps95']:,.0f}/s at 95% ({best['checks95']:,.0f} scored)" if best else "; never 95%"),
                  {"rounds": rounds, "per_centre": per_centre, "seconds": built, "inside": inside, "pools": pools})
    rep.finish()


def main():
    common.unlock_held_back()
    if sys.argv[1:2] == ["grouping"]:
        return grouping()
    full = sys.argv[1:2] == ["full"]
    rep = Report("step40-full" if full else "step40")
    ways, pools, probes = (FULL_WAYS, FULL_POOLS, FULL_PROBES) if full else (WAYS, POOLS, PROBES)
    for name in ["glove"] if full else (sys.argv[1:] or ["glove", "nytimes"]):
        metric = metric_of(name)
        X, Qt = load(name, "items_full" if full else "slice"), load(name, "q_tune")
        tdt = truth(name, "full_tune" if full else "slice_tune")
        idx, _ = build_index(name, PART3, train=True, size="full" if full else "slice")
        links = []
        for T in TARGETS:
            ids, _, checks = idx.search(Qt, k=10, target=T, threads=1)
            links.append({"accuracy": accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric)), "checks": float(checks.mean()),
                          "sec": timed(lambda: idx.search(Qt, k=10, target=T, threads=1), len(Qt))})
        lw = curve_at_95(links)
        res, best = {"links95": lw, "ways": []}, None
        for per, eta in ways:
            t0 = time.time()
            idx.build_buckets(per, eta)
            way = {"per_bucket": per, "eta": eta or "default", "build_seconds": time.time() - t0, "pools": {}}
            for pool in pools:
                pts = []
                for p in probes:
                    if p > len(X) // per:
                        break
                    ids, _, checks = idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1)
                    a = accuracy(correct_counts(ids, Qt, X, tdt["dists"], metric))
                    pts.append({"n_probe": p, "accuracy": a, "checks": float(checks.mean()),
                                "sec": timed(lambda: idx.search_buckets(Qt, k=10, n_probe=p, pool=pool, threads=1), len(Qt))})
                    if a >= 0.99:
                        break
                way["pools"][pool] = {"points": pts, **curve_at_95(pts)}
                r = way["pools"][pool]
                if r["qps95"] and (best is None or r["qps95"] > best[2]["qps95"]):
                    best = (per, eta or "default", r, pool)
            res["ways"].append(way)
            rep.note(f"{name}: {per} per bucket, weight {eta or 'default'}: " + "; ".join(
                f"pool {p}: {v['qps95'] or 0:,.0f}/s ({v['checks95'] or 0:,.0f} scored)" for p, v in way["pools"].items()))
        ratio = best[2]["qps95"] / lw["qps95"] if best and lw["qps95"] else 0
        detail = (f"links {lw['qps95']:,.0f}/s at 95% ({lw['checks95']:,.0f} checks); best buckets: {best[0]} per bucket, "
                  f"weight {best[1]}, pool {best[3]}: {best[2]['qps95']:,.0f}/s ({best[2]['checks95']:,.0f} scored), "
                  f"{ratio:.2f}x the links") if best else "no way reached 95%"
        rep.check(f"{name}: the best bucket search against the link walk at 95% (tuning questions)",
                  ratio >= 1.0 or name != "glove", detail, res)
        del idx
    rep.finish()


if __name__ == "__main__":
    main()
