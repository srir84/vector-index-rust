"""Step 24: the re-check count, picked after the build.
  - on the five, the tuned index (every part 3 default) against the same index with each re-check count from 10 to
    100, all timed on one core at 95% on the tuning questions, taking turns (5 rounds);
  - the picked count must be within 3% of the best count's speed, and the 95% target must give at least 94%;
  - the map does not depend on the count: the index built with the picked count and no tuning gives the tuned
    index's answers exactly;
  - the extra build time is reported.

.venv/bin/python -I validate/step24.py [dataset ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, Report, accuracy, correct_counts, load, truth
from step18 import build, plan95, sec95
from step21 import PART2_21

COUNTS = [10, 15, 25, 50, 100]   # the tuning tries 15, 25 and 50; 10 and 100 show nothing outside them is better


def one(rep, name):
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    tuned, secs = build(name, dict(PART2_21, tune="auto"))
    t = tuned.tuning()
    r, a, bits, pick = t["chosen_r"], t["chosen_alpha"], t["chosen_code_bits"], t["chosen_rerank"]
    trial_s = sum(1 for x in t["trials"] if x["links"] == "full size, re-check")
    acc = accuracy(correct_counts(tuned.search(Q, k=10, target=0.95, threads=16)[0], Q, X, td, metric))
    grid = {}
    for c in COUNTS:
        idx, _ = build(name, dict(PART2_21, tune="off", r=r, alpha=a, code_bits=bits, rerank=c))
        grid[c] = idx
    # the map does not depend on the count: no tuning, the picked count, the same answers
    same = all(np.array_equal(tuned.search(Q, k=10, list_size=L, threads=16)[0], grid[pick].search(Q, k=10, list_size=L, threads=16)[0])
               for L in (30, 100))
    same = same and np.array_equal(tuned.search(Q, k=10, target=0.95, threads=16)[0], grid[pick].search(Q, k=10, target=0.95, threads=16)[0])
    rep.check(f"{name}: built with the picked count and no tuning, the same answers", same,
              f"{r} links, slack {a:.1f}, {bits}-bit copies, re-check {pick}: answers equal at list sizes 30 and 100 and "
              f"at the 95% target")
    runs = {"picked": tuned, **{c: grid[c] for c in COUNTS}}
    plans = {k: plan95(i, name) for k, i in runs.items()}
    best = {k: float("inf") for k in runs}
    for _ in range(5):
        for k, i in runs.items():
            if plans[k] is not None:
                best[k] = min(best[k], sec95(i, Q, plans[k], runs=1))
    sps = {k: (1 / v if np.isfinite(v) else None) for k, v in best.items()}
    top = max(COUNTS, key=lambda c: sps[c] or 0)
    ratio = sps["picked"] / sps[top]
    rep.check(f"{name}: the picked re-check count is within 3% of the best count", ratio >= 0.97,
              f"picked {pick}: {sps['picked']:,.0f}/s; best {top}: {sps[top]:,.0f}/s ({ratio:.1%}); "
              + "; ".join(f"{c}: {sps[c]:,.0f}" if sps[c] else f"{c}: never 95%" for c in COUNTS)
              + f"; 50 (part 2): {sps['picked'] / sps[50]:.2f}x",
              {"tuning": t, "searches_per_s": {str(k): v for k, v in sps.items()}, "picked": pick, "build_s": secs})
    rep.check(f"{name}: the 95% target gives at least 94%", acc >= 0.94, f"{acc:.2%} on the tuning questions")
    rep.note(f"{name}: build {secs:.1f} s with tuning ({t['seconds']:.1f} s in the trial before it; "
             f"{trial_s} re-check counts timed after it)")


def main():
    rep = Report("step24")
    for name in sys.argv[1:] or list(DATASETS):
        one(rep, name)
    rep.finish()


if __name__ == "__main__":
    main()
