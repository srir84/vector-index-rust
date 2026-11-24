"""Step 20: cheaper checks, same results.
  - fetch-ahead hints in the chip's own 128-byte lines, and items renumbered after the build so linked items sit
    near each other in memory;
  - results identical: builds and answers on all five equal step 12's recording (part 1's design, renumbered);
  - speed: one core at 95% on the tuning questions, against the speed recorded with the code just before these
    changes (validate/results/step20-before/). Each speed is the fastest of 5 runs of at least half a second.

  .venv/bin/python -I validate/step20.py record    # once, with the code before step 20 (done before building it)
  .venv/bin/python -I validate/step20.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, Report, build_index, load
from step11 import FINAL
from step12 import BEFORE, answers
from step18 import plan95, sec95

SPEED_BEFORE = OUT / "step20-before"


def speed(idx, name, runs=5):
    p = plan95(idx, name)
    Q = load(name, "q_tune")
    return 1 / min(sec95(idx, Q, p, runs=1) for _ in range(runs)), p[3]


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        if any(SPEED_BEFORE.glob("*.json")):
            sys.exit(f"{SPEED_BEFORE} already holds the recording; move it away first to record again")
        SPEED_BEFORE.mkdir(parents=True, exist_ok=True)
        import vindex
        # the code before step 20 has no `renumber` setting, and nothing to renumber with
        settings = {k: v for k, v in FINAL.items() if k != "renumber"}
        for name in DATASETS:
            X = load(name, "slice")
            idx = vindex.Index(X.shape[1], DATASETS[name], seed=7, **settings)
            idx.build(X, np.arange(len(X)), train_questions=load(name, "q_train"))
            sps, chk = speed(idx, name)
            (SPEED_BEFORE / f"{name}.json").write_text(json.dumps({"searches_per_s": sps, "checks": chk}, indent=1))
            print(f"recorded {name}: {sps:,.0f} searches/s at 95% (tuning questions, one core), {chk:,.0f} checks")
        sys.exit(0)
    rep = Report("step20")
    ratios = []
    for name in DATASETS:
        ref = np.load(BEFORE / f"{name}.npz")
        before = json.loads((BEFORE / f"{name}.json").read_text())
        idx, _ = build_index(name, dict(FINAL, renumber=True), train=True)
        now = answers(idx, load(name, "q_test"))
        diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
        same_build = idx.stats()["build_checks"] == before["build_checks"]
        rep.check(f"{name}: renumbered, build and answers identical to step 12's recording", not diff and same_build,
                  "identical" if not diff and same_build else f"differ: {diff[:4]}, build checks same: {same_build}")
        old = json.loads((SPEED_BEFORE / f"{name}.json").read_text())
        sps, chk = speed(idx, name)
        ratios.append(sps / old["searches_per_s"])
        rep.check(f"{name}: no slower than before (within 3%)", ratios[-1] >= 0.97,
                  f"{sps:,.0f} searches/s at 95% against {old['searches_per_s']:,.0f} ({ratios[-1]:.2f}x); "
                  f"checks {chk:,.0f} (before {old['checks']:,.0f})")
    mean = float(np.mean(ratios))
    rep.check("all five: at least 10% faster on average", mean >= 1.10,
              f"{mean:.2f}x on average (" + ", ".join(f"{n} {r:.2f}x" for n, r in zip(DATASETS, ratios)) + ")")
    rep.finish()
