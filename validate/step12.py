"""Step 12: speed tuning. Tuning may change only time, never results: after each change, the build and every
search must give exactly what the untuned version gave (same build work, same items, same distances, same checks)
at every learned-effort target and at a fixed list size, on all five datasets. Then speed is measured as in
Step 11 (one core, middle of 5 timed runs after a warm-up, read off the curve at 95%). Faster means at least
10% faster: the untuned build, timed twice, differed by 6-8% between runs (0.92x and 0.94x on GloVe and Llama).

  .venv/bin/python -I validate/step12.py record          # once, with the untuned build (done 2026-11-04)
  .venv/bin/python -I validate/step12.py [dataset ...]   # after each change

`record` saved the untuned answers and speeds in validate/results/step12-before/; the untuned code itself is
gone, so these files are the fixed reference.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, TARGETS, Report, build_index, load
from step11 import FINAL, measure

BEFORE = OUT / "step12-before"


def answers(idx, Q):
    out = {}
    for i, T in enumerate(TARGETS):
        out[f"ids_{i}"], out[f"d_{i}"], out[f"c_{i}"] = idx.search(Q, k=10, target=T, threads=16)
    out["ids_fixed"], out["d_fixed"], out["c_fixed"] = idx.search(Q, k=10, list_size=100, threads=16)
    return out


def speed(m):
    return {"qps95": m["qps95"], "checks95": m["checks95"], "qps_all": m["qps_all"],
            "median_ms": m["median_ms"], "p99_ms": m["p99_ms"]}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["record"]:
        if any(BEFORE.glob("*.npz")):
            sys.exit(f"{BEFORE} already holds the untuned recording; recording again would compare the tuned "
                     "build with itself. Move that folder away first if you really mean to replace it.")
        BEFORE.mkdir(parents=True, exist_ok=True)
        for name in (args[1:] or DATASETS):
            idx, build_s = build_index(name, FINAL, train=True)
            np.savez_compressed(BEFORE / f"{name}.npz", **answers(idx, load(name, "q_test")))
            (BEFORE / f"{name}.json").write_text(json.dumps(
                {"build_checks": idx.stats()["build_checks"], "build_seconds": build_s, **speed(measure(name, idx))}, indent=1))
            print(f"recorded {name}")
        sys.exit(0)
    rep = Report("step12")
    for name in (args or DATASETS):
        ref = np.load(BEFORE / f"{name}.npz")
        before = json.loads((BEFORE / f"{name}.json").read_text())
        idx, build_s = build_index(name, FINAL, train=True)
        now = answers(idx, load(name, "q_test"))
        diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
        same_build = idx.stats()["build_checks"] == before["build_checks"]
        rep.check(f"{name}: same build work, and the same items, distances and checks at every effort",
                  same_build and not diff, "identical" if same_build and not diff else
                  f"build work {'same' if same_build else 'differs'}; differing: {', '.join(diff[:6])}")
        m = speed(measure(name, idx))
        rep.check(f"{name}: at least 10% faster at 95% (one core) than before tuning", m["qps95"] > 1.1 * before["qps95"],
                  f"{before['qps95']:,.0f} -> {m['qps95']:,.0f} searches/s ({m['qps95'] / before['qps95']:.2f}x); "
                  f"all cores {before['qps_all']:,.0f} -> {m['qps_all']:,.0f}; typical {before['median_ms']:.3f} -> "
                  f"{m['median_ms']:.3f} ms, slow {before['p99_ms']:.3f} -> {m['p99_ms']:.3f} ms; "
                  f"build {before['build_seconds']:.1f} -> {build_s:.1f} s",
                  {"before": before, "after": {**m, "build_seconds": build_s}})
    rep.finish()
