"""Step 25: cheaper checks, same results (2).
  - results identical: on the five, builds and answers equal a recording made with the code just before this step
    (validate/results/step25-before/), with each dataset's settings fixed as step 24 picked them (no tuning, so
    that timing cannot change a choice); and step 12's recording of part 1's design;
  - speed: one core at 95% on the tuning questions, the code before this step (kept in the recording) against the
    code now, on the same saved index, taking turns in separate processes, 5 rounds each; the fastest round counts.

  .venv/bin/python -I validate/step25.py record    # once, with the code before step 25 installed
  .venv/bin/python -I validate/step25.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, ROOT, Report, build_index, load
from step11 import FINAL
from step12 import BEFORE as STEP12, answers
from step18 import build
from step21 import PART2_21

RECORD = OUT / "step25-before"
PY = str(ROOT / ".venv" / "bin" / "python")
RUNNER = str(Path(__file__).resolve().parent / "time_saved.py")
BUILDER = str(Path(__file__).resolve().parent / "build_saved.py")
ROUNDS = 5


def picks():
    """Each dataset's settings as step 24's tuned build picked them."""
    rep = json.loads((OUT / "step24.json").read_text())
    out = {}
    for c in rep["checks"]:
        t = (c.get("data") or {}).get("tuning")
        if t:
            name = c["check"].split(":")[0]
            out[name] = dict(r=t["chosen_r"], alpha=round(t["chosen_alpha"], 2), code_bits=t["chosen_code_bits"],
                             rerank=t["chosen_rerank"])
    return out


def fixed(name, chosen):
    return dict(PART2_21, tune="off", **chosen[name])


def record():
    if any(RECORD.glob("*.npz")):
        sys.exit(f"{RECORD} already holds the recording; move it away first to record again")
    RECORD.mkdir(parents=True, exist_ok=True)
    pkg = Path(vindex.__file__).parent
    shutil.copytree(pkg, RECORD / "package" / pkg.name)
    chosen = picks()
    (RECORD / "settings.json").write_text(json.dumps(chosen, indent=1))
    for name in DATASETS:
        idx, _ = build(name, fixed(name, chosen))
        np.savez(RECORD / f"{name}.npz", **answers(idx, load(name, "q_test")))
        (RECORD / f"{name}.json").write_text(json.dumps({"build_checks": idx.stats()["build_checks"]}, indent=1))
        print(f"recorded {name}: {chosen[name]}", flush=True)


def head_to_head(name, settings):
    """Seconds per search at 95%, before and now, on the same saved index; the fastest of ROUNDS rounds each. The
    code before this step builds and saves it (the same index, as the identity check shows), so that it reads
    even when later code saves settings the earlier code does not know."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / name)
        before = {k: v for k, v in settings.items() if k != "popular_links"}   # off there: it has no such setting
        subprocess.run([PY, "-I", BUILDER, str(RECORD / "package"), name, json.dumps(before), path],
                       capture_output=True, text=True, check=True)
        best = {"before": float("inf"), "now": float("inf")}
        checks = {}
        for r in range(ROUNDS):
            order = ("before", "now") if r % 2 == 0 else ("now", "before")
            for who in order:
                pkg = str(RECORD / "package") if who == "before" else "installed"
                run = subprocess.run([PY, "-I", RUNNER, pkg, path, name], capture_output=True, text=True, check=True)
                out = json.loads(run.stdout.strip().splitlines()[-1])
                best[who] = min(best[who], out["sec"])
                checks[who] = out["checks"]
    return best, checks


def main():
    rep = Report("step25")
    chosen = json.loads((RECORD / "settings.json").read_text())
    ratios = []
    for name in DATASETS:
        # part 1's design against step 12's recording
        idx, _ = build_index(name, dict(FINAL, renumber=True), train=True)
        ref, now = np.load(STEP12 / f"{name}.npz"), answers(idx, load(name, "q_test"))
        diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
        rep.check(f"{name}: part 1's design: answers identical to step 12's recording", not diff,
                  "identical" if not diff else f"differ: {diff[:4]}")
        # part 3's settings against this step's recording
        idx, _ = build(name, fixed(name, chosen))
        ref, now = np.load(RECORD / f"{name}.npz"), answers(idx, load(name, "q_test"))
        diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
        same_build = idx.stats()["build_checks"] == json.loads((RECORD / f"{name}.json").read_text())["build_checks"]
        rep.check(f"{name}: part 3's settings: build and answers identical to the recording", not diff and same_build,
                  f"{chosen[name]}: " + ("identical" if not diff and same_build
                                         else f"differ: {diff[:4]}, build checks same: {same_build}"))
        best, checks = head_to_head(name, fixed(name, chosen))
        ratios.append(best["before"] / best["now"])
        rep.check(f"{name}: no more than 2% slower", ratios[-1] >= 0.98,
                  f"{1 / best['now']:,.0f} searches/s at 95% against {1 / best['before']:,.0f} before "
                  f"({ratios[-1]:.3f}x); checks {checks['now']:,.0f} (before {checks['before']:,.0f}); "
                  f"one core, the same saved index, {ROUNDS} rounds each, taking turns",
                  {"sec_before": best["before"], "sec_now": best["now"]})
    mean = float(np.mean(ratios))
    rep.check("all five: at least 5% faster on average", mean >= 1.05,
              f"{mean:.3f}x on average (" + ", ".join(f"{n} {r:.3f}x" for n, r in zip(DATASETS, ratios)) + ")")
    rep.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        record()
    else:
        main()
