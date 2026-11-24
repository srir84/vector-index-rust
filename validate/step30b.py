"""Step 30b: the biggest moves from the samples, timed again head to head (part 4).
  .venv/bin/python -I validate/step30b.py [dataset ...]      # default: imagenet yandex
In the samples we were level with Glass on ImageNet and Yandex; at full size step 30 has us well ahead. Our runs
(step 29) and Glass's (step 30) were timed hours apart, so here each pair is timed again one right after the other:
Glass's step 30 run again (the experiments' own worker, the same settings and search effort, the same measuring),
then ours (step 29's build and measuring), on the same machine. Nothing is written to step 29's or step 30's files.
Pass: the ratio between the two comes out within 10% of step 30's.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import EXP, Report, build_index, frozen_changes
from step11 import measure, others_cpu
from step27 import PART3
from step30 import FULL, ours_full, rival

TOOL = "glass"


def glass_again(name, work):
    """Glass's step 30 final run on `name` again, into `work`; returns its speed at 95% and the other programs' load."""
    rec = json.loads((FULL / "final" / TOOL / name / "best.json").read_text())
    out = work / f"{TOOL}-{name}.json"
    job = {"tool": TOOL, "dataset": name, "params": rec["params"], "qset": "test", "out": str(out), "final": True,
           "effort": rec["final"]["effort"], "reps": 5, "size": "full"}
    env = dict(os.environ, BENCH_UNLOCK_HELD_BACK="1", BENCH_SCRATCH=str(work))
    before = others_cpu()
    subprocess.run([str(EXP / ".venv" / "bin" / "python"), "-I", "-m", "bench.runners.worker", json.dumps(job)],
                   cwd=EXP, env=env, check=True)
    r = json.loads(out.read_text())
    return {"qps95": 1 / r["sec_per_query_at95"], "status": r["status_at95"], "params": rec["params"],
            "effort": job["effort"], "other_programs_cores": [before, others_cpu()]}


def ours_again(name):
    idx, build_s = build_index(name, PART3, train=True, size="full")
    before = others_cpu()
    m = measure(name, idx, size="full")
    return {"qps95": m["qps95"], "checks95": m["checks95"], "target": m["target"], "build_seconds": build_s,
            "other_programs_cores": [before, others_cpu()]}


def main():
    common.unlock_held_back()
    rep = Report("step30b")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    recorded = ours_full()
    with tempfile.TemporaryDirectory(prefix="vindex-step30b-") as work:
        for name in sys.argv[1:] or ["imagenet", "yandex"]:
            t = time.time()
            g = glass_again(name, Path(work))
            o = ours_again(name)
            was_o, was_g = recorded[name]["ours"]["qps95"], rival(TOOL, name)["qps95"]
            ratio, was = o["qps95"] / g["qps95"], was_o / was_g
            rep.check(f"{name}: timed again head to head, the ratio to Glass is within 10% of step 30's",
                      abs(ratio / was - 1) <= 0.10,
                      f"ours {o['qps95']:,.0f}/s (step 29: {was_o:,.0f}), Glass {g['qps95']:,.0f}/s (step 30: {was_g:,.0f}), "
                      f"so {ratio:.2f}x (step 30: {was:.2f}x); other programs using "
                      f"{max(g['other_programs_cores'] + o['other_programs_cores']):.1f} cores at most; "
                      f"{(time.time() - t) / 60:.0f} min",
                      {"dataset": name, "ours": o, "glass": g, "ratio": ratio, "recorded": {"ours": was_o, "glass": was_g,
                                                                                            "ratio": was}})
    rep.finish()


if __name__ == "__main__":
    main()
