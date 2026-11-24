"""Step 27: the held-back test and report (part 3).
  .venv/bin/python -I validate/step27.py rivals   # first: the rivals' settings search and final runs on the three new
                                                  # held-back datasets (the experiments' step 3 script; ScaNN needs Docker)
  .venv/bin/python -I validate/step27.py          # then: part 3's index on all twelve, measured as steps 11 and 22
Part 3's index: every default (it picks its own settings from its sample questions), built with tools/build_py_pgo.sh.
Measured exactly as steps 11 and 22: test questions, one core, middle of 5 timed runs, read off at 95%; all cores;
time per question; build time; memory; the SSD mode.
Targets (SPEC.md, part 3): still ahead on the six datasets won in part 2; faster than HNSW, DiskANN and CoreNN
wherever they reach 95%; the aim, ahead on Yandex and Llama too, and closer to ScaNN on GloVe. The new held-back
datasets' results are recorded as they come; part 2's four are run a second time (their first run is step 22's).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import DATASETS, EXP, HELD_BACK, HELD_BACK_3, OUT, Report, build_index
from step11 import NAMES, measure, measure_ssd, others_cpu, rivals

# every part 3 default: common.build_index otherwise gives part 1's design
PART3 = dict(effort_apart=True, tune="auto", renumber=True, question_bits=16, popular_links="auto")
TOOLS = ["hnsw", "diskann", "corenn", "glass", "scann", "roargraph"]
WON_IN_PART2 = ["gooaq", "landmark", "arxiv", "laion", "yi", "gist"]


def run_rivals():
    env = dict(os.environ, BENCH_DATASETS=",".join(HELD_BACK_3), BENCH_UNLOCK_HELD_BACK="1",
               BENCH_SCRATCH=os.environ.get("BENCH_SCRATCH", tempfile.gettempdir()))
    py = str(EXP / ".venv" / "bin" / "python")
    for phase in ("grid", "final"):
        subprocess.run([py, "-I", "scripts/step3_baselines.py", phase, *TOOLS], cwd=EXP, env=env, check=True)


def part2_ratio(name):
    """Part 2's speed over the fastest rival's, from step 22's recorded run."""
    r = json.loads((OUT / "step22.json").read_text())
    row = next(c for c in r["checks"] if c["check"].split(" ")[0].rstrip(":") == name and "data" in c and c["data"])
    ours, rv = row["data"]["ours"]["qps95"], row["data"]["rivals"]
    return ours / max(x["qps95"] for x in rv.values() if x["qps95"])


def main():
    common.unlock_held_back()
    rep = Report("step27")
    work = Path(tempfile.mkdtemp(prefix="vindex-step27-"))
    try:
        for name in (sys.argv[1:] or [*DATASETS, *HELD_BACK, *HELD_BACK_3]):
            idx, build_s = build_index(name, PART3, train=True)
            others_before = others_cpu()
            m = measure(name, idx)
            m["other_programs_cores"] = [others_before, others_cpu()]
            mem = idx.memory()
            per = mem["total_bytes"] / mem["items"]
            ssd = measure_ssd(name, idx, m["target"], work)
            t, st = idx.tuning(), idx.stats()
            rv = rivals(name)
            first3 = [r["qps95"] for k, r in rv.items() if k in ("hnsw", "diskann", "corenn") and r["qps95"]]
            reach = {k: r for k, r in rv.items() if r["qps95"]}
            best_t = max(reach, key=lambda k: reach[k]["qps95"], default=None)
            tag = " (held back)" if name in HELD_BACK_3 else " (held back in part 2)" if name in HELD_BACK else ""
            detail = (f"ours {m['qps95'] or 0:,.0f}/s at 95% (1 core; {m['checks95'] or 0:,.0f} checks); all cores "
                      f"{m['qps_all']:,.0f}/s; typical {m['median_ms']:.3f} ms, slow {m['p99_ms']:.3f} ms; build {build_s:.1f} s; "
                      f"{per:,.0f} bytes/item; picked " + (f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, "
                      f"{t['chosen_code_bits']}-bit copies, re-check {t['chosen_rerank']}, popular links "
                      + (f"on ({st['popular_items']:,} items)" if st["popular_items"] else "off") if t else "nothing (too little data)")
                      + f"; other programs using {m['other_programs_cores'][0]:.1f} -> {m['other_programs_cores'][1]:.1f} cores"
                      + "".join(f"\n            {NAMES[k]:9s} " + (f"{r['qps95']:>9,.0f}/s" if r["qps95"] else f"never 95% (best {r['best']:.0%})")
                                + f"  {r['bytes_per_item']:>6,.0f} bytes/item  build {r['build_s']:.0f} s" for k, r in rv.items()))
            data = {"ours": m, "build_seconds": build_s, "bytes_per_item": per, "ssd": ssd, "rivals": rv, "tuning": t, "stats": st}
            if first3:
                rep.check(f"{name}{tag}: faster at 95% than HNSW, DiskANN and CoreNN", m["qps95"] and m["qps95"] > max(first3), detail, data)
            else:
                rep.check(f"{name}{tag}: reaches 95% (HNSW, DiskANN and CoreNN never do)", m["qps95"] is not None, detail, data)
            if best_t and m["qps95"]:
                ratio = m["qps95"] / reach[best_t]["qps95"]
                if name in HELD_BACK_3:
                    rep.note(f"{name} (held back): {ratio:.2f}x the fastest rival ({NAMES[best_t]})")
                    continue
                before = part2_ratio(name)
                line = f"{ratio:.2f}x {NAMES[best_t]}'s speed (part 2: {before:.2f}x)"
                if name in WON_IN_PART2:
                    rep.check(f"{name}: still ahead of the fastest rival", ratio > 1.0, line)
                else:
                    rep.note(f"{name}: {line}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["rivals"]:
        run_rivals()
    else:
        main()
