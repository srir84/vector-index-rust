"""Step 22: the held-back test and report.
  .venv/bin/python -I validate/step22.py rivals   # first: the rivals' settings search and final runs on the four
                                                  # held-back datasets (the experiments' step 3 script)
  .venv/bin/python -I validate/step22.py          # then: part 2's index on all nine, measured as step 11
Part 2's index: every part 2 default (it picks its own settings from its sample questions), built with
tools/build_py_pgo.sh. Measured exactly as step 11 and the rivals were: test questions, one core, middle of 5 timed
runs, read off at 95%; all cores; time per question; build time; memory; the SSD mode.
Targets (SPEC.md, part 2): on each of the five, closer to the fastest rival than part 1 was (step 11's recorded
run); faster than HNSW, DiskANN and CoreNN wherever they reach 95%; the held-back results recorded as they come.
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
from common import DATASETS, EXP, HELD_BACK, OUT, Report, build_index
from step11 import NAMES, measure, measure_ssd, others_cpu, rivals

# every part 2 default: common.build_index otherwise gives part 1's design
PART2 = dict(effort_apart=True, tune="auto", renumber=True, question_bits=16)
TOOLS = ["hnsw", "diskann", "corenn", "glass", "scann", "roargraph"]


def run_rivals():
    env = dict(os.environ, BENCH_DATASETS=",".join(HELD_BACK), BENCH_UNLOCK_HELD_BACK="1",
               BENCH_SCRATCH=os.environ.get("BENCH_SCRATCH", tempfile.gettempdir()))
    py = str(EXP / ".venv" / "bin" / "python")
    for phase in ("grid", "final"):
        subprocess.run([py, "-I", "scripts/step3_baselines.py", phase, *TOOLS], cwd=EXP, env=env, check=True)


def part1_ratio(name):
    """Part 1's speed over the fastest rival's, from step 11's recorded run."""
    r = json.loads((OUT / "step11.json").read_text())
    row = next(c for c in r["checks"] if c["check"].startswith(f"{name}: faster") or c["check"].startswith(f"{name}: reaches"))
    ours, rv = row["data"]["ours"]["qps95"], row["data"]["rivals"]
    fastest = max(x["qps95"] for x in rv.values() if x["qps95"])
    return ours / fastest


def main():
    common.unlock_held_back()
    rep = Report("step22")
    work = Path(tempfile.mkdtemp(prefix="vindex-step22-"))
    try:
        for name in (sys.argv[1:] or [*DATASETS, *HELD_BACK]):
            idx, build_s = build_index(name, PART2, train=True)
            others_before = others_cpu()
            m = measure(name, idx)
            m["other_programs_cores"] = [others_before, others_cpu()]
            mem = idx.memory()
            per = mem["total_bytes"] / mem["items"]
            ssd = measure_ssd(name, idx, m["target"], work)
            t = idx.tuning()
            rv = rivals(name)
            first3 = [r["qps95"] for k, r in rv.items() if k in ("hnsw", "diskann", "corenn") and r["qps95"]]
            reach = {k: r for k, r in rv.items() if r["qps95"]}
            best_t = max(reach, key=lambda k: reach[k]["qps95"], default=None)
            held = " (held back)" if name in HELD_BACK else ""
            detail = (f"ours {m['qps95'] or 0:,.0f}/s at 95% (1 core; {m['checks95'] or 0:,.0f} checks); all cores "
                      f"{m['qps_all']:,.0f}/s; typical {m['median_ms']:.3f} ms, slow {m['p99_ms']:.3f} ms; build {build_s:.1f} s; "
                      f"{per:,.0f} bytes/item; picked " + (f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, "
                      f"{t['chosen_code_bits']}-bit copies" if t else "nothing (too little data)")
                      + f"; other programs using {m['other_programs_cores'][0]:.1f} -> {m['other_programs_cores'][1]:.1f} cores"
                      + "".join(f"\n            {NAMES[k]:9s} " + (f"{r['qps95']:>9,.0f}/s" if r["qps95"] else f"never 95% (best {r['best']:.0%})")
                                + f"  {r['bytes_per_item']:>6,.0f} bytes/item  build {r['build_s']:.0f} s" for k, r in rv.items()))
            data = {"ours": m, "build_seconds": build_s, "bytes_per_item": per, "ssd": ssd, "rivals": rv, "tuning": t}
            if first3:
                rep.check(f"{name}{held}: faster at 95% than HNSW, DiskANN and CoreNN", m["qps95"] and m["qps95"] > max(first3), detail, data)
            else:
                rep.check(f"{name}{held}: reaches 95% (HNSW, DiskANN and CoreNN never do)", m["qps95"] is not None, detail, data)
            if best_t and m["qps95"]:
                ratio = m["qps95"] / reach[best_t]["qps95"]
                if name in DATASETS:
                    before = part1_ratio(name)
                    rep.check(f"{name}: closer to the fastest rival than part 1", ratio > before,
                              f"{ratio:.2f}x {NAMES[best_t]}'s speed (part 1: {before:.2f}x)")
                else:
                    rep.note(f"{name} (held back): {ratio:.2f}x the fastest rival ({NAMES[best_t]})")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["rivals"]:
        run_rivals()
    else:
        main()
