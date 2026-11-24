"""Step 42: frozen again; full size; the held-back test (part 5).
  .venv/bin/python -I validate/step42.py freeze    # first: the fingerprint (validate/results/step42-frozen.json)
  .venv/bin/python -I validate/step42.py rivals    # GloVe-100's rivals: the experiments' scripts/step7_baselines.py
  .venv/bin/python -I validate/step42.py ours      # our index at full size on the twelve and GloVe-100 (as step 29)
  .venv/bin/python -I validate/step42.py changes   # the change test at full size on all thirteen (as step 32)
  .venv/bin/python -I validate/step42.py           # then: against part 4 and the rivals; SPEC part 5's targets
Part 5's index (every default: the steady setting search of step 37, the repair of step 38) is measured as part 4
measured part 3's: test questions, one core, middle of 5 timed runs, read off at exactly 95%. The rivals' part 4 runs
stand for the twelve; GloVe-100's rivals (ScaNN, Glass, HNSW) run here, each with its own settings search, as step 30.
Pass: SPEC.md part 5's targets; step 36's expectations are marked right or wrong.
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
from common import EXP, OUT, Report, build_index, fingerprints, frozen_changes
from step11 import NAMES, measure, measure_ssd, others_cpu
from step27 import PART3
from step29 import ALL, picks
from step30 import docker, ours_full, rival
from step38 import change_test

PART5 = dict(PART3, repair="occlude", n_replace=8)   # build_index starts from part 1's repair: set part 5's
NEW = "glove100"
EVERY = [*ALL, NEW]
FROZEN5 = OUT / "step42-frozen.json"
FULL7 = EXP / "results" / "step7"


def freeze():
    if FROZEN5.exists():
        raise SystemExit(f"{FROZEN5.name} is already recorded; it is never written twice")
    FROZEN5.write_text(json.dumps(fingerprints(), indent=1))
    print("wrote", FROZEN5)


def frozen_check(rep):
    changed = frozen_changes(frozen=FROZEN5)
    rep.check("the code is as frozen in step 42", not changed, "unchanged" if not changed else "; ".join(changed[:5]))


def rivals():
    env = dict(os.environ, BENCH_UNLOCK_HELD_BACK_5="1", BENCH_SCRATCH=os.environ.get("BENCH_SCRATCH", tempfile.gettempdir()))
    py = str(EXP / ".venv" / "bin" / "python")
    run = lambda *a: subprocess.run([py, "-I", "scripts/step7_baselines.py", *a], cwd=EXP, env=env, check=True)
    try:   # ScaNN first, with Docker up only for it
        docker(True)
        run("grid", "scann")
        run("final", "scann")
    except Exception as e:
        print(f"ScaNN runs failed: {e}", flush=True)
    finally:
        docker(False)
    run("grid", "glass", "hnsw")
    run("final", "glass", "hnsw")


def ours():
    common.unlock_held_back()
    common.unlock_held_back_5()
    rep = Report("step42-ours")
    frozen_check(rep)
    work = Path(tempfile.mkdtemp(prefix="vindex-step42-"))
    try:
        for name in EVERY:
            idx, build_s = build_index(name, PART5, train=True, size="full")
            before = others_cpu()
            m = measure(name, idx, size="full")
            m["other_programs_cores"] = [before, others_cpu()]
            mem = idx.memory()
            ssd = measure_ssd(name, idx, m["target"], work)
            t, st = idx.tuning(), idx.stats()
            rep.check(f"{name}: at full size, the 95% target picked on the tuning questions finds at least 94%",
                      m["accuracy_at_target"] >= 0.94 and ssd["same_answers"],
                      f"{mem['items']:,} items; {m['qps95'] or 0:,.0f}/s at 95% (1 core; {m['checks95'] or 0:,.0f} checks); "
                      f"target {m['target']} finds {m['accuracy_at_target']:.2%}; all cores {m['qps_all']:,.0f}/s; build "
                      f"{build_s:.0f} s; {mem['total_bytes'] / mem['items']:,.0f} bytes/item; picked {picks(t, st)}; SSD mode "
                      f"{'same answers' if ssd['same_answers'] else 'DIFFERENT answers'}, {ssd['qps']:,.0f}/s",
                      {"ours": m, "build_seconds": build_s, "bytes_per_item": mem["total_bytes"] / mem["items"], "memory": mem,
                       "ssd": ssd, "tuning": t, "stats": st})
            del idx
            shutil.rmtree(work / name, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()


def changes():
    common.unlock_held_back()
    common.unlock_held_back_5()
    rep = Report("step42-changes")
    frozen_check(rep)
    for name in EVERY:
        r = change_test(name, "full", {"repair": "occlude", "n_replace": 8})
        ok = (r["drift"] > -0.01 and r["memory_after_vs_fresh"] <= 1.05
              and all(c["deleted_returned"] == 0 and c["stale_returned"] == 0 for c in r["checkpoints"]))
        rep.check(f"{name}: drift under 1 point, nothing deleted or outdated returned, memory within 5%", ok,
                  f"drift {r['drift'] * 100:+.2f} ({r['checkpoints'][-1]['accuracy']:.2%} after {r['changes']:,} changes, "
                  f"fresh {r['fresh_accuracy']:.2%}); memory {r['memory_after_vs_fresh']:.3f}x", r)
    rep.finish()


def part5_full():
    r = json.loads((OUT / "step42-ours.json").read_text())
    return {c["check"].split(":")[0]: c["data"] for c in r["checks"] if c.get("data") and "at full size" in c["check"]}


def check():
    rep = Report("step42")
    frozen_check(rep)
    now, before = part5_full(), ours_full()
    ch = {c["data"]["dataset"]: (c["data"], c["pass"]) for c in json.loads((OUT / "step42-changes.json").read_text())["checks"]
          if isinstance(c.get("data"), dict) and "dataset" in c["data"]}
    ratios, rows = {}, []
    for name in EVERY:
        where = FULL7 if name == NEW else EXP / "results" / "step6"
        rv = {t: r for t in ("hnsw", "glass", "scann", "roargraph", "corenn") if (r := rival(t, name, where))}
        reach = {t: r for t, r in rv.items() if r["qps95"]}
        best = max(reach, key=lambda t: reach[t]["qps95"], default=None)
        q = now[name]["ours"]["qps95"]
        ratios[name] = (q / reach[best]["qps95"], best) if best else (None, None)
        was = before.get(name, {}).get("ours", {}).get("qps95")
        rows.append(f"{name}: {q:,.0f}/s" + (f" (part 4: {was:,.0f}/s, {q / was:.2f}x)" if was else "")
                    + (f"; {q / reach[best]['qps95']:.2f}x {NAMES[best]}" if best else "; no rival reached 95%")
                    + (f"; HNSW {q / rv['hnsw']['qps95']:.1f}x" if rv.get("hnsw", {}).get("qps95") else "; HNSW never 95%"))
    rep.note("\n      ".join(rows))
    won4 = [n for n in ALL if n != "glove"]
    lost = [n for n in won4 if not ratios[n][0] or ratios[n][0] < 1.05]
    rep.check("still ahead of the fastest rival on the eleven datasets won in part 4", not lost,
              "all eleven ahead (" + ", ".join(f"{n} {ratios[n][0]:.2f}x" for n in won4) + ")" if not lost
              else "not ahead on " + ", ".join(f"{n} ({ratios[n][0] or 0:.2f}x)" for n in lost), {"ratios": ratios})
    bad = [n for n in EVERY if not ch[n][1]]
    rep.check("the change test passes on all thirteen at full size", not bad,
              "; ".join(f"{n} {ch[n][0]['drift'] * 100:+.2f}" for n in EVERY), {n: ch[n][0]["drift"] for n in EVERY})
    g = ratios["glove"][0]
    rep.check("GloVe: the bucket search picked, and 1.25x part 4's speed (not built in, step 41)", False,
              f"the link walk as before: {now['glove']['ours']['qps95']:,.0f}/s, {now['glove']['ours']['qps95'] / before['glove']['ours']['qps95']:.2f}x "
              f"part 4, {g:.2f}x ScaNN")
    r100, best100 = ratios[NEW]
    h100 = rival("hnsw", NEW, FULL7)
    exp = [
        {"expected": "GloVe-100: ScaNN is the fastest rival, then Glass", "right": best100 == "scann",
         "what happened": f"fastest: {NAMES.get(best100, best100)}"},
        {"expected": "GloVe-100: ours within 10% of ScaNN (0.9-1.1x)", "right": bool(r100) and best100 == "scann" and 0.9 <= r100 <= 1.1,
         "what happened": f"{r100:.2f}x {NAMES.get(best100, best100)}" if r100 else "no rival reached 95%"},
        {"expected": "GloVe-100: at least 5x HNSW", "right": bool(h100 and h100["qps95"]) and now[NEW]["ours"]["qps95"] / h100["qps95"] >= 5,
         "what happened": f"{now[NEW]['ours']['qps95'] / h100['qps95']:.1f}x HNSW" if h100 and h100["qps95"] else "HNSW never reached 95%"},
        {"expected": "GloVe-100: the change test passes", "right": ch[NEW][1], "what happened": f"drift {ch[NEW][0]['drift'] * 100:+.2f}"},
        {"expected": "GloVe: 1.25-1.5x part 4's speed, within 10% of ScaNN", "right": False,
         "what happened": f"{now['glove']['ours']['qps95'] / before['glove']['ours']['qps95']:.2f}x part 4, {g:.2f}x ScaNN: the bucket search was not built in"},
        {"expected": "The bucket search is picked on GloVe and GloVe-100 and on none of the other eleven", "right": False,
         "what happened": "not built in (step 41)"},
        {"expected": "The other eleven: still ahead of the fastest rival", "right": not lost,
         "what happened": ", ".join(f"{n} {ratios[n][0]:.2f}x" for n in won4)},
    ]
    for e in exp:
        rep.note(f"{'RIGHT' if e['right'] else 'WRONG'}  {e['expected']}: {e['what happened']}")
    (OUT / "step42-expectations.json").write_text(json.dumps(exp, indent=1))
    rep.finish()


if __name__ == "__main__":
    {"freeze": freeze, "rivals": rivals, "ours": ours, "changes": changes}.get(sys.argv[1] if len(sys.argv) > 1 else "", check)()
