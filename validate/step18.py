"""Step 18: picking its own settings.
  - the true best: every choice the tuning can make (links per item 24-96 x slack 1.0-1.3) built at full size on
    the five datasets, each timed on one core at 95% on the tuning questions;
  - the self-picked choice (tune="auto", with the build's own sample questions only) must be within 10% of the
    best's speed; it is also compared with part 1's single setting (32 links, slack 1.2);
  - its extra build time is reported;
  - a memory limit keeps the choice within it; the tuning record comes back after save and load.

.venv/bin/python -I validate/step18.py [dataset ...]
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, TARGETS, Report, accuracy, correct_counts, cost_at_target, load, truth
from step11 import FINAL

# part 2's design without the tuning; each choice of the grid is built with it
PART2 = dict(FINAL, effort_apart=True, audit_budget=0, track_usage=False)
GRID = [(r, a) for r in (24, 32, 48, 64, 96) for a in (1.0, 1.1, 1.2, 1.3)]


def plan95(idx, name):
    """Accuracy on the tuning questions at every target (16 threads): the two targets around 95%, the weight between
    them, and checks per search at 95%. None when 95% is never reached."""
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    accs, checks = [], []
    for T in TARGETS:
        ids, _, c = idx.search(Q, k=10, target=T, threads=16)
        accs.append(accuracy(correct_counts(ids, Q, X, td, metric)))
        checks.append(float(c.mean()))
    accs = np.maximum.accumulate(accs)
    hit = next((i for i, a in enumerate(accs) if a >= 0.95), None)
    if hit is None:
        return None
    chk, _ = cost_at_target(list(accs), checks)
    if hit == 0:
        return TARGETS[0], TARGETS[0], 0.0, chk
    w = (0.95 - accs[hit - 1]) / (accs[hit] - accs[hit - 1]) if accs[hit] > accs[hit - 1] else 1.0
    return TARGETS[hit - 1], TARGETS[hit], w, chk


def sec_at(idx, Q, target, min_s=0.3):
    """One core: seconds per search at a target, over one run of at least min_s (the questions repeated to fill it)."""
    t = time.perf_counter()
    idx.search(Q, k=10, target=target, threads=1)
    repeat = max(1, int(np.ceil(min_s / max(time.perf_counter() - t, 1e-6))))
    t = time.perf_counter()
    for _ in range(repeat):
        idx.search(Q, k=10, target=target, threads=1)
    return (time.perf_counter() - t) / repeat / len(Q)


def sec95(idx, Q, plan, runs=3):
    """Seconds per search at exactly 95%: between the two targets around it, each the fastest of `runs` runs."""
    lo, hi, w, _ = plan
    s0 = min(sec_at(idx, Q, lo) for _ in range(runs))
    return s0 if hi == lo else s0 + w * (min(sec_at(idx, Q, hi) for _ in range(runs)) - s0)


def speed95(idx, name):
    """(searches per second at 95% on one core, checks per search), tuning questions."""
    p = plan95(idx, name)
    if p is None:
        return None, None
    return 1 / sec95(idx, load(name, "q_tune"), p), p[3]


def head_to_head(name, configs, part2=None, rounds=5):
    """Build each (links, slack, copy bits) again and time them taking turns, `rounds` times; the fastest round of
    each."""
    part2 = part2 or PART2
    Q = load(name, "q_tune")
    built = {}
    for r, a, b in configs:
        idx, _ = build(name, dict(part2, tune="off", r=r, alpha=a, code_bits=b))
        built[(r, a, b)] = (idx, plan95(idx, name))
    best = {k: float("inf") for k in built}
    for _ in range(rounds):
        for k, (idx, p) in built.items():
            if p is not None:
                best[k] = min(best[k], sec95(idx, Q, p, runs=1))
    return {k: (1 / v if np.isfinite(v) else None) for k, v in best.items()}


def build(name, settings):
    X = load(name, "slice")
    idx = vindex.Index(X.shape[1], DATASETS[name], seed=7, **settings)
    t = time.perf_counter()
    idx.build(X, np.arange(len(X)), train_questions=load(name, "q_train"))
    return idx, time.perf_counter() - t


def grid(name, configs=None, part2=None, saved=None):
    """Every choice built with fixed settings and timed. With REUSE_GRID=1 and `saved` present, the saved grid is
    read instead: it does not depend on the tuning, and the pick is timed again head-to-head anyway."""
    if saved and os.environ.get("REUSE_GRID") == "1" and saved.exists():
        print(f"      {name}: reusing {saved.name}", flush=True)
        return json.loads(saved.read_text())
    part2 = part2 or PART2
    rows = []
    for r, a, b in configs or [(r, a, 8) for r, a in GRID]:
        idx, secs = build(name, dict(part2, tune="off", r=r, alpha=a, code_bits=b))
        sps, chk = speed95(idx, name)
        m = idx.memory()
        rows.append({"r": r, "alpha": a, "code_bits": b, "searches_per_s": sps, "checks": chk, "build_s": secs,
                     "bytes_per_item": m["total_bytes"] / m["items"]})
        print(f"      {name} r={r} alpha={a} bits={b}: {sps or 0:,.0f}/s, {chk or 0:,.0f} checks, build {secs:.1f} s", flush=True)
    return rows


def picked(rep, name, rows, part2=None, label="the self-picked choice is within 10% of the best"):
    part2 = part2 or PART2
    idx, secs = build(name, dict(part2, tune="auto"))
    t = idx.tuning()
    key = lambda r, a, b: (r, round(a, 2), b)
    choice = key(t["chosen_r"], t["chosen_alpha"], t.get("chosen_code_bits", 8))
    by = {key(x["r"], x["alpha"], x["code_bits"]): x for x in rows}
    top = sorted(rows, key=lambda x: -(x["searches_per_s"] or 0))[:3]
    start = key(32, 1.2, 8)
    configs = list(dict.fromkeys([choice, start] + [key(x["r"], x["alpha"], x["code_bits"]) for x in top]))
    final = head_to_head(name, configs, part2)
    best = max(final, key=lambda k: final[k] or 0)
    ratio = (final[choice] or 0) / final[best]
    part1 = final[start]
    plain = by[start]["build_s"]
    desc = lambda k: f"{k[0]} links, slack {k[1]:.1f}" + (", 4-bit copies" if k[2] == 4 else "")
    rep.check(f"{name}: {label}", ratio >= 0.9,
              f"picked {desc(choice)}: {final[choice]:,.0f}/s; best {desc(best)}: {final[best]:,.0f}/s ({ratio:.0%} of it); "
              f"32 links, slack 1.2: {part1:,.0f}/s ({final[choice] / part1:.2f}x from tuning). Timed head-to-head, "
              f"5 rounds: the pick, the starting setting and the grid's top 3. Build {secs:.1f} s with tuning "
              f"({t['seconds']:.1f} s of it, {len(t['trials'])} trials) against {plain:.1f} s ({secs / plain:.1f}x); memory "
              f"{by[choice]['bytes_per_item']:,.0f} bytes per item (starting setting: {by[start]['bytes_per_item']:,.0f})",
              {"tuning": t, "grid": rows, "head_to_head": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in final.items()}, "build_s": secs})
    return idx


def limits_and_saving(rep, name="glove"):
    X = load(name, "slice")
    cap = 376   # part 1's 32 links on GloVe: 41 slots x 4 + 4, 200 one-byte copies, an 8-byte id
    idx, _ = build(name, dict(PART2, tune="auto", max_bytes_per_item=cap))
    t = idx.tuning()
    m = idx.memory()
    per = m["total_bytes"] / m["items"]
    # the index's own count adds 8 bytes for the whole index, hence the small allowance
    rep.check(f"{name}: a memory limit keeps the choice within it", per <= cap + 0.01 and all(x["bytes_per_item"] <= cap for x in t["trials"]),
              f"limit {cap} bytes per item: picked {t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, "
              f"{per:,.0f} bytes per item; {len(t['trials'])} choices tried")
    d = Path(tempfile.mkdtemp(prefix="vindex-step18-"))
    try:
        idx.save(str(d / "ix"))
        back = vindex.Index.load(str(d / "ix"))
        Q = load(name, "q_tune")
        same = back.tuning() == t and np.array_equal(back.search(Q, k=10, target=0.95)[0], idx.search(Q, k=10, target=0.95)[0])
        rep.check(f"{name}: the tuning record comes back after save and load", same, "same record and answers" if same else "differ")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    rep = Report("step18")
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        saved = OUT / f"step18-grid-{name}.json"
        rows = grid(name, saved=saved)
        saved.write_text(json.dumps(rows, indent=1))
        picked(rep, name, rows)
    limits_and_saving(rep)
    rep.finish()


if __name__ == "__main__":
    main()
