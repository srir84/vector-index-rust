"""Step 37: steady picks (part 5).
  .venv/bin/python -I validate/step37.py measure [dataset ...]   # first, on the code as part 4 froze it
  .venv/bin/python -I validate/step37.py [dataset ...]           # after the fix: the twelve samples
  .venv/bin/python -I validate/step37.py full                    # after the fix: Yandex and ImageNet at full size
Each dataset is built three times from scratch as part 3's index (every default, the 20,000 training questions),
keeping every trial its setting search timed, and each build is measured as step 27 measured it (test questions,
one core, middle of 5 timed runs, read off at 95%).
`measure` writes validate/results/step37-before.json and judges nothing. It found that the same picks always give
the same index (the same checks per search), and that runs as short as step 27's move by up to 7% between
identical indexes; so speeds are compared head to head instead: each index at the cheapest target reaching 95% on
the tuning questions, the test questions repeated to fill half a second, taking turns, the fastest of 5 rounds.
Pass (after the fix): the three builds of each dataset pick the same settings (and so the same index); the three,
timed head to head, are within 3% of each other; and the slowest is no more than 3% slower, head to head, than the
fastest of the picks made before the fix (on the samples: step37-before.json's three builds; at full size: step 29's).
The first version of the fix (ties within 2% to the plainer choice, no longer timing for close calls) is kept in
step37-first-rule.json and step37-full-first-rule.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import (DATASETS, HELD_BACK, HELD_BACK_3, OUT, TARGET, TARGETS, Report, accuracy, build_index,
                    correct_counts, load, metric_of, truth)
from step11 import measure
from step27 import PART3
from step30 import ours_full

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]
BUILDS = 3
SLOWER = 0.03
ROUNDS, MIN_RUN_S = 5, 0.5


def settings_of(p):
    """A build's picks as fixed settings, with the setting search off."""
    return dict(PART3, tune="off", r=p["r"], alpha=p["alpha"], code_bits=p["code_bits"], rerank=p["rerank"],
                question_links="on" if p["question_links"] else "off", popular_links="on" if p["popular_links"] else "off")


def head_to_head(name, indexes, size):
    """Seconds per question of each index, one core, at its cheapest target reaching 95% on the tuning questions:
    the test questions repeated to fill MIN_RUN_S, the indexes taking turns, the fastest of ROUNDS rounds."""
    metric = metric_of(name)
    X = load(name, "slice" if size == "slice" else "items_full")
    Q, Qt = load(name, "q_test"), load(name, "q_tune")
    tdt = truth(name, f"{size}_tune")["dists"]
    targets = [next((t for t in TARGETS if accuracy(correct_counts(i.search(Qt, k=10, target=t, threads=16)[0], Qt, X, tdt,
                                                                   metric)) >= TARGET), TARGETS[-1]) for i in indexes]
    best = [float("inf")] * len(indexes)
    for _ in range(ROUNDS):
        for j, (idx, t) in enumerate(zip(indexes, targets)):
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < MIN_RUN_S:
                idx.search(Q, k=10, target=t, threads=1)
                n += 1
            best[j] = min(best[j], (time.perf_counter() - t0) / (n * len(Q)))
    return best, targets


def picks(idx):
    """What a build picked: the setting search's choices and the switches the build decided."""
    t, st = idx.tuning() or {}, idx.stats()
    return {"r": t.get("chosen_r"), "alpha": round(t["chosen_alpha"], 3) if t else None, "code_bits": t.get("chosen_code_bits"),
            "rerank": t.get("chosen_rerank"), "question_links": bool(st["question_links"]),
            "popular_links": bool(st["popular_items"])}


def builds(name, size, keep=False):
    out, kept = [], []
    for b in range(BUILDS):
        idx, build_s = build_index(name, PART3, train=True, size=size)
        m = measure(name, idx, size=size)
        out.append({"picks": picks(idx), "qps95": m["qps95"], "checks95": m["checks95"], "build_seconds": build_s,
                    "trials": (idx.tuning() or {}).get("trials", [])})
        if keep:
            kept.append(idx)
        del idx
    return out, kept


def before_picks(name, size):
    """The picks made before the fix: step37-before.json's three builds (samples), or step 29's (full size)."""
    if size == "slice":
        bs = json.loads((OUT / "step37-before.json").read_text())[name]
        return [p for i, p in enumerate(b["picks"] for b in bs) if p not in [b["picks"] for b in bs][:i]]
    rec = ours_full()[name]
    t, st = rec["tuning"], rec["stats"]
    return [{"r": t["chosen_r"], "alpha": round(t["chosen_alpha"], 3), "code_bits": t["chosen_code_bits"],
             "rerank": t["chosen_rerank"], "question_links": bool(st["question_links"]), "popular_links": bool(st["popular_items"])}]


def main():
    common.unlock_held_back()
    mode = sys.argv[1] if len(sys.argv) > 1 else "samples"
    if mode == "measure":
        rec = {}
        for name in sys.argv[2:] or ALL:
            rec[name] = builds(name, "slice")[0]
            ps = [b["picks"] for b in rec[name]]
            qs = [b["qps95"] for b in rec[name]]
            print(f"{name:9s} {'same' if all(p == ps[0] for p in ps) else 'DIFFER'}  speeds "
                  + ", ".join(f"{q:,.0f}" for q in qs) + f"  spread {max(qs) / min(qs) - 1:.1%}  "
                  + " | ".join(f"{p['r']}/{p['alpha']}/{p['code_bits']}b/rr{p['rerank']}/pl{int(p['popular_links'])}" for p in ps),
                  flush=True)
        (OUT / "step37-before.json").write_text(json.dumps(rec, indent=1))
        return
    size = "full" if mode == "full" else "slice"
    names = ["yandex", "imagenet"] if mode == "full" else (sys.argv[1:] if mode != "samples" else ALL)
    rep = Report("step37-full" if mode == "full" else "step37")
    for name in names:
        bs, idxs = builds(name, size, keep=True)
        ps, checks = [b["picks"] for b in bs], [b["checks95"] for b in bs]
        same = all(p == ps[0] for p in ps) and all(c == checks[0] for c in checks)
        rep.check(f"{name}: {BUILDS} fresh builds pick the same settings, so the same index", same,
                  f"picks {ps[0] if same else ps}; checks per search {', '.join(f'{c:,.2f}' for c in checks)}",
                  {"builds": bs})
        before = before_picks(name, size)
        olds = [p for p in before if p not in ps]   # a pick also made now is timed as that build
        others = [build_index(name, settings_of(p), train=True, size=size)[0] for p in olds]
        sec, targets = head_to_head(name, [*idxs, *others], size)
        mine = sec[:BUILDS]
        theirs = [mine[ps.index(p)] if p in ps else sec[BUILDS + olds.index(p)] for p in before]
        spread = max(mine) / min(mine) - 1
        rep.check(f"{name}: the {BUILDS} builds within 3% of each other, head to head", spread <= SLOWER,
                  f"{', '.join(f'{1 / s:,.0f}/s' for s in mine)}: {spread:.1%} apart",
                  {"sec_per_question": mine, "targets": targets[:BUILDS]})
        worst = max(mine) / min(theirs) - 1
        tag = lambda p: f"{p['r']}/{p['alpha']}/{p['code_bits']}b/rr{p['rerank']}"
        rep.check(f"{name}: the slowest build no more than 3% slower, head to head, than the fastest pick made before the fix",
                  worst <= SLOWER,
                  f"slowest now {1 / max(mine):,.0f}/s against " + ", ".join(f"{1 / s:,.0f}/s ({tag(p)})" for s, p in zip(theirs, before))
                  + (f": {worst:.1%} slower" if worst > 0 else f": {-worst:.1%} faster"),
                  {"now": ps, "before": before, "sec_per_question": theirs, "targets": targets[BUILDS:]})
        del idxs, others
    rep.finish()


if __name__ == "__main__":
    main()
