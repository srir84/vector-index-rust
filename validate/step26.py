"""Step 26: more links for popular items.
  - on the five, three indexes timed on one core at 95% on the tuning questions, taking turns (5 rounds):
      part 3's index, with every default (the tuning keeps popular items' extra links only where they pay);
      the same index without extra links: built with the settings part 3's tuning picked, popular_links="off",
        and no tuning (a second tuned build could pick differently by timing noise alone);
      part 2's index: its settings as part 2 picked them, 50 re-checks, no extra links, no tuning;
  - Llama must be at least 1.3x faster than part 2's index; the other four no more than 2% slower than without
    the extra links; the 95% target must give at least 94%;
  - what the tuning chose, and the memory the extra links take.

.venv/bin/python -I validate/step26.py [dataset ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, Report, accuracy, correct_counts, load, truth
from step18 import build, plan95, sec95
from step21 import PART2_21

# part 2's picks (STEPS.md steps 21 and 22: links, slack, copy size); part 2 always re-checked 50
PART2_PICKS = json.loads((OUT / "step25-before" / "settings.json").read_text())


def one(rep, name):
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    part3, secs = build(name, dict(PART2_21, tune="auto", popular_links="auto"))
    t, st = part3.tuning(), part3.stats()
    picks = dict(r=t["chosen_r"], alpha=round(t["chosen_alpha"], 2), code_bits=t["chosen_code_bits"], rerank=t["chosen_rerank"])
    without, _ = build(name, dict(PART2_21, tune="off", popular_links="off", **picks))
    part2, _ = build(name, dict(PART2_21, tune="off", popular_links="off", **dict(PART2_PICKS[name], rerank=50)))
    runs = {"part 3": part3, "without extra links": without, "part 2": part2}
    plans = {k: plan95(i, name) for k, i in runs.items()}
    best = {k: float("inf") for k in runs}
    for _ in range(5):
        for k, i in runs.items():
            best[k] = min(best[k], sec95(i, Q, plans[k], runs=1))
    sps = {k: 1 / v for k, v in best.items()}
    acc = accuracy(correct_counts(part3.search(Q, k=10, target=0.95, threads=16)[0], Q, X, td, metric))
    m = part3.memory()
    on = st["popular_items"] > 0
    trials = [x for x in t["trials"] if x["links"].startswith("full size, popular")]
    chose = ("kept: " if on else "dropped: ") + "; ".join(
        f"{x['links'][11:]} {x['searches_per_s'] or 0:,.0f}/s" for x in trials) if trials else "not tried (answers do not bunch up)"
    detail = (f"part 3 {sps['part 3']:,.0f}/s ({plans['part 3'][3]:,.0f} checks); without extra links "
              f"{sps['without extra links']:,.0f}/s ({plans['without extra links'][3]:,.0f}); part 2 {sps['part 2']:,.0f}/s "
              f"({plans['part 2'][3]:,.0f}); {sps['part 3'] / sps['without extra links']:.2f}x from the extra links, "
              f"{sps['part 3'] / sps['part 2']:.2f}x from part 3. Popular links {chose}; {st['popular_items']:,} popular "
              f"items, {st['extra_links']:,} extra links; picked {t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, "
              f"{t['chosen_code_bits']}-bit copies, re-check {t['chosen_rerank']}; {m['total_bytes'] / m['items']:,.0f} bytes "
              f"per item; build {secs:.1f} s")
    data = {"searches_per_s": sps, "checks": {k: p[3] for k, p in plans.items()}, "tuning": t, "stats": st,
            "bytes_per_item": m["total_bytes"] / m["items"], "build_s": secs}
    if name == "llama":
        rep.check(f"{name}: at least 1.3x faster than part 2's index", sps["part 3"] / sps["part 2"] >= 1.3, detail, data)
    else:
        rep.check(f"{name}: no more than 2% slower than without the extra links",
                  sps["part 3"] / sps["without extra links"] >= 0.98, detail, data)
    rep.check(f"{name}: the 95% target gives at least 94%", acc >= 0.94, f"{acc:.2%} on the tuning questions")


def main():
    rep = Report("step26")
    for name in sys.argv[1:] or list(DATASETS):
        one(rep, name)
    rep.finish()


if __name__ == "__main__":
    main()
