"""Step 15: usage stats.
  - the counts equal what the searches returned: a batch spread over 16 threads, a batch on one thread, and
    single questions from 8 Python threads at once;
  - the hardness bins: the 3,000 training questions the effort table was learned from fill its 8 bins equally;
  - "questions unlike items", measured on real use, agrees with the build's own test;
  - the time it reports agrees with the wall clock;
  - nothing changes the results: builds and answers on all five equal step 12's recording;
  - the cost: one core, with and without the counters, taking turns.

.venv/bin/python -I validate/step15.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, Report, build_index, load
from step11 import FINAL
from step12 import BEFORE, answers

TARGET = 0.95


def exact_counts(rep, name):
    idx, _ = build_index(name, FINAL, train=True)
    Q = load(name, "q_tune")
    u0 = idx.usage()
    rep.check(f"{name}: the build's own searches are not counted", u0["calls"] == 0 and u0["questions"] == 0,
              f"calls {u0['calls']}, questions {u0['questions']}")

    idx.reset_usage()
    _, _, c = idx.search(Q[:500], k=10, target=TARGET, threads=16)
    a = idx.usage()
    ok_a = (a["calls"], a["questions"], a["checks"], a["learned_questions"], sum(a["hardness_counts"])) == \
           (1, 500, int(c.sum()), 500, 500)

    idx.reset_usage()
    t = time.perf_counter()
    _, _, c = idx.search(Q[500:], k=10, list_size=100, threads=1)
    wall = time.perf_counter() - t
    b = idx.usage()
    ok_b = (b["calls"], b["questions"], b["checks"], b["fixed_questions"], b["rechecks"]) == \
           (1, 500, int(c.sum()), 500, 500 * 50)

    idx.reset_usage()
    totals, lock = [0, 0], threading.Lock()

    def worker(w):
        mine = 0
        for j in range(100):
            q = Q[(w * 100 + j) % len(Q)][None, :]
            kw = {"target": TARGET} if j % 2 else {"list_size": 60}
            mine += int(idx.search(q, k=10, threads=1, **kw)[2].sum())
        with lock:
            totals[0] += mine
            totals[1] += 100

    ths = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    [th.start() for th in ths]
    [th.join() for th in ths]
    cc = idx.usage()
    ok_c = (cc["calls"], cc["questions"], cc["checks"], cc["fixed_questions"]) == (800, 800, totals[0], 400)
    rep.check(f"{name}: counts equal what the searches returned", ok_a and ok_b and ok_c,
              f"16 threads: {a['questions']} questions, {a['checks']:,} checks; one thread: {b['questions']} "
              f"questions, {b['rechecks']:,} re-checks (50 each); 8 Python threads: {cc['calls']} calls, "
              f"{cc['checks']:,} checks (returned {totals[0]:,})", {"a": a, "b": b, "c": cc})
    share = b["seconds"] / wall
    rep.check(f"{name}: the time counted is the time spent", 0.85 <= share <= 1.0,
              f"{b['seconds'] * 1e3:.1f} ms counted over {wall * 1e3:.1f} ms of wall clock ({share:.0%}); "
              f"typical {b['typical_ms']:.3f} ms, slow {b['slow_ms']:.3f} ms per question")
    return idx


def hardness_and_unlike(rep, name, idx):
    """The table's cut points are the 8 quantiles of the first 3,000 training questions' hardness, so searching
    those same questions must put 375 in each bin. The build's 'unlike items' ratio came from 256 training
    questions; real use measures it on each search's best answer."""
    tq = load(name, "q_train")[:3000]
    idx.reset_usage()
    idx.search(tq, k=10, target=TARGET, threads=16)
    u = idx.usage()
    counts = u["hardness_counts"]
    rep.check(f"{name}: the sample questions fill the 8 hardness bins equally", all(abs(c - 375) <= 1 for c in counts),
              f"bins (hardest first): {counts}")
    idx.reset_usage()
    idx.search(load(name, "q_tune"), k=10, target=TARGET, threads=16)
    use, build = idx.usage()["unlike_items"], idx.stats()["ood_ratio"]
    if DATASETS[name] == "cos":
        ok = abs(use / build - 1) <= 0.15
        rule = "within 15%"
    else:  # best score: the best answer is not the nearest by angle, so only the decision must agree
        ok = (use > 1.5) == (build > 1.5)
        rule = "same side of the 1.5 switch"
    rep.check(f"{name}: 'questions unlike items' in use agrees with the build's test", ok,
              f"in use {use:.2f}, at build {build:.2f} (pass: {rule})")


def identical(rep, name):
    ref = np.load(BEFORE / f"{name}.npz")
    before = json.loads((BEFORE / f"{name}.json").read_text())
    idx, _ = build_index(name, FINAL, train=True)
    now = answers(idx, load(name, "q_test"))
    diff = [k for k in ref.files if not np.array_equal(ref[k], now[k])]
    same_build = idx.stats()["build_checks"] == before["build_checks"]
    rep.check(f"{name}: build and answers identical to step 12's recording", not diff and same_build,
              "identical" if not diff and same_build else f"differ: {diff[:4]}, build checks same: {same_build}")


def overhead(rep, name, rounds=9):
    """One core, 1,000 tuning questions at 95%: the index with the counters and an identical one without them
    take turns, so both see the same machine."""
    on, _ = build_index(name, FINAL, train=True)
    off, _ = build_index(name, dict(FINAL, track_usage=False), train=True)
    Q = load(name, "q_tune")
    for idx in (on, off):
        idx.search(Q, k=10, target=TARGET, threads=1)
    t_on, t_off = [], []
    for _ in range(rounds):
        for idx, ts in ((on, t_on), (off, t_off)):
            t = time.perf_counter()
            idx.search(Q, k=10, target=TARGET, threads=1)
            ts.append(time.perf_counter() - t)
    slow = float(np.median(t_on) / np.median(t_off) - 1)
    rep.check(f"{name}: the counters cost under 1% (one core)", slow < 0.01,
              f"{len(Q) / np.median(t_on):,.0f} searches/s with, {len(Q) / np.median(t_off):,.0f} without "
              f"({slow:+.2%}; middle of {rounds} turns each)", {"on": t_on, "off": t_off})


def main():
    rep = Report("step15")
    for name in ("glove", "yandex", "llama"):
        idx = exact_counts(rep, name)
        hardness_and_unlike(rep, name, idx)
    for name in DATASETS:
        identical(rep, name)
    for name in ("llama", "landmark"):
        overhead(rep, name)
    rep.finish()


if __name__ == "__main__":
    main()
