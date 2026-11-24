"""Step 17: learning from use.
  - at build: the effort table learns from sample questions the question links don't use (effort_apart);
  - wrong sample questions (taken from the items): after 2,000 real questions are spot-checked, the table is
    learned again on its own and accuracy comes back to the target;
  - right sample questions: the same use starts no relearn; a forced one moves accuracy by under 1 point;
  - searches run while the table is learned again; the new table comes back after save and load.
Accuracy is always measured on the tuning questions. The "real use" is 2,000 training questions the build never
saw (rows 17,000-19,000; the build gets the first 17,000 or, when the sample is wrong, items instead).

.venv/bin/python -I validate/step17.py
"""
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, DATASETS, Report, accuracy, build_index, correct_counts, load, truth
from step11 import FINAL

TARGET = 0.95
PART2 = dict(FINAL, effort_apart=True)
BUILD_ROWS = slice(0, 17000)     # the build's sample questions
USE = slice(17000, 19000)        # 2,000 real questions the build never saw


def build(name, settings, train):
    X = load(name, "slice")
    idx = vindex.Index(X.shape[1], DATASETS[name], seed=7, **settings)
    idx.build(X, np.arange(len(X)), train_questions=np.ascontiguousarray(train))
    return idx


def tune_accuracy(idx, name):
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    ids, _, checks = idx.search(Q, k=10, target=TARGET, threads=16)
    return accuracy(correct_counts(ids, Q, X, td, DATASETS[name])), float(checks.mean())


def at_build(rep):
    for name in ("yandex", "llama"):
        shared, _ = build_index(name, FINAL, train=True)
        apart, _ = build_index(name, PART2, train=True)
        (a0, c0), (a1, c1) = tune_accuracy(shared, name), tune_accuracy(apart, name)
        rep.check(f"{name}: with effort_apart the 95% target lands within 1 point", abs(a1 - TARGET) <= 0.01,
                  f"{a1:.2%} at {c1:,.0f} checks per search; sharing the links' questions: {a0:.2%} at {c0:,.0f}")
    for name in ("gooaq", "glove", "landmark"):
        a, _ = build_index(name, FINAL, train=True)
        b, _ = build_index(name, PART2, train=True)
        Q = load(name, "q_tune")
        same = all(np.array_equal(x, y) for x, y in zip(a.search(Q, k=10, target=TARGET, threads=16),
                                                      b.search(Q, k=10, target=TARGET, threads=16)))
        rep.check(f"{name}: no question links, so effort_apart changes nothing", same, "identical answers" if same else "differ")


def use(idx, name):
    """2,000 real questions, each spot-checked, then one audit (which may learn the table again)."""
    idx.search(load(name, "q_train")[USE], k=10, target=TARGET, threads=16)
    return idx.audit()


def wrong_sample(rep):
    for name in ("yandex", "llama"):
        spare = np.load(DATA / name / "spare_ids.npy")
        items_as_questions = np.ascontiguousarray(np.load(DATA / name / "items_full.npy", mmap_mode="r")[spare])
        idx = build(name, dict(PART2, audit_budget=1e12), items_as_questions)
        (a0, c0) = tune_accuracy(idx, name)
        r = use(idx, name)
        est = r["accuracy"].get(f"target {TARGET}")
        (a1, c1) = tune_accuracy(idx, name)
        saw = f"{est['accuracy']:.2%} ± {est['margin']:.2%}" if est else "nothing"
        rep.check(f"{name}: sample questions taken from the items, then real use: back to within 1 point",
                  r["relearned"] and abs(a1 - TARGET) <= 0.01,
                  f"before {a0:.2%} at {c0:,.0f} checks; spot checks saw {saw}; relearned on its own: "
                  f"{r['relearned']}; after {a1:.2%} at {c1:,.0f} checks (question links "
                  f"{'on' if idx.stats()['question_links'] else 'off'})")


def right_sample(rep):
    for name in DATASETS:
        idx = build(name, dict(PART2, audit_budget=1e12), load(name, "q_train")[BUILD_ROWS])
        a0, c0 = tune_accuracy(idx, name)
        r = use(idx, name)
        a1, _ = tune_accuracy(idx, name)
        est = r["accuracy"][f"target {TARGET}"]
        used = idx.relearn_effort()
        a2, c2 = tune_accuracy(idx, name)
        rep.check(f"{name}: right sample questions: no relearn on its own; forced, under 1 point",
                  not r["relearned"] and a1 == a0 and used is not None and abs(a2 - a0) < 0.01,
                  f"spot checks saw {est['accuracy']:.2%} ± {est['margin']:.2%}; relearned on its own: {r['relearned']}; "
                  f"forced on {used} questions: {a0:.2%} at {c0:,.0f} checks -> {a2:.2%} at {c2:,.0f}")


def side_by_side_and_saved(rep, name="glove"):
    idx = build(name, dict(PART2, audit_budget=1e12), load(name, "q_train")[BUILD_ROWS])
    use(idx, name)
    Q = load(name, "q_tune")
    stop, errors, rounds = threading.Event(), [], [0]

    def searcher():
        try:
            while not stop.is_set():
                ids = idx.search(Q[:200], k=10, target=TARGET, threads=1)[0]
                if (ids < 0).any():
                    errors.append("missing answers")
                rounds[0] += 1
        except Exception as e:   # noqa: BLE001 - any failure is the finding
            errors.append(repr(e))

    ths = [threading.Thread(target=searcher, daemon=True) for _ in range(4)]
    try:
        [th.start() for th in ths]
        used = idx.relearn_effort(threads=4)
    finally:
        stop.set()
        [th.join() for th in ths]
    rep.check(f"{name}: searches run while the table is learned again", not errors and used,
              f"{rounds[0]} batches of 200 searched during the relearn on {used} questions; errors: {errors[:2] or 'none'}")
    d = Path(tempfile.mkdtemp(prefix="vindex-step17-"))
    try:
        idx.save(str(d / "ix"))
        back = vindex.Index.load(str(d / "ix"))
        same = all(idx.effort_table(t) == back.effort_table(t) for t in (0.9, 0.95, 0.99))
        same = same and np.array_equal(idx.search(Q, k=10, target=TARGET)[0], back.search(Q, k=10, target=TARGET)[0])
        rep.check(f"{name}: the relearned table comes back after save and load", same, "same tables and answers" if same else "differ")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    rep = Report("step17")
    at_build(rep)
    wrong_sample(rep)
    right_sample(rep)
    side_by_side_and_saved(rep)
    rep.finish()


if __name__ == "__main__":
    main()
