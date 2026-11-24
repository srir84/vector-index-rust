"""Step 10: save, load, and the full numbers on SSD, on real data (final design).
  - save, load with numbers in memory, search again: identical answers, distances and checks;
  - the same with the full numbers left on SSD;
  - a damaged copy of the folder is refused;
  - a reloaded index takes the first 3,000 changes of the change script exactly like the original;
  - (GloVe) 8 Python threads searching at once give the same answers as one call; and 1,000 changes made while
    4 threads keep searching give no errors and exactly the index the same changes give with nobody searching.

.venv/bin/python -I validate/step10.py [dataset ...]
"""
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, DATASETS, Report, load

FINAL = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=2, build_space="auto", effort_apart=False, tune="off", renumber=False, question_bits=32, popular_links="off")


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def apply(idx, script, X0, upto):
    for k in range(upto):
        ty, i, v = int(script["ops_type"][k]), int(script["ops_id"][k]), int(script["ops_vec"][k])
        if ty == 1:
            idx.add(i, X0[i])
        elif ty == 2:
            idx.delete(i)
        else:
            idx.update(i, script["update_vectors"][v])


def thread_checks(rep, name, X0, Q, Qtr, script, init):
    metric = DATASETS[name]
    a = vindex.Index(X0.shape[1], metric, **FINAL)
    a.build(X0[init], init, train_questions=Qtr)
    whole = a.search(Q, k=10, target=0.95, threads=1)
    t0 = time.perf_counter()
    a.search(Q, k=10, target=0.95, threads=1)
    one = time.perf_counter() - t0
    chunks = np.array_split(np.arange(len(Q)), 8)
    with ThreadPoolExecutor(8) as ex:
        t0 = time.perf_counter()
        parts = list(ex.map(lambda c: a.search(Q[c], k=10, target=0.95, threads=1), chunks))
        par = time.perf_counter() - t0
    ok = all(np.array_equal(np.concatenate([p[j] for p in parts]), whole[j]) for j in range(3))
    rep.check(f"{name}: 8 Python threads searching at once give the same answers as one call", ok,
              f"{one / par:.1f}x the speed of one thread: the searches really run side by side")
    # changes from this thread while 4 others keep searching; then the same changes on a twin nobody searches
    stop, errors, counts = threading.Event(), [], [0] * 4

    def searcher(t):
        while not stop.is_set():
            try:
                a.search(Q[:100], k=10, target=0.95, threads=1)
                counts[t] += 1
            except Exception as e:   # noqa: BLE001 (any error fails the check)
                errors.append(repr(e))
                return

    ths = [threading.Thread(target=searcher, args=(t,), daemon=True) for t in range(4)]
    for t in ths:
        t.start()
    try:
        apply(a, script, X0, 1000)
    except Exception as e:   # noqa: BLE001 (a change refused while others search fails the check)
        errors.append(repr(e))
    finally:
        stop.set()
        for t in ths:
            t.join()
    b = vindex.Index(X0.shape[1], metric, **FINAL)
    b.build(X0[init], init, train_questions=Qtr)
    apply(b, script, X0, 1000)
    ok = not errors and same(a.search(Q, k=10, target=0.95, threads=16), b.search(Q, k=10, target=0.95, threads=16))
    rep.check(f"{name}: 1,000 changes made while 4 threads keep searching: no errors, and exactly the index made "
              f"with nobody searching", ok, f"{sum(counts)} searches of 100 questions ran during the changes; "
              f"errors: {errors[:1] or 'none'}")


if __name__ == "__main__":
    rep = Report("step10")
    work = Path(tempfile.mkdtemp(prefix="vindex-step10-"))
    try:
        for name in (sys.argv[1:] or ["glove", "yandex", "llama"]):
            metric = DATASETS[name]
            X0, Q, Qtr = load(name, "slice"), load(name, "q_test"), load(name, "q_train")
            script = dict(np.load(DATA / name / "changes.npz"))
            init = script["initial_ids"]
            a = vindex.Index(X0.shape[1], metric, **FINAL)
            a.build(X0[init], init, train_questions=Qtr)
            folder = work / name
            a.save(str(folder))
            size = sum(f.stat().st_size for f in folder.iterdir())
            for mode in ("ram", "ssd"):
                b = vindex.Index.load(str(folder), numbers=mode)
                ok = all(same(a.search(Q, k=10, **e, threads=16), b.search(Q, k=10, **e, threads=16))
                         for e in ({"target": 0.95}, {"list_size": 100}))
                rep.check(f"{name}: reloaded ({mode}) gives identical answers, distances and checks", ok,
                          f"folder {size / 1e6:,.0f} MB")
            # a damaged copy is refused
            bad = work / f"{name}-damaged"
            shutil.copytree(folder, bad)
            raw = (bad / "links.u32").read_bytes()
            (bad / "links.u32").write_bytes(raw[: len(raw) // 2])
            try:
                vindex.Index.load(str(bad))
                refused = False
            except ValueError as e:
                refused, msg = True, str(e)
            rep.check(f"{name}: a damaged folder is refused", refused, msg if refused else "it loaded!")
            # the first 3,000 changes, applied to the original and to a reloaded copy
            b = vindex.Index.load(str(folder), numbers="ram")
            for idx in (a, b):
                apply(idx, script, X0, 3000)
            rep.check(f"{name}: after 3,000 changes, the reloaded copy still matches the original",
                      same(a.search(Q, k=10, target=0.95, threads=16), b.search(Q, k=10, target=0.95, threads=16))
                      and a.stats()["changes"] == b.stats()["changes"],
                      f"question links redone at change 3,000: {a.stats()['question_links']}")
            if name == "glove":
                thread_checks(rep, name, X0, Q, Qtr, script, init)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()
