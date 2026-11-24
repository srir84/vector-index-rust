"""Step 32: changes at full size (part 4).
  .venv/bin/python -I validate/step32.py [dataset ...]
The full-size change script (step 28: 5% of the items set aside for full replacements; build with 90% of the rest,
then add, delete and update 10% each in one mixed order, half small edits and half full replacements), run as step 9
runs the samples' one, with part 3's index and every default:
  - the first build picks its own settings; the target is the cheapest reaching 95% on the tuning questions right
    after it (true answers among the first items), and the effort table is kept through every change;
  - 10 checkpoints on the test questions: accuracy, deleted items returned, items found by their old numbers;
  - then a fresh build on the final items with the first build's picks, so the comparison is not between two
    separate tunings, which can differ by timing noise alone (step 26's lesson).
Pass (Fix 7's rule, as in step 9): drift under 1 point (or better than the fresh build), no deleted or old items
returned at any checkpoint, memory within 5% of the fresh build.
"""
import sys
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import (DATA, DATASETS, HELD_BACK, HELD_BACK_3, TARGET, TARGETS, Report, accuracy, correct_counts,
                    frozen_changes, load, metric_of)
from step9 import tenth
from step27 import PART3

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]
ADD, DELETE = 1, 2


def dist(q, x, metric):
    """The index's own distance: 1 - dot (angle), -dot (best score), squared straight-line distance."""
    if metric == "l2":
        d = q - x
        return float(d @ d)
    s = float(q @ x)
    return -s if metric == "ip" else 1.0 - s


def stale_returned(found, reported, Q, X_now, X_before, updated, metric, tol=1e-3):
    """Returned updated items whose reported distance is closer to their old numbers than their new ones (step 9's
    count, for all three formulas)."""
    stale = 0
    for qi, row in enumerate(found):
        q = Q[qi].astype(np.float64)
        for j, v in enumerate(row):
            if v < 0 or int(v) not in updated:
                continue
            dn, do = dist(q, X_now[v].astype(np.float64), metric), dist(q, X_before[v].astype(np.float64), metric)
            if abs(dn - do) > tol and abs(reported[qi, j] - do) < abs(reported[qi, j] - dn):
                stale += 1
    return stale


def change_test(name):
    metric = metric_of(name)
    z = dict(np.load(DATA / name / "changes_full.npz"))
    X0 = np.ascontiguousarray(np.load(DATA / name / "items_full.npy", mmap_mode="r")[z["base_rows"]])
    Q, Qt, Qtr = load(name, "q_test"), load(name, "q_tune"), load(name, "q_train")
    init, ops, oid, ovec, upd = z["initial_ids"], z["ops_type"], z["ops_id"], z["ops_vec"], z["update_vectors"]
    n_ops = len(ops)
    cps = [n_ops // 10 * (c + 1) for c in range(9)] + [n_ops]
    res = {"dataset": name, "items": int(len(X0)), "changes": int(n_ops), "checkpoints": []}
    idx = vindex.Index(X0.shape[1], metric, **PART3)
    t = time.time()
    idx.build(X0[init], init, train_questions=Qtr)
    res["initial_build_seconds"] = time.time() - t
    tu, st = idx.tuning(), idx.stats()
    res["picks"] = picks = ({"r": tu["chosen_r"], "alpha": round(tu["chosen_alpha"], 2), "code_bits": tu["chosen_code_bits"],
                             "rerank": tu["chosen_rerank"]} if tu else {})
    picks.update(question_links="on" if st["question_links"] else "off", popular_links="on" if st["popular_items"] else "off")
    td_t = tenth(Qt, X0[init], metric)
    T = next((c for c in TARGETS if accuracy(correct_counts(idx.search(Qt, k=10, target=c, threads=16)[0], Qt, X0, td_t,
                                                             metric)) >= TARGET), TARGETS[-1])
    res["target"] = T
    ids, _, checks = idx.search(Q, k=10, target=T, threads=16)
    res["accuracy_before"] = accuracy(correct_counts(ids, Q, X0, tenth(Q, X0[init], metric), metric))
    res["checks_before"] = float(checks.mean())
    X = X0.copy()
    deleted, updated = set(), set()
    c_before = idx.stats()["change_checks"]
    pos, t0 = 0, time.time()
    for c, upto in enumerate(cps):
        for k in range(pos, upto):
            ty, i, v = int(ops[k]), int(oid[k]), int(ovec[k])
            if ty == ADD:
                idx.add(i, X0[i])
            elif ty == DELETE:
                idx.delete(i)
                deleted.add(i)
            else:
                idx.update(i, upd[v])
                X[i] = upd[v]
                updated.add(i)
        pos = upto
        cp = np.load(DATA / name / f"truth_change_full_cp{c + 1:02d}.npz")
        ids, d, checks = idx.search(Q, k=10, target=T, threads=16)
        res["checkpoints"].append({
            "after_changes": int(upto), "accuracy": accuracy(correct_counts(ids, Q, X, cp["dists"], metric)),
            "deleted_returned": int(np.isin(ids[ids >= 0], np.fromiter(deleted, dtype=np.int64)).sum()) if deleted else 0,
            "stale_returned": stale_returned(ids, d, Q, X, X0, updated, metric), "checks": float(checks.mean())})
    res["change_seconds"] = time.time() - t0
    res["change_checks_per_change"] = (idx.stats()["change_checks"] - c_before) / n_ops
    res["memory_after"] = idx.memory()["total_bytes"]
    final = np.flatnonzero(cp["alive"])
    idx.learn_effort(Qtr)
    ids, _, checks = idx.search(Q, k=10, target=T, threads=16)
    res["after_relearn"] = {"accuracy": accuracy(correct_counts(ids, Q, X, cp["dists"], metric)), "checks": float(checks.mean())}
    del idx
    fresh = vindex.Index(X0.shape[1], metric, **dict(PART3, tune="off", **picks))
    t = time.time()
    fresh.build(X[final], final, train_questions=Qtr)
    res["fresh_build_seconds"] = time.time() - t
    ids, _, checks = fresh.search(Q, k=10, target=T, threads=16)
    res["fresh_accuracy"], res["fresh_checks"] = accuracy(correct_counts(ids, Q, X, cp["dists"], metric)), float(checks.mean())
    res["fresh_memory"] = fresh.memory()["total_bytes"]
    res["drift"] = res["checkpoints"][-1]["accuracy"] - res["fresh_accuracy"]
    res["memory_after_vs_fresh"] = res["memory_after"] / res["fresh_memory"]
    return res


def main():
    common.unlock_held_back()
    rep = Report("step32")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    for name in sys.argv[1:] or ALL:
        b = change_test(name)
        dret = sum(c["deleted_returned"] for c in b["checkpoints"])
        sret = sum(c["stale_returned"] for c in b["checkpoints"])
        ok = (abs(b["drift"]) < 0.01 or b["drift"] > 0) and dret == 0 and sret == 0 and b["memory_after_vs_fresh"] <= 1.05
        rep.check(f"{name}: changes at full size, Fix 7's pass rule", ok,
                  f"{b['items']:,} items, {b['changes']:,} changes in {b['change_seconds']:.0f} s; target {b['target']}; "
                  f"before {b['accuracy_before']:.2%}, after {b['checkpoints'][-1]['accuracy']:.2%}, fresh build "
                  f"{b['fresh_accuracy']:.2%}: drift {b['drift'] * 100:+.2f} points; deleted returned {dret}; old numbers "
                  f"returned {sret}; memory vs fresh {b['memory_after_vs_fresh']:.3f}x; table learned again: "
                  f"{b['after_relearn']['accuracy']:.2%}; builds {b['initial_build_seconds']:.0f} s and "
                  f"{b['fresh_build_seconds']:.0f} s", b)
    rep.finish()


if __name__ == "__main__":
    main()
