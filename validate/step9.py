"""Step 9: changes. The experiments' change test (bench/runners/changes.py): build with 90% of the slice, then
10,000 adds, 10,000 deletes and 10,000 updates in one mixed order, checked at 10 checkpoints on the test questions,
then a fresh build on the final items. The search uses the learned effort at the target picked on the tuning
questions right after the first build; its table is kept through all changes.

Two runs per dataset:
  - with the prototype's own insertion orders (first build and fresh build): close to the prototype's own re-run
    (results/step5/changes/<final version>-final): same target, accuracy within 0.3 points, drift within 0.3 points;
  - with our own random order: Fix 7's pass rule — drift under 1 point (or better than fresh), no deleted or old
    items returned at any checkpoint, memory within 5% of a fresh build.

.venv/bin/python -I validate/step9.py [dataset ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import (DATA, DATASETS, REF, TARGET, TARGETS, Report, accuracy, correct_counts, load, prototype_order,
                    flipped_for)

FINAL = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=2, build_space="auto", effort_apart=False, tune="off", renumber=False, question_bits=32, popular_links="off")
PROTO_RUN = REF / "step5" / "changes" / "v0+build-mobius+learned-effort+qlinks-auto+tiny8+qrounds2-final"
ADD, DELETE, UPDATE = 1, 2, 3


def tenth(Q, X, metric, k=10):
    """True k-th distance of each question among the rows of X, by checking every row (64-bit)."""
    out = np.empty(len(Q))
    for a in range(0, len(Q), 250):
        q = Q[a : a + 250].astype(np.float64)
        s = q @ X.T.astype(np.float64) if metric != "l2" else None
        if metric == "cos":
            d = 1.0 - s
        elif metric == "ip":
            d = -s
        else:
            d = (q ** 2).sum(1)[:, None] + (X.astype(np.float64) ** 2).sum(1)[None, :] - 2 * q @ X.T.astype(np.float64)
        out[a : a + 250] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return np.stack([out] * k, axis=1)   # correct_counts reads column k-1


def stale_returned(found, reported, Q, X_now, X_before, updated, metric, tol=1e-3):
    """Returned updated items whose reported distance is closer to their OLD numbers than their new ones."""
    stale = 0
    for qi, row in enumerate(found):
        for j, v in enumerate(row):
            if v < 0 or int(v) not in updated:
                continue
            q = Q[qi].astype(np.float64)
            dn, do = (float(-(q @ X_now[v])) if metric == "ip" else 1.0 - float(q @ X_now[v])), \
                     (float(-(q @ X_before[v])) if metric == "ip" else 1.0 - float(q @ X_before[v]))
            if abs(dn - do) <= tol:
                continue
            if abs(reported[qi, j] - do) < abs(reported[qi, j] - dn):
                stale += 1
    return stale


def change_test(name, order_mode):
    metric = DATASETS[name]
    X0 = load(name, "slice")
    Q, Qt, Qtr = load(name, "q_test"), load(name, "q_tune"), load(name, "q_train")
    script = dict(np.load(DATA / name / "changes.npz"))
    init = script["initial_ids"]
    n_ops = len(script["ops_type"])
    cps = [n_ops // 10 * (c + 1) for c in range(9)] + [n_ops]
    res = {"dataset": name, "order": order_mode, "checkpoints": []}
    idx = vindex.Index(X0.shape[1], metric, **FINAL)
    kw = {"order": prototype_order(X0[init], metric, flipped_for(metric, FINAL))} if order_mode == "prototype" else {}
    t = time.time()
    idx.build(X0[init], init, train_questions=Qtr, **kw)
    res["initial_build_seconds"] = time.time() - t
    # the target: the cheapest one reaching 95% on the tuning questions (truth among the first items)
    td_t = tenth(Qt, X0[init], metric)
    T = TARGETS[-1]
    for cand in TARGETS:
        ids, _, _ = idx.search(Qt, k=10, target=cand, threads=16)
        if accuracy(correct_counts(ids, Qt, X0, td_t, metric)) >= TARGET:
            T = cand
            break
    res["target"] = T
    td0 = tenth(Q, X0[init], metric)
    ids, _, checks = idx.search(Q, k=10, target=T, threads=16)
    res["accuracy_before"], res["checks_before"] = accuracy(correct_counts(ids, Q, X0, td0, metric)), float(checks.mean())
    X = X0.copy()
    deleted, updated = set(), set()
    c_before = idx.stats()["change_checks"]
    pos, t0 = 0, time.time()
    for c, upto in enumerate(cps):
        for k in range(pos, upto):
            ty, i, v = int(script["ops_type"][k]), int(script["ops_id"][k]), int(script["ops_vec"][k])
            if ty == ADD:
                idx.add(i, X0[i])
            elif ty == DELETE:
                idx.delete(i)
                deleted.add(i)
            else:
                vec = script["update_vectors"][v]
                idx.update(i, vec)
                X[i] = vec
                updated.add(i)
        pos = upto
        z = np.load(DATA / name / f"truth_change_cp{c + 1:02d}.npz")
        ids, d, checks = idx.search(Q, k=10, target=T, threads=16)
        res["checkpoints"].append({
            "after_changes": upto, "accuracy": accuracy(correct_counts(ids, Q, X, z["dists"], metric)),
            "deleted_returned": int(np.isin(ids[ids >= 0], np.array(sorted(deleted))).sum()),
            "stale_returned": stale_returned(ids, d, Q, X, X0, updated, metric), "checks": float(checks.mean())})
    res["change_seconds"] = time.time() - t0
    res["change_checks_per_change"] = (idx.stats()["change_checks"] - c_before) / n_ops
    res["memory_after"] = idx.memory()["total_bytes"]
    # the final items
    alive = np.zeros(len(X0), bool)
    alive[init] = True
    for ty, i in zip(script["ops_type"], script["ops_id"]):
        if ty == ADD:
            alive[i] = True
        elif ty == DELETE:
            alive[i] = False
    final = np.nonzero(alive)[0]
    z = np.load(DATA / name / f"truth_change_cp{len(cps):02d}.npz")
    # learn the table again on the changed index: is any drop the links, or a table learned on the old items?
    idx.learn_effort(Qtr)
    ids, _, checks = idx.search(Q, k=10, target=T, threads=16)
    res["after_relearn"] = {"accuracy": accuracy(correct_counts(ids, Q, X, z["dists"], metric)), "checks": float(checks.mean())}
    fresh = vindex.Index(X0.shape[1], metric, **FINAL)
    kw = {"order": prototype_order(X[final], metric, flipped_for(metric, FINAL))} if order_mode == "prototype" else {}
    fresh.build(X[final], final, train_questions=Qtr, **kw)
    ids, _, checks = fresh.search(Q, k=10, target=T, threads=16)
    res["fresh_accuracy"], res["fresh_checks"] = accuracy(correct_counts(ids, Q, X, z["dists"], metric)), float(checks.mean())
    res["fresh_memory"] = fresh.memory()["total_bytes"]
    res["drift"] = res["checkpoints"][-1]["accuracy"] - res["fresh_accuracy"]
    res["memory_after_vs_fresh"] = res["memory_after"] / res["fresh_memory"]
    return res


if __name__ == "__main__":
    rep = Report("step9")
    for name in (sys.argv[1:] or DATASETS):
        proto = json.loads((PROTO_RUN / f"{name}.json").read_text())
        p_last = proto["checkpoints"][-1]["accuracy"]
        a = change_test(name, "prototype")
        last = a["checkpoints"][-1]["accuracy"]
        rep.check(f"{name}, prototype's order: same target, accuracy and drift as the prototype's re-run",
                  a["target"] == proto["effort"]["learned_target"]
                  and abs(a["accuracy_before"] - proto["accuracy_before"]) <= 0.003
                  and abs(last - p_last) <= 0.003 and abs(a["drift"] - proto["drift"]) <= 0.003,
                  f"target {a['target']} ({proto['effort']['learned_target']}); before {a['accuracy_before']:.2%} "
                  f"({proto['accuracy_before']:.2%}); after {last:.2%} ({p_last:.2%}); fresh {a['fresh_accuracy']:.2%} "
                  f"({proto['fresh_accuracy']:.2%}); drift {a['drift'] * 100:+.2f} ({proto['drift'] * 100:+.2f}) points; "
                  f"work per change {a['change_checks_per_change']:,.0f} ({proto['change_cmps_per_change']:,.0f})", a)
        b = change_test(name, "own")
        dret = sum(c["deleted_returned"] for c in b["checkpoints"])
        sret = sum(c["stale_returned"] for c in b["checkpoints"])
        ok = (abs(b["drift"]) < 0.01 or b["drift"] > 0) and dret == 0 and sret == 0 and b["memory_after_vs_fresh"] <= 1.05
        rep.check(f"{name}, our own order: Fix 7's pass rule", ok,
                  f"drift {b['drift'] * 100:+.2f} points; deleted returned {dret}; old numbers returned {sret}; "
                  f"memory vs fresh {b['memory_after_vs_fresh']:.2f}x; {b['change_seconds']:.0f} s for "
                  f"{len(b['checkpoints']) * 3000:,} changes; table learned again: {b['after_relearn']['accuracy']:.2%}", b)
    rep.finish()
