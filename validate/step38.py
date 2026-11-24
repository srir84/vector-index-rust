"""Step 38: the change test on Landmark (part 5).
  .venv/bin/python -I validate/step38.py diagnose      # before the fix: why Landmark drifts, and the samples' record
  .venv/bin/python -I validate/step38.py try           # then: Landmark's full-size test with each candidate fix
  .venv/bin/python -I validate/step38.py               # after the fix: the check
The change test as step 32 runs it (full size) and as step 9 runs it (samples), with part 3's index; each checkpoint
also records the map: the start item, items no walk from the start can reach, items no link points at, links per
item.
`diagnose` runs Landmark's full-size test as it is, then with one thing changed at a time:
  - no cleanup: links to deleted items are never cleared in a batch (slots are not reused);
  - more replacement links: an item that loses a link to a deleted item takes up to 8 of its links, not 3;
  - no learning from use: the checkpoints' searches are not spot-checked, so the effort is never learned again;
and the twelve samples' tests as they are, the record the check compares with. It writes
validate/results/step38-diagnose.json and judges nothing.
Pass (after the fix): Landmark's drift under 1 point at full size; on the samples, no dataset's drift worse than the
record by more than 0.2 points; nothing deleted or outdated returned; memory within 5% of a fresh build.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import (DATA, DATASETS, HELD_BACK, HELD_BACK_3, OUT, TARGET, TARGETS, Report, accuracy, correct_counts,
                    load, metric_of)
from step9 import tenth
from step27 import PART3
from step32 import ADD, DELETE, stale_returned

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]
VARIANTS = {"as it is": {}, "no cleanup": {"cleanup_batch": 10**9}, "more replacement links": {"n_replace": 8},
            "no learning from use": {"track_usage": False, "audit_budget": 0.0, "relearn_every": 0}}
# fixes tried after the diagnosis (`try`): how a delete repairs the links of the items that linked to the deleted one
FIXES = {"prune": {"repair": "prune"}, "occlude": {"repair": "occlude"}, "occlude, 8 replacements": {"repair": "occlude", "n_replace": 8}}
MAP = ("start", "unreachable", "no_links_in", "mean_links")


def change_test(name, size="full", extra=None):
    """Step 32's change test (step 9's on the samples), with `extra` settings on top of part 3's."""
    metric, extra = metric_of(name), extra or {}
    if size == "full":
        z = dict(np.load(DATA / name / "changes_full.npz"))
        X0 = np.ascontiguousarray(np.load(DATA / name / "items_full.npy", mmap_mode="r")[z["base_rows"]])
        cp_file = "truth_change_full_cp{:02d}.npz"
    else:
        z = dict(np.load(DATA / name / "changes.npz"))
        X0 = load(name, "slice")
        cp_file = "truth_change_cp{:02d}.npz"
    Q, Qt, Qtr = load(name, "q_test"), load(name, "q_tune"), load(name, "q_train")
    init, ops, oid, ovec, upd = z["initial_ids"], z["ops_type"], z["ops_id"], z["ops_vec"], z["update_vectors"]
    cps = [int(np.load(DATA / name / cp_file.format(c + 1))["upto"]) for c in range(10)]
    res = {"dataset": name, "size": size, "extra": extra, "changes": int(len(ops)), "checkpoints": []}
    idx = vindex.Index(X0.shape[1], metric, **{**PART3, **extra})
    idx.build(X0[init], init, train_questions=Qtr)
    tu, st = idx.tuning(), idx.stats()
    picks = ({"r": tu["chosen_r"], "alpha": round(tu["chosen_alpha"], 2), "code_bits": tu["chosen_code_bits"],
              "rerank": tu["chosen_rerank"]} if tu else {})
    picks.update(question_links="on" if st["question_links"] else "off", popular_links="on" if st["popular_items"] else "off")
    res["picks"] = picks
    res["map_before"] = {k: st[k] for k in MAP}
    td_t = tenth(Qt, X0[init], metric)
    T = next((c for c in TARGETS if accuracy(correct_counts(idx.search(Qt, k=10, target=c, threads=16)[0], Qt, X0, td_t,
                                                             metric)) >= TARGET), TARGETS[-1])
    res["target"] = T
    X = X0.copy()
    deleted, updated = set(), set()
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
        cp = np.load(DATA / name / cp_file.format(c + 1))
        ids, d, checks = idx.search(Q, k=10, target=T, threads=16)
        st = idx.stats()
        res["checkpoints"].append({
            "after_changes": int(upto), "accuracy": accuracy(correct_counts(ids, Q, X, cp["dists"], metric)),
            "deleted_returned": int(np.isin(ids[ids >= 0], np.fromiter(deleted, dtype=np.int64)).sum()) if deleted else 0,
            "stale_returned": stale_returned(ids, d, Q, X, X0, updated, metric), "checks": float(checks.mean()),
            **{k: st[k] for k in MAP}})
    res["change_seconds"] = time.time() - t0
    res["memory_after"] = idx.memory()["total_bytes"]
    final = np.flatnonzero(cp["alive"])
    del idx
    fresh = vindex.Index(X0.shape[1], metric, **{**PART3, **extra, "tune": "off", **picks})
    fresh.build(X[final], final, train_questions=Qtr)
    ids, _, checks = fresh.search(Q, k=10, target=T, threads=16)
    res["fresh_accuracy"], res["fresh_checks"] = accuracy(correct_counts(ids, Q, X, cp["dists"], metric)), float(checks.mean())
    res["fresh_map"] = {k: fresh.stats()[k] for k in MAP}
    res["fresh_memory"] = fresh.memory()["total_bytes"]
    res["drift"] = res["checkpoints"][-1]["accuracy"] - res["fresh_accuracy"]
    res["memory_after_vs_fresh"] = res["memory_after"] / res["fresh_memory"]
    return res


def line(r):
    last = r["checkpoints"][-1]
    return (f"drift {r['drift'] * 100:+.2f} (after {last['accuracy']:.2%}, fresh {r['fresh_accuracy']:.2%}); target "
            f"{r['target']}; checks {last['checks']:.0f} (fresh {r['fresh_checks']:.0f}); unreachable "
            f"{r['map_before']['unreachable']} -> {last['unreachable']} (fresh {r['fresh_map']['unreachable']}); links per "
            f"item {r['map_before']['mean_links']:.1f} -> {last['mean_links']:.1f} (fresh {r['fresh_map']['mean_links']:.1f}); "
            f"memory {r['memory_after_vs_fresh']:.3f}x fresh")


def main():
    common.unlock_held_back()
    if sys.argv[1:2] == ["try"]:
        path = OUT / "step38-diagnose.json"
        out = json.loads(path.read_text())
        for label, extra in FIXES.items():
            out.setdefault("landmark_fixes", {})[label] = r = change_test("landmark", "full", extra)
            print(f"landmark, {label}: {line(r)}", flush=True)
            path.write_text(json.dumps(out, indent=1))
        return
    if sys.argv[1:2] == ["diagnose"]:
        out = {"landmark": {}, "samples": {}}
        for label, extra in VARIANTS.items():
            out["landmark"][label] = r = change_test("landmark", "full", extra)
            print(f"landmark, {label}: {line(r)}", flush=True)
            for c in r["checkpoints"]:
                print(f"    {c['after_changes']:>7,}: {c['accuracy']:.2%}, checks {c['checks']:.0f}, start {c['start']}, "
                      f"unreachable {c['unreachable']}, no links in {c['no_links_in']}", flush=True)
        for name in ALL:
            out["samples"][name] = r = change_test(name, "slice")
            print(f"sample {name}: {line(r)}", flush=True)
        (OUT / "step38-diagnose.json").write_text(json.dumps(out, indent=1))
        return
    rep = Report("step38")
    before = json.loads((OUT / "step38-diagnose.json").read_text())["samples"]
    r = change_test("landmark", "full")
    ok = (r["drift"] > -0.01 and all(c["deleted_returned"] == 0 and c["stale_returned"] == 0 for c in r["checkpoints"])
          and r["memory_after_vs_fresh"] <= 1.05)
    rep.check("landmark: at full size, drift under 1 point, nothing deleted or outdated returned, memory within 5%", ok,
              line(r), r)
    for name in ALL:
        s = change_test(name, "slice")
        was = before[name]["drift"]
        ok = (s["drift"] >= was - 0.002 and all(c["deleted_returned"] == 0 and c["stale_returned"] == 0 for c in s["checkpoints"])
              and s["memory_after_vs_fresh"] <= 1.05)
        rep.check(f"{name} (sample): drift no worse than before by more than 0.2 points", ok,
                  f"{line(s)}; before: drift {was * 100:+.2f}", s)
    rep.finish()


if __name__ == "__main__":
    main()
