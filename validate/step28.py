"""Step 28: locked (code, rivals) and the full-size data (part 4).
  .venv/bin/python -I validate/step28.py freeze   # once, before any full-size run: the fingerprints (refuses to
                                                  # overwrite them)
  .venv/bin/python -I validate/step28.py          # the checks
The full-size data is made by the experiments' scripts/step6_prepare_full.py. Checks:
  - the fingerprints are recorded, and nothing locked has changed since;
  - on each of the twelve datasets:
      the sample's items are rows of the full items;
      on the full items, no tuning or test question's 10th true distance is further than on the sample;
      the full-size true answers agree in two precisions, up to ties and rounding;
      on Llama, Yi and Llama's second head: the training questions' true answers (for RoarGraph), on a spot check;
      the change script: 5% spares, none among its items; 90% built first, then 10% adds, deletes and updates each,
      half small edits and half full replacements; each checkpoint's items are the ones its changes leave.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import (DATA, DATASETS, FROZEN, HELD_BACK, HELD_BACK_3, Report, fingerprints, frozen_changes, load,
                    metric_of, truth)

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]
ROARGRAPH_SETS = ("llama", "yi", "llama2")


def freeze():
    if FROZEN.exists():
        sys.exit(f"{FROZEN} exists: the lock is made once, before any full-size run")
    FROZEN.write_text(json.dumps(fingerprints(), indent=1))
    print(f"wrote {FROZEN}")


def exact_dists(Q, X, metric):
    q, x = Q.astype(np.float64), X.astype(np.float64)
    if metric == "l2":
        return (q ** 2).sum(1)[:, None] + (x ** 2).sum(1)[None, :] - 2 * q @ x.T
    s = q @ x.T
    return 1.0 - s if metric == "cos" else -s


def one(rep, name):
    metric, d = metric_of(name), DATA / name
    items = np.load(d / "items_full.npy", mmap_mode="r")
    sample_in_full = np.array_equal(np.asarray(items[np.load(d / "slice_ids.npy")]), load(name, "slice"))
    further = {qs: float(np.max(truth(name, f"full_{qs}")["dists"][:, 9] - truth(name, f"slice_{qs}")["dists"][:, 9]))
               for qs in ("tune", "test")}
    card = json.loads((d / "card_full.json").read_text())
    not_ties = card["tune_precision_check"]["of_which_not_explained_by_ties"]
    not_ties_test = json.loads((d / "card.json").read_text())["precision_check"]["full_test"]["of_which_not_explained_by_ties"]
    rep.check(f"{name}: the sample is part of the full items, the full items only bring closer answers, and two "
              f"precisions agree", sample_in_full and max(further.values()) <= 1e-9 and not_ties == 0 and not_ties_test == 0,
              f"{len(items):,} items; sample rows equal: {sample_in_full}; 10th distance further than on the sample by at "
              f"most {max(further.values()):.1e} (tuning {further['tune']:.1e}, test {further['test']:.1e}); differences "
              f"between precisions not explained by ties: {not_ties} tuning, {not_ties_test} test")

    if name in ROARGRAPH_SETS:
        tr = truth(name, "full_train")
        Qtr = load(name, "q_train")
        rows = np.random.default_rng(28).choice(len(Qtr), size=50, replace=False)
        X = np.asarray(items)
        D = exact_dists(Qtr[rows], X, metric)
        want = np.sort(np.partition(D, 99, axis=1)[:, :100], axis=1)
        got = np.take_along_axis(D, tr["ids"][rows].astype(np.int64), axis=1)
        gap = float(np.max(np.abs(got - want)))
        rep.check(f"{name}: the training questions' true answers (for RoarGraph), spot check of 50", tr["ids"].shape == (len(Qtr), 100)
                  and gap <= 1e-6, f"{tr['ids'].shape[0]:,} questions x {tr['ids'].shape[1]} answers; largest distance gap {gap:.1e}")

    z = np.load(d / "changes_full.npz")
    base, spare, init, ops, ids = z["base_rows"], z["spare_rows"], z["initial_ids"], z["ops_type"], z["ops_id"]
    n = len(base)
    n_new = n - int(0.9 * n)
    cnt = {k: int((ops == v).sum()) for k, v in (("adds", 1), ("deletes", 2), ("updates", 3))}
    small, full = int((z["update_kind"] == 0).sum()), int((z["update_kind"] == 1).sum())
    shape_ok = (len(np.intersect1d(base, spare)) == 0 and len(base) + len(spare) == len(items)
                and len(spare) == len(items) // 20 and len(init) == n - n_new and all(v == n_new for v in cnt.values())
                and small == cnt["updates"] // 2 and full == cnt["updates"] - small)
    in_init = np.zeros(n, dtype=bool)
    in_init[init] = True
    roles_ok = (not in_init[ids[ops == 1]].any() and in_init[ids[ops == 2]].all() and in_init[ids[ops == 3]].all()
                and len(np.unique(ids[ops == 2])) == cnt["deletes"])
    alive, pos, same = in_init.copy(), 0, []
    for c in range(1, 11):
        cp = np.load(d / f"truth_change_full_cp{c:02d}.npz")
        upto = int(cp["upto"])
        alive[ids[pos:upto][ops[pos:upto] == 1]] = True
        alive[ids[pos:upto][ops[pos:upto] == 2]] = False
        pos = upto
        same.append(np.array_equal(alive, cp["alive"]))
    rep.check(f"{name}: the change script", shape_ok and roles_ok and all(same) and pos == len(ops),
              f"{n:,} items and {len(spare):,} spares (none shared); {len(init):,} built first, then {cnt['adds']:,} adds, "
              f"{cnt['deletes']:,} deletes, {cnt['updates']:,} updates ({small:,} small edits, {full:,} full replacements); "
              f"adds new, deletes and updates on built items: {roles_ok}; checkpoints match the changes: {sum(same)}/10")


def main():
    common.unlock_held_back()
    rep = Report("step28")
    changed = frozen_changes()
    rep.check("the code and the rivals' builds are as frozen", not changed,
              "fingerprints recorded and unchanged" if not changed else "; ".join(changed[:5]))
    for name in sys.argv[1:] or ALL:
        one(rep, name)
    rep.finish()


if __name__ == "__main__":
    freeze() if sys.argv[1:2] == ["freeze"] else main()
