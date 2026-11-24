"""Shared helpers for the step checks: the experiments' data, their accuracy rule, their reference numbers.

The accuracy rule and "cost at exactly 95%" are copied from the experiments (bench/measures.py), so a number
here means the same as a number there. Paths can be changed with VINDEX_EXPERIMENTS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXP = Path(os.environ.get("VINDEX_EXPERIMENTS", ROOT.parent / "vector-index-experiments"))
DATA = EXP / "data" / "prepared"
REF = EXP / "results"
OUT = ROOT / "validate" / "results"
DATASETS = {"gooaq": "cos", "glove": "cos", "yandex": "cos", "llama": "ip", "landmark": "cos"}
# Held-back datasets (SPEC.md, part 2): never tuned on; load() and truth() refuse them until step 22 unlocks them.
HELD_BACK = {"arxiv": "cos", "laion": "cos", "yi": "ip", "gist": "l2"}
# Part 3's held-back datasets (SPEC.md, part 3): the same lock, until step 27. Part 2's four stay locked too.
HELD_BACK_3 = {"nytimes": "cos", "imagenet": "cos", "llama2": "ip"}
_held_back_unlocked = False
# Part 5's held-back dataset (SPEC.md, part 5), with its own lock: unlock_held_back() does not open it, only
# unlock_held_back_5() does, in step 42.
HELD_BACK_5 = {"glove100": "cos"}
_held_back_5_unlocked = False
# Part 4 (SPEC.md) runs every dataset at full size, after both held-back tests: its results may name them, here
# (validate/results/step28* to step35*) and in the experiments' results/step6/.
PART4_RESULTS = tuple(f"step{n}" for n in range(28, 44))   # parts 4 and 5 (steps 28-35, 36-43)
PART4_PLACES = ("step6/", "step7/")                        # the experiments' full-size rival runs (parts 4 and 5)
TIE_EPS = 1e-3          # an item counts if its distance <= the true 10th distance + 0.001
TARGET = 0.95
L_SWEEP = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000]
TARGETS = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]

# the experiments' stages (results/step5/runs/<stage>/<dataset>.json), in the order the fixes were added
STAGE_V0 = "v0"
STAGE_FLIP = "v0+build-mobius"
STAGE_EFFORT = "v0+build-mobius+learned-effort"
STAGE_QLINKS = "v0+build-mobius+learned-effort+qlinks-auto"
STAGE_COPIES = "v0+build-mobius+learned-effort+qlinks-auto+tiny8"
STAGE_TWICE = "v0+build-mobius+learned-effort+qlinks-auto+tiny8+qrounds2"


FROZEN = OUT / "step28-frozen.json"


def fingerprints():
    """SHA-256 of what part 4 locks (step 28): the index's code and build scripts, the installed package and the
    rivals' builds. The experiments' runners are recorded too, but may be fixed (a fix is reported)."""
    import hashlib
    import vindex
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    rel = lambda ps, base: {str(p.relative_to(base)): sha(p) for p in sorted(set(ps))}
    code = [*ROOT.glob("crates/**/*.rs"), *ROOT.glob("crates/*/Cargo.toml"), *ROOT.glob("crates/*/pyproject.toml"),
            ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "rust-toolchain.toml", *ROOT.glob("tools/build_py*.sh"),
            ROOT / "tools" / "pgo_workload.py"]
    site = next((EXP / ".venv" / "lib").glob("python3*/site-packages"))
    rivals = [*(EXP / "baselines" / "bin").iterdir(), *site.glob("hnswlib*.so"), *site.glob("glass*.so"),
              *(site / "corenn_py").rglob("*.so"), EXP / "baselines" / "scann" / "Dockerfile"]
    runners = [*EXP.glob("bench/runners/*.py"), EXP / "scripts" / "step3_baselines.py", *EXP.glob("scripts/step6_*.py")]
    return {"index": rel(code, ROOT),
            "package": {p.name: sha(p) for p in sorted(Path(vindex.__file__).parent.glob("*.so"))},
            "rivals": rel(rivals, EXP),
            "rival_versions": sorted(p.name for p in site.glob("*.dist-info")
                                     if p.name.split("-")[0].lower() in ("hnswlib", "glass", "corenn_py")),
            "runners": rel(runners, EXP)}


def frozen_changes(parts=("index", "package", "rivals", "rival_versions"), frozen=None):
    """What has changed in the locked parts since step 28 recorded them (or since `frozen`, another fingerprint file,
    as part 5's step 42): [] when nothing has."""
    frozen = frozen or FROZEN
    if not frozen.exists():
        return [f"the fingerprint file {frozen.name} is missing"]
    then, now = json.loads(frozen.read_text()), fingerprints()
    out = []
    for part in parts:
        a, b = then[part], now[part]
        if isinstance(a, list):
            out += [f"{part}: {x} {'removed' if x in a else 'added'}" for x in sorted(set(a) ^ set(b))]
            continue
        out += [f"{part}: {k} {'changed' if k in b else 'removed'}" for k in a if a[k] != b.get(k)]
        out += [f"{part}: {k} added" for k in b if k not in a]
    return out


def unlock_held_back():
    """Steps 22 and 27 only: let load() and truth() read the held-back datasets."""
    global _held_back_unlocked
    _held_back_unlocked = True


def unlock_held_back_5():
    """Step 42 only: let load() and truth() read part 5's held-back dataset."""
    global _held_back_5_unlocked
    _held_back_5_unlocked = True


def metric_of(name):
    return {**DATASETS, **HELD_BACK, **HELD_BACK_3, **HELD_BACK_5}[name]


def _guard(name):
    if (name in HELD_BACK or name in HELD_BACK_3) and not _held_back_unlocked:
        raise PermissionError(f"{name} is a held-back dataset: only steps 22 and 27 may load it (SPEC.md, parts 2 and 3)")
    if name in HELD_BACK_5 and not _held_back_5_unlocked:
        raise PermissionError(f"{name} is part 5's held-back dataset: only step 42 may load it (SPEC.md, part 5)")


def load(name, what):
    _guard(name)
    return np.ascontiguousarray(np.load(DATA / name / f"{what}.npy"))


def truth(name, what):
    _guard(name)
    z = np.load(DATA / name / f"truth_{what}.npz")
    return {k: z[k] for k in z.files}


def reference(stage, name):
    """The experiments' result for a stage and dataset; earlier stages stand in where a stage left a dataset as is."""
    order = [STAGE_V0, STAGE_FLIP, STAGE_EFFORT, STAGE_QLINKS, STAGE_COPIES, STAGE_TWICE]
    for st in reversed(order[: order.index(stage) + 1]):
        f = REF / "step5" / "runs" / st / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text())
    raise FileNotFoundError(f"no reference for {stage} {name}")


def dist_pairs(Q, X, ids, metric):
    """Exact distance (64-bit) from Q[i] to X[ids[i, j]]; -1 ids give +inf."""
    safe = np.where(ids < 0, 0, ids)
    xs = X[safe].astype(np.float64)
    q = Q.astype(np.float64)
    if metric == "l2":
        d = np.einsum("nmd,nmd->nm", xs - q[:, None, :], xs - q[:, None, :])
    else:
        s = np.einsum("nmd,nd->nm", xs, q)
        d = 1.0 - s if metric == "cos" else -s
    d[ids < 0] = np.inf
    return d


def dedupe(ids):
    ids = np.array(ids, dtype=np.int64, copy=True)
    for row in ids:
        seen = set()
        for j, v in enumerate(row):
            if v < 0:
                continue
            if v in seen:
                row[j] = -1
            else:
                seen.add(v)
    return ids


def correct_counts(found, Q, X, truth_d, metric, k=10):
    """Per question: how many of the first k returned items are really in the top k (ties count)."""
    ids = dedupe(np.asarray(found)[:, :k])
    d = dist_pairs(Q, X, ids, metric)
    return np.minimum((d <= truth_d[:, k - 1][:, None] + TIE_EPS).sum(axis=1), k)


def accuracy(counts, k=10):
    return float(np.mean(counts) / k)


def cost_at_target(accs, costs, target=TARGET):
    """Cost at exactly the target accuracy (linear between the two points around it), as in the experiments."""
    r = np.asarray(accs, dtype=float)
    c = np.asarray(costs, dtype=float)
    order = np.argsort(c, kind="stable")
    r, c = np.maximum.accumulate(r[order]), c[order]
    hit = np.nonzero(r >= target)[0]
    if len(hit) == 0:
        return None, "never"
    i = int(hit[0])
    if i == 0:
        return float(c[0]), "below_min_effort"
    r0, r1, c0, c1 = r[i - 1], r[i], c[i - 1], c[i]
    if r1 == r0:
        return float(c1), "ok"
    return float(c0 + (target - r0) * (c1 - c0) / (r1 - r0)), "ok"


def within(a, b, rel):
    return a is not None and b is not None and abs(a - b) <= rel * abs(b)


class Report:
    """Collects PASS/FAIL lines for one step, prints them, and writes validate/results/<step>.json."""

    def __init__(self, step):
        self.step, self.rows, self.t0 = step, [], time.time()

    def check(self, name, ok, detail="", data=None):
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail, "data": data})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        return ok

    def note(self, text):
        print(f"      {text}", flush=True)

    def finish(self):
        OUT.mkdir(parents=True, exist_ok=True)
        ok = all(r["pass"] for r in self.rows)
        summary = {"step": self.step, "pass": ok, "seconds": round(time.time() - self.t0, 1), "checks": self.rows}
        (OUT / f"{self.step}.json").write_text(json.dumps(summary, indent=1, default=float))
        print(f"\n{self.step}: {'ALL PASS' if ok else 'SOME FAILED'} "
              f"({sum(r['pass'] for r in self.rows)}/{len(self.rows)}) in {summary['seconds']} s")
        sys.exit(0 if ok else 1)


def prototype_order(X, metric, flipped, seed=7):
    """The prototype's insertion order (bench/ours/index.py, build): its start item (closest to the average, in
    build space), then numpy's permutation with seed 7 without it. Our build reproduces the prototype's map exactly
    when given this order, so results can be compared one to one."""
    B = X.astype(np.float32)
    if flipped:
        B = (X / np.maximum(np.sum(X.astype(np.float64) ** 2, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    start = int(np.argmin(np.sum((B - B.mean(axis=0)) ** 2, axis=1)))
    order = np.random.default_rng(seed).permutation(len(X))
    return np.concatenate([[start], order[order != start]]).astype(np.int64)


def flipped_for(metric, settings):
    space = settings.get("build_space", "auto")
    return metric != "cos" and space != "same" and (space == "flipped" or metric == "ip")


# Part 2 changed a default that changes results; these settings give part 1's design back, so its checks still
# compare one to one with the prototype and with step 12's recording. Part 2's checks set what they change.
PART1 = {"effort_apart": False, "tune": "off", "renumber": False, "question_bits": 32, "popular_links": "off",
         "repair": "append", "n_replace": 3}   # the last two: how deletes repaired links in parts 1 to 4


def build_index(name, settings, order=None, seed=7, train=False, size="slice"):
    """Build our index on a slice (part 1's design unless `settings` says otherwise). order="prototype" uses the
    prototype's exact insertion order. size="full": on every item instead (part 4)."""
    import vindex
    metric = metric_of(name)
    X = load(name, "slice" if size == "slice" else "items_full")
    idx = vindex.Index(X.shape[1], metric, seed=seed, **{**PART1, **settings})
    kw = {"train_questions": load(name, "q_train")} if train else {}
    if order == "prototype":
        kw["order"] = prototype_order(X, metric, flipped_for(metric, settings))
    t = time.time()
    idx.build(X, np.arange(len(X)), **kw)
    return idx, time.time() - t


def sweep_list_sizes(idx, Q, X, td, metric, after=2):
    """Accuracy and checks per search for growing list sizes; stop two sizes after 95% is first reached."""
    curve, hit = [], None
    for L in L_SWEEP:
        ids, _, checks = idx.search(Q, k=10, list_size=L, threads=16)
        acc = accuracy(correct_counts(ids, Q, X, td, metric))
        curve.append({"L": L, "accuracy": acc, "checks": float(checks.mean())})
        if hit is None and acc >= TARGET:
            hit = len(curve) - 1
        if hit is not None and len(curve) - 1 - hit >= after:
            break
    return curve


def sweep_targets(idx, Q, X, td, metric):
    """Accuracy and checks per search for each target accuracy of the learned effort."""
    out = []
    for T in TARGETS:
        ids, _, checks = idx.search(Q, k=10, target=T, threads=16)
        out.append({"target": T, "accuracy": accuracy(correct_counts(ids, Q, X, td, metric)), "checks": float(checks.mean())})
    return out


def at95(curve):
    return cost_at_target([c["accuracy"] for c in curve], [c["checks"] for c in curve])


# The prototype's checks at 95% over its own seeds 7, 1, 2, 3, where measured (with the prototype, 2026-11-04).
PROTO_SEEDS = {(STAGE_QLINKS, "llama"): (1263, 1294, 1241, 1261)}


def compare_stage(rep, name, stage, settings, learned, own_tol=0.05, seeds=(1, 2, 3)):
    """The two checks of a stage on one dataset:
    1. built with the prototype's exact order: checks at 95% within 1% of the prototype (the port is exact);
    2. built with our own random order: within own_tol, or (for maps that swing with the order) the prototype's
       value inside our spread over four seeds, or our average over four seeds within own_tol of the prototype's
       average over its seeds."""
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    ref = reference(stage, name)
    sweep = sweep_targets if learned else sweep_list_sizes
    out = {}
    for label, order in (("prototype's order", "prototype"), ("our own order", None)):
        idx, secs = build_index(name, settings, order=order, train=learned or settings.get("question_links") != "off")
        curve = sweep(idx, Q, X, td, metric)
        cost, status = at95(curve)
        best = max(c["accuracy"] for c in curve)
        out[label] = (idx, cost, best, secs)
        if ref["cmps_at95"] is None:
            if order:
                ok = cost is None and abs(best - ref["best_accuracy"]) <= 0.002
                detail = f"best {best:.2%} vs prototype {ref['best_accuracy']:.2%} (pass: within 0.2 points)"
            else:
                ok = cost is None
                detail = f"best {best:.2%} (prototype {ref['best_accuracy']:.2%}; this varies with the order)"
            rep.check(f"{name}, {label}: never reaches 95%, like the prototype", ok, detail)
            continue
        rel = abs(cost - ref["cmps_at95"]) / ref["cmps_at95"] if cost else float("inf")
        if order:
            rep.check(f"{name}, {label}: checks at 95% within 1%", rel <= 0.01,
                      f"ours {cost:,.0f} vs prototype {ref['cmps_at95']:,.0f} ({rel:.2%} apart)", {"curve": curve})
        elif rel <= own_tol:
            rep.check(f"{name}, {label}: checks at 95% within {own_tol:.0%}", True,
                      f"ours {cost:,.0f} vs prototype {ref['cmps_at95']:,.0f} ({rel:.1%} apart)", {"curve": curve})
        else:
            costs = [cost]
            for sd in seeds:
                alt, _ = build_index(name, settings, seed=sd, train=learned or settings.get("question_links") != "off")
                costs.append(at95(sweep(alt, Q, X, td, metric))[0])
            proto = PROTO_SEEDS.get((stage, name), (ref["cmps_at95"],))
            inside = None not in costs and min(costs) * 0.98 <= ref["cmps_at95"] <= max(costs) * 1.02
            means = None not in costs and abs(np.mean(costs) - np.mean(proto)) <= own_tol * np.mean(proto)
            rep.check(f"{name}, {label}: prototype inside our seed spread, or averages over seeds within {own_tol:.0%}",
                      inside or means,
                      f"prototype {', '.join(f'{c:,.0f}' for c in proto)} (average {np.mean(proto):,.0f}); ours over "
                      f"seeds 7, {', '.join(map(str, seeds))}: " + ", ".join("never" if c is None else f"{c:,.0f}" for c in costs)
                      + ("" if None in costs else f" (average {np.mean(costs):,.0f}, "
                         f"{np.mean(costs) / np.mean(proto) - 1:+.1%})"))
    return out
