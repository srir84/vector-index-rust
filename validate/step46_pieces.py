"""Step 46: what each piece adds, at full size (after the freeze; the code is step 45's).
  .venv/bin/python -I validate/step46_pieces.py [dataset ...]     # default: gooaq glove laion llama
Each dataset is built with the settings its own tuned build picked in step 42 (validate/results/step42-ours.json),
with the tuning off so every row starts from the same choice, then again with one piece turned off at a time:
  fixed effort       the same index, every query searched with one list size (read off at 95%)
  8-bit codes        where the tuning picked 4-bit codes
  float32 walk       no codes: the beam search compares the full vectors
  no question links  where they were on
  no popular links   where they were on
  graph on x         where the graph was built on the Möbius-transformed vectors
  defaults           r 32, alpha 1.2, 8-bit codes, rerank 50: what the tuning would start from
Measured as step 42 measured: test queries, one core, median of 5 timed runs, read off at exactly 95% recall@10.
Run it on an otherwise idle machine.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import L_SWEEP, OUT, TARGET, Report, accuracy, build_index, correct_counts, cost_at_target, load, metric_of, truth
from step11 import measure, timed
from step42 import PART5

NAMES = ["gooaq", "glove", "laion", "llama"]


def picked(name):
    r = json.loads((OUT / "step42-ours.json").read_text())
    d = next(c["data"] for c in r["checks"] if c.get("data") and c["check"].startswith(f"{name}:"))
    t, st = d["tuning"], d["stats"]
    return {"r": t["chosen_r"], "alpha": round(t["chosen_alpha"], 2), "code_bits": t["chosen_code_bits"],
            "rerank": t["chosen_rerank"], "question_links": "on" if st["question_links"] else "off",
            "popular_links": "on" if st["popular_items"] else "off", "build_space": st["build_space"]}, d["ours"]["qps95"]


def fixed_effort(name, idx):
    """One list size for every query: accuracy and one-core time for growing sizes, stop two after 95%."""
    X, Q = load(name, "items_full"), load(name, "q_test")
    td, metric = truth(name, "full_test")["dists"], metric_of(name)
    curve, hit = [], None
    for L in L_SWEEP:
        ids, _, checks = idx.search(Q, k=10, list_size=L, threads=1)
        sec = timed(lambda: idx.search(Q, k=10, list_size=L, threads=1)) / len(Q)
        curve.append({"L": L, "accuracy": accuracy(correct_counts(ids, Q, X, td, metric)), "sec": sec,
                      "checks": float(checks.mean())})
        if hit is None and curve[-1]["accuracy"] >= TARGET:
            hit = len(curve) - 1
        if hit is not None and len(curve) - 1 - hit >= 2:
            break
    sec95, status = cost_at_target([c["accuracy"] for c in curve], [c["sec"] for c in curve])
    checks95, _ = cost_at_target([c["accuracy"] for c in curve], [c["checks"] for c in curve])
    return {"curve": curve, "qps95": 1 / sec95 if sec95 else None, "status95": status, "checks95": checks95}


def variants(p):
    out = [("as picked", {})]
    if p["code_bits"] == 4:
        out.append(("8-bit codes", {"code_bits": 8}))
    out.append(("float32 walk", {"code_bits": 0}))
    if p["question_links"] == "on":
        out.append(("no question links", {"question_links": "off"}))
    if p["popular_links"] == "on":
        out.append(("no popular links", {"popular_links": "off"}))
    if p["build_space"] == "flipped":
        out.append(("graph on x", {"build_space": "same"}))
    default = {"r": 32, "alpha": 1.2, "code_bits": 8, "rerank": 50}
    if {**p, **default} not in [{**p, **v} for _, v in out]:
        out.append(("defaults", default))
    return out


def main():
    common.unlock_held_back()
    rep = Report("step46-pieces")
    for name in sys.argv[1:] or NAMES:
        p, qps42 = picked(name)
        base = None
        for label, change in variants(p):
            s = {**PART5, **p, **change, "tune": "off"}
            idx, build_s = build_index(name, s, train=True, size="full")
            m = measure(name, idx, size="full")
            row = {"dataset": name, "variant": label, "settings": s, "build_seconds": build_s, "ours": m,
                   "step42_qps95": qps42}
            if label == "as picked":
                base = m["qps95"]
                fx = fixed_effort(name, idx)
                rep.check(f"{name}: as picked, tuning off", bool(m["qps95"]),
                          f"{m['qps95'] or 0:,.0f}/s at 95% (step 42's tuned build: {qps42:,.0f}/s); {p}", row)
                rep.check(f"{name}: fixed effort", True,
                          f"{fx['qps95'] or 0:,.0f}/s at 95% ({(fx['qps95'] or 0) / base:.2f}x as picked)" if base else "",
                          {"dataset": name, "variant": "fixed effort", "fixed": fx, "base_qps95": base})
            else:
                rep.check(f"{name}: {label}", True,
                          f"{m['qps95'] or 0:,.0f}/s at 95% ({(m['qps95'] or 0) / base:.2f}x as picked); {change}"
                          if base else str(change), row)
            del idx
    rep.finish()


if __name__ == "__main__":
    main()
