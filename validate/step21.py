"""Step 21: cheaper checks, new options.
  - 16-bit whole-number questions (now the default): on the five, accuracy at the 90%, 95% and 99% targets within
    half a point of float questions, and at least 10% faster on one core, timed head-to-head;
  - 4-bit copies (with 16-bit questions): where they can reach 95%, the learned effort lands within 1 point of it;
    memory per item and speed against 1-byte copies are reported;
  - step 18's check again with the new choices: the tuning also tries 4-bit copies, and the self-picked choice must
    be within 10% of the best of links x slack x copy size, timed head-to-head.
All on the tuning questions.

.venv/bin/python -I validate/step21.py [dataset ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, OUT, TARGETS, Report, accuracy, correct_counts, load, truth
from step18 import PART2, build, grid, picked, plan95, sec95

# part 2's design: renumbered, 16-bit questions (step 18's PART2 keeps part 1's float questions and order)
PART2_21 = dict(PART2, question_bits=16, renumber=True)


def accuracies(idx, name, targets=(0.90, 0.95, 0.99)):
    metric = DATASETS[name]
    X, Q, td = load(name, "slice"), load(name, "q_tune"), truth(name, "slice_tune")["dists"]
    return {t: accuracy(correct_counts(idx.search(Q, k=10, target=t, threads=16)[0], Q, X, td, metric)) for t in targets}


def whole_number_questions(rep, name):
    a, _ = build(name, dict(PART2_21, tune="off", question_bits=32))
    b, _ = build(name, dict(PART2_21, tune="off", question_bits=16))
    acc32, acc16 = accuracies(a, name), accuracies(b, name)
    worst = max(abs(acc16[t] - acc32[t]) for t in acc32)
    rep.check(f"{name}: 16-bit questions keep accuracy at the targets (within half a point)", worst <= 0.005,
              "; ".join(f"target {t:.0%}: {acc32[t]:.2%} -> {acc16[t]:.2%}" for t in acc32))
    Q = load(name, "q_tune")
    pa, pb = plan95(a, name), plan95(b, name)
    ta = tb = float("inf")
    for _ in range(5):
        ta, tb = min(ta, sec95(a, Q, pa, runs=1)), min(tb, sec95(b, Q, pb, runs=1))
    rep.check(f"{name}: 16-bit questions at least 10% faster (one core, at 95%)", ta / tb >= 1.10,
              f"{1 / ta:,.0f} -> {1 / tb:,.0f} searches/s ({ta / tb:.2f}x); checks per search {pa[3]:,.0f} -> {pb[3]:,.0f}")


def four_bit(rep, name):
    b, _ = build(name, dict(PART2_21, tune="off", code_bits=4))
    acc = accuracies(b, name, TARGETS)
    best = max(acc.values())
    m = b.memory()
    per = m["total_bytes"] / m["items"]
    if best < 0.95:
        rep.check(f"{name}: 4-bit copies: never reach 95% here, so the tuning must not pick them", True,
                  f"best {best:.2%} at any target; {per:,.0f} bytes per item")
        return
    rep.check(f"{name}: 4-bit copies: the learned effort lands within 1 point of 95%", abs(acc[0.95] - 0.95) <= 0.01,
              f"target 95% gives {acc[0.95]:.2%}; {per:,.0f} bytes per item")


def main():
    rep = Report("step21")
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        whole_number_questions(rep, name)
    for name in names:
        four_bit(rep, name)
    configs = [(r, a, bits) for r in (24, 32, 48, 64, 96) for a in (1.0, 1.1, 1.2, 1.3) for bits in (8, 4)]
    for name in names:
        saved = OUT / f"step21-grid-{name}.json"
        rows = grid(name, configs, PART2_21, saved=saved)
        saved.write_text(json.dumps(rows, indent=1))
        picked(rep, name, rows, PART2_21, "the self-picked choice (4-bit copies allowed) is within 10% of the best")
    rep.finish()


if __name__ == "__main__":
    main()
