"""Step 7: 1-byte copies with the re-check of the best 50 (Fix 6), against the prototype's stage with them
(results/step5/runs/v0+build-mobius+learned-effort+qlinks-auto+tiny8), on all five slices:
  - with the prototype's own insertion order: checks at 95% within 1%;
  - with our own random order: within 5%, or (maps that swing with the order) by seed spread or averages;
  - memory per item within 1% of the prototype's.

.venv/bin/python -I validate/step7.py [dataset ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, STAGE_COPIES, Report, compare_stage, reference

SETTINGS = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=1, build_space="auto")

if __name__ == "__main__":
    rep = Report("step7")
    for name in (sys.argv[1:] or DATASETS):
        out = compare_stage(rep, name, STAGE_COPIES, SETTINGS, learned=True)
        idx = out["our own order"][0]
        m, ref = idx.memory(), reference(STAGE_COPIES, name)
        per = m["total_bytes"] / m["items"]
        rep.check(f"{name}: memory per item within 1% of the prototype's", abs(per - ref["memory_per_item"]) <= 0.01 * ref["memory_per_item"],
                  f"ours {per:,.0f} bytes (links {m['links_bytes'] / m['items']:,.0f}, copies {m['numbers_bytes'] / m['items']:,.0f}, "
                  f"other {m['other_bytes'] / m['items']:,.0f}) vs prototype {ref['memory_per_item']:,.0f}")
    rep.finish()
