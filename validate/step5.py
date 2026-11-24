"""Step 5: the learned effort, against the prototype's stage with it
(results/step5/runs/v0+build-mobius+learned-effort), on all five slices, sweeping the target accuracy:
  - with the prototype's own insertion order: checks at 95% within 1%;
  - with our own random order: within 5%, or the prototype inside our seed spread where maps swing;
  - the hardest group gets at least as much effort as the easiest.

.venv/bin/python -I validate/step5.py [dataset ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, STAGE_EFFORT, Report, compare_stage

SETTINGS = dict(code_bits=0, learned_effort=True, question_links="off", build_space="auto")

if __name__ == "__main__":
    rep = Report("step5")
    for name in (sys.argv[1:] or DATASETS):
        out = compare_stage(rep, name, STAGE_EFFORT, SETTINGS, learned=True)
        table, _ = out["our own order"][0].effort_table(0.95)
        rep.check(f"{name}: the hardest group gets at least the easiest group's effort", table[0] >= table[-1],
                  f"table for 95%, hardest group first: {table}")
    rep.finish()
