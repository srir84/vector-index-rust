"""Step 8: question links made twice at build (Fix 8) — the final design — against the prototype's stage with it
(results/step5/runs/v0+build-mobius+learned-effort+qlinks-auto+tiny8+qrounds2), on Yandex and Llama:
  - with the prototype's own insertion order: checks at 95% within 1%;
  - with our own random order: within 5%, or (maps that swing with the order) by seed spread or averages.

.venv/bin/python -I validate/step8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import STAGE_TWICE, Report, compare_stage

FINAL = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=2, build_space="auto", effort_apart=False, tune="off", renumber=False, question_bits=32, popular_links="off")

if __name__ == "__main__":
    rep = Report("step8")
    for name in (sys.argv[1:] or ["yandex", "llama"]):
        compare_stage(rep, name, STAGE_TWICE, FINAL, learned=True)
    rep.finish()
