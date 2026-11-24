"""Step 6: question links. The automatic test switches them on for Yandex and Llama only (the prototype's ratios:
GooAQ 0.93, GloVe 0.99, Yandex 2.08, Llama 24.8, Landmark 1.00); with one round of them, Yandex and Llama match the
prototype's stage (results/step5/runs/v0+build-mobius+learned-effort+qlinks-auto):
  - with the prototype's own insertion order: checks at 95% within 1%;
  - with our own random order: within 5%, or the prototype inside our seed spread where maps swing.

.venv/bin/python -I validate/step6.py [dataset ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, STAGE_QLINKS, Report, build_index, compare_stage

SETTINGS = dict(code_bits=0, learned_effort=True, question_links="auto", q_rounds=1, build_space="auto")
PROTO_RATIO = {"gooaq": 0.93, "glove": 0.99, "yandex": 2.08, "llama": 24.8, "landmark": 1.00}

if __name__ == "__main__":
    rep = Report("step6")
    for name in (sys.argv[1:] or DATASETS):
        want_on = PROTO_RATIO[name] > 1.5
        if want_on:
            out = compare_stage(rep, name, STAGE_QLINKS, SETTINGS, learned=True)
            idx = out["our own order"][0]
        else:
            idx, _ = build_index(name, SETTINGS, train=True)
        st = idx.stats()
        rep.check(f"{name}: question links {'on' if want_on else 'off'}", st["question_links"] == want_on,
                  f"ratio {st['ood_ratio']:.2f} (prototype {PROTO_RATIO[name]:.2f}); "
                  f"{st['question_link_items']:,} items got question links")
    rep.finish()
