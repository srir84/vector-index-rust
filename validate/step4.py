"""Step 4: links built on flipped copies (x / |x|^2) make Llama work (results/step5/runs/v0+build-mobius):
  - with the prototype's own insertion order: checks at 95% within 1%;
  - with our own random order: within 5%, or the prototype inside our spread over four seeds (Llama's maps swing
    with the order: seed to seed, checks at list size 10 range from about 480 to 650 in both implementations);
  - almost no item is left without links in.

.venv/bin/python -I validate/step4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import STAGE_FLIP, Report, compare_stage, reference

SETTINGS = dict(code_bits=0, learned_effort=False, question_links="off", build_space="auto")

if __name__ == "__main__":
    rep = Report("step4")
    out = compare_stage(rep, "llama", STAGE_FLIP, SETTINGS, learned=False)
    st = out["our own order"][0].stats()
    rep.check("the build used the flipped copies", st["build_space"] == "flipped", st["build_space"])
    rep.check("items with no links in: under 10", st["no_links_in"] < 10,
              f"ours {st['no_links_in']} (prototype {reference(STAGE_FLIP, 'llama')['graph']['orphans']}; "
              f"before the flip about 60,000)")
    rep.finish()
