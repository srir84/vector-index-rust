"""Step 3: the plain map (32 links, build effort 200, slack 1.2) against the prototype's starting version
(results/step5/runs/v0), on all five slices, with a list-size sweep:
  - built with the prototype's own insertion order: checks at 95% within 1% (the port is exact);
  - built with our own random order: within 5%, or the prototype inside our seed spread where maps swing.

.venv/bin/python -I validate/step3.py [dataset ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, STAGE_V0, Report, compare_stage, reference

PLAIN = dict(code_bits=0, learned_effort=False, question_links="off", build_space="same")

if __name__ == "__main__":
    rep = Report("step3")
    for name in (sys.argv[1:] or DATASETS):
        out = compare_stage(rep, name, STAGE_V0, PLAIN, learned=False)
        idx, _, _, secs = out["our own order"]
        st, ref = idx.stats(), reference(STAGE_V0, name)
        rep.note(f"{name}: built in {secs:.1f} s; {st['build_checks'] / st['items']:,.0f} build checks per item "
                 f"(prototype {ref['build_cmps_per_item']:,.0f}); links per item {st['mean_links']:.1f} "
                 f"(prototype {ref['graph']['mean_links']:.1f})")
    rep.finish()
