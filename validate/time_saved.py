"""Time a saved index on one core at 95% on the tuning questions, with the vindex package found in a given folder
(step 25's head-to-head between the code before it and after it). Prints seconds per search, one run.

.venv/bin/python -I validate/time_saved.py <package folder, or "installed"> <saved index> <dataset>
"""
import json
import sys
from pathlib import Path

if sys.argv[1] != "installed":
    sys.path.insert(0, sys.argv[1])
import vindex  # noqa: E402  (the chosen package, before step18 imports it)

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import load  # noqa: E402
from step18 import plan95, sec95  # noqa: E402

idx = vindex.Index.load(sys.argv[2])
name = sys.argv[3]
p = plan95(idx, name)
print(json.dumps({"sec": sec95(idx, load(name, "q_tune"), p, runs=1), "checks": p[3], "module": vindex.__file__}))
