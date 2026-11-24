"""Build one dataset's index with the vindex package found in a given folder, and save it (step 25's head-to-head:
the code before a step builds the index, so that both codes can load it).

.venv/bin/python -I validate/build_saved.py <package folder, or "installed"> <dataset> <settings as JSON> <folder>
"""
import json
import sys
from pathlib import Path

if sys.argv[1] != "installed":
    sys.path.insert(0, sys.argv[1])
import vindex  # noqa: E402  (the chosen package, before step18 imports it)

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from step18 import build  # noqa: E402

idx, _ = build(sys.argv[2], json.loads(sys.argv[3]))
idx.save(sys.argv[4])
print(json.dumps({"module": vindex.__file__}))
