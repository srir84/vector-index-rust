"""Step 2: the walk, on the prototype's own map of GloVe, gives the prototype's answers with the same checks.

First, once, in the experiments environment:  ../vector-index-experiments/.venv/bin/python -I validate/export_prototype_map.py
Then:  .venv/bin/python -I validate/step2.py
"""
import sys
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import Report, load

rep = Report("step2")
fx = np.load(Path(__file__).resolve().parent / "fixtures" / "glove_v0_map.npz")
X, Q = load("glove", "slice"), load("glove", "q_tune")
for L in fx["L"]:
    ids, _, checks = vindex.debug_search_graph("cos", X, fx["nbrs"], fx["deg"], int(fx["entry"]), Q, int(L), 10)
    want = fx[f"ids_{L}"]
    same = np.mean([set(a) == set(b) for a, b in zip(ids, want)])
    c_ours, c_proto = int(checks.sum()), int(fx[f"checks_{L}"].sum())
    rel = abs(c_ours - c_proto) / c_proto
    rep.check(f"list size {L}: same top 10", same >= 0.995, f"{same:.2%} of 1,000 questions (pass: 99.5%)")
    rep.check(f"list size {L}: same checks", rel <= 0.005,
              f"ours {c_ours:,} vs prototype {c_proto:,} ({rel:.3%} apart; pass: 0.5%)")
rep.finish()
