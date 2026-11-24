"""Export the Python prototype's plain map of GloVe, with its own search results, for step 2.

Runs in the EXPERIMENTS environment, because it uses the prototype itself:
  ../vector-index-experiments/.venv/bin/python -I validate/export_prototype_map.py
Writes validate/fixtures/glove_v0_map.npz (about 20 MB).
With arguments `llama mobius` it exports Llama's map built on flipped copies instead (llama_mobius_map.npz).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "vector-index-experiments"
sys.path.insert(0, str(EXP))
from bench.data import prepared  # noqa: E402
from bench.ours.index import OurIndex, Settings  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "glove"
space = sys.argv[2] if len(sys.argv) > 2 else "same"
p = prepared(name)
X = np.ascontiguousarray(p.load("slice"))
Q = np.ascontiguousarray(p.load("q_tune"))
s = Settings(R=32, L_build=200, alpha=1.2, build_space=space)
idx = OurIndex(X.shape[1], p.metric, len(X), s).build(X, np.arange(len(X)))
out = {"nbrs": idx.nbrs[: len(X)].astype(np.uint32), "deg": idx.deg[: len(X)].astype(np.uint32),
       "entry": np.int64(idx.medoid), "L": np.array([10, 50, 200])}
for L in (10, 50, 200):
    ids, ds, cf, cc, hops = idx.search(Q, L, k=10, patience=0)
    out[f"ids_{L}"], out[f"checks_{L}"] = ids.astype(np.int64), (cf + cc).astype(np.int64)
(HERE / "fixtures").mkdir(exist_ok=True)
tag = "v0" if space == "same" else space
np.savez(HERE / "fixtures" / f"{name}_{tag}_map.npz", **out)
print("saved", {k: v.shape for k, v in out.items() if hasattr(v, "shape")}, "entry", int(idx.medoid))
