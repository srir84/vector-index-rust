"""Step 44: build the same index (40,000 random items, tune off, cos/ip/l2) and save it to OUT_DIR, so the saved files
from two versions of the package can be compared with diff -r.
    .venv/bin/python -I validate/step44_same_build.py OUT_DIR
"""
import sys, numpy as np, vindex
rng = np.random.default_rng(5)
X = rng.standard_normal((40000, 48), dtype=np.float32); T = rng.standard_normal((3000, 48), dtype=np.float32)
for metric in ("cos", "ip", "l2"):
    idx = vindex.Index(48, metric=metric, tune="off")
    idx.build(X, np.arange(len(X)), train_questions=T)
    idx.save(f"{sys.argv[1]}/{metric}")
