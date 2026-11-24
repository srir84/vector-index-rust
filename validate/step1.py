"""Step 1: distances and 1-byte copies agree with numpy on real data (1,000 items and questions per dataset).

.venv/bin/python -I validate/step1.py
"""
import sys
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATASETS, Report, load

rep = Report("step1")
for name, metric in DATASETS.items():
    X = load(name, "slice")[:1000]
    Q = load(name, "q_tune")[:1000]
    # full numbers: against 64-bit numpy
    got = vindex.debug_distances(metric, Q, X).astype(np.float64)
    q64, x64 = Q.astype(np.float64), X.astype(np.float64)
    if metric == "l2":
        want = (q64 ** 2).sum(1)[:, None] + (x64 ** 2).sum(1)[None, :] - 2 * q64 @ x64.T
    else:
        s = q64 @ x64.T
        want = 1.0 - s if metric == "cos" else -s
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))
    # numpy's own 32-bit result, for scale: where large terms cancel (Llama is not scaled to length 1),
    # any 32-bit sum loses relative accuracy, so the limit is 1e-5 or twice numpy's 32-bit error
    s32 = (Q @ X.T).astype(np.float64)
    np32 = (q64 ** 2).sum(1)[:, None] + (x64 ** 2).sum(1)[None, :] - 2 * s32 if metric == "l2" else (
        1.0 - s32 if metric == "cos" else -s32)
    err32 = np.max(np.abs(np32 - want) / np.maximum(1.0, np.abs(want)))
    limit = max(1e-5, 2 * err32)
    rep.check(f"{name}: full-number distances", err <= limit,
              f"worst error {err:.1e} (numpy 32-bit: {err32:.1e}; limit {limit:.1e})")

    # 1-byte copies: same scale as numpy computes, codes round half to even, distances by the same formula
    low, step, codes, cd = vindex.debug_codes(metric, X, Q)
    lo, hi = X.min(0), X.max(0)
    st = np.where(hi > lo, (hi - lo) / 255.0, 1.0).astype(np.float32)
    want_codes = np.clip(np.rint((X - lo) / st), 0, 255).astype(np.uint8)
    same = np.mean(codes == want_codes)
    decoded = low[None, :] + codes.astype(np.float32) * step[None, :]
    half = np.max(np.abs(decoded - X) - 0.5 * step[None, :])
    rep.check(f"{name}: copies match numpy and decode within half a step", same > 0.9999 and half <= 1e-6,
              f"same codes {same:.5f}, worst overshoot {half:.1e}")
    dec64 = decoded.astype(np.float64)
    s = q64 @ dec64.T
    if metric == "cos":
        want_c = 1.0 - s
    elif metric == "ip":
        want_c = -s
    else:
        want_c = (q64 ** 2).sum(1)[:, None] + (dec64 ** 2).sum(1)[None, :] - 2 * s
    errc = np.max(np.abs(cd - want_c) / np.maximum(1.0, np.abs(want_c)))
    rep.check(f"{name}: 1-byte distances", errc <= 1e-4, f"worst error {errc:.1e} (limit 1e-4)")
rep.finish()
