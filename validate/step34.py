"""Step 34: the GooAQ demo (part 4). After demo/prepare.py, in the demo's own environment:
    demo/.venv/bin/python -I validate/step34.py
It starts demo/server.py on a spare port and asks it 100 of GooAQ's own questions (a fixed random draw), then checks:
  - each text the demo shows is the one its numbers were made from: the model, run here on every text shown for
    the 100 questions (by either index), gives numbers within 0.999 (angle similarity) of that item's numbers;
  - the demo's answers are the index's own: the same index files, searched here with the same question numbers
    (checked to be the model's, run here), give the same 10 items in the same order, for vindex and for HNSW;
  - GooAQ's licence allows showing its text (its LICENSE, downloaded with it, is Apache 2.0) and the page says what
    the licence asks: credit, and no commercial use;
  - nearly every item has its text (step 1 of demo/prepare.py matched it): at least 99%.
"""
import json
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, ROOT, Report, frozen_changes

DEMO = ROOT / "demo"
sys.path.insert(0, str(DEMO))
from prepare import TEXT, WORK, model   # noqa: E402

PORT = 8035
SAME = 0.999


def ask(q):
    u = f"http://127.0.0.1:{PORT}/search?" + urllib.parse.urlencode({"q": q, "numbers": 1})
    with urllib.request.urlopen(u, timeout=60) as r:
        return json.loads(r.read())


def questions(n=100):
    import pyarrow.parquet as pq
    qs = sorted({q for f in sorted(TEXT.glob("train-*.parquet")) for q in pq.read_table(f, columns=["question"]).column("question").to_pylist()})
    return random.Random(34).sample(qs, n)


def main():
    import hnswlib
    import vindex
    rep = Report("step34")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    srv = subprocess.Popen([sys.executable, "-I", str(DEMO / "server.py"), "--port", str(PORT)], stdout=subprocess.PIPE, text=True)
    try:
        line = srv.stdout.readline()
        assert "ready" in line, f"the server did not start: {line!r}"
        qs = questions()
        t = time.time()
        got = [ask(q) for q in qs]
        ask_s = time.time() - t
    finally:
        srv.terminate()
        srv.wait()
    m = model()
    X = np.load(DATA / "gooaq" / "items_full.npy", mmap_mode="r")
    # the question numbers the server searched are the model's
    mine = m.encode(qs, convert_to_numpy=True, normalize_embeddings=True, batch_size=1).astype(np.float32)
    theirs = np.array([g["numbers"] for g in got], dtype=np.float32)
    q_diff = float(np.abs(mine - theirs).max())
    # the texts shown are the ones their numbers were made from
    shown = {}
    for g in got:
        for name in ("vindex", "hnsw"):
            for r in g[name]["results"]:
                shown[r["item"]] = r["text"]
    no_text = [i for i, s in shown.items() if s is None]
    items = sorted(i for i, s in shown.items() if s is not None)
    v = m.encode([shown[i] for i in items], convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    sim = np.einsum("ij,ij->i", v.astype(np.float64), np.asarray(X[items], dtype=np.float64))
    rep.check("each text the demo shows is the one its numbers were made from",
              not no_text and float(sim.min()) >= SAME,
              f"{len(items):,} different items shown for {len(qs)} questions; the model on their texts gives numbers "
              f"within {sim.min():.5f} (lowest), {np.median(sim):.6f} (middle) of the items'"
              + (f"; {len(no_text)} shown without text" if no_text else ""),
              {"items": len(items), "lowest": float(sim.min()), "median": float(np.median(sim)), "no_text": len(no_text)})
    # the demo's answers are the index's own
    idx = vindex.Index.load(WORK / "vindex")
    h = json.loads((WORK / "hnsw.json").read_text())
    hn = hnswlib.Index(space="ip", dim=768)
    hn.load_index(str(WORK / "hnsw.bin"))
    hn.set_ef(max(int(h["effort"]), 10))
    target = got[0]["settings"]["vindex_target"]
    ids, _, _ = idx.search(theirs, k=10, target=target, threads=1)
    hl, _ = hn.knn_query(theirs, k=10, num_threads=1)
    diff_v = [i for i, g in enumerate(got) if [r["item"] for r in g["vindex"]["results"]] != [int(x) for x in ids[i] if x >= 0]]
    diff_h = [i for i, g in enumerate(got) if [r["item"] for r in g["hnsw"]["results"]] != [int(x) for x in hl[i]]]
    rep.check("for 100 questions, the demo's answers are the index's own (vindex and HNSW, same order)",
              not diff_v and not diff_h and q_diff < 1e-4,
              f"vindex differs on {len(diff_v)}, HNSW on {len(diff_h)}; the server's question numbers are the model's "
              f"to within {q_diff:.1e}; target {target}, HNSW effort {h['effort']} ({h['params']}); "
              f"{len(qs)} questions answered in {ask_s:.1f} s",
              {"vindex_ms": [g["vindex"]["ms"] for g in got], "hnsw_ms": [g["hnsw"]["ms"] for g in got],
               "embed_ms": [g["embed_ms"] for g in got],
               "same_answers": [len({r["item"] for r in g["vindex"]["results"]} & {r["item"] for r in g["hnsw"]["results"]})
                                for g in got]})
    lic = (DEMO / "data" / "gooaq-licence" / "LICENSE").read_text()
    readme = (DEMO / "data" / "gooaq-licence" / "README.md").read_text()
    page = (DEMO / "index.html").read_text()
    ok = ("Apache License" in lic and "Version 2.0" in lic and "should not be used for any commercial purposes" in readme
          and "Apache 2.0" in page and "commercial" in page and "AllenAI" in page)
    rep.check("GooAQ's licence allows showing its text, and the page credits it and says it is not for commercial use", ok,
              "LICENSE: Apache License 2.0; its README: \"should not be used for any commercial purposes\"; the page "
              "credits GooAQ by AllenAI, the licence and the request")
    ia = np.load(WORK / "item_answer.npy")
    share = float((ia >= 0).mean())
    rep.check("at least 99% of the items have their answer text", share >= 0.99,
              f"{int((ia >= 0).sum()):,} of {len(ia):,} ({share:.3%})", {"share": share})
    t_v, t_h = np.median([g["vindex"]["ms"] for g in got]), np.median([g["hnsw"]["ms"] for g in got])
    rep.note(f"times shown (middle over the 100 questions): vindex {t_v:.3f} ms, HNSW {t_h:.3f} ms, "
             f"question to numbers {np.median([g['embed_ms'] for g in got]):.1f} ms")
    rep.finish()


if __name__ == "__main__":
    main()
