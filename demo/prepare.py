"""Step 34: the GooAQ demo's data (part 4). Run in the demo's own environment (README in this folder):
    demo/.venv/bin/python -I demo/prepare.py
Each stage writes to demo/work/ and is skipped when its output is already there:
  1. answers   the unique answers in GooAQ's text (demo/data/gooaq/pair/*.parquet, downloaded from Hugging Face's
               sentence-transformers/gooaq), in first-seen order -> answers.bin (UTF-8) + answers_at.npy (offsets)
  2. numbers   each answer turned into numbers by all-distilroberta-v1, the model that made the benchmark's GooAQ
               file, scaled to length 1, on the Apple GPU -> answer_numbers.npy
  3. match     each benchmark item (the full GooAQ items) to the answer it was made from: our index over the
               answers' numbers, nearest one, accepted at an angle similarity of 0.999 or more; rows it misses get an
               exact check over every answer -> item_answer.npy (answer number per item, -1 if none)
  4. vindex    our index on the full GooAQ items, built as in step 29 (part 3's index, every default, the 20,000
               training questions) -> vindex/
  5. hnsw      HNSW (hnswlib 0.8.0) on the same items, with the settings its full-size search picked in step 30
               -> hnsw.bin + hnsw.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT.parent / "vector-index-experiments"
DATA = EXP / "data" / "prepared" / "gooaq"
TEXT = HERE / "data" / "gooaq" / "pair"
WORK = HERE / "work"
MODEL = "sentence-transformers/all-distilroberta-v1"
PART3 = dict(effort_apart=True, tune="auto", renumber=True, question_bits=16, popular_links="auto")   # step 27's
SAME = 0.999


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def answers():
    if (WORK / "answers_at.npy").exists():
        return
    import pyarrow.parquet as pq
    seen, out = set(), []
    for f in sorted(TEXT.glob("train-*.parquet")):
        for a in pq.read_table(f, columns=["answer"]).column("answer").to_pylist():
            if a not in seen:
                seen.add(a)
                out.append(a)
    blob = [a.encode("utf-8") for a in out]
    at = np.zeros(len(blob) + 1, dtype=np.int64)
    at[1:] = np.cumsum([len(b) for b in blob])
    (WORK / "answers.bin").write_bytes(b"".join(blob))
    np.save(WORK / "answers_at.npy", at)
    log(f"answers: {len(out):,} unique")


def read_answers():
    blob = (WORK / "answers.bin").read_bytes()
    at = np.load(WORK / "answers_at.npy")
    return [blob[at[i]:at[i + 1]].decode("utf-8") for i in range(len(at) - 1)]


def model():
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")


def numbers():
    out = WORK / "answer_numbers.npy"
    if out.exists():
        return
    texts = read_answers()
    m = model()
    t = time.time()
    v = np.lib.format.open_memmap(WORK / "answer_numbers.part.npy", mode="w+", dtype=np.float32, shape=(len(texts), 768))
    step = 65_536
    for i in range(0, len(texts), step):
        v[i:i + step] = m.encode(texts[i:i + step], batch_size=256, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
        log(f"numbers: {min(i + step, len(texts)):,} of {len(texts):,} ({time.time() - t:.0f} s)")
    v.flush()
    del v
    (WORK / "answer_numbers.part.npy").rename(out)


def match():
    out = WORK / "item_answer.npy"
    if out.exists():
        return
    import vindex
    A = np.load(WORK / "answer_numbers.npy", mmap_mode="r")
    X = np.load(DATA / "items_full.npy", mmap_mode="r")
    t = time.time()
    # a helper index, searched with a fixed list size (so no sample questions); every miss gets the exact check below
    idx = vindex.Index(A.shape[1], "cos", tune="off", learned_effort=False)
    idx.build(np.ascontiguousarray(A), np.arange(len(A), dtype=np.int64))
    log(f"match: index over the answers built in {time.time() - t:.0f} s")
    best = np.full(len(X), -1, dtype=np.int64)
    sim = np.zeros(len(X), dtype=np.float32)
    step = 100_000
    for i in range(0, len(X), step):
        q = np.ascontiguousarray(X[i:i + step])
        ids, _, _ = idx.search(q, k=1, list_size=64, threads=16)
        j = ids[:, 0]
        ok = j >= 0
        best[i:i + step][ok] = j[ok]
        sim[i:i + step][ok] = np.einsum("ij,ij->i", q[ok], np.asarray(A[j[ok]]))
    miss = np.flatnonzero(sim < SAME)
    log(f"match: {len(X) - len(miss):,} of {len(X):,} matched by the index; exact check for the other {len(miss):,}")
    for i in range(0, len(miss), 2_000):   # every answer, in blocks
        rows = miss[i:i + 2_000]
        q = np.asarray(X[rows])
        top, top_s = np.full(len(rows), -1), np.full(len(rows), -2.0, dtype=np.float32)
        for b in range(0, len(A), 100_000):
            s = q @ np.asarray(A[b:b + 100_000]).T
            k = s.argmax(1)
            v = s[np.arange(len(rows)), k]
            better = v > top_s
            top[better], top_s[better] = k[better] + b, v[better]
        best[rows], sim[rows] = top, top_s
    best[sim < SAME] = -1
    np.save(out, best.astype(np.int32))
    np.save(WORK / "item_answer_similarity.npy", sim)
    log(f"match: {int((best >= 0).sum()):,} of {len(X):,} items have their answer text "
        f"(similarity: lowest accepted {sim[best >= 0].min():.5f}, median {np.median(sim):.6f})")


def build_vindex():
    out = WORK / "vindex"
    if out.exists():
        return
    import vindex
    X = np.ascontiguousarray(np.load(DATA / "items_full.npy"))
    Qtr = np.load(DATA / "q_train.npy")
    t = time.time()
    idx = vindex.Index(X.shape[1], "cos", **PART3)
    idx.build(X, np.arange(len(X), dtype=np.int64), train_questions=Qtr)
    log(f"vindex: built in {time.time() - t:.0f} s")
    idx.save(WORK / "vindex.part")
    (WORK / "vindex.part").rename(out)


def build_hnsw():
    out = WORK / "hnsw.bin"
    if out.exists():
        return
    import hnswlib
    r = json.loads((EXP / "results" / "step6" / "final" / "hnsw" / "gooaq" / "best.json").read_text())
    p = r["params"]
    X = np.ascontiguousarray(np.load(DATA / "items_full.npy"))
    t = time.time()
    idx = hnswlib.Index(space="ip", dim=X.shape[1])   # as the experiments' runner: the items are scaled to length 1
    idx.init_index(max_elements=len(X), ef_construction=p["build"], M=max(2, p["links"] // 2), random_seed=100)
    idx.set_num_threads(16)
    idx.add_items(X, np.arange(len(X)), num_threads=16)
    log(f"hnsw: built in {time.time() - t:.0f} s with {p}")
    idx.save_index(str(WORK / "hnsw.part"))
    (WORK / "hnsw.part").rename(out)
    (WORK / "hnsw.json").write_text(json.dumps({"params": p, "effort": r["final"]["effort"], "from": "step 30 (step6/final/hnsw/gooaq)"}))


def main():
    WORK.mkdir(exist_ok=True)
    stages = {"answers": answers, "numbers": numbers, "match": match, "vindex": build_vindex, "hnsw": build_hnsw}
    for name in sys.argv[1:] or stages:
        stages[name]()
    log("done")


if __name__ == "__main__":
    main()
