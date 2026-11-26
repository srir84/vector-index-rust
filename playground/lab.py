"""The playground's work, shared by the command line and the browser page (both in play.py): download a dataset,
build an index on it, search it, and check the answers against the true ones.

Files: playground/data/<name>.hdf5 (the datasets), playground/work/<name>/ (the index, and what the build and the
last search found). A dataset file holds `train` (the items), and may hold `test` (questions), `neighbors` (their true
100 closest items), `learn` (sample questions, in files where questions look unlike the items) and a `distance`
attribute: VIBE's layout (https://huggingface.co/datasets/vector-index-bench/vibe).
"""
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK = HERE / "work"
VIBE = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/"
SAMPLE_QUESTIONS = 20_000   # as in every result in REPORT.md
TEST_QUESTIONS = 1_000      # set aside from the items when a file has no test questions
K = 10
BLOCK = 200_000             # rows read at a time when checking every item
METRIC_OF = {"normalized": "cos", "cosine": "cos", "angular": "cos", "euclidean": "l2", "ip": "ip"}


class Spec(NamedTuple):
    kind: str
    items: int
    dims: int
    gb: float
    unlike: bool       # questions look unlike the items (VIBE's "out-of-distribution")
    in_report: bool    # one of the 13 datasets in REPORT.md


# VIBE's files of plain float numbers (its README, November 2026). Its multi-vector, 8-bit and bit files are left out:
# vindex takes float numbers and "cos", "ip" or "l2".
CATALOG = {
    "yahoo-minilm-384-normalized": Spec("text: Yahoo Answers", 677_305, 384, 1.04, False, False),
    "ccnews-nomic-768-normalized": Spec("text: news", 495_328, 768, 1.53, False, False),
    "agnews-mxbai-1024-euclidean": Spec("text: news, l2", 769_382, 1024, 3.16, False, False),
    "simplewiki-openai-3072-normalized": Spec("text: Wikipedia, OpenAI model", 260_372, 3072, 3.21, False, False),
    "codesearchnet-jina-768-cosine": Spec("code", 1_374_067, 768, 4.23, False, False),
    "msmarco-qwen-1024-normalized": Spec("text: web passages", 8_840_823, 1024, 36.22, False, False),
    "dpr-jina-768-normalized": Spec("text: Wikipedia passages", 20_969_760, 768, 64.42, False, False),
    "hotpotqa-harrier-640-normalized": Spec("text: Wikipedia, searched with questions", 5_233_329, 640, 13.69, True, False),
    "celeba-resnet-2048-cosine": Spec("photos: faces", 201_599, 2048, 1.66, False, False),
    "imagenet-clip-512-normalized": Spec("photos", 1_281_167, 512, 2.63, False, False),
    "landmark-dino-768-cosine": Spec("photos: landmarks", 760_757, 768, 2.34, False, False),
    "inaturalist-resnet-2048-cosine": Spec("photos: nature", 499_000, 2048, 4.10, False, False),
    "coco-nomic-768-normalized": Spec("photos searched with text", 282_360, 768, 1.27, True, False),
    "yi-128-ip": Spec("chatbot attention", 187_843, 128, 0.27, True, True),
    "llama-128-ip": Spec("chatbot attention", 256_921, 128, 0.37, True, True),
    "glove-200-cosine": Spec("words", 1_192_514, 200, 0.96, False, True),
    "yandex-200-cosine": Spec("photos searched with text", 1_000_000, 200, 2.00, True, True),
    "landmark-nomic-768-normalized": Spec("photos: landmarks", 760_757, 768, 2.34, False, True),
    "arxiv-nomic-768-normalized": Spec("text: paper summaries", 1_344_643, 768, 4.14, False, True),
    "imagenet-align-640-normalized": Spec("photos searched with text", 1_281_167, 640, 4.22, True, True),
    "gooaq-distilroberta-768-normalized": Spec("text: Google answers", 1_475_024, 768, 4.54, False, True),
    "laion-clip-512-normalized": Spec("photos searched with text", 1_000_448, 512, 4.50, True, True),
}

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


def check_name(name):
    """A dataset name is a file name in playground/data/ (without .hdf5): no folders, nothing outside it."""
    if not isinstance(name, str) or not NAME.match(name) or ".." in name:
        raise ValueError(f"not a dataset name: {name!r}")
    return name


def data_file(name):
    return DATA / f"{check_name(name)}.hdf5"


def work_dir(name):
    return WORK / check_name(name)


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def datasets():
    """Every dataset in the catalog, and any other .hdf5 file in playground/data/, with what has been done to it."""
    names = list(CATALOG) + sorted(p.stem for p in DATA.glob("*.hdf5") if p.stem not in CATALOG and NAME.match(p.stem))
    out = []
    for name in names:
        spec = CATALOG.get(name)
        out.append({
            "name": name,
            "kind": spec.kind if spec else "your own file",
            "items": spec.items if spec else None,
            "dims": spec.dims if spec else None,
            "gb": spec.gb if spec else round(data_file(name).stat().st_size / 1e9, 2),
            "unlike": spec.unlike if spec else None,
            "in_report": spec.in_report if spec else False,
            "downloaded": data_file(name).exists(),
            "built": (work_dir(name) / "build.json").exists(),
            "searched": (work_dir(name) / "search.json").exists(),
        })
    return out


def info(name):
    """What the last build and the last search found."""
    return {"build": _json(work_dir(name) / "build.json"), "search": _json(work_dir(name) / "search.json")}


def download(name, report):
    """Download a VIBE file into playground/data/. `report` gets {phase, step, done, total} (bytes)."""
    if name not in CATALOG:
        raise ValueError(f"{name} is not in the catalog; put your own file in {DATA}/ instead")
    path = data_file(name)
    if path.exists():
        return {"file": str(path), "already_there": True}
    DATA.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    url = VIBE + name + ".hdf5"
    # macOS's curl, not urllib: python.org's Python trusts no HTTPS certificate until its own installer step is run
    head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, timeout=60).stdout
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", head)
    total = int(sizes[-1]) if sizes else 0
    p = subprocess.Popen(["curl", "-fsSL", "-o", str(part), url], stderr=subprocess.PIPE, text=True)
    while p.poll() is None:
        report({"phase": "downloading", "step": f"{name}.hdf5",
                "done": part.stat().st_size if part.exists() else 0, "total": total})
        time.sleep(0.25)
    if p.returncode != 0:
        raise RuntimeError(f"download failed: {p.stderr.read().strip()}")
    part.rename(path)
    return {"file": str(path), "bytes": path.stat().st_size}


def _open(name):
    import h5py
    path = data_file(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is not there; download it first")
    f = h5py.File(path, "r")
    if "train" not in f:
        f.close()
        raise ValueError(f"{path} has no 'train' (the items)")
    point = f.attrs.get("point_type", "float")
    if isinstance(point, bytes):
        point = point.decode()
    if point != "float":
        f.close()
        raise ValueError(f"{path} holds {point} numbers; vindex takes float numbers")
    return f


def _metric(f, given):
    if given:
        if given not in ("cos", "ip", "l2"):
            raise ValueError("metric must be cos, ip or l2")
        return given
    d = f.attrs.get("distance")
    if isinstance(d, bytes):
        d = d.decode()
    if d not in METRIC_OF:
        raise ValueError(f"the file's distance is {d!r}, not one vindex knows; give --metric cos, ip or l2"
                         if d else "the file says no distance; give --metric cos, ip or l2")
    return METRIC_OF[d]


def _read_rows(ds, rows):
    """Rows of an HDF5 dataset in the order given (HDF5 reads them in increasing order)."""
    order = np.argsort(rows)
    out = np.empty((len(rows), ds.shape[1]), dtype=np.float32)
    out[order] = ds[np.asarray(rows)[order]]
    return out


def build(name, report, threads=None, sample_questions=SAMPLE_QUESTIONS, metric=None):
    """Build an index on a dataset and save it in playground/work/<name>/. Every setting is vindex's default.

    Sample questions: the file's own (`learn`) when it has them, else items set aside (and so not stored). When the
    file has no test questions, 1,000 more items are set aside as test questions. `report` gets the build's
    progress ({phase, step, done, total}) about 4 times a second.
    """
    import vindex
    f = _open(name)
    report({"phase": "reading the file", "step": "", "done": 0, "total": 0})
    metric = _metric(f, metric)
    items = f["train"][:].astype(np.float32, copy=False)
    n, dim = items.shape
    rng = np.random.default_rng(0)
    if "learn" in f:
        pick = rng.choice(len(f["learn"]), min(sample_questions, len(f["learn"])), replace=False)
        train_q = _read_rows(f["learn"], pick)
        sample_rows = np.zeros(0, dtype=np.int64)
        sample_source = f"{len(train_q):,} of the file's own sample questions"
    else:
        sample_rows = rng.choice(n, min(sample_questions, n // 10), replace=False)
        train_q = items[sample_rows]
        sample_source = f"{len(sample_rows):,} items set aside"
    test_rows = np.zeros(0, dtype=np.int64)
    if "test" not in f:
        rest = np.setdiff1d(np.arange(n), sample_rows)
        test_rows = rng.choice(rest, min(TEST_QUESTIONS, len(rest) // 10), replace=False)
    f.close()
    keep = np.ones(n, dtype=bool)
    keep[sample_rows] = False
    keep[test_rows] = False
    ids = np.flatnonzero(keep)
    if len(ids) < n:
        items = items[keep]

    out = work_dir(name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    np.savez(out / "set_aside.npz", sample_rows=sample_rows, test_rows=test_rows)

    idx = vindex.Index(dim, metric=metric)
    failed = []

    def run():
        try:
            idx.build(items, ids, train_questions=train_q, threads=threads)
        except Exception as e:   # handed to the caller below
            failed.append(e)

    t0 = time.time()
    worker = threading.Thread(target=run)
    worker.start()
    while worker.is_alive():
        report(idx.build_progress())
        worker.join(0.25)
    if failed:
        raise failed[0]
    seconds = time.time() - t0
    report({"phase": "saving", "step": "", "done": 0, "total": 0})
    idx.save(str(out / "index"))
    t = idx.tuning()
    result = {
        "name": name,
        "metric": metric,
        "items": int(len(ids)),
        "dims": int(dim),
        "sample_questions": sample_source,
        "test_questions": "the file's own" if not len(test_rows) else f"{len(test_rows):,} items set aside",
        "build_seconds": round(seconds, 1),
        "threads": threads or "all cores",
        "memory": idx.memory(),
        "stats": idx.stats(),
        "chosen": {k: t[k] for k in ("chosen_r", "chosen_alpha", "chosen_code_bits", "chosen_rerank")} if t else None,
        "tuning_seconds": round(t["seconds"], 1) if t else None,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "build.json").write_text(json.dumps(result, indent=1, default=str))
    _forget(name)
    return result


def _brute_force(f, ds_rows, keep, Q, metric, report):
    """The true K closest kept items to each question, checking every item, a block of rows at a time."""
    q = Q / np.linalg.norm(Q, axis=1, keepdims=True) if metric == "cos" else Q
    best_s = np.full((len(Q), K), -np.inf, dtype=np.float32)
    best_i = np.full((len(Q), K), -1, dtype=np.int64)
    ds = f["train"]
    for a in range(0, ds_rows, BLOCK):
        x = ds[a:a + BLOCK].astype(np.float32, copy=False)
        if metric == "cos":
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        s = q @ x.T
        if metric == "l2":   # closest = smallest |x|^2 - 2 q.x
            s = 2 * s - (x * x).sum(axis=1)[None, :]
        s[:, ~keep[a:a + BLOCK]] = -np.inf
        cat_s = np.concatenate([best_s, s], axis=1)
        cat_i = np.concatenate([best_i, np.broadcast_to(np.arange(a, a + len(x)), s.shape)], axis=1)
        top = np.argpartition(-cat_s, K - 1, axis=1)[:, :K]
        best_s = np.take_along_axis(cat_s, top, axis=1)
        best_i = np.take_along_axis(cat_i, top, axis=1)
        report({"phase": "finding the true answers", "step": "checking every item",
                "done": min(a + BLOCK, ds_rows), "total": ds_rows})
    return [set(row.tolist()) for row in best_i]


def _questions_and_truth(name, report):
    """The test questions, and the true K closest stored items of each (worked out once, then kept)."""
    out = work_dir(name)
    aside = np.load(out / "set_aside.npz")
    meta = _json(out / "build.json")
    f = _open(name)
    try:
        n = f["train"].shape[0]
        Q = f["test"][:].astype(np.float32) if "test" in f else _read_rows(f["train"], aside["test_rows"])
        cached = out / "truth.npy"
        if cached.exists():
            return Q, [set(r.tolist()) for r in np.load(cached)]
        keep = np.ones(n, dtype=bool)
        keep[aside["sample_rows"]] = False
        keep[aside["test_rows"]] = False
        truth = None
        if "test" in f and "neighbors" in f and f["neighbors"].shape[1] >= K:
            # the file's own true answers, less the items set aside; a question left with fewer than K is checked again
            nb = f["neighbors"][:]
            truth = [[r for r in row if r >= 0 and keep[r]][:K] for row in nb]
            short = [i for i, t in enumerate(truth) if len(t) < K]
            if short:
                again = _brute_force(f, n, keep, Q[short], meta["metric"], report)
                for i, t in zip(short, again):
                    truth[i] = sorted(t)
            truth = [set(t) for t in truth]
        else:
            truth = _brute_force(f, n, keep, Q, meta["metric"], report)
        np.save(cached, np.array([sorted(t) for t in truth], dtype=np.int64))
        return Q, truth
    finally:
        f.close()


_loaded = {}
_loaded_lock = threading.Lock()


def _index(name):
    """The saved index, loaded once and kept (one at a time) for the next search."""
    import vindex
    with _loaded_lock:
        if name not in _loaded:
            _loaded.clear()
            path = work_dir(name) / "index"
            if not path.exists():
                raise FileNotFoundError(f"{name} has no index yet; build it first")
            _loaded[name] = vindex.Index.load(str(path))
        return _loaded[name]


def _forget(name):
    with _loaded_lock:
        _loaded.pop(name, None)


def _found_share(found, truth):
    return float(np.mean([len(set(row.tolist()) & t) / K for row, t in zip(found, truth)]))


def search(name, report, targets=(0.90, 0.95, 0.99), list_sizes=(), threads=1):
    """Search the test questions at each target (and each fixed list size), timed, and check every answer."""
    report({"phase": "loading the index", "step": "", "done": 0, "total": 0})
    idx = _index(name)
    Q, truth = _questions_and_truth(name, report)
    runs = [("target", t) for t in targets] + [("list_size", s) for s in list_sizes]
    rows = []
    for i, (how, value) in enumerate(runs):
        report({"phase": "searching", "step": f"{how} {value}", "done": i, "total": len(runs)})
        kw = {how: value}
        idx.search(Q[:200], k=K, threads=threads, **kw)   # warm up: the first touch of memory is slow
        t0 = time.perf_counter()
        found, _, checks = idx.search(Q, k=K, threads=threads, **kw)
        secs = time.perf_counter() - t0
        rows.append({how: value, "found": round(_found_share(found, truth), 4),
                     "questions_per_s": round(len(Q) / secs), "checks": round(float(checks.mean()), 1)})
    result = {"name": name, "questions": len(Q), "threads": threads, "runs": rows,
              "usage": idx.usage(), "searched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work_dir(name) / "search.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def question(name, i, target=0.95):
    """One test question in detail: what the index found, and the true answers."""
    idx = _index(name)
    Q, truth = _questions_and_truth(name, lambda p: None)
    if not 0 <= i < len(Q):
        raise ValueError(f"question must be 0 to {len(Q) - 1}")
    found, dists, checks = idx.search(Q[i:i + 1], k=K, target=target)
    t = truth[i]
    return {"question": i, "target": target, "checks": int(checks[0]),
            "found": [{"item": int(a), "distance": round(float(d), 4), "true": int(a) in t}
                      for a, d in zip(found[0], dists[0])],
            "missed": sorted(int(a) for a in t - set(found[0].tolist()))}
