"""Step 11: speed, measured exactly as the rivals were in the experiments' Step 3, on the test questions:
  - one core, batches of 1,000 questions, middle of 5 timed runs after a warm-up;
  - speed read off the speed-vs-accuracy curve at exactly 95% (learned-effort targets 88% to 99.5%);
  - all-core speed and typical / slow time per question, at the target picked on the tuning questions;
  - build time (all cores), memory per item (counted, and measured in a fresh process after loading);
  - how many cores' worth of CPU other programs were using, before and after each dataset's timing;
  - the SSD mode: saved, loaded with the full numbers left in the file, and the file dropped from the operating
    system's memory cache before every timed batch (and before every single question for the time per question),
    so the re-checks really read from the SSD;
then compared with HNSW, DiskANN, CoreNN, Glass, ScaNN and RoarGraph (results/step3/final/<tool>/<dataset>/best.json).

.venv/bin/python -I validate/step11.py [dataset ...]
"""
import ctypes
import ctypes.util
import json
import mmap
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import (DATASETS, REF, TARGET, TARGETS, Report, accuracy, build_index, correct_counts, cost_at_target,
                    load, metric_of, truth)

FINAL = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=2, build_space="auto", effort_apart=False, tune="off", renumber=False, question_bits=32, popular_links="off")
RIVALS = ["hnsw", "diskann", "corenn", "glass", "scann", "roargraph"]
NAMES = {"hnsw": "HNSW", "diskann": "DiskANN", "corenn": "CoreNN", "glass": "Glass", "scann": "ScaNN", "roargraph": "RoarGraph"}

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
MS_INVALIDATE = 0x2
PAGE = os.sysconf("SC_PAGE_SIZE")


def _mapped(path, fn):
    size = path.stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        p = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        try:
            return fn(p, size)
        finally:
            _libc.munmap(p, size)
    finally:
        os.close(fd)


def drop_from_cache(path):
    """Drop a file's pages from macOS's memory cache, so the next reads come from the SSD."""
    if _mapped(path, lambda p, size: _libc.msync(p, size, MS_INVALIDATE)) != 0:
        raise OSError(ctypes.get_errno(), "msync failed")


def cached_share(path):
    """Share of a file's pages in the memory cache."""
    def share(p, size):
        v = ctypes.create_string_buffer((size + PAGE - 1) // PAGE)
        _libc.mincore(p, size, v)
        return sum(b & 1 for b in v.raw) / len(v.raw)
    return _mapped(path, share)


def others_cpu():
    """Cores' worth of CPU other programs are using right now (macOS's recent average per process, this one left out)."""
    out = subprocess.run(["ps", "-Ao", "pid=,pcpu="], capture_output=True, text=True, check=True).stdout.split()
    me = os.getpid()
    return sum(float(c) for p, c in zip(out[::2], out[1::2]) if int(p) != me) / 100


def timed(fn, reps=5, before=None):
    if before:
        before()
    fn()
    ts = []
    for _ in range(reps):
        if before:
            before()
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def rivals(name):
    out = {}
    for t in RIVALS:
        f = REF / "step3" / "final" / t / name / "best.json"
        if f.exists():
            r = json.loads(f.read_text())
            acc = r.get("memory_accounting", {})
            out[t] = {"qps95": 1 / r["sec_per_query_at95"] if r.get("sec_per_query_at95") else None,
                      "status": r.get("status_at95"), "best": r["best_accuracy"],
                      "bytes_per_item": (acc.get("total_bytes") or r.get("rss_index_bytes") or 0) / r["n_items"],
                      "qps_all": 1 / r["final"]["sec_per_query_allcores"], "build_s": r["build_seconds"]}
    return out


def measure(name, idx, size="slice"):
    """size="full": the index holds every item (part 4), graded with the full-size true answers."""
    metric = metric_of(name)
    X, Q = load(name, "slice" if size == "slice" else "items_full"), load(name, "q_test")
    Qt = load(name, "q_tune")
    td, tdt = truth(name, f"{size}_test")["dists"], truth(name, f"{size}_tune")["dists"]
    curve = []
    for T in TARGETS:
        ids, _, checks = idx.search(Q, k=10, target=T, threads=1)
        sec = timed(lambda: idx.search(Q, k=10, target=T, threads=1)) / len(Q)
        curve.append({"target": T, "accuracy": accuracy(correct_counts(ids, Q, X, td, metric)),
                      "sec": sec, "checks": float(checks.mean())})
    sec95, status = cost_at_target([c["accuracy"] for c in curve], [c["sec"] for c in curve])
    checks95, _ = cost_at_target([c["accuracy"] for c in curve], [c["checks"] for c in curve])
    # the target picked on the tuning questions, as the rivals' effort was
    T = next((c for c in TARGETS if accuracy(correct_counts(idx.search(Qt, k=10, target=c, threads=16)[0], Qt, X, tdt, metric)) >= TARGET), TARGETS[-1])
    sec_all = timed(lambda: idx.search(Q, k=10, target=T, threads=16)) / len(Q)
    lat = []
    for q in Q[:50]:
        idx.search(q[None], k=10, target=T, threads=1)
    for q in Q:
        t = time.perf_counter()
        idx.search(q[None], k=10, target=T, threads=1)
        lat.append(time.perf_counter() - t)
    ids, _, _ = idx.search(Q, k=10, target=T, threads=1)
    return {"curve": curve, "qps95": 1 / sec95 if sec95 else None, "status95": status, "checks95": checks95,
            "target": T, "accuracy_at_target": accuracy(correct_counts(ids, Q, X, td, metric)),
            "qps_all": 1 / sec_all, "median_ms": float(np.median(lat) * 1e3), "p99_ms": float(np.percentile(lat, 99) * 1e3)}


MEASURE = """
import ctypes, os, sys, numpy as np, vindex
libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
def footprint():
    # macOS's own count of a process's memory (what Activity Monitor shows): it leaves out file pages the operating
    # system caches and can drop, so the SSD mode's full numbers are not counted just because they were read
    buf = ctypes.create_string_buffer(512)
    if libproc.proc_pid_rusage(os.getpid(), 2, buf) != 0:
        raise OSError("proc_pid_rusage failed")
    return int.from_bytes(buf.raw[72:80], "little")   # rusage_info_v2.ri_phys_footprint
q = np.load(sys.argv[3])
base = footprint()
idx = vindex.Index.load(sys.argv[1], numbers=sys.argv[2])
idx.search(q, k=10, target=float(sys.argv[4]), threads=1)
print(footprint() - base)
"""


def measured_bytes(folder, mode, Q, T, work):
    """Memory taken by loading the saved index and running the test questions once, in a fresh process (macOS's
    footprint count)."""
    qf = work / "q.npy"
    np.save(qf, Q)
    out = subprocess.run([sys.executable, "-I", "-c", MEASURE, str(folder), mode, str(qf), str(T)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.split()[-1])


def measure_ssd(name, idx, T, work):
    Q = load(name, "q_test")
    folder = work / name
    idx.save(str(folder))
    nums = folder / "numbers.f32"
    b = vindex.Index.load(str(folder), numbers="ssd")
    same = all(np.array_equal(x, y) for x, y in zip(b.search(Q, k=10, target=T, threads=16), idx.search(Q, k=10, target=T, threads=16)))
    drop = lambda: drop_from_cache(nums)
    drop()
    share_after_drop = cached_share(nums)
    sec = timed(lambda: b.search(Q, k=10, target=T, threads=1), before=drop) / len(Q)
    sec_all = timed(lambda: b.search(Q, k=10, target=T, threads=16), before=drop) / len(Q)
    lat = []
    for q in Q[:200]:
        drop()
        t = time.perf_counter()
        b.search(q[None], k=10, target=T, threads=1)
        lat.append(time.perf_counter() - t)
    n = idx.memory()["items"]
    return {"same_answers": same, "cached_share_after_drop": share_after_drop, "qps": 1 / sec, "qps_all": 1 / sec_all,
            "median_ms_cold": float(np.median(lat) * 1e3), "p99_ms_cold": float(np.percentile(lat, 99) * 1e3),
            "bytes_per_item_counted": b.memory()["total_bytes"] / n,
            "bytes_per_item_measured": measured_bytes(folder, "ssd", Q, T, work) / n,
            "ram_bytes_per_item_measured": measured_bytes(folder, "ram", Q, T, work) / n,
            "folder_mb": sum(f.stat().st_size for f in folder.iterdir()) / 1e6}


if __name__ == "__main__":
    rep = Report("step11")
    work = Path(tempfile.mkdtemp(prefix="vindex-step11-"))
    try:
        for name in (sys.argv[1:] or DATASETS):
            idx, build_s = build_index(name, FINAL, train=True)
            others_before = others_cpu()   # recorded with the numbers: other programs slow the all-core runs most
            m = measure(name, idx)
            m["other_programs_cores"] = [others_before, others_cpu()]
            mem = idx.memory()
            per = mem["total_bytes"] / mem["items"]
            ssd = measure_ssd(name, idx, m["target"], work)
            rv = rivals(name)
            first3 = [r["qps95"] for t, r in rv.items() if t in ("hnsw", "diskann", "corenn") and r["qps95"]]
            fastest = max((r["qps95"] for r in rv.values() if r["qps95"]), default=None)
            best_t = max((t for t in rv if rv[t]["qps95"]), key=lambda t: rv[t]["qps95"], default=None)
            detail = (f"ours {m['qps95']:,.0f}/s at 95% (1 core; {m['checks95']:,.0f} checks); all cores {m['qps_all']:,.0f}/s; "
                      f"typical {m['median_ms']:.3f} ms, slow {m['p99_ms']:.3f} ms; build {build_s:.1f} s; {per:,.0f} bytes/item; "
                      f"other programs using {m['other_programs_cores'][0]:.1f} -> {m['other_programs_cores'][1]:.1f} cores"
                      + "".join(f"\n            {NAMES[t]:9s} " + (f"{r['qps95']:>9,.0f}/s" if r["qps95"] else f"never 95% (best {r['best']:.0%})")
                                + f"  {r['bytes_per_item']:>6,.0f} bytes/item  build {r['build_s']:.0f} s" for t, r in rv.items()))
            beats_first3 = bool(first3) and m["qps95"] is not None and m["qps95"] > max(first3)
            rep.check(f"{name}: faster at 95% than HNSW, DiskANN and CoreNN" if first3 else f"{name}: reaches 95% (HNSW, DiskANN and CoreNN never do)",
                      beats_first3 if first3 else m["qps95"] is not None, detail,
                      {"ours": m, "build_seconds": build_s, "bytes_per_item": per, "ssd": ssd, "rivals": rv})
            rep.check(f"{name}: the SSD mode gives the same answers, reading the full numbers from the SSD",
                      ssd["same_answers"] and ssd["cached_share_after_drop"] < 0.01,
                      f"{ssd['qps']:,.0f}/s one core, {ssd['qps_all']:,.0f}/s all cores (file dropped from the cache before "
                      f"each batch); one question from a cold cache: typical {ssd['median_ms_cold']:.2f} ms, slow "
                      f"{ssd['p99_ms_cold']:.2f} ms; memory {ssd['bytes_per_item_counted']:,.0f} bytes/item counted, "
                      f"{ssd['bytes_per_item_measured']:,.0f} measured (with the full numbers in memory: "
                      f"{ssd['ram_bytes_per_item_measured']:,.0f} measured)")
            if fastest and m["qps95"]:
                rep.note(f"{name}: against the fastest rival ({NAMES[best_t]}): {m['qps95'] / fastest:.2f}x its speed; "
                         f"memory {per / rv[best_t]['bytes_per_item']:.2f}x its bytes per item")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()
