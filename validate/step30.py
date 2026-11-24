"""Step 30: the rivals at full size (part 4).
  .venv/bin/python -I validate/step30.py rivals   # first: the experiments' scripts/step6_baselines.py, settings search
                                                  # then test questions; ScaNN first, with Docker Desktop started for
                                                  # it and quit after, so it is not running while the others are timed
  .venv/bin/python -I validate/step30.py          # then: ours (step 29) against each rival, and step 28's expectations
Every number as in step 27: test questions, one core, read off at exactly 95%.
Pass: faster than HNSW at 95% wherever it reaches 95%. Every other comparison is reported as it comes, and each
expectation written in step 28 (STEPS.md) is marked right or wrong.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, DATASETS, EXP, HELD_BACK, HELD_BACK_3, OUT, Report, frozen_changes
from step11 import NAMES

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]
FULL = EXP / "results" / "step6"
ATTENTION = ["llama", "yi", "llama2"]
GLASS_AHEAD = ["gooaq", "landmark", "laion", "gist", "nytimes"]
GLASS_LEVEL = ["yandex", "arxiv", "imagenet"]


def docker(up):
    if up:
        subprocess.run(["open", "-a", "Docker"], check=True)
        for _ in range(120):
            if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
                return
            time.sleep(5)
        raise RuntimeError("Docker Desktop did not start")
    subprocess.run(["osascript", "-e", 'quit app "Docker Desktop"'])
    for _ in range(60):
        if subprocess.run(["pgrep", "-f", "Docker Desktop.app"], capture_output=True).returncode != 0:
            return
        time.sleep(5)


def run_rivals():
    env = dict(os.environ, BENCH_UNLOCK_HELD_BACK="1", BENCH_SCRATCH=os.environ.get("BENCH_SCRATCH", tempfile.gettempdir()))
    py = str(EXP / ".venv" / "bin" / "python")
    run = lambda *a: subprocess.run([py, "-I", "scripts/step6_baselines.py", *a], cwd=EXP, env=env, check=True)
    try:   # a ScaNN failure (say, Docker not starting) must not stop the other rivals' runs; it is rerun after
        docker(True)
        info = subprocess.run(["docker", "info", "--format", "{{.NCPU}} cores, {{.MemTotal}} bytes"], capture_output=True, text=True)
        image = subprocess.run(["docker", "image", "inspect", "vibench-scann:1.4.2", "--format", "{{.Id}}"],
                               capture_output=True, text=True)
        (OUT / "step30-docker.txt").write_text(f"Docker VM: {info.stdout.strip()}\nScaNN image: {image.stdout.strip()}\n")
        run("grid", "scann")
        run("final", "scann")
    except Exception as e:
        print(f"ScaNN runs failed: {e}", flush=True)
    finally:
        docker(False)
    others = ["glass", "roargraph", "corenn", "hnsw"]
    run("grid", *others)
    run("final", *others)


def rival(tool, name, where=FULL):
    f = where / "final" / tool / name / "best.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    acc = r.get("memory_accounting", {})
    return {"qps95": 1 / r["sec_per_query_at95"] if r.get("sec_per_query_at95") else None, "best": r["best_accuracy"],
            "status": r.get("status_at95"), "bytes_per_item": (acc.get("total_bytes") or r.get("rss_index_bytes") or 0) / r["n_items"],
            "qps_all": 1 / r["final"]["sec_per_query_allcores"], "build_s": r["build_seconds"], "params": r["params"],
            "n_items": r["n_items"]}


def ours_full():
    r = json.loads((OUT / "step29.json").read_text())
    return {c["check"].split(":")[0]: c["data"] for c in r["checks"] if c.get("data") and "at full size" in c["check"]}


def ours_sample():
    r = json.loads((OUT / "step27.json").read_text())
    return {c["check"].split(" ")[0].rstrip(":"): c["data"] for c in r["checks"] if c.get("data")}


def items(name):
    return json.loads((DATA / name / "card.json").read_text())["items_full"]


def expectations(ours, sample, rv):
    """Step 28's expectations, each with what happened. rv[name][tool] holds the full-size rivals."""
    q = lambda n: ours[n]["ours"]["qps95"]
    qs = lambda n: sample[n]["ours"]["qps95"]
    ratio = lambda n, t: q(n) / rv[n][t]["qps95"] if rv[n].get(t) and rv[n][t]["qps95"] else None
    s_ratio = lambda n, t: qs(n) / sample[n]["rivals"][t]["qps95"] if sample[n]["rivals"].get(t, {}).get("qps95") else None
    big = [n for n in ALL if items(n) >= 700_000]
    small = [n for n in ALL if n not in big]
    fall = {n: qs(n) / q(n) for n in ALL}
    out = []

    def add(text, right, what):
        out.append({"expected": text, "right": bool(right), "what happened": what})

    add("Our speed falls 1.5-3x on the eight big datasets, less on the four smaller ones",
        all(1.5 <= fall[n] <= 3 for n in big) and all(fall[n] < min(fall[b] for b in big) for n in small),
        "; ".join(f"{n} {fall[n]:.2f}x" for n in ALL))
    g = {n: (ratio(n, "glass"), s_ratio(n, "glass")) for n in GLASS_AHEAD + GLASS_LEVEL}
    add("Our ratio to Glass stays within about 15% of the samples'",
        all(a and b and 0.85 <= a / b <= 1.15 for a, b in g.values()),
        "; ".join(f"{n} {a:.2f}x (sample {b:.2f}x)" for n, (a, b) in g.items() if a and b))
    add("Still ahead of Glass on GooAQ, Landmark, LAION, GIST1M and NYTimes",
        all(g[n][0] and g[n][0] > 1 for n in GLASS_AHEAD), "; ".join(f"{n} {g[n][0]:.2f}x" for n in GLASS_AHEAD if g[n][0]))
    add("Level with Glass on Yandex, arXiv and ImageNet (within 5%)",
        all(g[n][0] and 0.95 <= g[n][0] <= 1.05 for n in GLASS_LEVEL), "; ".join(f"{n} {g[n][0]:.2f}x" for n in GLASS_LEVEL if g[n][0]))
    add("Still ahead of RoarGraph on the three attention datasets, by 1.3x or more",
        all(ratio(n, "roargraph") and ratio(n, "roargraph") >= 1.3 for n in ATTENTION),
        "; ".join(f"{n} {ratio(n, 'roargraph') or 0:.2f}x" for n in ATTENTION))
    sg = ratio("glove", "scann")
    add("ScaNN still ahead on GloVe, by less than in the samples (1.25x)", sg and 0.8 < sg < 1.0,
        f"ours {sg:.2f}x ScaNN's speed (ScaNN {1 / sg:.2f}x ours)" if sg else "ScaNN did not reach 95%")
    ny = rv["nytimes"]
    add("On NYTimes, ScaNN still slower than Glass", ny.get("scann") and ny.get("glass") and ny["scann"]["qps95"]
        and ny["glass"]["qps95"] and ny["scann"]["qps95"] < ny["glass"]["qps95"],
        f"ScaNN {ny['scann']['qps95'] or 0:,.0f}/s, Glass {ny['glass']['qps95'] or 0:,.0f}/s" if ny.get("scann") and ny.get("glass") else "missing")
    h = {n: ratio(n, "hnsw") for n in ALL}
    add("At least 5x faster than HNSW wherever it reaches 95%; it never does on the three attention datasets",
        all(v >= 5 for v in h.values() if v) and all(h[n] is None for n in ATTENTION),
        "; ".join(f"{n} {v:.1f}x" if v else f"{n} never" for n, v in h.items()))
    mem = {n: (ours[n]["bytes_per_item"], sample[n]["bytes_per_item"], rv[n]["hnsw"]["bytes_per_item"] if rv[n].get("hnsw") else None)
           for n in ALL}
    add("Memory per item within 10% of the samples'; 0.15-0.6x HNSW's with the full numbers in a file",
        all(abs(a / b - 1) <= 0.10 for a, b, _ in mem.values()) and all(c and 0.15 <= a / c <= 0.6 for a, _, c in mem.values()),
        "; ".join(f"{n} {a:,.0f} (sample {b:,.0f}, {a / c:.2f}x HNSW)" if c else f"{n} {a:,.0f}" for n, (a, b, c) in mem.items()))
    on = [n for n in ALL if ours[n]["stats"]["popular_items"]]
    add("The extra links switch on only on the three attention datasets", sorted(on) == sorted(ATTENTION),
        "on for: " + (", ".join(on) or "none"))
    b = {n: (ours[n]["build_seconds"], sample[n]["build_seconds"]) for n in big}
    add("Builds take 10-25x the samples' time on the big datasets, 5-25 minutes",
        all(10 <= x / y <= 25 and 300 <= x <= 1500 for x, y in b.values()),
        "; ".join(f"{n} {x / 60:.1f} min ({x / y:.0f}x)" for n, (x, y) in b.items()))
    return out


def main():
    rep = Report("step30")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    ours, sample = ours_full(), ours_sample()
    rv = {n: {t: r for t in ("hnsw", "glass", "scann", "roargraph", "corenn") if (r := rival(t, n))} for n in ALL}
    for n in ALL:
        m = ours[n]["ours"]
        reach = {t: r for t, r in rv[n].items() if r["qps95"]}
        best_t = max(reach, key=lambda t: reach[t]["qps95"], default=None)
        detail = (f"ours {m['qps95']:,.0f}/s at 95% (1 core), {ours[n]['bytes_per_item']:,.0f} bytes/item, build "
                  f"{ours[n]['build_seconds']:.0f} s" + "".join(
                      f"\n            {NAMES[t]:9s} " + (f"{r['qps95']:>9,.0f}/s ({m['qps95'] / r['qps95']:.2f}x)" if r["qps95"]
                                                       else f"never 95% (best {r['best']:.1%})")
                      + f"  {r['bytes_per_item']:>6,.0f} bytes/item  build {r['build_s']:.0f} s" for t, r in rv[n].items()))
        data = {"ours": m, "rivals": rv[n], "fastest_rival": best_t}
        hn = rv[n].get("hnsw")
        if hn and hn["qps95"]:
            rep.check(f"{n}: faster than HNSW at 95%", m["qps95"] > hn["qps95"], detail, data)
        else:
            rep.check(f"{n}: reaches 95% (HNSW never does)", m["qps95"] is not None, detail, data)
        if best_t:
            rep.note(f"{n}: {m['qps95'] / reach[best_t]['qps95']:.2f}x the fastest rival here ({NAMES[best_t]})")
    exp = expectations(ours, sample, rv)
    for e in exp:
        rep.note(f"{'RIGHT' if e['right'] else 'WRONG'}  {e['expected']}: {e['what happened']}")
    (OUT / "step30-expectations.json").write_text(json.dumps(exp, indent=1))
    rep.finish()


if __name__ == "__main__":
    run_rivals() if sys.argv[1:2] == ["rivals"] else main()
