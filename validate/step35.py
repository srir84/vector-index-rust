"""Step 35: REPORT.md's part 4 and the one-page write-up (part 4).
  .venv/bin/python -I validate/step35.py
Checks:
  - the code is as frozen in step 28;
  - REPORT.md part 4's three tables of numbers (speed, all cores, memory): every cell, row by row, recomputed here
    from the results files;
  - every other number in REPORT.md part 4 and in WRITE-UP.md is one the results files give (recomputed here, or
    one of the results page's, which validate/step33.py recomputes), a constant of the method, or a step's number;
  - the CoreNN sentence is word for word step 31's;
  - nothing is left unfinished: no placeholder in them or in demo/README.md;
  - no raw diagram is left in either (each is baked to a picture), and every picture they show exists.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import EXP, OUT, ROOT, Report, frozen_changes
from step30 import ALL, ours_full, rival
from step33 import PLAN, TOOL, expected, f_int, f_items, f_pct, f_pts, f_time

NAMES = {"GooAQ": "gooaq", "GloVe": "glove", "Yandex": "yandex", "Llama": "llama", "Landmark": "landmark", "arXiv": "arxiv",
         "LAION": "laion", "Yi": "yi", "GIST1M": "gist", "NYTimes": "nytimes", "ImageNet": "imagenet",
         "Llama, second head": "llama2"}
# a number, with the unit words the documents put after it
TOKEN = re.compile(r"(?<![A-Za-z0-9.])[−+-]?\d[\d,]*(?:\.\d+)?(?![\d])")
# the method's own constants and the plan's, and references to steps and parts (not results)
CONSTANTS = ["95%", "true top 10", "10 closest", "batches of 1,000", "middle of 5", "1,000 tuning", "20,000 training",
             "100,000-item", "100,000 items", "within 5%", "1.3×", "1-point", "1 point", "limit of 1", "10 checkpoints",
             "90% of its items", "10% more", "deleting 10%", "updating 10%", "10% deleted", "10% updated", "16 cores",
             "12 of them", "128 GB", "macOS 27.0.1", "M3 Max", "AVX-512", "Llama 3", "2.6×", "1.5–3×", "10–25×",
             "about 15%", "at least 5×", "at least 10×", "1.3× or more", "1.25×", "up to 9 settings", "CC BY 4.0",
             "16 cores", "Part 4", "part 4", "Four parts", "four parts", "1 in 100", "all four", "the 10 closest",
             "finds 95%", "within 1 point", "first 8 KB"]
STEPREF = re.compile(r"\b(?:[Ss]teps?|section|[Pp]arts?)\s+\d+[a-z]?(?:\s*(?:and|to|–|-)\s*\d+[a-z]?)?")


def core(tok):
    """The digits of a number as written: sign, thousands commas and unit words aside."""
    return tok.replace("−", "").replace("+", "").replace("-", "").replace(",", "").rstrip(".")


def nums(text):
    return {core(t) for t in TOKEN.findall(text)}


def tables(text):
    out, cur = [], []
    for ln in text.splitlines():
        if ln.startswith("|"):
            cur.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def main():
    rep = Report("step35")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    report = (ROOT / "REPORT.md").read_text()
    p4 = report[report.index("# Part 4"):report.index("# Part 3")]
    write_up = (ROOT / "WRITE-UP.md").read_text()
    ours = ours_full()
    s27 = {c["check"].split(" ")[0].rstrip(":"): c["data"] for c in json.loads((OUT / "step27.json").read_text())["checks"]
           if c.get("data")}
    s30 = {c["check"].split(":")[0]: c["data"] for c in json.loads((OUT / "step30.json").read_text())["checks"]
           if c.get("data")}
    cards = {n: json.loads((EXP / "data" / "prepared" / n / "card.json").read_text()) for n in ALL}

    # the three tables, cell by cell
    want = {}
    for n in ALL:
        o, f = ours[n], s30[n]["fastest_rival"]
        rv, h = rival(f, n), rival("hnsw", n)
        mo = o["memory"]
        want[("Items", n)] = [f"{cards[n]['items_full']:,}", f_int(o["ours"]["qps95"]), f_int(s27[n]["ours"]["qps95"]),
                              f"{TOOL[f]} {f_int(rv['qps95'])}", f"{o['ours']['qps95'] / rv['qps95']:.2f}",
                              f"{s27[n]['ours']['qps95'] / s27[n]['rivals'][f]['qps95']:.2f}", f_int(o["ours"]["checks95"])]
        want[("All 16 cores (searches/s)", n)] = [f_int(o["ours"]["qps_all"]),
                                                  f"{o['ours']['median_ms']:.3f} / {o['ours']['p99_ms']:.3f}",
                                                  f_time(o["build_seconds"])]
        want[("Ours, full numbers on SSD", n)] = [f_int(mo["total_bytes"] / mo["items"]),
                                                  f_int((mo["total_bytes"] + mo["full_numbers_on_ssd_bytes"]) / mo["items"]),
                                                  f"{TOOL[f]} {f_int(rv['bytes_per_item'])}", f_int(h["bytes_per_item"])]
    bad, rows = [], 0
    for t in tables(p4):
        key = t[0][1] if len(t[0]) > 1 else ""
        for row in t[2:]:
            if (key, NAMES.get(row[0])) in want:
                rows += 1
                if row[1:] != want[(key, NAMES[row[0]])]:
                    bad.append(f"{row[0]} ({key}): {row[1:]} against {want[(key, NAMES[row[0]])]}")
    rep.check("REPORT.md part 4's tables: every cell matches the results files", rows == 3 * len(ALL) and not bad,
              f"{rows} rows checked" + "".join(f"\n      {b}" for b in bad[:8]))

    # every other number: one the results files give
    e, _, _ = expected()
    allowed = {core(v) for v in e.values() if isinstance(v, str)} | {core(t) for v in e.values() if isinstance(v, str)
                                                                    for t in TOKEN.findall(v)}
    exps = json.loads((OUT / "step30-expectations.json").read_text())
    s31 = next(c["data"] for c in json.loads((OUT / "step31.json").read_text())["checks"] if c.get("data"))
    s32 = {c["data"]["dataset"]: c["data"] for c in json.loads((OUT / "step32.json").read_text())["checks"]
           if isinstance(c.get("data"), dict) and "dataset" in c["data"]}
    s30b = [c["data"] for c in json.loads((OUT / "step30b.json").read_text())["checks"] if c.get("data")]
    s30c = {}
    for c in json.loads((OUT / "step30c.json").read_text())["checks"]:
        if c.get("data"):
            s30c.setdefault(c["data"]["dataset"], {}).update(c["data"])
    machine = json.loads((OUT / "machine.json").read_text())
    runs = [json.loads(f.read_text()) for f in (EXP / "results" / "step6").glob("*/*/*/*.json")]
    ratio = {n: ours[n]["ours"]["qps95"] / rival(s30[n]["fastest_rival"], n)["qps95"] for n in ALL}
    wins = [r for r in ratio.values() if r >= 1.05]
    hn = {n: ours[n]["ours"]["qps95"] / rival("hnsw", n)["qps95"] for n in ALL if rival("hnsw", n)["qps95"]}
    fall = {n: s27[n]["ours"]["qps95"] / ours[n]["ours"]["qps95"] for n in ALL}
    big = [n for n in ALL if cards[n]["items_full"] >= 700_000]
    per = lambda sec, ch: sec / ch
    g_up, g_ch, o_ch = [], {}, {}
    for n in PLAN["glass"]:
        s, f = (json.loads((EXP / "results" / w / "final" / "glass" / n / "best.json").read_text()) for w in ("step3", "step6"))
        g_up.append(per(f["sec_per_query_at95"], f["cmps_at95"]) / per(s["sec_per_query_at95"], s["cmps_at95"]))
        g_ch[n] = f["cmps_at95"] / s["cmps_at95"]
        o_ch[n] = ours[n]["ours"]["checks95"] / s27[n]["ours"]["checks95"]
    grew = [n for n in PLAN["glass"] if g_ch[n] > 1.1 * o_ch[n]]   # where Glass's checks grew clearly more than ours
    level = [n for n in PLAN["glass"] if abs(s27[n]["ours"]["qps95"] / s27[n]["rivals"]["glass"]["qps95"] - 1) < 0.05]
    mem_vs = [(ours[n]["memory"]["total_bytes"] + ours[n]["memory"]["full_numbers_on_ssd_bytes"]) / ours[n]["memory"]["items"]
              / rival(s30[n]["fastest_rival"], n)["bytes_per_item"] for n in ALL]
    ssd_vs_hnsw = [ours[n]["bytes_per_item"] / rival("hnsw", n)["bytes_per_item"] for n in ALL]
    lm = s32["landmark"]
    derived = {
        "wins": [f"{min(wins):.2f}", f"{max(wins):.2f}", str(len(wins)), str(len(ALL))],
        "GloVe": [f"{ratio['glove']:.2f}", f"{1 / ratio['glove']:.2f}"],
        "HNSW": [f"{min(hn.values()):.2f}", f"{max(hn.values()):.2f}", str(len(hn))]
                + [f"{rival('hnsw', n)['best']:.1%}" for n in ALL if n not in hn],
        "level in the samples": [f"{s27[n]['ours']['qps95'] / s27[n]['rivals']['glass']['qps95']:.2f}" for n in level]
                                + [f"{ratio[n]:.2f}" for n in level],
        "speed fell": [f"{min(fall[n] for n in big):.2f}", f"{max(fall.values()):.2f}", f"{min(fall.values()):.2f}",
                       f"{sorted(fall.values())[1]:.2f}"]
                      + [f_int(x) for n in ("glove", "gist") for x in (ours[n]["ours"]["checks95"], s27[n]["ours"]["checks95"])],
        "others": [f"{min(c for n in ALL for c in ours[n]['ours']['other_programs_cores']):.1f}",
                   f"{max(c for n in ALL for c in ours[n]['ours']['other_programs_cores']):.1f}"],
        "accuracy at the target": [f"{min(ours[n]['ours']['accuracy_at_target'] for n in ALL):.2%}",
                                   f"{max(ours[n]['ours']['accuracy_at_target'] for n in ALL):.2%}"],
        "re-check": [f"{r['ours']['qps95'] / r['glass']['qps95']:.2f}" for r in s30b]
                    + [f"{max(abs(r['glass']['qps95'] / rival('glass', r['dataset'])['qps95'] - 1) for r in s30b):.1%}",
                       f"{max(abs(r['ours']['qps95'] / ours[r['dataset']]['ours']['qps95'] - 1) for r in s30b):.1%}"]
                    + [f_int(ours[r["dataset"]]["ours"]["checks95"]) for r in s30b] + [f_int(r["ours"]["checks95"]) for r in s30b]
                    + [f"{max(abs(f['a']['qps95'] / ours[n]['ours']['qps95'] - 1) for n, f in s30c.items()):.1%}"],
        "why": [f"{min(g_up):.2f}", f"{max(g_up):.2f}", str(len(g_up)),
                f"{min(g_ch[n] for n in grew):.2f}", f"{max(g_ch[n] for n in grew):.2f}",
                f"{min(o_ch[n] for n in grew):.2f}", f"{max(o_ch[n] for n in grew):.2f}"]
               + [f"{(f['a']['qps95'] - f['b']['qps95']) / f['a']['qps95']:.1%}" for f in s30c.values()],
        "ScaNN on NYTimes": [f_int(rival(t, "nytimes")["qps95"]) for t in ("scann", "glass")] + [f_int(ours["nytimes"]["ours"]["qps95"])],
        "CoreNN": [f"{s31['speed_ratio']:.1f}", f"{s31['memory_less_ssd']:.0%}"],
        "memory": [f"{min(mem_vs):.2f}", f"{max(mem_vs):.2f}", f"{min(ssd_vs_hnsw):.2f}", f"{max(ssd_vs_hnsw):.2f}",
                   f_int(ours["imagenet"]["bytes_per_item"]), f_int(s27["imagenet"]["bytes_per_item"])],
        "SSD": [f_int(x) for k in ("qps", "qps_all") for x in (min(ours[n]["ssd"][k] for n in ALL), max(ours[n]["ssd"][k] for n in ALL))],
        "changes": [f_pts(d["drift"]) for d in s32.values()]
                   + [f"{min(d['memory_after_vs_fresh'] for d in s32.values()):.3f}",
                      f"{max(d['memory_after_vs_fresh'] for d in s32.values()):.3f}",
                      f_pct(lm["checkpoints"][-1]["accuracy"]), f_pct(lm["fresh_accuracy"]), f_int(lm["changes"])],
        "expectations": [str(len(exps) + 2), str(sum(x["right"] for x in exps) + (s31["speed_ratio"] >= 10)),
                         str(sum(not x["right"] for x in exps) + 1)]
                        + [t for x in exps for t in TOKEN.findall(x["expected"] + " " + x["what happened"])],
        "runs": [str(sum(1 for r in runs if r.get("qset") == "tune")), str(sum(1 for r in runs if r.get("qset") == "test")),
                 f"{sum(r.get('wall_seconds', 0) for r in runs) / 3600:.1f}",
                 f"{json.loads((OUT / 'step29.json').read_text())['seconds'] / 60:.0f}",
                 f"{json.loads((OUT / 'step32.json').read_text())['seconds'] / 3600:.0f}"],
        "items": [f_items(min(c["items_full"] for c in cards.values())), f_items(max(c["items_full"] for c in cards.values()))],
        "the results page": re.findall(r"\((\d[\d,]*) different\)", json.loads((OUT / "step33.json").read_text())["checks"][2]["detail"]),
        "the demo": [str(len(next(c["data"]["vindex_ms"] for c in json.loads((OUT / "step34.json").read_text())["checks"]
                                  if c.get("data") and "vindex_ms" in c["data"])))],
        "machine": [str(machine["cores"]), str(machine["performance_cores"]), str(machine["memory_gb"]), machine["macos"]],
        "picks changed": [str(sum(ours[n]["tuning"]["chosen_r"] != s27[n]["tuning"]["chosen_r"]
                                  or ours[n]["tuning"]["chosen_code_bits"] != s27[n]["tuning"]["chosen_code_bits"]
                                  or ours[n]["tuning"]["chosen_rerank"] != s27[n]["tuning"]["chosen_rerank"]
                                  or abs(ours[n]["tuning"]["chosen_alpha"] - s27[n]["tuning"]["chosen_alpha"]) > 1e-6
                                  for n in ALL))],
    }
    for vs in derived.values():
        allowed |= {core(t) for v in vs for t in TOKEN.findall(v)}
    unexplained = {}
    for name, text in (("REPORT.md part 4", p4), ("WRITE-UP.md", write_up)):
        body = re.sub(r"<details>.*?</details>", " ", text, flags=re.S)
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)|`[^`]*`|https?://\S+", " ", body)
        body = "\n".join(ln for ln in body.splitlines() if not (ln.startswith("|") and any(ln.startswith(f"| {k} |") for k in NAMES)))
        body = body.replace((OUT / "step31-sentence.txt").read_text().strip(), " ")
        for c in sorted(CONSTANTS, key=len, reverse=True):
            body = body.replace(c, " ")
        body = STEPREF.sub(" ", body)
        unexplained[name] = sorted({t for t in TOKEN.findall(body) if core(t) not in allowed})
    rep.check("every other number in REPORT.md part 4 and WRITE-UP.md comes from the results files",
              not any(unexplained.values()),
              "; ".join(f"{k}: " + (", ".join(v[:15]) if v else "all from the results files") for k, v in unexplained.items()),
              {"unexplained": unexplained, "derived": derived})
    sentence = (OUT / "step31-sentence.txt").read_text().strip()
    rep.check("the CoreNN sentence is word for word step 31's", sentence in p4, "found in REPORT.md part 4"
              if sentence in p4 else "not found")
    demo = (ROOT / "demo" / "README.md").read_text()
    left = {k: sorted(set(re.findall(r"\b[A-Z]{3,}(?:-[A-Z]{3,})+\b", t)))
            for k, t in (("REPORT.md", report), ("WRITE-UP.md", write_up), ("demo/README.md", demo))}
    rep.check("nothing is left unfinished (no placeholders)", not any(left.values()),
              "; ".join(f"{k}: {', '.join(v)}" for k, v in left.items() if v) or "none")
    pics = [m for t in (p4, write_up) for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", t)]
    raw = [k for k, t in (("REPORT.md", report), ("WRITE-UP.md", write_up))
           if re.search(r"^```mermaid", re.sub(r"<details>.*?</details>", "", t, flags=re.S), re.M)]
    missing = [p for p in pics if not (ROOT / p).exists() or (ROOT / p).stat().st_size == 0]
    rep.check("no raw diagram is left, and every picture shown exists", not raw and not missing,
              f"{len(pics)} pictures: {', '.join(pics)}" + (f"; raw diagrams in {raw}" if raw else "")
              + (f"; missing {missing}" if missing else ""))
    rep.finish()


if __name__ == "__main__":
    main()
