"""Step 43: REPORT.md's part 5 and the one-page write-up (part 5). The results page is checked by validate/step33.py.
  .venv/bin/python -I validate/step43.py
Checks:
  - the code is as frozen in step 42;
  - REPORT.md part 5's two tables of numbers (speed; GloVe-100's indexes): every cell, row by row, recomputed here
    from the results files;
  - every other number in REPORT.md part 5 and in WRITE-UP.md is one the results files give (recomputed here, or
    one of the results page's, which validate/step33.py recomputes), a setting in the index's code, a constant of
    the method, or a step's number;
  - nothing is left unfinished: no placeholder in them;
  - no raw diagram is left in either (each is baked to a picture), and every picture they show exists.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import EXP, OUT, ROOT, Report, frozen_changes
from step30 import ALL, FULL, ours_full, rival
from step33 import TOOL, expected, f_int, f_pct, f_pts
from step35 import CONSTANTS as CONSTANTS4, STEPREF, TOKEN, core, tables
from step42 import EVERY, FROZEN5, FULL7, NEW, part5_full

NAMES = {"GooAQ": "gooaq", "GloVe": "glove", "Yandex": "yandex", "Llama": "llama", "Landmark": "landmark", "arXiv": "arxiv",
         "LAION": "laion", "Yi": "yi", "GIST1M": "gist", "NYTimes": "nytimes", "ImageNet": "imagenet",
         "Llama, second head": "llama2", "GloVe-100": NEW}
CONSTANTS = CONSTANTS4 + ["Five parts", "five parts", "Part 5", "part 5", "GloVe-100", "within 10%", "1.25–1.5×",
                          "at least 1.25×", "within 3%", "Three fresh builds", "three fresh builds", "three builds",
                          "100 numbers per word", "3 replacement links", "up to 8", "13 public datasets",
                          "all thirteen", "all 13 datasets", "eight datasets", "a few percent"]


STEP8 = ["faiss_hnsw", "faiss_hnsw_sq8", "faiss_ivfpq", "lorann"]


def runs(step):
    return {c["check"].split(":")[0]: c for c in json.loads((OUT / f"{step}.json").read_text())["checks"]}


def main():
    rep = Report("step43")
    changed = frozen_changes(frozen=FROZEN5)
    rep.check("the code is as frozen in step 42", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    report = (ROOT / "REPORT.md").read_text()
    p5 = report[report.index("# Part 5"):report.index("# Part 4")]
    write_up = (ROOT / "WRITE-UP.md").read_text()
    now, p4 = part5_full(), ours_full()
    where = lambda n: FULL7 if n == NEW else FULL
    rv = {n: {t: r for t in TOOL if (r := rival(t, n, where(n)))} for n in EVERY}
    fastest = {n: max((t for t in rv[n] if rv[n][t]["qps95"]), key=lambda t: rv[n][t]["qps95"]) for n in EVERY}
    q = {n: now[n]["ours"]["qps95"] for n in EVERY}
    ratio = {n: q[n] / rv[n][fastest[n]]["qps95"] for n in EVERY}
    cards = {n: json.loads((EXP / "data" / "prepared" / n / "card.json").read_text()) for n in EVERY}

    # the two tables, cell by cell
    want = {}
    for n in EVERY:
        f = fastest[n]
        want[("Items", n)] = [f"{cards[n]['items_full']:,}", f_int(q[n]), f_int(p4[n]["ours"]["qps95"]) if n in p4 else "",
                              f"{q[n] / p4[n]['ours']['qps95']:.2f}" if n in p4 else "", f"{TOOL[f]} {f_int(rv[n][f]['qps95'])}",
                              f"{ratio[n]:.2f}", f_int(now[n]["ours"]["checks95"])]
    mo, t = now[NEW]["memory"], now[NEW]["tuning"]
    ram = lambda m: (m["total_bytes"] + m["full_numbers_on_ssd_bytes"]) / m["items"]
    settings = lambda p: f"{p['leaves']:,} buckets" if "leaves" in p else f"{p['links']} links"
    for tool in ("scann", "glass", "hnsw"):
        r = rv[NEW][tool]
        want[("One core at 95%", TOOL[tool])] = [f_int(r["qps95"]), f_int(r["qps_all"]), f_int(r["bytes_per_item"]),
                                                 settings(r["params"])]
    want[("One core at 95%", "**vindex**")] = [f"**{f_int(q[NEW])}**", f"**{f_int(now[NEW]['ours']['qps_all'])}**",
                                               f"**{f_int(ram(mo))} ({f_int(mo['total_bytes'] / mo['items'])} with the full "
                                               "numbers on SSD)**",
                                               f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, {t['chosen_code_bits']}-bit copies"]
    bad, rows = [], 0
    for tb in tables(p5):
        key = tb[0][1] if len(tb[0]) > 1 else ""
        for row in tb[2:]:
            k = (key, NAMES.get(row[0], row[0]))
            if k in want:
                rows += 1
                if row[1:] != want[k]:
                    bad.append(f"{row[0]} ({key}): {row[1:]} against {want[k]}")
    rep.check("REPORT.md part 5's tables: every cell matches the results files", rows == len(EVERY) + 4 and not bad,
              f"{rows} rows checked" + "".join(f"\n      {b}" for b in bad[:8]))

    # every other number: one the results files give
    e, _, _ = expected()
    allowed = {core(v) for v in e.values() if isinstance(v, str)} | {core(x) for v in e.values() if isinstance(v, str)
                                                                    for x in TOKEN.findall(v)}
    tune = (ROOT / "crates" / "vindex" / "src" / "tune.rs").read_text()
    const = lambda name: float(re.search(rf"const {name}: \w+ = ([\d.]+);", tune).group(1))
    ch5 = {c["data"]["dataset"]: c["data"] for c in runs("step42-changes").values() if isinstance(c.get("data"), dict)
           and "dataset" in c["data"]}
    ch4 = {c["data"]["dataset"]: c["data"] for c in runs("step32").values() if isinstance(c.get("data"), dict)
           and "dataset" in c["data"]}
    same = lambda bs: len({json.dumps(b["picks"], sort_keys=True) for b in bs}) == 1
    first = [c for c in json.loads((OUT / "step37-first-rule.json").read_text())["checks"] if "same settings" in c["check"]]
    s37 = json.loads((OUT / "step37.json").read_text())["checks"]
    sec = {c["check"].split(":")[0]: c["data"]["sec_per_question"] for c in json.loads((OUT / "step37-full.json").read_text())["checks"]
           if "of each other" in c["check"]}
    imagenet_trials = next(c["data"]["builds"] for c in s37 if c["check"].startswith("imagenet:") and "same settings" in c["check"])
    slack_gain = []   # in each ImageNet sample build: slack 1.3 over 1.2, at 32 links, in the setting search's trial
    for b in imagenet_trials:
        sp = {round(x["alpha"], 1): x["searches_per_s"] for x in b["trials"] if x["links"] == "auto" and x["r"] == 32}
        slack_gain.append(sp[1.3] / sp[1.2] - 1)
    y42 = [x for x in now["yandex"]["tuning"]["trials"] if x["links"] == "auto" and abs(x["alpha"] - 1.2) < 1e-3]
    links = {x["r"]: x["searches_per_s"] for x in y42}
    same_picks = [n for n in ALL if all(now[n]["tuning"][k] == p4[n]["tuning"][k] for k in ("chosen_r", "chosen_code_bits", "chosen_rerank"))
                  and abs(now[n]["tuning"]["chosen_alpha"] - p4[n]["tuning"]["chosen_alpha"]) < 1e-6]
    moved_mem = [n for n in ALL if abs(ram(now[n]["memory"]) / ram(p4[n]["memory"]) - 1) > 0.001]
    hn = {n: q[n] / rv[n]["hnsw"]["qps95"] for n in EVERY if rv[n]["hnsw"]["qps95"]}
    ssd_vs_hnsw = [now[n]["bytes_per_item"] / rv[n]["hnsw"]["bytes_per_item"] for n in EVERY]
    s31 = next(c["data"] for c in json.loads((OUT / "step31.json").read_text())["checks"] if c.get("data"))
    grouping = json.loads((OUT / "step40-grouping.json").read_text())["checks"]
    best_buckets = max(float(re.search(r"([\d,]+)/s at 95%", c["detail"]).group(1).replace(",", "")) for c in grouping)
    links_full = float(re.search(r"links ([\d,]+)/s", runs("step40-full")["glove"]["detail"]).group(1).replace(",", ""))
    exps = json.loads((OUT / "step42-expectations.json").read_text())
    step7 = [json.loads(f.read_text()) for f in (EXP / "results" / "step7").glob("*/*/*/*.json")]
    others = [c for n in EVERY for c in now[n]["ours"]["other_programs_cores"]]
    derived = {
        "rule": [f"{(const('TIE') - 1):.0%}", f"{const('CLOSE'):.1%}", str(int(const("MORE_ROUNDS"))), f"{const('CLEAR_WIN') - 1:.0%}"],
        "wins": [f"{min(ratio[n] for n in ALL if n != 'glove'):.2f}", f"{max(ratio.values()):.2f}",
                 str(sum(r >= 1.05 for r in ratio.values())), str(len(EVERY))],
        "losses": [f"{1 / ratio[n]:.2f}" for n in ("glove", NEW)] + [f"{ratio[n]:.2f}" for n in ("glove", NEW)]
                  + [f"{q[NEW] / rv[NEW]['glass']['qps95']:.2f}", f"{hn[NEW]:.1f}", f"{q['glove'] / p4['glove']['ours']['qps95']:.2f}"],
        "HNSW": [f"{min(hn.values()):.2f}", f"{max(hn.values()):.2f}"],
        "steady": [str(sum(same(c["data"]["builds"]) for c in s37 if "same settings" in c["check"])),
                   str(sum("same settings" in c["check"] for c in s37)),
                   str(sum(same(bs) for bs in json.loads((OUT / "step37-before.json").read_text()).values())),
                   str(sum(same(c["data"]["builds"]) for c in first)), e["steady.apart"], e["steady.rebuild_move"],
                   e["steady.yandex_slower"], f"{min(slack_gain):.1%}", f"{max(slack_gain):.1%}",
                   f"{max(sec['yandex']) / min(sec['yandex']) - 1:.1%}", f"{max(sec['imagenet']) / min(sec['imagenet']) - 1:.1%}",
                   f"{abs(links[48] / links[96] - 1):.2%}", f"{const('CLOSE') * 100 + 2:.1f}%",
                   f"{1 - q['yandex'] / p4['yandex']['ours']['qps95']:.0%}", f"{q['yandex'] / p4['yandex']['ours']['qps95']:.2f}"],
        "yandex": [f_int(now["yandex"]["ours"]["checks95"]), f_int(p4["yandex"]["ours"]["checks95"]),
                   str(now["yandex"]["tuning"]["chosen_r"]), f"{now['yandex']['tuning']['chosen_alpha']:.1f}", "48"],
        "same picks": [f"{min(q[n] / p4[n]['ours']['qps95'] for n in same_picks):.2f}",
                       f"{max(q[n] / p4[n]['ours']['qps95'] for n in same_picks):.2f}", f"{min(others):.2f}", f"{max(others):.2f}",
                       str(len(ALL) - len(same_picks)), str(len(moved_mem))],
        "accuracy": [f"{min(now[n]['ours']['accuracy_at_target'] for n in EVERY):.2%}",
                     f"{max(now[n]['ours']['accuracy_at_target'] for n in EVERY):.2%}"],
        "changes": [f_pts(d["drift"]) for d in ch5.values()] + [f_pts(ch4["landmark"]["drift"])]
                   + [f"{min(d['memory_after_vs_fresh'] for d in ch5.values()):.3f}",
                      f"{max(d['memory_after_vs_fresh'] for d in ch5.values()):.3f}",
                      f"{max(abs(ch5[n]['drift'] - ch4[n]['drift']) for n in ALL if n != 'landmark') * 100:.2f}"],
        "memory": [f_int(ram(now[n]["memory"])) for n in moved_mem] + [f_int(ram(p4[n]["memory"])) for n in moved_mem]
                  + [str(now[n]["tuning"]["chosen_r"]) for n in moved_mem] + [str(p4[n]["tuning"]["chosen_r"]) for n in moved_mem]
                  + [f"{min(ssd_vs_hnsw):.2f}", f"{max(ssd_vs_hnsw):.2f}", e["limits.slow_lo"], e["limits.slow_hi"]],
        "buckets": [re.search(r"([\d.]+)x the links", runs("step40")["glove"]["detail"]).group(1),
                    f"{best_buckets / links_full:.2f}", f"{best_buckets / rival('scann', 'glove')['qps95']:.2f}"],
        "expectations": [str(len(exps) + 1), str(sum(x["right"] for x in exps)), str(sum(not x["right"] for x in exps) - 1), "1"],
        "CoreNN": [f"{s31['speed_ratio']:.1f}", f"{s31['memory_less_ssd']:.0%}"],
        "time": [f"{json.loads((OUT / 'step42-ours.json').read_text())['seconds'] / 60:.0f}",
                 f"{json.loads((OUT / 'step42-changes.json').read_text())['seconds'] / 3600:.1f}",
                 f"{sum(r.get('wall_seconds', 0) for r in step7) / 60:.0f}"],
    }
    # step 47 (after the freeze): Faiss and LoRANN on all 13 (the experiments' results/step8), counted as rivals;
    # hnswlib no longer counts, Faiss HNSW stands in
    s8 = {n: {t: r for t in STEP8 if (r := rival(t, n, EXP / "results" / "step8"))} for n in EVERY}
    rv8 = {n: {**{t: r for t, r in rv[n].items() if t != "hnsw"}, **s8[n]} for n in EVERY}
    ratio8 = {n: q[n] / max(r["qps95"] for r in rv8[n].values() if r["qps95"]) for n in EVERY}
    ahead = [r for r in ratio8.values() if r >= 1.05]   # the results page's rule: within 5% is level
    fh = [q[n] / s8[n]["faiss_hnsw"]["qps95"] for n in EVERY if s8[n]["faiss_hnsw"]["qps95"]]
    derived["step 47"] = [str(len(ahead)), f"{min(ahead):.2f}", f"{max(ahead):.2f}", f"{ratio8['yandex']:.2f}",
                          f"{ratio8['gist']:.2f}", f"{ratio['gist']:.2f}", f"{q['glove'] / s8['glove']['faiss_ivfpq']['qps95']:.2f}",
                          f"{min(fh):.2f}", f"{max(fh):.2f}"]
    for vs in derived.values():
        allowed |= {core(x) for v in vs for x in TOKEN.findall(v)}
    unexplained = {}
    for name, text in (("REPORT.md part 5", p5), ("WRITE-UP.md", write_up)):
        body = re.sub(r"<details>.*?</details>", " ", text, flags=re.S)
        body = re.sub(r"!\[[^\]]*\]\([^)]*\)|`[^`]*`|https?://\S+", " ", body)
        body = "\n".join(ln for ln in body.splitlines() if not (ln.startswith("|") and any(ln.startswith(f"| {k} |") for k in NAMES)))
        for c in sorted(CONSTANTS, key=len, reverse=True):
            body = body.replace(c, " ")
        body = STEPREF.sub(" ", body)
        unexplained[name] = sorted({x for x in TOKEN.findall(body) if core(x) not in allowed})
    rep.check("every other number in REPORT.md part 5 and WRITE-UP.md comes from the results files or the code",
              not any(unexplained.values()),
              "; ".join(f"{k}: " + (", ".join(v[:15]) if v else "all accounted for") for k, v in unexplained.items()),
              {"unexplained": unexplained, "derived": derived})
    left = {k: sorted(set(re.findall(r"\b[A-Z]{3,}(?:-[A-Z]{3,})+\b", x))) for k, x in (("REPORT.md", report), ("WRITE-UP.md", write_up))}
    rep.check("nothing is left unfinished (no placeholders)", not any(left.values()),
              "; ".join(f"{k}: {', '.join(v)}" for k, v in left.items() if v) or "none")
    pics = [m for x in (p5, write_up) for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", x)]
    raw = [k for k, x in (("REPORT.md", report), ("WRITE-UP.md", write_up))
           if re.search(r"^```mermaid", re.sub(r"<details>.*?</details>", "", x, flags=re.S), re.M)]
    missing = [p for p in pics if not (ROOT / p).exists() or (ROOT / p).stat().st_size == 0]
    rep.check("no raw diagram is left, and every picture shown exists", not raw and not missing,
              f"{len(pics)} pictures: {', '.join(pics)}" + (f"; raw diagrams in {raw}" if raw else "")
              + (f"; missing {missing}" if missing else ""))
    rep.finish()


if __name__ == "__main__":
    main()
