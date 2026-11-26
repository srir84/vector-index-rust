"""Steps 33 and 43: the results page, made from the results files only (part 5, with part 4's numbers beside).
    .venv/bin/python -I tools/results_page.py [--allow-missing]     # -> page/results.html
It reads, and only reads:
  validate/results/step42-ours.json, step42-changes.json   ours at full size in part 5: speed, memory, SSD, picks; changes
  validate/results/step29.json, step32.json        the same in part 4, shown beside
  validate/results/step42-expectations.json        step 36's expectations, marked
  validate/results/step37-before.json, step37.json, step37-full.json, step30b.json   how steady fresh builds are
  validate/results/step31.json, step31-sentence.txt   the CoreNN sentence (part 4)
  validate/results/step42-frozen.json, step30-docker.txt, machine.json          versions and the machine
  the experiments' results/step6/final/<rival>/<dataset>/best.json             the rivals at full size (part 4)
  the experiments' results/step7/final/<rival>/glove100/best.json              GloVe-100's rivals (part 5)
  the experiments' data/prepared/<dataset>/card.json and items_full.npy (its shape only)   the datasets
Every number on the page sits in an element with a data-n key, and every passage copied from a results file in one
with a data-src key; validate/step33.py recomputes or re-reads each one and checks the page. The pictures are drawn
here as HTML and SVG, so the page reads in full before any script runs; a short script adds the hover readouts.
--allow-missing draws a draft while rival runs are still going (marked on the page).
"""
import argparse
import html
import json
import math
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "validate" / "results"
EXP = Path(os.environ.get("VINDEX_EXPERIMENTS", ROOT.parent / "vector-index-experiments"))
DATA = EXP / "data" / "prepared"
NEW = "glove100"   # part 5's held-back dataset: its rivals ran in part 5 (step 42), the others' in part 4
RUNS = lambda n: EXP / "results" / ("step7" if n == NEW else "step6")
OUT = ROOT / "page" / "results.html"

ORDER = ["gooaq", "glove", "yandex", "llama", "landmark", "arxiv", "laion", "yi", "gist", "nytimes", "imagenet", "llama2",
         NEW]
FIRST_TESTED = {"arxiv": 2, "laion": 2, "yi": 2, "gist": 2, "nytimes": 3, "imagenet": 3, "llama2": 3, NEW: 5}   # held back
NAME = {"gooaq": "GooAQ", "glove": "GloVe", "yandex": "Yandex", "llama": "Llama", "landmark": "Landmark", "arxiv": "arXiv",
        "laion": "LAION", "yi": "Yi", "gist": "GIST1M", "nytimes": "NYTimes", "imagenet": "ImageNet",
        "llama2": "Llama, 2nd head", NEW: "GloVe-100"}
WHAT = {   # from each card's "what", in everyday words; the count goes in front
    "gooaq": "answers from Google's “people also ask” boxes",
    "glove": "words from Twitter",
    "yandex": "photos, searched with text",
    "llama": "lookups inside the Llama 3 language model as it reads a novel",
    "landmark": "photos of famous places",
    "arxiv": "summaries of science papers",
    "laion": "web photos, searched with text",
    "yi": "lookups inside the Yi language model",
    "gist": "photos, described by hand-made features",
    "nytimes": "New York Times articles, as word counts",
    "imagenet": "photos, searched with text captions",
    "llama2": "lookups from another part of Llama 3",
    NEW: "words from Twitter, from a second word model",
}
DISTANCE = {"cos": "angle", "ip": "best score", "l2": "straight-line distance"}
TOOL = {"ours": "vindex", "hnsw": "HNSW", "glass": "Glass", "scann": "ScaNN", "roargraph": "RoarGraph", "corenn": "CoreNN"}
PLAN = {"glass": ["gooaq", "yandex", "landmark", "arxiv", "laion", "gist", "nytimes", "imagenet", NEW],
        "scann": ["glove", "nytimes", NEW], "roargraph": ["llama", "yi", "llama2"], "corenn": ["gist"], "hnsw": ORDER}
LEVEL = 0.05          # within 5% counts as level: about the timing noise of one core
X0, X1 = 0.85, 1.0    # the accuracy shown on the speed-accuracy charts
esc = html.escape


# ---------- numbers, as the page prints them (validate/step33.py prints them the same way) ----------
def f_int(v):
    return f"{v:,.0f}"


def f_items(n):
    return (f"{n / 1e6:.2f}".rstrip("0").rstrip(".") + " million") if n >= 1_000_000 else f"{n:,}"


def f_x(v):
    return f"{v:.2f}×"


def f_pct(v):
    return f"{v * 100:.2f}%"


def f_pts(v):
    return f"{v * 100:+.2f}".replace("-", "−")


def f_time(s):
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


class Nums:
    """Every printed number goes through here, keyed, so the check can find it."""

    def __init__(self):
        self.keys = {}

    def __call__(self, key, text, tag="span", cls=None):
        if self.keys.setdefault(key, text) != text:
            raise ValueError(f"{key} printed two ways: {self.keys[key]!r} and {text!r}")
        c = f' class="{cls}"' if cls else ""
        return f'<{tag}{c} data-n="{key}">{esc(text)}</{tag}>'


# ---------- reading the results ----------
def checks(step):
    return json.loads((RES / f"{step}.json").read_text())["checks"]


def frontier(points):
    """(accuracy, seconds per question) -> the line the 95% read-off uses (common.cost_at_target): sorted by time,
    each point's accuracy the best reached at that time or less."""
    pts = sorted(points, key=lambda p: p[1])
    r = np.maximum.accumulate([a for a, _ in pts]).tolist()
    return r, [s for _, s in pts]


def full_size(step):
    return {c["check"].split(":")[0]: c["data"] for c in checks(step) if c.get("data") and "at full size" in c["check"]}


def change_runs(step):
    return {c["data"]["dataset"]: dict(c["data"], passed=c["pass"]) for c in checks(step)
            if isinstance(c.get("data"), dict) and "dataset" in c["data"]}


def ours_of(o):
    t, st = o["tuning"], o["stats"]
    return {"qps": o["ours"]["qps95"], "qps_all": o["ours"]["qps_all"], "build": o["build_seconds"],
            "mem_ssd": o["memory"]["total_bytes"] / o["memory"]["items"],
            "mem_ram": (o["memory"]["total_bytes"] + o["memory"]["full_numbers_on_ssd_bytes"]) / o["memory"]["items"],
            "ssd_qps": o["ssd"]["qps"], "ssd_qps_all": o["ssd"]["qps_all"],
            "others": o["ours"]["other_programs_cores"],
            "curve": frontier([(c["accuracy"], c["sec"]) for c in o["ours"]["curve"]]),
            "picks": (f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, {t['chosen_code_bits']}-bit copies, "
                      f"re-check {t['chosen_rerank']}" + (", question links" if st["question_links"] else "")
                      + (", extra links for popular items" if st["popular_items"] else ""))}


def load(allow_missing):
    ours, ours4 = full_size("step42-ours"), full_size("step29")
    changes, changes4 = change_runs("step42-changes"), change_runs("step32")
    missing, ds = [], {}
    for n in ORDER:
        card = json.loads((DATA / n / "card.json").read_text())
        d = {"name": n, "items": card["items_full"], "dim": int(np.load(DATA / n / "items_full.npy", mmap_mode="r").shape[1]),
             "distance": DISTANCE[card["formula"]], "licence": card.get("licence", ""), "rivals": {},
             "ours": ours_of(ours[n]), "p4": ours_of(ours4[n]) if n in ours4 else None,
             "changes": changes.get(n), "changes4": changes4.get(n)}
        for tool, names in PLAN.items():
            if n not in names:
                continue
            f = RUNS(n) / "final" / tool / n / "best.json"
            if not f.exists():
                missing.append(f"{tool}/{n}")
                continue
            r = json.loads(f.read_text())
            acc = r.get("memory_accounting") or {}
            sec = r.get("sec_per_query_at95")
            p = r["params"]
            curve = frontier([(c["accuracy"], c["sec_per_query_1core"]) for c in r["curve"]])
            d["rivals"][tool] = {
                "qps": 1 / sec if sec else None, "best": r["best_accuracy"], "build": r["build_seconds"],
                # its quickest setting already finds more than 95%: its speed is that setting's (common.cost_at_target)
                "floor": curve[0][0] if r.get("status_at95") == "below_min_effort" else None,
                "mem": (acc.get("total_bytes") or r.get("rss_index_bytes") or 0) / r["n_items"],
                "mem_from": "counted" if acc.get("total_bytes") else "process",
                "qps_all": 1 / r["final"]["sec_per_query_allcores"], "threads": r["final"].get("threads_allcores"),
                "curve": curve,
                "settings": (f"{p['leaves']:,} buckets, {p['dims_per_block']} number{'s' if p['dims_per_block'] > 1 else ''} "
                             f"per code" if "leaves" in p else f"{p['links']} links, build list {p['build']}"
                             + (f", slack {p['slack']}" if p.get("slack") else "")),
                "effort": r["final"]["effort"]}
        reach = {t: v for t, v in d["rivals"].items() if v["qps"]}
        d["fastest"] = max(reach, key=lambda t: reach[t]["qps"]) if reach else None
        d["ratio"] = d["ours"]["qps"] / reach[d["fastest"]]["qps"] if reach else None
        d["ratio4"] = d["p4"]["qps"] / reach[d["fastest"]]["qps"] if reach and d["p4"] else None
        ds[n] = d
    if missing and not allow_missing:
        raise SystemExit(f"rival results missing: {', '.join(missing)} (use --allow-missing for a draft)")
    extra = {"machine": json.loads((RES / "machine.json").read_text()),
             "frozen": json.loads((RES / "step42-frozen.json").read_text()),
             "docker": (RES / "step30-docker.txt").read_text() if (RES / "step30-docker.txt").exists() else "",
             "expectations": json.loads((RES / "step42-expectations.json").read_text())
             if (RES / "step42-expectations.json").exists() else None,
             "sentence31": (RES / "step31-sentence.txt").read_text().strip() if (RES / "step31-sentence.txt").exists() else None,
             "questions": json.loads((DATA / "gooaq" / "card.json").read_text())["questions"],
             "slice": json.loads((DATA / "gooaq" / "card.json").read_text())["slice_items"],
             "tried": {t: {n: [json.loads(f.read_text())["params"] for f in sorted((RUNS(n) / "grid" / t / n).glob("*.json"))]
                           for n in names} for t, names in PLAN.items()}}
    # how steady fresh builds are: part 4's rebuilds (step 30b), three builds of each sample before the fix and after
    # (step 37), and Yandex at full size: step 37's three builds against part 5's frozen run
    rc = [c["data"] for c in checks("step30b") if c.get("data")]
    same = lambda bs: all(b["picks"] == bs[0]["picks"] for b in bs)
    picks37 = {c["check"].split(":")[0]: c for c in checks("step37") if "same settings" in c["check"]}
    apart = {c["check"].split(":")[0]: max(c["data"]["sec_per_question"]) / min(c["data"]["sec_per_question"]) - 1
             for c in checks("step37") if "of each other" in c["check"]}
    y37 = next(c["data"]["builds"] for c in checks("step37-full") if c["check"].startswith("yandex") and "same settings" in c["check"])
    extra["steady"] = {
        "rebuild_move": max(abs(r["ours"]["qps95"] / ours4[r["dataset"]]["ours"]["qps95"] - 1) for r in rc),
        "before": sum(same(bs) for bs in json.loads((RES / "step37-before.json").read_text()).values()),
        "now": sum(same(c["data"]["builds"]) for c in picks37.values()), "samples": len(picks37),
        "apart": max(apart[n] for n, c in picks37.items() if same(c["data"]["builds"])),
        "flips": [n for n, c in picks37.items() if not same(c["data"]["builds"])],
        "yandex_slower": 1 - ours["yandex"]["ours"]["qps95"] / min(b["qps95"] for b in y37)}
    v =(EXP / "baselines" / "VERSIONS.md").read_text()
    extra["scann_version"] = re.search(r"scann (\d+\.\d+\.\d+) from PyPI", v).group(1)
    extra["roargraph_commit"] = re.search(r"matchyc/RoarGraph commit ([0-9a-f]{7})", v).group(1)
    return ds, extra, missing


def verdict(ratio):
    return "faster" if ratio >= 1 + LEVEL else "slower" if ratio <= 1 - LEVEL else "level"


# ---------- pictures ----------
def ratio_chart(ds, N):
    rows = sorted((d for d in ds.values() if d["ratio"]), key=lambda d: -d["ratio"])
    rs = [d["ratio"] for d in rows] + [d["ratio4"] for d in rows if d["ratio4"]]
    lo, hi = min(0.5, min(rs) / 1.3), max(2.0, max(rs) * 1.4)   # room for the value at each bar's tip
    pos = lambda v: (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 100
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi]
    grid = "".join(f'<span class="gl{" one" if t == 1 else ""}" style="left:{pos(t):.2f}%"></span>' for t in ticks)
    axis = "".join(f'<span class="rtick" style="left:{pos(t):.2f}%">{t:g}×</span>' for t in ticks)
    out = []
    for d in rows:
        n, r, rv = d["name"], d["ratio"], TOOL[d["fastest"]]
        a, b = sorted((pos(1), pos(r)))
        up = r >= 1
        r4 = d["ratio4"]
        m = pos(r4) if r4 else None   # the value goes past part 4's mark when that lies beyond the bar's tip
        tip = (f'<span class="tipval" style="left:calc({max(b, m or b):.2f}% + 6px)">' if up
               else f'<span class="tipval" style="right:calc({100 - min(a, m or a):.2f}% + 6px)">')
        words = (f"{f_x(r)} as fast as {rv}" if up else f"{rv} is {f_x(1 / r)} as fast")
        was = (f'<span class="then">part 4: {N(f"{n}.ratio4", f_x(r4))}</span>' if r4 else
               '<span class="then">new in part 5</span>')
        mark = f'<span class="p4" style="left:{pos(r4):.2f}%"></span>' if r4 else ""
        out.append(
            f'<li class="rrow" aria-label="{esc(NAME[n])}: {esc(words)}' + (f", part 4 {f_x(r4)}" if r4 else "") + '">'
            f'<span class="rname">{esc(NAME[n])} <span class="vs">vs {esc(rv)}</span>{was}</span>'
            f'<span class="track">{grid}<span class="bar {"up" if up else "down"}" style="left:{a:.2f}%;width:{b - a:.2f}%"></span>'
            f'{mark}{tip}{N(f"{n}.ratio", f_x(r))}</span></span></li>')
    return (f'<ol class="ratios" role="list">{"".join(out)}</ol>'
            f'<div class="raxis" aria-hidden="true"><span class="rname"></span><span class="track">{axis}</span></div>')


def nice_ticks(lo, hi):
    c = [k * 10 ** e for e in range(0, 8) for k in (1, 2, 5) if lo <= k * 10 ** e <= hi]
    return c if len(c) <= 6 else [v for v in c if str(int(v)).startswith("1")]


def compact(v):
    return f"{v:g}" if v < 1000 else f"{v / 1000:g}K" if v < 1e6 else f"{v / 1e6:g}M"


def panel(d, N):
    n = d["name"]
    W, H, ML, MR, MT, MB = 340, 230, 40, 78, 10, 34
    PW, PH = W - ML - MR, H - MT - MB
    series = [("ours", d["ours"], True)] + [(t, v, False) for t, v in d["rivals"].items()]
    lines = []
    for t, v, mine in series:   # each segment split in six, linear in time as the read-off is
        r, c = v["curve"]
        pts = [(r[0], c[0])]
        for i in range(1, len(r)):
            pts += [(r[i - 1] + (r[i] - r[i - 1]) * k / 6, c[i - 1] + (c[i] - c[i - 1]) * k / 6) for k in range(1, 7)]
        lines.append((t, v, mine, pts))
    seen = [1 / s for _, _, _, pts in lines for a, s in pts if a >= X0] + [v["qps"] for _, v, _ in series if v["qps"]]
    ylo, yhi = min(seen) / 1.5, max(seen) * 1.5
    lx = lambda a: ML + (a - X0) / (X1 - X0) * PW
    ly = lambda q: MT + (math.log(yhi) - math.log(q)) / (math.log(yhi) - math.log(ylo)) * PH
    g = [f'<svg viewBox="0 0 {W} {H}" data-x0="{X0}" data-x1="{X1}" role="img" '
         f'aria-label="{esc(NAME[n])}: questions per second against share of the true 10 found">',
         f'<defs><clipPath id="c-{n}"><rect x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></clipPath></defs>']
    for q in nice_ticks(ylo, yhi):
        g.append(f'<line class="grid" x1="{ML}" x2="{ML + PW}" y1="{ly(q):.1f}" y2="{ly(q):.1f}"/>'
                 f'<text class="tick" x="{ML - 6}" y="{ly(q) + 3.5:.1f}" text-anchor="end">{compact(q)}</text>')
    for a in (0.85, 0.90, 0.95, 1.0):
        g.append(f'<text class="tick" x="{lx(a):.1f}" y="{H - MB + 15}" text-anchor="middle">{a * 100:.0f}%</text>')
    g.append(f'<line class="mark95" x1="{lx(0.95):.1f}" x2="{lx(0.95):.1f}" y1="{MT}" y2="{MT + PH}"/>'
             f'<text class="axis" x="{ML + PW / 2:.1f}" y="{H - 4}" text-anchor="middle">share of the true 10 found</text>')
    ends, notes = [], []
    for t, v, mine, pts in sorted(lines, key=lambda l: l[2]):   # ours drawn last, on top
        vis = [(a, s) for a, s in pts if a >= X0 - 0.02]
        if not any(a >= X0 for a, _ in vis):
            notes.append(f'{esc(TOOL[t])} never reaches {X0 * 100:.0f}% (best {N(f"{n}.{t}.best", f_pct(v["best"]))})')
            continue
        path = " ".join(f"{'M' if i == 0 else 'L'}{lx(a):.1f},{ly(1 / s):.1f}" for i, (a, s) in enumerate(vis))
        g.append(f'<path class="{"ours" if mine else "rival"}" clip-path="url(#c-{n})" d="{path}"/>')
        a_end, s_end = vis[-1]   # the line's last point: its most careful search
        ends.append([ly(1 / s_end), lx(min(a_end, X1)), TOOL[t], mine, ly(1 / s_end)])
    for t, v, mine in sorted(series, key=lambda s: s[2]):
        if v["qps"]:   # a rival whose quickest setting finds more than 95% has its dot where that setting is
            g.append(f'<circle class="dot {"ours" if mine else "rival"}" cx="{lx(v.get("floor") or 0.95):.1f}" '
                     f'cy="{ly(v["qps"]):.1f}" r="4"/>')
        if v.get("floor"):
            notes.append(f'{esc(TOOL[t])}’s quickest setting already finds {N(f"{n}.{t}.floor", f_pct(v["floor"]))}')
    ends.sort()
    for i in range(1, len(ends)):   # keep end labels 12 apart, top to bottom, inside the plot's height
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
    over = ends[-1][0] - (MT + PH) if ends else 0
    if over > 0:
        for e in ends:
            e[0] -= over
    for y, x, label, mine, y0 in ends:
        g.append(f'<line class="leader" x1="{x + 2:.1f}" x2="{ML + PW + 4}" y1="{y0:.1f}" y2="{y:.1f}"/>'
                 f'<text class="end{" mine" if mine else ""}" x="{ML + PW + 7}" y="{y + 3.5:.1f}">{esc(label)}</text>')
    g.append(f'<line class="cross" x1="0" x2="0" y1="{MT}" y2="{MT + PH}" visibility="hidden"/>'
             f'<rect class="hit" x="{ML}" y="{MT}" width="{PW}" height="{PH}"/></svg>')
    badge = (f'<span class="badge" title="Held back: not used while the index was built; first tested in part '
             f'{FIRST_TESTED[n]}">held back</span>') if n in FIRST_TESTED else ""
    note = f'<p class="pnote">{"; ".join(notes)}.</p>' if notes else ""
    return (f'<figure class="panel" data-ds="{n}" tabindex="0"><figcaption><span class="ptitle">{esc(NAME[n])}{badge}</span>'
            f'<span class="psub">{N(f"{n}.items", f_items(d["items"]))} items · {N(f"{n}.dim", str(d["dim"]))} numbers '
            f'each · {d["distance"]}</span></figcaption>{"".join(g)}{note}'
            f'<div class="readout" hidden></div></figure>')


# ---------- the page ----------
def sentence(d, N):
    n, r = d["name"], d["ratio"]
    head = f'<b>{esc(NAME[n])}</b>, {N(f"{n}.items", f_items(d["items"]))} {esc(WHAT[n])}: '
    if r is None:
        return head + "rival runs still going."
    rv = TOOL[d["fastest"]]
    v = verdict(r)
    fl = d["rivals"][d["fastest"]]["floor"]
    floor = (f' {rv}’s quickest setting already finds {N(f"{n}.{d["fastest"]}.floor", f_pct(fl))}, so its speed is '
             "that setting’s.") if fl else ""
    if v == "faster":
        return head + f'we are {N(f"{n}.ratio", f_x(r))} as fast as {rv}, the fastest rival here.' + floor
    if v == "level":
        return head + f'level with {rv} ({N(f"{n}.ratio", f_x(r))} its speed).' + floor
    return head + f'{rv} is {N(f"{n}.inverse", f_x(1 / r))} as fast as we are.' + floor


def speed_table(ds, N):
    rows = []
    for n in ORDER:
        d = ds[n]
        o = d["ours"]
        rows.append(f'<tr class="group"><th colspan="6" scope="rowgroup">{esc(NAME[n])}'
                    + (' <span class="badge">held back</span>' if n in FIRST_TESTED else "") + "</th></tr>")
        rows.append(f'<tr><th scope="row"><span class="key ours"></span>vindex, all in memory</th>'
                    f'<td>{N(f"{n}.ours.qps", f_int(o["qps"]))}</td><td>{N(f"{n}.ours.qps_all", f_int(o["qps_all"]))}</td>'
                    f'<td>{N(f"{n}.ours.mem_ram", f_int(o["mem_ram"]))}</td><td>{N(f"{n}.ours.build", f_time(o["build"]))}</td>'
                    f'<td class="set">{N(f"{n}.ours.picks", o["picks"])}</td></tr>')
        if p4 := d["p4"]:
            rows.append(f'<tr class="was"><th scope="row"><span class="key ours was"></span>vindex in part 4, all in memory</th>'
                        f'<td>{N(f"{n}.p4.qps", f_int(p4["qps"]))}</td><td>{N(f"{n}.p4.qps_all", f_int(p4["qps_all"]))}</td>'
                        f'<td>{N(f"{n}.p4.mem_ram", f_int(p4["mem_ram"]))}</td><td>{N(f"{n}.p4.build", f_time(p4["build"]))}</td>'
                        f'<td class="set">{N(f"{n}.p4.picks", p4["picks"])}</td></tr>')
        rows.append(f'<tr><th scope="row"><span class="key ours"></span>vindex, full numbers on SSD</th>'
                    f'<td>{N(f"{n}.ours.ssd_qps", f_int(o["ssd_qps"]))}</td>'
                    f'<td>{N(f"{n}.ours.ssd_qps_all", f_int(o["ssd_qps_all"]))}</td>'
                    f'<td>{N(f"{n}.ours.mem_ssd", f_int(o["mem_ssd"]))}</td><td></td><td class="set">same index</td></tr>')
        for t, v in d["rivals"].items():
            speed = (N(f"{n}.{t}.qps", f_int(v["qps"])) + ("<sup>q</sup>" if v["floor"] else "") if v["qps"]
                     else f'never (best {N(f"{n}.{t}.best", f_pct(v["best"]))})')
            rows.append(f'<tr><th scope="row"><span class="key rival"></span>{esc(TOOL[t])}</th><td>{speed}</td>'
                        f'<td>{N(f"{n}.{t}.qps_all", f_int(v["qps_all"]))}</td><td>{N(f"{n}.{t}.mem", f_int(v["mem"]))}'
                        f'{"<sup>p</sup>" if v["mem_from"] == "process" else ""}</td>'
                        f'<td>{N(f"{n}.{t}.build", f_time(v["build"]))}</td>'
                        f'<td class="set">{N(f"{n}.{t}.settings", v["settings"])}</td></tr>')
    return ('<div class="tablewrap"><table class="speeds"><thead><tr><th scope="col">Index</th>'
            '<th scope="col">Questions per second at 95%, one core</th><th scope="col">All cores</th>'
            '<th scope="col">Memory per item, bytes</th><th scope="col">Build, all cores</th><th scope="col">Settings</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def changes_table(ds, N):
    rows = []
    for n in ORDER:
        b, b4 = ds[n]["changes"], ds[n]["changes4"]
        ok = b["passed"]
        rows.append(f'<tr><th scope="row">{esc(NAME[n])}</th><td>{N(f"{n}.changes.count", f_int(b["changes"]))}</td>'
                    f'<td>{N(f"{n}.changes.after", f_pct(b["checkpoints"][-1]["accuracy"]))}</td>'
                    f'<td>{N(f"{n}.changes.fresh", f_pct(b["fresh_accuracy"]))}</td>'
                    f'<td>{N(f"{n}.changes.drift", f_pts(b["drift"]))}</td>'
                    f'<td class="was">{N(f"{n}.changes4.drift", f_pts(b4["drift"])) if b4 else "new"}</td>'
                    f'<td class="{"ok" if ok else "miss"}"><span aria-hidden="true">{"&#10003;" if ok else "&#10005;"}</span> '
                    f'{"passed" if ok else "missed"}</td></tr>')
    return ('<div class="tablewrap"><table class="changes"><thead><tr><th scope="col">Dataset</th>'
            '<th scope="col">Changes</th><th scope="col">Found after them</th><th scope="col">Fresh build</th>'
            '<th scope="col">Drift, points</th><th scope="col">Drift in part 4</th><th scope="col">Rule</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def expectations(extra, ds, N):
    items = []
    for i, e in enumerate(extra["expectations"] or []):
        items.append((e["right"], f'<span data-src="expectations.{i}.expected">{esc(e["expected"])}</span>',
                      f'<span class="what" data-src="expectations.{i}.what">{esc(e["what happened"])}</span>'))
    ch = [d["changes"] for d in ds.values()]
    missed = [d["name"] for d in ds.values() if not d["changes"]["passed"]]
    items.append((not missed, "The change test passes on every dataset",
                  f'<span class="what">passed on {N("exp.changes.passed", str(len(ch) - len(missed)))} of '
                  f'{N("exp.changes.all", str(len(ch)))}' + "".join(
                      f'; {esc(NAME[m])} drifted {N(f"{m}.changes.drift", f_pts(ds[m]["changes"]["drift"]))} points'
                      for m in missed) + "</span>"))
    if not extra["expectations"]:
        items.append((None, "The rest of the expectations", '<span class="what">marked once the runs finish</span>'))
    li = "".join(f'<li class="{"right" if r else "wrong" if r is False else "open"}"><span class="mark">'
                 f'{"&#10003; right" if r else "&#10005; wrong" if r is False else "pending"}</span>'
                 f'<span class="exp">{t}</span>{w}</li>' for r, t, w in items)
    return f'<ul class="expect">{li}</ul>'


def limits(ds, extra, N):
    slower = [d for d in ds.values() if d["ratio"] and verdict(d["ratio"]) == "slower"]
    level = [d for d in ds.values() if d["ratio"] and verdict(d["ratio"]) == "level"]
    mem = [d["ours"]["mem_ram"] / d["ours"]["mem_ssd"] for d in ds.values()]
    slow = [d["ours"]["qps"] / d["ours"]["ssd_qps"] for d in ds.values()]
    vs = [d["ours"]["mem_ram"] / d["rivals"][d["fastest"]]["mem"] for d in ds.values() if d["fastest"]]
    oth = [c for d in ds.values() for c in d["ours"]["others"]]
    tried = {t: sorted({len(v) for v in per.values() if v}) for t, per in extra["tried"].items()}
    span = lambda s: (str(s[0]) if len(s) == 1 else f"{s[0]}–{s[-1]}") if s else "?"
    cores = extra["docker"].split(" cores")[0].split()[-1] if " cores" in extra["docker"] else None
    li = []
    if slower:
        li.append("<b>Not beaten:</b> " + "; ".join(
            f'on {esc(NAME[d["name"]])}, {TOOL[d["fastest"]]} is {N(f"{d["name"]}.inverse", f_x(1 / d["ratio"]))} as fast as we are'
            for d in slower) + ".")
    if level:
        li.append("<b>Level, not ahead:</b> " + "; ".join(
            f'{esc(NAME[d["name"]])} ({N(f"{d["name"]}.ratio", f_x(d["ratio"]))} {TOOL[d["fastest"]]}’s speed)'
            for d in level) + ". Within 5% is about the timing noise of one core.")
    li.append("<b>Memory:</b> our speeds are with everything in memory, where we use "
              f'{N("limits.vs_lo", f"{min(vs):.2f}")} to {N("limits.vs_hi", f"{max(vs):.2f}")}× the fastest '
              "rival’s memory per item. With the full numbers left on the SSD we use "
              f'{N("limits.mem_lo", f"{min(mem):.1f}")} to {N("limits.mem_hi", f"{max(mem):.1f}")}× less, but answer '
              f'{N("limits.slow_lo", f"{min(slow):.0f}")} to {N("limits.slow_hi", f"{max(slow):.0f}")}× fewer '
              "questions per second.")
    li.append("<b>The bucket search was left out:</b> on GloVe and GloVe-100, ScaNN groups the items into buckets and "
              "scans the closest buckets’ short codes instead of walking links. We built that way of searching too, "
              "but at full size on GloVe it did not beat ScaNN, so it is not part of the index.")
    li.append("<b>Which rivals:</b> at full size we ran only the rivals that were fastest on each dataset in the "
              f'{N("slice", f_int(extra["slice"]))}-item samples, or within 1.3× of the fastest (ScaNN on NYTimes), '
              "plus HNSW everywhere and CoreNN on GIST1M. That list "
              "was fixed before any full-size run. Their runs are part 4’s; only GloVe-100’s rivals (ScaNN, Glass "
              "and HNSW) ran in part 5. DiskANN, slower in the samples, was not rerun. SymphonyQG needs "
              "Intel’s AVX-512 and cannot run on this Mac.")
    graph = sorted(set(tried["hnsw"]) | set(tried["glass"]) | set(tried["roargraph"]))
    li.append(f'<b>Their settings:</b> each rival tried {N("tried.graph", span(graph))} settings on the tuning '
              f'questions (ScaNN {N("tried.scann", span(tried["scann"]))}) and kept its fastest at 95%. Their authors '
              "may know better ones. CoreNN ran on its author’s own settings only.")
    li.append("<b>Containers and patches:</b> ScaNN has no Mac version, so it ran in a Linux container on the same Mac"
              + (f' ({N("docker.cores", cores)} cores)' if cores else "") + ", which may cost it some speed. Glass, "
              "RoarGraph and CoreNN needed small changes to build on a Mac; their search code was not changed, "
              "except that RoarGraph’s Intel-only distance code runs through a translation layer (SIMDe), "
              "which may be slower than code written for Apple chips.")
    s = extra["steady"]
    li.append("<b>Fresh builds are steadier, not fully steady:</b> in part 4 a fresh build could be up to "
              f'{N("steady.rebuild_move", f"{s["rebuild_move"]:.1%}")} faster or slower than the last, because the '
              "index’s setting search, which times trial builds, kept the faster of two close choices by any margin. "
              "Now ties go to the plainer choice, and close calls are timed longer. Three fresh builds of each sample made "
              f'the same picks on {N("steady.now", str(s["now"]))} of {N("steady.samples", str(s["samples"]))} datasets '
              f'(before: {N("steady.before", str(s["before"]))}), and those were within '
              f'{N("steady.apart", f"{s["apart"]:.1%}")} of each other. '
              + "".join(f"{esc(NAME[n])}’s sample still flips: one choice sits right on the bar. " for n in s["flips"])
              + "At full size, the frozen run’s Yandex build picked differently from three builds made just before it, "
              f'and is {N("steady.yandex_slower", f"{s["yandex_slower"]:.1%}")} slower than the slowest of them.')
    li.append("<b>A busy machine:</b> a macOS storage scan, not ours, used up to "
              f'{N("limits.others_max", f"{max(oth):.1f}")} other cores during our runs. Speeds are on one core, '
              "where this mattered little in earlier checks.")
    return f'<ul class="limits-list">{"".join(f"<li>{x}</li>" for x in li)}</ul>'


CSS = """
/* Layout: one reading column for words, a wider band for the charts and tables; summary first, detail after. */
:root {
  --bg: #f4f5f7; --surface: #ffffff; --ink: #15181c; --ink-2: #4b5159; --ink-3: #646b74; --rule: #e0e3e8;
  --ours: #2a78d6; --rival: #8a9099; --good: #1a7a43; --bad: #b5302c;
  --display: "Schibsted Grotesk", "Helvetica Neue", Arial, sans-serif;
  --body: "Atkinson Hyperlegible Next", -apple-system, "Segoe UI", system-ui, sans-serif;
  --mono: "Atkinson Hyperlegible Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #101216; --surface: #181b20; --ink: #e8eaed; --ink-2: #b1b7bf; --ink-3: #8b929b; --rule: #2a2f37;
    --ours: #5197ea; --rival: #6a7079; --good: #4cc27a; --bad: #f0746f; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #101216; --surface: #181b20; --ink: #e8eaed; --ink-2: #b1b7bf; --ink-3: #8b929b; --rule: #2a2f37;
  --ours: #5197ea; --rival: #6a7079; --good: #4cc27a; --bad: #f0746f; color-scheme: dark;
}
body { background: var(--bg); color: var(--ink); font: 16px/1.55 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 40px 72px; display: grid; gap: 56px; }
section { display: grid; gap: 16px; min-width: 0; }
.prose { max-width: 68ch; }
h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: clamp(30px, 5vw, 46px); line-height: 1.08; font-weight: 700; max-width: 22ch; }
h2 { font-size: clamp(22px, 3vw, 28px); line-height: 1.2; font-weight: 700; }
h3 { font-size: 18px; font-weight: 600; }
p { margin: 0; }
.eyebrow { font: 500 12px/1.4 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3); }
.lede { font-size: 19px; color: var(--ink-2); max-width: 62ch; }
header { display: grid; gap: 16px; }
.draft { border: 1px solid var(--bad); color: var(--bad); padding: 8px 12px; border-radius: 6px; font-weight: 600; }
.caption { color: var(--ink-2); max-width: 68ch; }
.divider { border-top: 1px solid var(--rule); padding-top: 32px; }
.kicker { font: 600 13px/1.4 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); }
/* the one picture */
.ratios { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.rrow, .raxis { display: grid; grid-template-columns: minmax(0, 15rem) minmax(0, 1fr); align-items: center; gap: 12px; }
.rname { font-weight: 600; line-height: 1.25; }
.vs { font-weight: 400; color: var(--ink-3); }
.track { position: relative; height: 26px; }
.gl { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule); }
.gl.one { background: var(--ink-3); }
.bar { position: absolute; top: 6px; height: 14px; background: var(--ours); }
.bar.up { border-radius: 0 4px 4px 0; }
.bar.down { border-radius: 4px 0 0 4px; }
.tipval { position: absolute; top: 3px; font: 600 13px/20px var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
.raxis .track { height: 18px; }
.rtick { position: absolute; transform: translateX(-50%); font: 12px/18px var(--mono); color: var(--ink-3); }
.rrow:hover .bar { filter: brightness(1.12); }
.then { display: block; font: 400 12.5px/1.3 var(--mono); color: var(--ink-3); }
.p4 { position: absolute; top: 2px; bottom: 2px; width: 2px; margin-left: -1px; border-radius: 1px; background: var(--ink-2); }
tr.was th, tr.was td { color: var(--ink-3); }
td.was { color: var(--ink-3); }
.key.was { background: var(--ink-3); }
/* sentences */
.sentences { margin: 0; padding-left: 1.1em; display: grid; gap: 8px; max-width: 72ch; }
.aside31 { max-width: 72ch; color: var(--ink-2); }
/* the honesty box: the one framed element */
.limits { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 20px 22px; display: grid; gap: 12px; }
.limits-list { margin: 0; padding-left: 1.1em; display: grid; gap: 8px; max-width: 76ch; }
.fair { margin: 0; padding-left: 1.1em; display: grid; gap: 6px; max-width: 72ch; }
/* small multiples */
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; color: var(--ink-2); font-size: 14px; align-items: center; }
.lk { display: inline-flex; align-items: center; gap: 8px; }
.lk i { display: inline-block; width: 22px; height: 2px; border-radius: 1px; }
.lk i.ours { background: var(--ours); } .lk i.rival { background: var(--rival); }
.lk i.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--ink-2); }
.panels { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: 16px; }
.panel { position: relative; margin: 0; background: var(--surface); border-radius: 8px; padding: 12px 10px 10px; min-width: 0; outline: none; }
.panel:focus-visible { box-shadow: 0 0 0 2px var(--ours); }
.panel figcaption { display: grid; gap: 2px; padding: 0 4px 4px; }
.ptitle { font: 600 16px/1.3 var(--display); display: flex; gap: 8px; align-items: center; }
.psub { font-size: 13px; color: var(--ink-3); }
.badge { font: 500 11px/1.6 var(--mono); letter-spacing: 0.04em; text-transform: uppercase; color: var(--ink-2);
  border: 1px solid var(--rule); border-radius: 4px; padding: 0 5px; }
.panel svg { display: block; width: 100%; height: auto; overflow: visible; }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .tick { fill: var(--ink-3); font: 10px var(--mono); }
svg .axis { fill: var(--ink-3); font: 10.5px var(--body); }
svg .mark95 { stroke: var(--ink-3); stroke-width: 1; }
svg path { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg path.ours { stroke: var(--ours); } svg path.rival { stroke: var(--rival); }
svg .dot { stroke: var(--surface); stroke-width: 2; }
svg .dot.ours { fill: var(--ours); } svg .dot.rival { fill: var(--rival); }
svg .leader { stroke: var(--rule); stroke-width: 1; }
svg .end { fill: var(--ink-2); font: 11px var(--body); }
svg .end.mine { fill: var(--ink); font-weight: 600; }
svg .cross { stroke: var(--ink-2); stroke-width: 1; }
svg .hit { fill: transparent; cursor: crosshair; }
.pnote { font-size: 12.5px; color: var(--ink-3); padding: 2px 4px 0; }
.readout { position: absolute; top: 44px; z-index: 2; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; padding: 8px 10px; font-size: 13px; box-shadow: 0 4px 16px #0000001f; pointer-events: none; min-width: 150px; }
.readout .rh { color: var(--ink-3); margin-bottom: 4px; font-family: var(--mono); font-size: 12px; }
.readout .rr { display: grid; grid-template-columns: 16px auto 1fr; gap: 6px; align-items: center; }
.readout .rr i { height: 2px; border-radius: 1px; }
.readout .rr i.ours { background: var(--ours); } .readout .rr i.rival { background: var(--rival); }
.readout .rr b { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.readout .rr span { color: var(--ink-2); }
/* tables */
.tablewrap { overflow-x: auto; background: var(--surface); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
td { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.set { font-family: var(--body); white-space: normal; min-width: 16rem; color: var(--ink-2); }
thead th { font-weight: 600; color: var(--ink-2); font-size: 13px; }
tr.group th { padding-top: 16px; font: 600 15px var(--display); border-bottom-color: var(--ink-3); }
tbody th { font-weight: 400; white-space: nowrap; }
.key { display: inline-block; width: 14px; height: 2px; margin-right: 8px; vertical-align: middle; border-radius: 1px; }
.key.ours { background: var(--ours); } .key.rival { background: var(--rival); }
td.ok { color: var(--good); font-family: var(--body); font-weight: 600; }
td.miss { color: var(--bad); font-family: var(--body); font-weight: 600; }
.note { font-size: 14px; color: var(--ink-3); max-width: 76ch; }
.expect { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; max-width: 90ch; }
.expect li { display: grid; grid-template-columns: 6.5rem minmax(0, 1fr); gap: 2px 12px; }
.expect .mark { font-weight: 600; grid-row: span 2; }
.expect .right .mark { color: var(--good); } .expect .wrong .mark { color: var(--bad); } .expect .open .mark { color: var(--ink-3); }
.expect .what { color: var(--ink-3); font: 13px/1.5 var(--mono); overflow-wrap: anywhere; }
footer { color: var(--ink-3); font-size: 14px; display: grid; gap: 8px; max-width: 80ch; }
a { color: inherit; text-underline-offset: 2px; }
@media (max-width: 640px) {
  .rrow, .raxis { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .raxis .rname { display: none; }
  .lede { font-size: 17px; }
  .expect li { grid-template-columns: minmax(0, 1fr); }
  .expect .mark { grid-row: auto; }
}
@media (prefers-reduced-motion: no-preference) { .bar { transition: filter .15s; } }
"""

JS = """
(() => {
  const data = JSON.parse(document.getElementById('curves').textContent);
  const fmt = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });
  const costAt = (s, a) => {            // the read-off rule of the results files (common.cost_at_target)
    const i = s.r.findIndex(v => v >= a);
    if (i < 0) return null;
    if (i === 0) return s.c[0];
    const [r0, r1, c0, c1] = [s.r[i - 1], s.r[i], s.c[i - 1], s.c[i]];
    return r1 === r0 ? c1 : c0 + (a - r0) * (c1 - c0) / (r1 - r0);
  };
  document.querySelectorAll('figure.panel').forEach(fig => {
    const svg = fig.querySelector('svg'), hit = svg.querySelector('.hit'), cross = svg.querySelector('.cross');
    const out = fig.querySelector('.readout'), series = data[fig.dataset.ds] || [];
    const x0 = +svg.dataset.x0, x1 = +svg.dataset.x1, pl = +hit.getAttribute('x'), pw = +hit.getAttribute('width');
    let cur = null;
    const hide = () => { out.hidden = true; cross.setAttribute('visibility', 'hidden'); cur = null; };
    const show = a => {
      cur = a = Math.min(x1, Math.max(x0, a));
      const x = pl + (a - x0) / (x1 - x0) * pw;
      cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
      const head = document.createElement('div'); head.className = 'rh';
      head.textContent = `${(a * 100).toFixed(1)}% of the true 10 found`;
      const rows = series.map(s => {
        const c = costAt(s, a), row = document.createElement('div'); row.className = 'rr';
        const k = document.createElement('i'); k.className = s.ours ? 'ours' : 'rival';
        const v = document.createElement('b'); v.textContent = c ? `${fmt.format(1 / c)}/s` : 'never';
        const l = document.createElement('span'); l.textContent = s.label;
        row.append(k, v, l); return row;
      });
      out.replaceChildren(head, ...rows); out.hidden = false;
      const fr = fig.getBoundingClientRect(), hr = hit.getBoundingClientRect();
      const px = hr.left - fr.left + (a - x0) / (x1 - x0) * hr.width, w = out.offsetWidth;
      out.style.left = `${px > fr.width / 2 ? Math.max(4, px - w - 10) : Math.min(fr.width - w - 4, px + 10)}px`;
    };
    hit.addEventListener('pointermove', e => {
      const r = hit.getBoundingClientRect();
      show(x0 + (e.clientX - r.left) / r.width * (x1 - x0));
    });
    hit.addEventListener('pointerdown', e => {
      const r = hit.getBoundingClientRect();
      show(x0 + (e.clientX - r.left) / r.width * (x1 - x0));
    });
    hit.addEventListener('pointerleave', hide);
    fig.addEventListener('focus', () => show(0.95));
    fig.addEventListener('blur', hide);
    fig.addEventListener('keydown', e => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { show((cur ?? 0.95) + (e.key === 'ArrowRight' ? 0.005 : -0.005)); e.preventDefault(); }
      if (e.key === 'Escape') hide();
    });
  });
})();
"""


def build(ds, extra, missing):
    N = Nums()
    done = [d for d in ds.values() if d["ratio"]]
    counts = {v: sum(verdict(d["ratio"]) == v for d in done) for v in ("faster", "level", "slower")}
    total = len(ORDER)
    m = extra["machine"]
    q = extra["questions"]
    items = [d["items"] for d in ds.values()]
    h1 = (f'Fastest on {N("summary.faster", str(counts["faster"]))} of {N("summary.datasets", str(total))} datasets '
          f"at full size")
    L, S = counts["level"], counts["slower"]
    names = lambda v: ", ".join(NAME[d["name"]] for d in done if verdict(d["ratio"]) == v)
    parts = ([f'level with the fastest rival on {N("summary.level", str(L))} ({names("level")})'] if L else []) + (
        [f'behind on {N("summary.slower", str(S))} ({names("slower")})'] if S else [])
    rest = f' It is {" and ".join(parts)}.' if parts else ""
    floors = any(v["floor"] for d in ds.values() for v in d["rivals"].values())
    clean = all(c["deleted_returned"] == 0 and c["stale_returned"] == 0
                for d in ds.values() if d["changes"] for c in d["changes"]["checkpoints"])
    cores = extra["docker"].split(" cores")[0].split()[-1] if " cores" in extra["docker"] else None
    g = [p for t in ("hnsw", "glass", "roargraph") for per in extra["tried"][t].values() for p in per]
    sc = extra["tried"]["scann"]
    words = lambda xs: (", ".join(map(str, xs[:-1])) + f" or {xs[-1]}" if len(xs) > 1 else str(xs[0])) if xs else "?"
    tried = (f'Settings tried: {N("tried.links", words(sorted({p["links"] for p in g})))} links with build lists of '
             f'{N("tried.build", words(sorted({p["build"] for p in g})))} (HNSW, Glass, RoarGraph); for ScaNN, '
             + "; ".join(f'{esc(NAME[n])} {N(f"tried.buckets.{n}", words([f"{b:,}" for b in sorted({p["leaves"] for p in per})]))}'
                         for n, per in sc.items() if per)
             + f' buckets, with {N("tried.dims", words(sorted({p["dims_per_block"] for per in sc.values() for p in per})))}'
             " numbers per code.")
    curves = {d["name"]: [{"label": TOOL[t], "ours": t == "ours", "r": v["curve"][0], "c": v["curve"][1]}
                          for t, v in [("ours", d["ours"]), *d["rivals"].items()]] for d in ds.values()}
    order = sorted(done, key=lambda d: -d["ratio"]) + [d for d in ds.values() if not d["ratio"]]
    versions = ", ".join(v.replace(".dist-info", "").replace("-", " ").replace("_py", "")
                         for v in extra["frozen"]["rival_versions"])
    sentence31 = (f'<p class="aside31">In part 4: <span data-src="sentence31">{esc(extra["sentence31"])}</span></p>'
                  if extra["sentence31"] else "")
    draft =(f'<p class="draft">Draft: rival runs still going ({len(missing)} results missing). Numbers will change.</p>'
             if missing else "")
    body = f"""
<div class="wrap">
<header>
  <p class="eyebrow">vindex &middot; benchmark results</p>
  <h1>{h1}</h1>
  <p class="lede">vindex is a search index for AI embeddings, the lists of numbers that stand for texts, photos or
  words. Given a question, it finds the 10 closest items without checking them all. Here it races the fastest
  rivals on {N("summary.datasets", str(total))} public datasets of {N("summary.items_min", f_items(min(items)))} to
  {N("summary.items_max", f_items(max(items)))} items.{rest}</p>
  <p class="lede">All on one Apple-silicon Mac, which favours vindex: read the box below before trusting the numbers.
  These are the final version’s numbers (part 5); an earlier version’s (part 4) are shown beside.</p>
  {draft}
</header>

<section aria-labelledby="h-picture">
  <h2 id="h-picture">How many times as fast as the fastest rival</h2>
  <p class="caption">Questions answered per second on one processor core, when each index finds 95% of the true 10
  closest items. Right of the line, we are faster. The thin mark is part 4’s build.</p>
  {ratio_chart(ds, N)}
</section>

<section aria-labelledby="h-first" class="limits">
  <h2 id="h-first">Read this before trusting the numbers</h2>
  <ul>
    <li><b>The machine favours vindex.</b> vindex has hand-written vector code for Apple chips (NEON) and was built
      with profile-guided optimization. The rivals were written for Intel and AMD chips first. HNSW (hnswlib) has no
      vector code for Apple chips at all, so its numbers here say little. Glass’s own Apple-chip code is switched off
      upstream, so it runs plain loops that the compiler vectorizes. RoarGraph’s Intel code runs through a
      translation layer, and ScaNN in a Linux container. On an Intel or AMD server the results may differ.</li>
    <li><b>Not every strong index was tried.</b> SymphonyQG needs Intel’s AVX-512. NGT-QG, LoRANN and Faiss were not
      run. “Fastest rival” means the fastest of Glass, ScaNN, RoarGraph, CoreNN and HNSW.</li>
    <li><b>vindex learns from sample questions</b> drawn from the same source as the test questions, for its search
      effort and its extra links. RoarGraph uses them too; Glass, ScaNN and HNSW do not.</li>
    <li><b>Held back for one round each.</b> The index was built in five parts. Each held-back dataset was kept out of
      one part, then used like the others; only GloVe-100 stayed locked until the code was frozen.</li>
    <li><b>Rerun it yourself:</b> the rivals’ scripts, versions, patches and every result are in
      <a href="https://github.com/srir84/vector-index-experiments">vector-index-experiments</a>, and ours in
      <a href="https://github.com/srir84/vector-index-rust">vector-index-rust</a>.</li>
  </ul>
</section>

<section aria-labelledby="h-datasets">
  <h2 id="h-datasets">Dataset by dataset</h2>
  <ul class="sentences">{"".join(f"<li>{sentence(d, N)}</li>" for d in order)}</ul>
  {sentence31}
</section>

<section aria-labelledby="h-limits" class="limits">
  <h2 id="h-limits">What we did not beat, and other limits</h2>
  {limits(ds, extra, N)}
</section>

<section aria-labelledby="h-fair">
  <h2 id="h-fair">How the race was kept fair</h2>
  <ul class="fair">
    <li>Every index ran on the same Mac: {N("machine.chip", m["chip"])}, {N("machine.cores", str(m["cores"]))} cores,
      {N("machine.memory_gb", str(m["memory_gb"]))} GB of memory, macOS {N("machine.macos", m["macos"])}.</li>
    <li>For each test question we know the true 10 closest items, found by checking every item. An index finds 95%
      when, on average, 9.5 of its 10 answers are among them.</li>
    <li>Every index picked its settings on {N("questions.tune", f_int(q["tune"]))} tuning questions. The speeds here
      are on {N("questions.test", f_int(q["test"]))} other test questions, read off at exactly 95%, the middle of 5
      timed runs.</li>
    <li>Our code and the rivals’ builds were frozen, with a fingerprint of every file, before the full-size runs, in
      part 4 and again in part 5. GloVe-100 stayed locked until then.</li>
    <li>Before part 5’s changes we wrote down what we expected. Each guess is marked right or wrong below.</li>
  </ul>
</section>

<section aria-labelledby="h-tech" class="divider">
  <p class="kicker">For technical readers</p>
  <h2 id="h-tech">Speed against accuracy, every index</h2>
  <p class="caption">Each line is one index, from its fastest to its most careful search. The dot is the speed at
  exactly 95%. Hover a chart, or focus it and use the arrow keys, to read every index at one accuracy.</p>
  <div class="legend"><span class="lk"><i class="ours"></i>vindex (ours)</span>
    <span class="lk"><i class="rival"></i>rivals, named at each line’s end</span>
    <span class="lk"><i class="dot"></i>speed at 95%</span>
    <span class="lk"><span class="badge">held back</span>not used while the index was built</span></div>
  <div class="panels">{"".join(panel(ds[n], N) for n in ORDER)}</div>
</section>

<section aria-labelledby="h-table">
  <h2 id="h-table">Every number behind the charts</h2>
  <p class="caption">One core at 95% unless marked. Memory is counted from what each index holds; for ours, both
  ways it can run: everything in memory (the speeds above), or the full numbers left on the SSD and read when needed.</p>
  {speed_table(ds, N)}
  <p class="note">All cores: ours used {N("machine.cores", str(m["cores"]))} threads{
  f'; ScaNN’s container had {N("docker.cores", cores)} cores' if cores else ""}. <sup>p</sup> CoreNN’s
  memory is its process’s after the build, since it keeps its items in a database on disk.{
  " <sup>q</sup> At its quickest setting, which already finds more than 95%." if floors else ""}</p>
</section>

<section aria-labelledby="h-changes">
  <h2 id="h-changes">Adding, deleting and updating items</h2>
  <p class="caption">Each index was built on 90% of its items, then 10% more were added, 10% deleted and 10%
  updated, in one mixed order. Its accuracy was then compared with a fresh build of the final items. The rule:
  drift under 1 point, and no deleted or outdated item ever returned. {"None ever came back." if clean else
  "Some came back; the rule column shows where."}</p>
  {changes_table(ds, N)}
</section>

<section aria-labelledby="h-expected">
  <h2 id="h-expected">What we expected, written before part 5’s changes</h2>
  {expectations(extra, ds, N)}
  <p class="note">Two more, about which choices flip and why Landmark drifted, are marked in the project’s step
  notes. Part 4’s expectations are in its report.</p>
</section>

<footer>
  <p>Rivals: {N("versions", versions)}; ScaNN {N("version.scann", extra["scann_version"])} in a Linux container;
  RoarGraph from its source code, commit {N("version.roargraph", extra["roargraph_commit"])}. {tried}</p>
  <p>Datasets: vector files from VIBE (CC BY 4.0) and ann-benchmarks. Credits and licences per dataset are in each
  dataset’s card. Every number on this page comes from the project’s results files and is checked by
  <code>validate/step33.py</code>.</p>
</footer>
</div>
<script type="application/json" id="curves">{json.dumps(curves, separators=(",", ":"))}</script>
<script>{JS}</script>
"""
    head = ('<title>vindex at Full Size</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<meta name="description" content="Full-size speed results for the vindex search index against its fastest rivals.">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Mono:wght@400;500;600'
            '&family=Atkinson+Hyperlegible+Next:wght@400;600;700&family=Schibsted+Grotesk:wght@600;700&display=swap">\n'
            f"<style>{CSS}</style>\n")
    return head + body, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    ds, extra, missing = load(a.allow_missing)
    page, N = build(ds, extra, missing)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KB, {len(N.keys)} numbers{', DRAFT: ' + str(len(missing)) + ' rival results missing' if missing else ''})")


if __name__ == "__main__":
    main()
