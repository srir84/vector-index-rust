"""Steps 33 and 43: the results page (part 5, with part 4's numbers beside).
  .venv/bin/python -I tools/results_page.py      # first: build page/results.html from the results files
  .venv/bin/python -I validate/step33.py [--no-browser]
Checks, with its own reading of the results files (not the page builder's):
  - every number on the page (each element with a data-n key) is recomputed and printed the same way;
  - every passage copied from a results file (data-src) is word for word that file's;
  - no other number is on the page, apart from the method's own constants (95%, the top 10, 5 timed runs ...) and
    the charts' axis ticks;
  - the hover data holds each index's speed-accuracy line exactly as the results files give it;
  - in Chrome, at phone width (390 px) and desktop width, light and dark: no errors, no sideways scroll, and the
    hover readout works.
Pass: all of the above, on a page that is not a draft.
"""
import html
import json
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, EXP, OUT, ROOT, Report, frozen_changes
from step30 import ALL, FULL, ours_full, rival
from step42 import EVERY, FROZEN5, FULL7, NEW, part5_full

PAGE = ROOT / "page" / "results.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PLAN = {"glass": ["gooaq", "yandex", "landmark", "arxiv", "laion", "gist", "nytimes", "imagenet", NEW],
        "scann": ["glove", "nytimes", NEW], "roargraph": ["llama", "yi", "llama2"], "corenn": ["gist"], "hnsw": EVERY}
WHERE = lambda n: FULL7 if n == NEW else FULL   # GloVe-100's rivals ran in part 5, the others' in part 4
TOOL = {"hnsw": "HNSW", "glass": "Glass", "scann": "ScaNN", "roargraph": "RoarGraph", "corenn": "CoreNN"}
NAME = {"gooaq": "GooAQ", "glove": "GloVe", "yandex": "Yandex", "llama": "Llama", "landmark": "Landmark", "arxiv": "arXiv",
        "laion": "LAION", "yi": "Yi", "gist": "GIST1M", "nytimes": "NYTimes", "imagenet": "ImageNet",
        "llama2": "Llama, 2nd head", NEW: "GloVe-100"}
LEVEL = 0.05
# The method's own constants, the only numbers allowed outside a data-n element (each phrase is cut out of the
# visible text before it is searched for numbers).
CONSTANTS = ["part 4", "Part 4", "part 5", "Part 5", "GloVe-100", "95%", "true 10", "10 closest", "of its 10 answers", "9.5", "middle of 5 timed runs",
             "Llama 3", "AVX-512", "CC BY 4.0", "90% of its items", "10% more were added, 10% deleted and 10%",
             "drift under 1 point", "Within 5%", "never reaches 85%", "at least 10\u00d7 as fast",
             "within 1.3\u00d7 of the fastest"]
NUMBER = re.compile(r"(?<![A-Za-z0-9])[−+-]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")


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


def words(xs):
    return (", ".join(map(str, xs[:-1])) + f" or {xs[-1]}" if len(xs) > 1 else str(xs[0])) if xs else "?"


def line(points):
    """The speed-accuracy line as common.cost_at_target reads it: by time, best accuracy so far."""
    pts = sorted(points, key=lambda p: p[1])
    r, best = [], -1.0
    for a, _ in pts:
        best = max(best, a)
        r.append(best)
    return r, [s for _, s in pts]


def picks_text(o):
    t, st = o["tuning"], o["stats"]
    return (f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, {t['chosen_code_bits']}-bit copies, "
            f"re-check {t['chosen_rerank']}" + (", question links" if st["question_links"] else "")
            + (", extra links for popular items" if st["popular_items"] else ""))


def ram_of(mo):
    return (mo["total_bytes"] + mo["full_numbers_on_ssd_bytes"]) / mo["items"]


def change_runs(name):
    return {c["data"]["dataset"]: (c["data"], c["pass"]) for c in json.loads((OUT / name).read_text())["checks"]
            if isinstance(c.get("data"), dict) and "dataset" in c["data"]}


def expected():
    """Every data-n key the page may hold, with its text, and every data-src passage, from the results files."""
    e, src, curves = {}, {}, {}
    ours, ours4 = part5_full(), ours_full()
    s42, s32 = change_runs("step42-changes.json"), change_runs("step32.json")
    cards = {n: json.loads((DATA / n / "card.json").read_text()) for n in EVERY}
    ratios, mem_vs, mem_x, slow_x, others = {}, [], [], [], []
    for n in EVERY:
        o, c = ours[n], cards[n]
        mo, ss = o["memory"], o["ssd"]
        e[f"{n}.items"] = f_items(c["items_full"])
        e[f"{n}.dim"] = str(np.load(DATA / n / "items_full.npy", mmap_mode="r").shape[1])
        e[f"{n}.ours.qps"] = f_int(o["ours"]["qps95"])
        e[f"{n}.ours.qps_all"] = f_int(o["ours"]["qps_all"])
        e[f"{n}.ours.build"] = f_time(o["build_seconds"])
        ram, ssd = ram_of(mo), mo["total_bytes"] / mo["items"]
        e[f"{n}.ours.mem_ram"], e[f"{n}.ours.mem_ssd"] = f_int(ram), f_int(ssd)
        e[f"{n}.ours.ssd_qps"], e[f"{n}.ours.ssd_qps_all"] = f_int(ss["qps"]), f_int(ss["qps_all"])
        e[f"{n}.ours.picks"] = picks_text(o)
        if (o4 := ours4.get(n)):   # part 4's build, shown beside
            e[f"{n}.p4.qps"], e[f"{n}.p4.qps_all"] = f_int(o4["ours"]["qps95"]), f_int(o4["ours"]["qps_all"])
            e[f"{n}.p4.mem_ram"], e[f"{n}.p4.build"] = f_int(ram_of(o4["memory"])), f_time(o4["build_seconds"])
            e[f"{n}.p4.picks"] = picks_text(o4)
        mem_x.append(ram / ssd)
        slow_x.append(o["ours"]["qps95"] / ss["qps"])
        others += o["ours"]["other_programs_cores"]
        curves[n] = [("vindex", True, line([(p["accuracy"], p["sec"]) for p in o["ours"]["curve"]]))]
        best = None
        for tool, names in PLAN.items():
            if n not in names or not (r := rival(tool, n, WHERE(n))):
                continue
            raw = json.loads((WHERE(n) / "final" / tool / n / "best.json").read_text())
            curves[n].append((TOOL[tool], False, line([(p["accuracy"], p["sec_per_query_1core"]) for p in raw["curve"]])))
            k = f"{n}.{tool}"
            if r["qps95"]:
                e[f"{k}.qps"] = f_int(r["qps95"])
                if best is None or r["qps95"] > best[1]["qps95"]:
                    best = (tool, r)
            e[f"{k}.best"] = f_pct(r["best"])
            if r["status"] == "below_min_effort":   # already past 95% at its quickest setting: that setting's accuracy
                e[f"{k}.floor"] = f_pct(min(raw["curve"], key=lambda p: p["sec_per_query_1core"])["accuracy"])
            e[f"{k}.qps_all"], e[f"{k}.mem"], e[f"{k}.build"] = f_int(r["qps_all"]), f_int(r["bytes_per_item"]), f_time(r["build_s"])
            p = r["params"]
            e[f"{k}.settings"] = (f"{p['leaves']:,} buckets, {p['dims_per_block']} number{'s' if p['dims_per_block'] > 1 else ''} "
                                  "per code" if "leaves" in p else f"{p['links']} links, build list {p['build']}"
                                  + (f", slack {p['slack']}" if p.get("slack") else ""))
        if best:
            ratios[n] = o["ours"]["qps95"] / best[1]["qps95"]
            e[f"{n}.ratio"], e[f"{n}.inverse"] = f_x(ratios[n]), f_x(1 / ratios[n])
            if n in ours4:
                e[f"{n}.ratio4"] = f_x(ours4[n]["ours"]["qps95"] / best[1]["qps95"])
            mem_vs.append(ram / best[1]["bytes_per_item"])
        b, ok = s42[n]
        e[f"{n}.changes.count"] = f_int(b["changes"])
        e[f"{n}.changes.after"] = f_pct(b["checkpoints"][-1]["accuracy"])
        e[f"{n}.changes.fresh"] = f_pct(b["fresh_accuracy"])
        e[f"{n}.changes.drift"] = f_pts(b["checkpoints"][-1]["accuracy"] - b["fresh_accuracy"])
        if n in s32:
            b4 = s32[n][0]
            e[f"{n}.changes4.drift"] = f_pts(b4["checkpoints"][-1]["accuracy"] - b4["fresh_accuracy"])
    v = {k: sum((r >= 1 + LEVEL) if k == "faster" else (r <= 1 - LEVEL) if k == "slower" else (1 - LEVEL < r < 1 + LEVEL)
                for r in ratios.values()) for k in ("faster", "level", "slower")}
    e.update({f"summary.{k}": str(x) for k, x in v.items()})
    e["summary.datasets"] = str(len(EVERY))
    e["summary.items_min"] = f_items(min(c["items_full"] for c in cards.values()))
    e["summary.items_max"] = f_items(max(c["items_full"] for c in cards.values()))
    qs = {json.dumps(c["questions"]["tune"]) + json.dumps(c["questions"]["test"]) for c in cards.values()}
    assert len(qs) == 1, "the datasets differ in question counts"
    e["questions.tune"], e["questions.test"] = f_int(cards["gooaq"]["questions"]["tune"]), f_int(cards["gooaq"]["questions"]["test"])
    e["slice"] = f_int(cards["gooaq"]["slice_items"])
    m = json.loads((OUT / "machine.json").read_text())
    e.update({"machine.chip": m["chip"], "machine.cores": str(m["cores"]), "machine.memory_gb": str(m["memory_gb"]),
              "machine.macos": m["macos"]})
    e.update({"limits.vs_lo": f"{min(mem_vs):.2f}" if mem_vs else None, "limits.vs_hi": f"{max(mem_vs):.2f}" if mem_vs else None,
              "limits.mem_lo": f"{min(mem_x):.1f}", "limits.mem_hi": f"{max(mem_x):.1f}",
              "limits.slow_lo": f"{min(slow_x):.0f}", "limits.slow_hi": f"{max(slow_x):.0f}",
              "limits.others_max": f"{max(others):.1f}"})
    # how steady fresh builds are: part 4's rebuilds (step 30b), step 37's three builds per sample before and after the
    # fix (same picks = the same chosen settings in all three), and Yandex: step 37's full-size builds against step 42's
    rc = [c["data"] for c in json.loads((OUT / "step30b.json").read_text())["checks"] if c.get("data")]
    e["steady.rebuild_move"] = f"{max(abs(r['ours']['qps95'] / ours4[r['dataset']]['ours']['qps95'] - 1) for r in rc):.1%}"
    keyset = lambda p: json.dumps(p, sort_keys=True)
    before = json.loads((OUT / "step37-before.json").read_text())
    e["steady.before"] = str(sum(len({keyset(b["picks"]) for b in bs}) == 1 for bs in before.values()))
    s37 = json.loads((OUT / "step37.json").read_text())["checks"]
    builds = {c["check"].split(":")[0]: c["data"]["builds"] for c in s37 if "same settings" in c["check"]}
    steady = [n for n, bs in builds.items() if len({keyset(b["picks"]) for b in bs}) == 1]
    e["steady.now"], e["steady.samples"] = str(len(steady)), str(len(builds))
    secs = {c["check"].split(":")[0]: c["data"]["sec_per_question"] for c in s37 if "of each other" in c["check"]}
    e["steady.apart"] = f"{max(max(secs[n]) / min(secs[n]) - 1 for n in steady):.1%}"
    y = next(c["data"]["builds"] for c in json.loads((OUT / "step37-full.json").read_text())["checks"]
             if c["check"].startswith("yandex:") and "same settings" in c["check"])
    e["steady.yandex_slower"] = f"{1 - ours['yandex']['ours']['qps95'] / min(b['qps95'] for b in y):.1%}"
    dk = (OUT / "step30-docker.txt").read_text()
    e["docker.cores"] = re.search(r"Docker VM: (\d+) cores", dk).group(1)
    grid = {t: {n: [json.loads(f.read_text())["params"] for f in sorted((WHERE(n) / "grid" / t / n).glob("*.json"))]
                for n in names} for t, names in PLAN.items()}
    counts = lambda ts: sorted({len(p) for t in ts for p in grid[t].values() if p})
    span = lambda s: (str(s[0]) if len(s) == 1 else f"{s[0]}–{s[-1]}") if s else "?"
    e["tried.graph"], e["tried.scann"] = span(counts(["hnsw", "glass", "roargraph"])), span(counts(["scann"]))
    g = [p for t in ("hnsw", "glass", "roargraph") for per in grid[t].values() for p in per]
    e["tried.links"], e["tried.build"] = words(sorted({p["links"] for p in g})), words(sorted({p["build"] for p in g}))
    e["tried.dims"] = words(sorted({p["dims_per_block"] for per in grid["scann"].values() for p in per}))
    for n, per in grid["scann"].items():
        e[f"tried.buckets.{n}"] = words([f"{b:,}" for b in sorted({p["leaves"] for p in per})])
    fz = json.loads(FROZEN5.read_text())
    e["versions"] = ", ".join(re.sub(r"^(\w+?)(?:_py)?-([\d.]+)\.dist-info$", r"\1 \2", x) for x in fz["rival_versions"])
    vtext = (EXP / "baselines" / "VERSIONS.md").read_text()
    e["version.scann"] = re.search(r"scann (\d+\.\d+\.\d+) from PyPI", vtext).group(1)
    e["version.roargraph"] = re.search(r"matchyc/RoarGraph commit ([0-9a-f]{7})", vtext).group(1)
    if (OUT / "step31-sentence.txt").exists():
        src["sentence31"] = (OUT / "step31-sentence.txt").read_text().strip()
    e["exp.changes.passed"], e["exp.changes.all"] = str(sum(ok for _, ok in s42.values())), str(len(s42))
    for i, x in enumerate(json.loads((OUT / "step42-expectations.json").read_text())):
        src[f"expectations.{i}.expected"], src[f"expectations.{i}.what"] = x["expected"], x["what happened"]
    return e, src, curves


class Page(HTMLParser):
    """Collects the keyed elements' text, the copied passages, the other visible text, and the hover data."""

    SKIP = {"script", "style", "title"}
    VOID = {"meta", "link", "br", "hr", "img", "input", "source", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.nums, self.srcs, self.free, self.ticks, self.json = [], {}, {}, [], [], None
        self.cur_json = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.VOID:
            return
        self.stack.append((tag, a.get("data-n"), a.get("data-src"), a.get("class") or "", []))
        if tag == "script" and a.get("id") == "curves":
            self.cur_json = []

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        while self.stack:
            t = self.stack.pop()
            if t[0] == "script" and self.cur_json is not None:
                self.json, self.cur_json = "".join(self.cur_json), None
            if t[1]:
                self.nums.setdefault(t[1], []).append("".join(t[4]))
            elif t[2]:
                self.srcs.setdefault(t[2], []).append(" ".join("".join(t[4]).split()))
            if t[0] == tag:
                break

    def handle_data(self, data):
        if self.cur_json is not None:
            self.cur_json.append(data)
            return
        if any(t[0] in self.SKIP for t in self.stack):
            return
        own = next((t for t in reversed(self.stack) if t[1] or t[2]), None)
        if own:
            own[4].append(data)
        elif any("tick" in t[3].split() or "rtick" in t[3].split() for t in self.stack):
            self.ticks.append(data)
        else:
            self.free.append(data)


def parse(text):
    p = Page()
    p.feed(text)
    return p


def browser(page_text):
    """Chrome, headless: per width and theme, errors, sideways scroll and the first chart's readout."""
    head = ("<script>window.__e=[];addEventListener('error',e=>__e.push(String(e.message||(e.target&&(e.target.src||"
            "e.target.href))||'error')),true);(()=>{const c=console.error;console.error=(...a)=>{__e.push(a.map(String)"
            ".join(' '));c.apply(console,a)}})();</script>")
    tail = ("<script>addEventListener('load',()=>setTimeout(()=>{const d=document.documentElement,f=document."
            "querySelector('figure.panel');let ro=null;if(f){f.focus();const o=f.querySelector('.readout');ro={shown:!o."
            "hidden,rows:o.querySelectorAll('.rr').length,series:(JSON.parse(document.getElementById('curves')."
            "textContent)[f.dataset.ds]||[]).length}}const p=document.createElement('pre');p.id='__check';p.textContent="
            "JSON.stringify({width:innerWidth,scroll:d.scrollWidth,errors:__e,bg:getComputedStyle(document.body)."
            "backgroundColor,readout:ro});document.body.append(p)},400))</script>")
    # the skeleton the artifact host wraps a page in (charset, viewport, a small reset)
    skel = ('<!doctype html><html lang="en" data-theme="{theme}"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{{color-scheme:light;'
            'padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}body{{margin:0;'
            'font:14px system-ui,sans-serif}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>{head}'
            '</head><body>{page}{tail}</body></html>')
    # headless Chrome keeps its window at least 500 px wide, so the page runs in a frame of exactly the width tested
    # (a srcdoc frame shares the wrapper's origin, so the wrapper can copy the frame's result out for --dump-dom)
    frame = ('<!doctype html><html><body style="margin:0"><iframe id="f" style="width:{width}px;height:900px;border:0" '
             'srcdoc="{doc}"></iframe><script>setTimeout(()=>{{const d=document.getElementById("f").contentDocument,'
             'c=d&&d.getElementById("__check"),p=document.createElement("pre");p.id="__check";p.textContent=c?c.'
             'textContent:"{{}}";document.body.append(p)}},3000)</script></body></html>')
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for theme in ("light", "dark"):
            doc = html.escape(skel.format(theme=theme, head=head, page=page_text, tail=tail), quote=True)
            for width in (390, 1280):
                f = Path(tmp) / f"{theme}-{width}.html"
                f.write_text(frame.format(width=width, doc=doc))
                r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                                    f"--user-data-dir={tmp}/profile", f"--window-size={max(width, 500) + 40},960",
                                    "--virtual-time-budget=6000", "--dump-dom", f.as_uri()],
                                   capture_output=True, text=True, timeout=120)
                m = re.search(r'<pre id="__check">(.*?)</pre>', r.stdout, re.S)
                out[(theme, width)] = json.loads(html.unescape(m.group(1))) if m else {"failed": r.stderr[-500:]}
    return out


def main():
    rep = Report("step33")
    changed = frozen_changes(frozen=FROZEN5)
    rep.check("the code is as frozen in step 42", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    text = PAGE.read_text()
    p = parse(text)
    rep.check("the page is complete, not a draft", 'class="draft"' not in text and "<title>" in text[:8192],
              "no draft banner; title in the first 8 KB" if 'class="draft"' not in text else "it is a draft")
    e, src, curves = expected()
    shown = p.nums
    wrong = [f"{k}: page {t!r}, files {e.get(k)!r}" for k, ts in shown.items() for t in ts if e.get(k) != t]
    rep.check("every number on the page is recomputed from the results files and matches", shown and not wrong,
              f"{sum(map(len, shown.values()))} numbers ({len(shown)} different), {len(wrong)} wrong"
              + ("".join(f"\n      {w}" for w in wrong[:12])))
    must = [f"{n}.{w}" for n in EVERY for w in ("items", "ours.qps", "ours.mem_ram", "ours.mem_ssd", "changes.drift")]
    must += [f"{n}.{w}" for n in ALL for w in ("p4.qps", "ratio4", "changes4.drift")]
    must += [f"{n}.ratio" for n in EVERY if f"{n}.ratio" in e] + [f"{n}.{t}.mem" for t, ns in PLAN.items() for n in ns]
    absent = [k for k in must if k not in shown]
    rep.check("the page shows every dataset's headline numbers", not absent, "all there" if not absent else ", ".join(absent[:10]))
    psrc = {k: v[0] if len(v) == 1 else v for k, v in p.srcs.items()}
    bad = [k for k in set(src) | set(psrc) if psrc.get(k) != (" ".join(src[k].split()) if k in src else None)]
    rep.check("every passage copied from a results file is word for word", not bad,
              f"{len(psrc)} passages" + (f"; differ: {', '.join(sorted(bad)[:8])}" if bad else ""))
    free = " ".join(" ".join(p.free).split())
    for c in CONSTANTS:
        free = free.replace(c, " ")
    stray = sorted(set(NUMBER.findall(free)))
    tick_ok = all(re.fullmatch(r"\s*(\d+(\.\d+)?[KM]?|\d+%|[\d.]+×)?\s*", t) for t in p.ticks)
    rep.check("no other number is on the page (only the method's constants and axis ticks)", not stray and tick_ok,
              f"{len(p.ticks)} axis ticks" + (f"; unexplained: {', '.join(stray[:12])}" if stray else "")
              + ("" if tick_ok else "; a tick is not an axis value"))
    page_curves = json.loads(p.json) if p.json else {}
    diff = []
    for n, series in curves.items():
        got = page_curves.get(n, [])
        if [(s["label"], s["ours"], s["r"], s["c"]) for s in got] != [(lab, mine, r, c) for lab, mine, (r, c) in series]:
            diff.append(n)
    rep.check("the hover data is each index's speed-accuracy line from the results files", page_curves and not diff,
              f"{sum(len(s) for s in curves.values())} lines on {len(curves)} datasets" + (f"; differ: {diff}" if diff else ""))
    if "--no-browser" not in sys.argv:
        b = browser(text)
        probs = []
        for (theme, width), r in b.items():
            if "failed" in r:
                probs.append(f"{theme} {width}px: did not load ({r['failed'][:120]})")
                continue
            if r["errors"]:
                probs.append(f"{theme} {width}px errors: {r['errors'][:3]}")
            if r["scroll"] > r["width"]:
                probs.append(f"{theme} {width}px: {r['scroll']} px wide in a {r['width']} px window")
            ro = r["readout"]
            if not ro or not ro["shown"] or ro["rows"] != ro["series"]:
                probs.append(f"{theme} {width}px: readout {ro}")
        bgs = {r.get("bg") for (t, w), r in b.items() if w == 390}
        if len(bgs) != 2:
            probs.append(f"light and dark share a background: {bgs}")
        rep.check("in Chrome, phone and desktop widths, light and dark: no errors, no sideways scroll, the readout works",
                  not probs, "; ".join(f"{t} {w}px {r.get('scroll')}/{r.get('width')} px" for (t, w), r in b.items())
                  + ("".join(f"\n      {x}" for x in probs)))
    rep.finish()


if __name__ == "__main__":
    main()
