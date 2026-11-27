"""Draw REPORT.md's speed pictures: searches per second at 95% of the true top 10, one core, ours against the six
rivals, one panel per dataset.
  part 1: validate/results/step11.json -> diagrams/speed.png
  part 2: validate/results/step22.json (and step 11 for part 1's dots) -> diagrams/speed-part2.png
  part 3: validate/results/step27.json (and step 22 for part 2's dots) -> diagrams/speed-part3.png, one row per
          dataset: our speed over the fastest rival's
  part 4: validate/results/step30.json (and step 27 for the samples' dots) -> diagrams/speed-part4.png, as part 3
          but at full size
  part 5: validate/results/step42-ours.json (and step 30 for the rivals and part 4's dots; GloVe-100's rivals from the
          experiments' results/step7) -> diagrams/speed-part5.png
  readme: the same runs against the fastest other index -> diagrams/benchmark.png, and the README's table printed

Needs matplotlib (the experiments' environment has it):
    ../vector-index-experiments/.venv/bin/python -I tools/draw_report.py [part2 | part3 | part4 | part5 | readme]
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "diagrams" / "speed.png"

# one accent for ours, gray for the rivals (identity comes from the row label, not from colour)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OURS, OURS1, RIVAL = "#2a78d6", "#8fb6e6", "#8d8b85"
ROWS = [("ours", "vindex (ours)"), ("glass", "Glass"), ("scann", "ScaNN"), ("roargraph", "RoarGraph"),
        ("diskann", "DiskANN"), ("hnsw", "HNSW"), ("corenn", "CoreNN")]
DATASETS = [("gooaq", "GooAQ: 768 numbers, angle"), ("glove", "GloVe: 200 numbers, angle"),
            ("yandex", "Yandex: 200 numbers, angle; questions unlike items"),
            ("llama", "Llama: 128 numbers, best score"), ("landmark", "Landmark: 768 numbers, angle")]


HELD_BACK = [("arxiv", "arXiv (held back): 768 numbers, angle"), ("laion", "LAION (held back): 512 numbers, angle; questions unlike items"),
             ("yi", "Yi (held back): 128 numbers, best score"), ("gist", "GIST1M (held back): 960 numbers, straight line")]
ROWS2 = [("ours", "vindex, part 2"), ("part1", "vindex, part 1"), *ROWS[1:]]


def per_dataset(step="step11"):
    rows = json.loads((ROOT / "validate" / "results" / f"{step}.json").read_text())["checks"]
    out = {}
    for r in rows:
        if r.get("data") and "ours" in r["data"]:
            name = r["check"].split(":")[0].split(" ")[0]
            speeds = {t: v["qps95"] for t, v in r["data"]["rivals"].items()}
            speeds["ours"] = r["data"]["ours"]["qps95"]
            out[name] = speeds
    return out


NEW = [("nytimes", "NYTimes (new, held back)"), ("imagenet", "ImageNet (new, held back)"),
       ("llama2", "Llama, 2nd head (new, held back)")]
PLAIN = {"gooaq": "GooAQ", "glove": "GloVe", "yandex": "Yandex", "llama": "Llama", "landmark": "Landmark",
         "arxiv": "arXiv", "laion": "LAION", "yi": "Yi", "gist": "GIST1M"}
RIVAL_NAMES = {"glass": "Glass", "scann": "ScaNN", "roargraph": "RoarGraph", "diskann": "DiskANN", "hnsw": "HNSW",
               "corenn": "CoreNN", "faiss_hnsw": "Faiss HNSW", "faiss_hnsw_sq8": "Faiss HNSW-SQ8",
               "faiss_ivfpq": "Faiss IVF-PQ", "lorann": "LoRANN"}


def part3():
    now, before = per_dataset("step27"), per_dataset("step22")
    rows = [(n, PLAIN[n]) for n, _ in DATASETS + HELD_BACK] + NEW
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, (name, label) in enumerate(rows):
        rv = {t: v for t, v in now[name].items() if t != "ours" and v}
        best = max(rv, key=rv.get)
        r3 = now[name]["ours"] / rv[best]
        r2 = before[name]["ours"] / rv[best] if name in before else None
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        if r2:
            ax.plot([r2, r3], [y, y], color=OURS1, lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([r2], [y], s=46, color=OURS1, zorder=2, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([r3], [y], s=70, color=OURS, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(r3, r2 or 0) * 1.06, y, f"{r3:.2f}x", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")
        labels.append(f"{label} vs {RIVAL_NAMES[best]}")
    ax.axvline(1.0, color=INK2, lw=1, zorder=0)
    ax.text(1.0, -0.9, "level with the fastest rival", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.4, -1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.0)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0)
    ax.scatter([], [], s=70, color=OURS, label="part 3")
    ax.scatter([], [], s=46, color=OURS1, label="part 2")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_xlabel("our speed over the fastest rival's (log scale)")
    fig.suptitle("Our speed against the fastest rival on each dataset", x=0.02, y=0.99, ha="left", va="top",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.945, "At 95% of the true top 10, one core of an M3 Max, test questions, 100,000 items per dataset. "
             "Right of the line: we are faster.", fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_file = OUT.with_name("speed-part3.png")
    fig.savefig(out_file, dpi=160, facecolor=SURFACE)
    print("wrote", out_file)


def part4():
    """One row per dataset: at full size (step 30), our speed over the fastest rival's; and in the samples (step 27),
    over the same rival's."""
    now, before = per_dataset("step30"), per_dataset("step27")
    names = {**PLAIN, "nytimes": "NYTimes", "imagenet": "ImageNet", "llama2": "Llama, 2nd head"}
    rows = [n for n, _ in DATASETS + HELD_BACK + NEW]
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, name in enumerate(rows):
        rv = {t: v for t, v in now[name].items() if t != "ours" and v}
        best = max(rv, key=rv.get)
        full = now[name]["ours"] / rv[best]
        sample = before[name]["ours"] / before[name][best]
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        ax.plot([sample, full], [y, y], color=OURS1, lw=2, zorder=1, solid_capstyle="round")
        ax.scatter([sample], [y], s=46, color=OURS1, zorder=2, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([full], [y], s=70, color=OURS, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(full, sample) * 1.06, y, f"{full:.2f}x", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")
        labels.append(f"{names[name]} vs {RIVAL_NAMES[best]}")
    ax.axvline(1.0, color=INK2, lw=1, zorder=0)
    ax.text(1.0, -0.9, "level with the fastest rival", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.4, -1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.0)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0)
    ax.scatter([], [], s=70, color=OURS, label="full size")
    ax.scatter([], [], s=46, color=OURS1, label="100,000-item sample")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)   # above the rows
    ax.set_xlabel("our speed over the fastest rival's (log scale)")
    fig.suptitle("Our speed against the fastest rival, at full size", x=0.02, y=0.99, ha="left", va="top",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.945, "At 95% of the true top 10, one core of an M3 Max, test questions, every item of each dataset. "
             "Right of the line: we are faster.", fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_file = OUT.with_name("speed-part4.png")
    fig.savefig(out_file, dpi=160, facecolor=SURFACE)
    print("wrote", out_file)


def part5():
    """One row per dataset: part 5's full-size run (step 42) over the fastest rival's, and part 4's (step 30) over the
    same rival's. GloVe-100's rivals are the experiments' results/step7 runs."""
    p4 = per_dataset("step30")
    now = {c["check"].split(":")[0]: c["data"]["ours"]["qps95"]
           for c in json.loads((ROOT / "validate" / "results" / "step42-ours.json").read_text())["checks"]
           if c.get("data") and "at full size" in c["check"]}
    step7 = ROOT.parent / "vector-index-experiments" / "results" / "step7" / "final"
    p4["glove100"] = {t: 1 / r["sec_per_query_at95"] if r.get("sec_per_query_at95") else None
                      for t in ("scann", "glass", "hnsw") for r in [json.loads((step7 / t / "glove100" / "best.json").read_text())]}
    names = {**PLAIN, "nytimes": "NYTimes", "imagenet": "ImageNet", "llama2": "Llama, 2nd head",
             "glove100": "GloVe-100 (new, held back)"}
    rows = [n for n, _ in DATASETS + HELD_BACK + NEW] + ["glove100"]
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, name in enumerate(rows):
        rv = {t: v for t, v in p4[name].items() if t != "ours" and v}
        best = max(rv, key=rv.get)
        r5 = now[name] / rv[best]
        r4 = p4[name]["ours"] / rv[best] if "ours" in p4[name] else None
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        if r4:
            ax.plot([r4, r5], [y, y], color=OURS1, lw=2, zorder=1, solid_capstyle="round")
            ax.scatter([r4], [y], s=46, color=OURS1, zorder=2, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter([r5], [y], s=70, color=OURS, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(r5, r4 or 0) * 1.06, y, f"{r5:.2f}x", va="center", ha="left", fontsize=8.5, color=INK, fontweight="bold")
        labels.append(f"{names[name]} vs {RIVAL_NAMES[best]}")
    ax.axvline(1.0, color=INK2, lw=1, zorder=0)
    ax.text(1.0, -0.9, "level with the fastest rival", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.4, -1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 3.0)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0)
    ax.scatter([], [], s=70, color=OURS, label="part 5")
    ax.scatter([], [], s=46, color=OURS1, label="part 4")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)   # above the rows
    ax.set_xlabel("our speed over the fastest rival's (log scale)")
    fig.suptitle("Our speed against the fastest rival, at full size, part 5", x=0.02, y=0.99, ha="left", va="top",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.945, "At 95% of the true top 10, one core of an M3 Max, test questions, every item of each dataset. "
             "Right of the line: we are faster.", fontsize=9, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_file = OUT.with_name("speed-part5.png")
    fig.savefig(out_file, dpi=160, facecolor=SURFACE)
    print("wrote", out_file)


def main(part2=False):
    data = per_dataset("step22" if part2 else "step11")
    rows_spec, datasets, out_file = (ROWS, DATASETS, OUT)
    if part2:
        before = per_dataset("step11")
        for name in data:
            data[name]["part1"] = before.get(name, {}).get("ours")
        rows_spec, datasets, out_file = (ROWS2, DATASETS + HELD_BACK, OUT.with_name("speed-part2.png"))
    ROWS_, DATASETS_ = rows_spec, datasets
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, axes = plt.subplots(len(DATASETS_), 1, figsize=(8.2, 2.12 * len(DATASETS_) * len(ROWS_) / 7), sharex=True, facecolor=SURFACE)
    for ax, (name, title) in zip(axes, DATASETS_):
        speeds = data[name]
        ax.set_facecolor(SURFACE)
        ys = range(len(ROWS_))
        fastest = max((v for t, v in speeds.items() if t not in ("ours", "part1") and v), default=None)
        for y, (tool, label) in zip(ys, ROWS_):
            v = speeds.get(tool)
            ax.axhline(y, color=GRID, lw=0.8, zorder=0)
            if v is None:
                note = "not measured in part 1" if tool == "part1" else "never reaches 95%"
                ax.text(330, y, note, va="center", ha="left", color=INK2, fontsize=8.5, style="italic")
                continue
            ours = tool in ("ours", "part1")
            ax.scatter([v], [y], s=70 if ours else 46, color=(OURS if tool == "ours" else OURS1) if ours else RIVAL,
                       zorder=3, edgecolors=SURFACE, linewidths=1.5)
            if ours or v == fastest:
                ax.text(v * 1.18, y, f"{v:,.0f}/s", va="center", ha="left", fontsize=8.5,
                        color=INK if ours else INK2, fontweight="bold" if ours else "normal")
        ax.set_yticks(list(ys), [label for _, label in ROWS_])
        for t in ax.get_yticklabels():
            if t.get_text().startswith("vindex, part 2") or t.get_text() == "vindex (ours)":
                t.set_fontweight("bold")
        ax.set_ylim(len(ROWS_) - 0.4, -0.6)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(300, 250_000)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", which="both", length=0)
    axes[-1].set_xlabel("searches per second, one core (log scale)")
    fig.suptitle("Speed at 95% of the true top 10 found, one core of an M3 Max", x=0.02, y=0.995, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.02, 0.968, "Test questions, 100,000 items per dataset. Rivals measured in the experiments' Step 3; "
             "same machine, same rules.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2)
    fig.subplots_adjust(top=0.925)   # tight_layout also reserves room for the title; one gap is enough
    out_file.parent.mkdir(exist_ok=True)
    fig.savefig(out_file, dpi=160, facecolor=SURFACE)
    print("wrote", out_file)


README_NAMES = {"gooaq": "GooAQ: Google answers", "glove": "GloVe: words", "yandex": "Yandex: text to photos",
                "llama": "Llama: chatbot attention", "landmark": "Landmark: photos", "arxiv": "arXiv: paper summaries",
                "laion": "LAION: text to photos", "yi": "Yi: chatbot attention", "gist": "GIST1M: photo features",
                "nytimes": "NYTimes: news", "imagenet": "ImageNet: text to photos",
                "llama2": "Llama, 2nd head: chatbot attention", "glove100": "GloVe-100: words"}


def _qps_at(curve, target, sec="sec"):
    """Searches per second at exactly `target` recall, read off an effort sweep as the experiments do
    (bench/measures.py cost_at_target): linear between the last point below and the first at or above it."""
    pts = sorted((c[sec], c["accuracy"]) for c in curve)
    best = 0.0
    for i, (cost, acc) in enumerate(pts):
        prev = best
        best = max(best, acc)
        if best >= target:
            if i == 0 or best == prev:
                return 1 / cost
            c0 = pts[i - 1][0]
            return 1 / (c0 + (target - prev) * (cost - c0) / (best - prev))
    return None


def readme():
    """The README's chart and table: vindex (part 5, step 42) against the fastest other index tried on each dataset
    (the experiments' results/step6, and results/step7 for GloVe-100), at full size. Chart: searches per second at
    95% recall@10, one core. Table: the same, plus the ratio at 90%, the ratio on all cores, and Faiss HNSW (step 8's
    Faiss and LoRANN runs count as rivals; hnswlib, which has no ARM vector code, no longer does)."""
    exp = ROOT.parent / "vector-index-experiments" / "results"
    ours = {c["check"].split(":")[0]: c["data"]["ours"]
            for c in json.loads((ROOT / "validate" / "results" / "step42-ours.json").read_text())["checks"]
            if c.get("data") and "at full size" in c["check"]}
    rows = [n for n, _ in DATASETS + HELD_BACK + NEW] + ["glove100"]
    table = []
    for name in rows:
        o = ours[name]
        rv = {}
        files = [*(exp / ("step7" if name == "glove100" else "step6") / "final").glob(f"*/{name}/best.json"),
                 *(exp / "step8" / "final").glob(f"*/{name}/best.json")]   # step 8: Faiss and LoRANN
        for f in files:
            r = json.loads(f.read_text())
            rv[r["tool"]] = {"95": _qps_at(r["curve"], 0.95, "sec_per_query_1core"),
                             "90": _qps_at(r["curve"], 0.90, "sec_per_query_1core"),
                             "all": 1 / r["final"]["sec_per_query_allcores"]}
        others = {t: v for t, v in rv.items() if t != "hnsw"}   # hnswlib has no ARM vector code: Faiss HNSW stands in
        best = max((t for t in others if others[t]["95"]), key=lambda t: others[t]["95"])
        best90 = max((others[t]["90"] for t in others if others[t]["90"]), default=None)
        best_all = max(v["all"] for v in others.values())
        table.append({"name": name, "ours": o["qps95"], "best": best, "rival": others[best]["95"],
                      "r90": _qps_at(o["curve"], 0.90) / best90 if best90 else None,
                      "r_all": o["qps_all"] / best_all, "hnsw": rv.get("faiss_hnsw", {}).get("95")})
    table.sort(key=lambda r: r["ours"] / r["rival"], reverse=True)

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(9.2, 0.5 * len(table) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.36
    for y, r in enumerate(table):
        ours_v, rival = r["ours"], r["rival"]
        for dy, v, colour in ((-h / 2, ours_v, OURS), (h / 2, rival, RIVAL)):
            ax.barh(y + dy, v, height=h * 0.92, color=colour, zorder=2)
        ax.text(ours_v * 1.08, y - h / 2, f"{ours_v / rival:.2f}x", va="center", fontsize=8.5,
                color=INK if ours_v > rival else "#b4471c", fontweight="bold")
        ax.text(rival * 1.08, y + h / 2, RIVAL_NAMES[r["best"]], va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(table)), [README_NAMES[r["name"]] for r in table])
    ax.set_ylim(len(table) - 0.5, -0.6)
    ax.set_xscale("log")
    ax.set_xlim(1_000, 300_000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", which="both", length=0)
    from matplotlib.patches import Patch
    keys = [Patch(color=c, label=k) for k, c in (("vindex", OURS), ("fastest other index tried", RIVAL))]
    ax.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_xlabel("queries per second, one core, at 95% recall@10 (log scale; longer is faster)")
    fig.suptitle("vindex against the fastest other index tried, 13 public datasets", x=0.02, y=0.99, ha="left",
                 va="top", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.02, 0.962, "Every item of each dataset (187,843 to 1.45 million), 1,000 test queries, one core of an "
             "Apple M3 Max.\nOthers tried: Glass, ScaNN, RoarGraph, Faiss (HNSW, HNSW-SQ8, IVF-PQ), LoRANN, CoreNN.", fontsize=9,
             color=INK2, va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_file = OUT.with_name("benchmark.png")
    fig.savefig(out_file, dpi=160, facecolor=SURFACE)
    print("wrote", out_file)
    ratio = lambda v: "–" if v is None else (f"**{v:.2f}×**" if v > 1 else f"{v:.2f}×")
    print("| Dataset | vindex | Fastest other index | vindex ÷ it | at 90% | all 16 cores | Faiss HNSW |")
    print("|---|---|---|---|---|---|---|")
    for r in table:
        hn = f"{r['hnsw']:,.0f}" if r["hnsw"] else "never 95%"
        print(f"| {README_NAMES[r['name']]} | {r['ours']:,.0f} | {RIVAL_NAMES[r['best']]} {r['rival']:,.0f} | "
              f"{ratio(r['ours'] / r['rival'])} | {ratio(r['r90'])} | {ratio(r['r_all'])} | {hn} |")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["readme"]:
        readme()
    elif sys.argv[1:2] == ["part3"]:
        part3()
    elif sys.argv[1:2] == ["part4"]:
        part4()
    elif sys.argv[1:2] == ["part5"]:
        part5()
    else:
        main(part2=sys.argv[1:2] == ["part2"])
