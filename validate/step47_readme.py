"""Step 47: the benchmark's two tables (docs/BENCHMARK.md since step 49; the README before), checked against the results files.
    ../vector-index-experiments/.venv/bin/python -I validate/step47_readme.py      (needs matplotlib, as draw_report does)
Checks:
  - "The numbers": every row is what `tools/draw_report.py readme` prints from the results files (step 42 and
    the experiments' results/current.json: steps 6 to 9);
  - "What each piece adds": every cell, recomputed from validate/results/step46-pieces.json.
"""
import json
import subprocess
from itertools import takewhile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import OUT, ROOT, Report

DS = ["gooaq", "glove", "laion", "llama"]
ROWS = [("as picked", "As the tuning picked"), ("fixed effort", "One list size for every query"),
        ("8-bit codes", "8-bit codes instead of 4-bit"), ("float32 walk", "No codes"),
        ("no question links", "No question links"), ("no popular links", "No popular links"),
        ("graph on x", "Graph built on x"), ("defaults", "Default settings")]


def table_rows(text, header):
    """The rows of the table that starts with `header`, without the header and its rule."""
    return list(takewhile(lambda ln: ln.startswith("|"), text[text.index(header):].splitlines()[2:]))


def pieces():
    """Each variant's speed at 95%, as a share of the first row; "–" where the piece was not in use."""
    r = json.loads((OUT / "step46-pieces.json").read_text())
    base, cell = {}, {}
    for c in r["checks"]:
        d = c["data"]
        n, v = d["dataset"], d["variant"]
        q = d["fixed"]["qps95"] if v == "fixed effort" else d["ours"]["qps95"]
        if v == "as picked":
            base[n] = q
        cell[(v, n)] = q if q else f"never 95% (best {max(x['accuracy'] for x in d['ours']['curve']):.0%})"
    cell[("defaults", "gooaq")] = cell[("8-bit codes", "gooaq")]   # GooAQ's defaults are its 8-bit build
    want = {}
    for v, _ in ROWS:
        cs = []
        for n in DS:
            x = cell.get((v, n))
            cs.append("–" if x is None else x if isinstance(x, str) else f"{x:,.0f}/s" if v == "as picked"
                      else f"{x / base[n]:.2f}×")
        want[v] = cs
    return want


def main():
    rep = Report("step47-readme")
    readme = (ROOT / "docs" / "BENCHMARK.md").read_text()

    printed = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "draw_report.py"), "readme"],
                             capture_output=True, text=True, check=True).stdout
    want = [ln.replace("| Dataset | vindex |", "| Dataset | vindex (queries/s) |") for ln in printed.splitlines()
            if ln.startswith("|")][2:]
    have = table_rows(readme, "| Dataset | vindex (queries/s) |")
    bad = [f"{h!r} against {w!r}" for h, w in zip(have, want) if h != w]
    rep.check("docs/BENCHMARK.md 'The numbers': every row matches the results files", len(have) == len(want) == 13 and not bad,
              f"{len(have)} rows checked" + "".join(f"\n      {b}" for b in bad[:5]))

    want = pieces()
    have = table_rows(readme, "| Change | GooAQ | GloVe | LAION | Llama |")
    bad = []
    for v, label in ROWS:
        row = next((ln for ln in have if ln.startswith(f"| {label}")), None)
        cells = [c.strip() for c in row.strip("|").split("|")][1:] if row else None
        if cells is None or [c.split(" (the 8-bit row)")[0] for c in cells] != want[v]:
            bad.append(f"{label}: {cells} against {want[v]}")
    rep.check("docs/BENCHMARK.md 'What each piece adds': every cell matches step 46's results", len(have) == len(ROWS) and not bad,
              f"{len(have)} rows checked" + "".join(f"\n      {b}" for b in bad))
    rep.finish()


if __name__ == "__main__":
    main()
