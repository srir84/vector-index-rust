"""Step 29: our index at full size (part 4).
  .venv/bin/python -I validate/step29.py [dataset ...]
Part 3's index with every default (it picks its own settings from its sample questions), built on each dataset's
full items with its 20,000 training questions, by the package frozen in step 28 (a tools/build_py_pgo.sh build).
Measured exactly as in step 27, graded with the full-size true answers: test questions, one core, middle of 5 timed
runs, read off at 95%; all cores; time per question; build time; memory, counted and measured; the SSD mode.
Pass: the SSD mode gives the same answers while reading the full numbers from the SSD; the target picked on the
tuning questions finds at least 94% on the test questions. The rivals are step 30's.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import DATASETS, HELD_BACK, HELD_BACK_3, OUT, Report, build_index, frozen_changes
from step11 import measure, measure_ssd, others_cpu
from step27 import PART3

ALL = [*DATASETS, *HELD_BACK, *HELD_BACK_3]


def sample_result(name):
    """Step 27's record for the same dataset, on its 100,000-item sample."""
    r = json.loads((OUT / "step27.json").read_text())
    return next((c["data"] for c in r["checks"] if c.get("data") and c["check"].split(" ")[0].rstrip(":") == name), None)


def picks(t, st):
    if not t:
        return "nothing (too little data)"
    return (f"{t['chosen_r']} links, slack {t['chosen_alpha']:.1f}, {t['chosen_code_bits']}-bit copies, re-check "
            f"{t['chosen_rerank']}, popular links " + (f"on ({st['popular_items']:,} items)" if st["popular_items"] else "off"))


def main():
    common.unlock_held_back()
    rep = Report("step29")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    work = Path(tempfile.mkdtemp(prefix="vindex-step29-"))
    try:
        for name in sys.argv[1:] or ALL:
            idx, build_s = build_index(name, PART3, train=True, size="full")
            others_before = others_cpu()
            m = measure(name, idx, size="full")
            m["other_programs_cores"] = [others_before, others_cpu()]
            mem = idx.memory()
            per = mem["total_bytes"] / mem["items"]
            ssd = measure_ssd(name, idx, m["target"], work)
            t, st = idx.tuning(), idx.stats()
            s = sample_result(name)
            vs = (f"; sample: {s['ours']['qps95']:,.0f}/s, so {s['ours']['qps95'] / m['qps95']:.2f}x faster than here, "
                  f"build {s['build_seconds']:.0f} s, {s['bytes_per_item']:,.0f} bytes/item, picked "
                  f"{picks(s['tuning'], s['stats'])}") if s and s["ours"]["qps95"] and m["qps95"] else ""
            rep.check(f"{name}: at full size, the 95% target picked on the tuning questions finds at least 94%",
                      m["accuracy_at_target"] >= 0.94,
                      f"{mem['items']:,} items; ours {m['qps95'] or 0:,.0f}/s at 95% (1 core; {m['checks95'] or 0:,.0f} "
                      f"checks); target {m['target']} finds {m['accuracy_at_target']:.2%}; all cores {m['qps_all']:,.0f}/s; "
                      f"typical {m['median_ms']:.3f} ms, slow {m['p99_ms']:.3f} ms; build {build_s:.0f} s; {per:,.0f} "
                      f"bytes/item; picked {picks(t, st)}" + (f" (trial {t['trial_items']:,} items)" if t else "")
                      + f"; other programs using {m['other_programs_cores'][0]:.1f} -> {m['other_programs_cores'][1]:.1f} cores"
                      + vs,
                      {"ours": m, "build_seconds": build_s, "bytes_per_item": per, "memory": mem, "ssd": ssd,
                       "tuning": t, "stats": st, "sample": s and {"qps95": s["ours"]["qps95"], "build_seconds": s["build_seconds"],
                                                                  "bytes_per_item": s["bytes_per_item"]}})
            rep.check(f"{name}: the SSD mode gives the same answers, reading the full numbers from the SSD",
                      ssd["same_answers"] and ssd["cached_share_after_drop"] < 0.01,
                      f"{ssd['qps']:,.0f}/s one core, {ssd['qps_all']:,.0f}/s all cores (file dropped from the cache before "
                      f"each batch); one question from a cold cache: typical {ssd['median_ms_cold']:.2f} ms, slow "
                      f"{ssd['p99_ms_cold']:.2f} ms; memory {ssd['bytes_per_item_counted']:,.0f} bytes/item counted, "
                      f"{ssd['bytes_per_item_measured']:,.0f} measured (with the full numbers in memory: "
                      f"{ssd['ram_bytes_per_item_measured']:,.0f} measured)")
            del idx
            shutil.rmtree(work / name, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rep.finish()


if __name__ == "__main__":
    main()
