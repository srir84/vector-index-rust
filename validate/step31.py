"""Step 31: GIST1M against CoreNN (part 4).
  .venv/bin/python -I validate/step31.py
CoreNN ran in step 30's rival runs (the experiments' scripts/step6_baselines.py) on GIST1M's full items with its
author's own settings (libcorenn's defaults: 64 links, build list 128, slack 1.1) and no settings search; its search
effort is the smallest that reaches 95% on the tuning questions, as for every rival. Ours is step 29's.
Speed: test questions, one core, read off at exactly 95%. Memory per item: CoreNN's process memory after its build
(it keeps the items in RocksDB on disk and caches them); ours measured in a fresh process (macOS's footprint count),
with the full numbers on SSD and with them in memory too.
Pass: the sentence's numbers come straight from the two results files.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import OUT, Report, frozen_changes
from step30 import FULL, ours_full, rival


def main():
    rep = Report("step31")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    cn = rival("corenn", "gist")
    raw = json.loads((FULL / "final" / "corenn" / "gist" / "best.json").read_text())
    ours = ours_full()["gist"]
    q, ssd = ours["ours"]["qps95"], ours["ssd"]
    if cn["qps95"]:
        speed = q / cn["qps95"]
        less_ssd = 1 - ssd["bytes_per_item_measured"] / cn["bytes_per_item"]
        less_ram = 1 - ssd["ram_bytes_per_item_measured"] / cn["bytes_per_item"]
        sentence = (f"On GIST1M, the dataset CoreNN's author tested on, our index answers {speed:.1f}x as many questions "
                    f"per second as CoreNN at 95% of the true top 10 (one core), and uses {less_ssd:.0%} less memory "
                    f"with the full numbers on SSD ({less_ram:.0%} less with them in memory too).")
    else:
        speed = less_ssd = less_ram = None
        sentence = (f"On GIST1M, CoreNN on its author's settings never reached 95% of the true top 10 (best "
                    f"{cn['best']:.1%}); ours does, at {q:,.0f} questions per second on one core.")
    detail = (f"ours {q:,.0f}/s, CoreNN {cn['qps95'] or 0:,.0f}/s at 95% (one core, test questions; CoreNN's settings "
              f"{cn['params']}, effort {raw['final']['effort']}); memory per item: CoreNN {cn['bytes_per_item']:,.0f} "
              f"(process, after its build; {raw.get('disk_bytes', 0) / cn['n_items']:,.0f} on disk), ours "
              f"{ssd['bytes_per_item_measured']:,.0f} with the full numbers on SSD, {ssd['ram_bytes_per_item_measured']:,.0f} "
              f"with them in memory too; build: CoreNN {cn['build_s'] / 60:.0f} min, ours {ours['build_seconds'] / 60:.0f} min")
    rep.check("GIST1M against CoreNN on its author's settings: the sentence", True, f"{sentence}\n      {detail}",
              {"sentence": sentence, "speed_ratio": speed, "memory_less_ssd": less_ssd, "memory_less_ram": less_ram,
               "corenn": cn, "ours": {"qps95": q, "ssd": ssd, "build_seconds": ours["build_seconds"]}})
    rep.note(f"expected (step 28): at least 10x faster and less memory; "
             + ("RIGHT" if speed and speed >= 10 and less_ssd and less_ssd > 0 else "WRONG"))
    (OUT / "step31-sentence.txt").write_text(sentence + "\n")
    rep.finish()


if __name__ == "__main__":
    main()
