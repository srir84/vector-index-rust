"""Step 30c: what moves our full-size speed: the timing, the picks, or the memory layout? (part 4)
  .venv/bin/python -I validate/step30c.py [dataset ...]      # default: gooaq yandex
Step 30b rebuilt our index from scratch and timed it again: its speed moved by about 11%, more than Glass's, and on
Yandex its checks per search moved too, so the setting search picked differently (it picks by timing trial builds,
and close calls can go either way: DESIGN.md section 16). Here each dataset is built twice more, one right after the
other, with step 29's picks fixed and the setting search off, so both builds have step 29's settings:
  A. as in step 29, renumbering on: the same index as step 29's, timed again, so the timing alone;
  B. renumbering off: after the build, our index renumbers its items in the order a walk of the links meets them, so
     items that link to each other sit close together in memory. DESIGN.md (section on renumbering): it gained 0-2%
     at 100,000 items, "where most of the data sits in the chip's caches anyway; it should matter more beyond them".
Renumbering never changes the answers or the work, so A against B is the layout alone.
Pass: A gives step 29's accuracy and checks per search at every target; B gives A's.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import Report, build_index, frozen_changes
from step11 import measure, others_cpu
from step27 import PART3
from step30 import ours_full


def fixed(rec):
    """Step 29's picks as settings, with the setting search off."""
    t, st = rec["tuning"], rec["stats"]
    return dict(PART3, tune="off", r=t["chosen_r"], alpha=round(t["chosen_alpha"], 3), code_bits=t["chosen_code_bits"],
                rerank=t["chosen_rerank"], popular_links="on" if st["popular_items"] else "off")


def run(name, settings):
    idx, build_s = build_index(name, settings, train=True, size="full")
    before = others_cpu()
    m = measure(name, idx, size="full")
    m["other_programs_cores"] = [before, others_cpu()]
    m["build_seconds"] = build_s
    return m


def same(a, b):
    return all(x["accuracy"] == y["accuracy"] and x["checks"] == y["checks"] for x, y in zip(a["curve"], b["curve"]))


def main():
    common.unlock_held_back()
    rep = Report("step30c")
    changed = frozen_changes()
    rep.check("the code is as frozen in step 28", not changed, "unchanged" if not changed else "; ".join(changed[:5]))
    recorded = ours_full()
    for name in sys.argv[1:] or ["gooaq", "yandex"]:
        rec = recorded[name]
        s = fixed(rec)
        a = run(name, s)
        b = run(name, dict(s, renumber=False))
        was = rec["ours"]["qps95"]
        rep.check(f"{name}: with step 29's picks fixed, the same accuracy and work as step 29", same(a, rec["ours"]),
                  f"settings {', '.join(f'{k}={v}' for k, v in s.items() if k in ('r', 'alpha', 'code_bits', 'rerank'))}; "
                  f"{a['qps95']:,.0f}/s at 95% (1 core) against step 29's {was:,.0f}/s ({a['qps95'] / was:.2f}x), "
                  f"checks per search {a['checks95']:,.0f} (step 29: {rec['ours']['checks95']:,.0f})",
                  {"dataset": name, "settings": s, "a": a, "recorded_qps95": was})
        rep.check(f"{name}: with renumbering off, the same accuracy and work; only the memory layout changes",
                  same(a, b),
                  f"renumbering on {a['qps95']:,.0f}/s, off {b['qps95']:,.0f}/s, so the layout gives {a['qps95'] / b['qps95']:.2f}x "
                  f"on one core ({a['qps_all'] / b['qps_all']:.2f}x on all cores); other programs using "
                  f"{max(a['other_programs_cores'] + b['other_programs_cores']):.1f} cores at most",
                  {"dataset": name, "b": b, "gain": a["qps95"] / b["qps95"], "gain_all": a["qps_all"] / b["qps_all"]})
    rep.finish()


if __name__ == "__main__":
    main()
