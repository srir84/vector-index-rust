"""Step 19: suggestions.
  - poor settings (GloVe with 16 links): after real use, suggest() names more links; rebuilt as suggested, the
    measured gain matches the trial's within 15%;
  - wrong sample questions (Yandex, Llama: taken from the items, so question links stay off): suggest() notes that
    real questions look unlike the items; a rebuild with the spot-checked questions as sample questions is
    suggested exactly when it really gains at least 15%, and then with the gain predicted within 15%;
  - well-set indexes (the five, tuned at build): no rebuild is suggested;
  - many deleted items: a note to rebuild for memory.
"Real use" is 2,000 training questions the build never saw (rows 17,000-19,000), each spot-checked. Speeds are
measured as in step 18: one core, tuning questions, read off at 95%.

.venv/bin/python -I validate/step19.py
"""
import sys
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, DATASETS, Report, load
from step11 import FINAL
from step18 import speed95

TARGET = 0.95
PART2 = dict(FINAL, effort_apart=True, audit_budget=1e12)   # every searched question spot-checked
BUILD_ROWS, USE = slice(0, 17000), slice(17000, 19000)


def build(name, settings, train):
    X = load(name, "slice")
    idx = vindex.Index(X.shape[1], DATASETS[name], seed=7, **settings)
    idx.build(X, np.arange(len(X)), train_questions=np.ascontiguousarray(train))
    return idx


def use(idx, name):
    idx.search(load(name, "q_train")[USE], k=10, target=TARGET, threads=16)
    idx.audit()


def rebuild_as_suggested(name, sug, settings, train, idx):
    s = dict(settings, **{k: (float(v) if k == "alpha" else int(v) if k == "r" else v) for k, v in sug["settings"].items()})
    q = idx.checked_questions() if sug["use_checked_questions"] else train
    return build(name, s, q)


def matches_reality(rep, label, name, idx, sug, settings, train):
    """A suggested rebuild, carried out as suggested: it must really gain at least 15%, and the trial's gain must
    be within 15% of the measured one."""
    new = rebuild_as_suggested(name, sug, settings, train, idx)
    (v0, _), (v1, _) = speed95(idx, name), speed95(new, name)
    predicted, actual = sug["after"] / sug["now"], v1 / v0
    ok = actual >= 1.15 and abs(actual / predicted - 1) <= 0.15
    rep.check(f"{name}: {label}: the suggested rebuild gains what the trial said, within 15%", ok,
              f"trial {predicted:.2f}x, measured {actual:.2f}x ({v0:,.0f} -> {v1:,.0f} searches/s at 95%, one core); "
              f"suggested: {sug['what']}")


def poor_settings(rep, name="glove"):
    settings = dict(PART2, r=16)
    train = load(name, "q_train")[BUILD_ROWS]
    idx = build(name, settings, train)
    use(idx, name)
    sugs = idx.suggest()
    reb = [s for s in sugs if s["kind"] == "rebuild"]
    more = bool(reb) and int(reb[0]["settings"].get("r", 16)) > 16
    rep.check(f"{name} built with 16 links: more links are suggested", more,
              reb[0]["what"] + " " + reb[0]["why"] if reb else "no rebuild suggested: " + "; ".join(s["what"] for s in sugs))
    if more:
        matches_reality(rep, "16 links", name, idx, reb[0], settings, train)


def wrong_sample(rep):
    """Sample questions taken from the items: the note must appear; a rebuild is suggested exactly when one really
    gains at least 15% (when none is, the rebuild the note points to must really gain less)."""
    for name in ("yandex", "llama"):
        spare = np.load(DATA / name / "spare_ids.npy")
        items_as_questions = np.ascontiguousarray(np.load(DATA / name / "items_full.npy", mmap_mode="r")[spare])
        idx = build(name, PART2, items_as_questions)
        use(idx, name)
        sugs = idx.suggest()
        unlike = [s for s in sugs if "unlike the items" in s["what"]]
        rep.check(f"{name} with sample questions taken from the items: real questions noted as unlike the items",
                  bool(unlike), unlike[0]["why"] if unlike else "no note")
        reb = [s for s in sugs if s["kind"] == "rebuild"]
        if reb:
            matches_reality(rep, "sample questions from the items", name, idx, reb[0], PART2, items_as_questions)
        else:
            real = build(name, PART2, idx.checked_questions())   # the rebuild the note points to
            (v0, _), (v1, _) = speed95(idx, name), speed95(real, name)
            rep.check(f"{name}: no rebuild suggested, and the note's rebuild really gains under 15%", v1 / v0 < 1.15,
                      f"measured {v1 / v0:.2f}x ({v0:,.0f} -> {v1:,.0f} searches/s at 95%, one core)")


def well_set(rep):
    """Tuned at build: no rebuild should be worth it. Any rebuild suggested is carried out and must really gain
    what it says (then the build's tuning missed something, and the suggestion is right to say so)."""
    for name in DATASETS:
        settings, train = dict(PART2, tune="auto"), load(name, "q_train")[BUILD_ROWS]
        idx = build(name, settings, train)
        use(idx, name)
        sugs = idx.suggest()
        reb = [s for s in sugs if s["kind"] == "rebuild"]
        t = idx.tuning()
        if reb:
            matches_reality(rep, f"tuned at build ({t['chosen_r']} links, slack {t['chosen_alpha']:.1f})", name, idx,
                            reb[0], dict(settings, tune="off"), train)
        else:
            rep.check(f"{name}, tuned at build: no rebuild suggested", True,
                      f"built with {t['chosen_r']} links, slack {t['chosen_alpha']:.1f}; notes: "
                      + ("; ".join(s["what"] for s in sugs) or "none"))


def deleted(rep, name="yandex"):
    idx = build(name, dict(FINAL, effort_apart=True), load(name, "q_train")[BUILD_ROWS])
    for i in range(0, 100_000, 4):   # a quarter of the items
        idx.delete(i)
    sugs = idx.suggest()
    mem = [s for s in sugs if s["what"].startswith("Rebuild to free memory")]
    rep.check(f"{name} with a quarter of its items deleted: a note to rebuild for memory", bool(mem),
              mem[0]["why"] if mem else "; ".join(s["what"] for s in sugs))


def main():
    rep = Report("step19")
    poor_settings(rep)
    wrong_sample(rep)
    well_set(rep)
    deleted(rep)
    rep.finish()


if __name__ == "__main__":
    main()
