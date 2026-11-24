"""Step 36: part 5's held-back dataset is downloaded, checked, prepared like the others, and locked.
  - the download matches ann-benchmarks' server tag (the MD5 of its 8 MiB pieces' MD5s), as NYTimes' did;
  - it is prepared like the others: 100,000-item sample, the same question groups, our true answers agree with the
    file's own apart from the items set aside as questions, no difference between the two precisions beyond ties,
    and the full-size true answers and change script are there;
  - both loaders refuse it, even with the earlier held-back datasets unlocked;
  - nothing has been run on it: no result here or in the experiments names it, apart from its preparation;
  - the index's code is still as frozen in step 28, so the expectations in STEPS.md (step 36) come before any part 5
    code change.
Nothing here searches it; it reads only the download, its record and the preparation cards.

.venv/bin/python -I validate/step36.py
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
import common
from common import DATA, EXP, OUT, Report, frozen_changes, load

NAME, FILE = "glove100", "glove-100-angular"
TAG = "9ef1a69a639809d4ae43efd5e6707fa1-58"   # ann-benchmarks' server tag
RAW = EXP / "data" / "raw" / FILE


def multipart_tag(path, part=8 * 1024 * 1024):
    digests = []
    with open(path, "rb") as fh:
        while chunk := fh.read(part):
            digests.append(hashlib.md5(chunk).digest())
    return hashlib.md5(b"".join(digests)).hexdigest() + f"-{len(digests)}"


def names_it(path):
    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        return False
    return NAME in text or FILE in text


def main():
    rep = Report("step36")
    f = RAW / f"{FILE}.hdf5"
    tag, sha = multipart_tag(f), hashlib.sha256(f.read_bytes()).hexdigest()
    rep.check("the download matches the server's tag", tag == TAG and (RAW / "OK").read_text().strip() == sha,
              f"tag {tag} (server {TAG}); sha256 {sha[:16]}... as recorded")
    card = json.loads((DATA / NAME / "card.json").read_text())
    full = json.loads((DATA / NAME / "card_full.json").read_text())
    q, prec = card["questions"], card["precision_check"]
    agree = card["agreement_with_vibe_top10_full_counting_ties"] + card["vibe_top10_items_we_set_aside_as_questions"]
    ok = (card["slice_items"] == 100_000 and (q["test"], q["tune"], q["train"]) == (1_000, 1_000, 20_000)
          and agree >= 0.999 and all(p["of_which_not_explained_by_ties"] == 0 for p in prec.values())
          and (DATA / NAME / "changes_full.npz").exists() and (DATA / NAME / "truth_full_tune.npz").exists())
    rep.check("prepared like the others", ok,
              f"{card['items_full']:,} items of {card['health']['numbers_per_item']} numbers; questions {q['test']:,} test "
              f"(of {q['test_in_file']:,} in the file), {q['tune']:,} tuning, {q['train']:,} training ({q['train_source']}); "
              f"our true top 10 agree with the file's on {card['agreement_with_vibe_top10_full_counting_ties']:.2%}, and "
              f"{card['vibe_top10_items_we_set_aside_as_questions']:.2%} are items set aside as questions; full size: "
              f"{', '.join(f'{k} {v:,}' if isinstance(v, int) else k for k, v in full.items() if not isinstance(v, (dict, list)))[:200]}")
    common.unlock_held_back()   # the earlier held-back datasets are unlocked for good; this one must stay locked
    try:
        load(NAME, "slice")
        ours = "loaded"
    except PermissionError:
        ours = "refused"
    env = {k: v for k, v in os.environ.items() if k != "BENCH_UNLOCK_HELD_BACK_5"}
    env["BENCH_UNLOCK_HELD_BACK"] = "1"
    r = subprocess.run([str(EXP / ".venv" / "bin" / "python"), "-I", "-c",
                        f"from bench.data import prepared\ntry:\n    prepared({NAME!r}); print('loaded')\n"
                        "except PermissionError:\n    print('refused')"], cwd=EXP, env=env, capture_output=True, text=True)
    theirs = r.stdout.strip() or r.stderr.strip()[-200:]
    rep.check("both loaders refuse it, even with the earlier held-back datasets unlocked",
              ours == "refused" and theirs == "refused", f"ours {ours}; the experiments' {theirs}")
    own = {"prepare_full-glove100.log", "download_part5.log"}
    # step 42 is its held-back test (validate/results/step42*, the experiments' results/step7/): only it may name it
    hits = [str(p) for base in (OUT, EXP / "results") for p in base.rglob("*")
            if p.is_file() and p.suffix in (".json", ".log", ".txt", ".csv") and p.name not in own
            and not p.name.startswith(("step36", "step42")) and "step7" not in p.parts and names_it(p)]
    rep.check("nothing has been run on it", not hits, "no result names it" if not hits else "; ".join(hits[:5]))
    changed = frozen_changes(("index", "package"))
    rep.check("the index's code is as frozen in step 28: the expectations come before any part 5 change", not changed,
              "unchanged" if not changed else "; ".join(changed[:5]))
    rep.finish()


if __name__ == "__main__":
    main()
