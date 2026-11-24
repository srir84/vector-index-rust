"""Step 23: part 3's held-back datasets are downloaded, checked, prepared like the five, and locked.
  - each download matches its published checksum (NYTimes: its server's 8 MiB-piece checksum);
  - Llama's second head is the head the spec's draw names, made into a file by VIBE's rule, and that rule, run on
    the head VIBE's Llama came from, gives VIBE's own file exactly;
  - each is prepared like the five: the same files, 100,000-item slice, the same question groups, our true answers
    agree with the file's own, and no question's answers differ beyond ties;
  - both loaders refuse all seven held-back datasets (part 2's four stay locked);
  - nothing has been run on the three: no result here or in the experiments names them.
Nothing here searches a held-back dataset; it reads only download records, the raw files and preparation cards.

.venv/bin/python -I validate/step23.py
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import DATA, EXP, HELD_BACK, HELD_BACK_3, OUT, PART4_PLACES, PART4_RESULTS, REF, Report, load, truth
from step14 import file_hashes

RAW = EXP / "data" / "raw"
PY = str(EXP / ".venv" / "bin" / "python")
FILES = {"nytimes": "nytimes-256-angular", "imagenet": "imagenet-align-640-normalized", "llama2": "llama2-128-ip"}
NYT_TAG = "dec76e37150d602463a8ac07e1044c44-38"   # ann-benchmarks' server tag: MD5 of the 8 MiB pieces' MD5s
PUBLISHED = {  # Hugging Face's file listings
    "imagenet-align-640-normalized.hdf5": "84f1fb99ba8611a37fc254cd4ec50b06c969efff07c11e272444f3e6e89cee7e",
    "K_15_18.npy": "dfa5b9a294dcd9663cb850f2fcf9083ee4f416df93c1a3d3b253134b2208790f",
    "Q_15_18.npy": "888648dd0db968c88805a8f8294131f7075715cd4464953c8c617e5caaa69ded",
    "K_12_15.npy": "0eb97418ce7407359d439d5c97c6fd5d1f97331d5c0fef3d2eb9cb08882f5e29",
    "Q_12_15.npy": "3761f5aa21801c250061a7eec1e6fa8654ece2e26dac34589456180499ecc2e4",
}
# the 20 heads in ejaasaari/llama_heads (Hugging Face's listing), and the draw SPEC.md part 3 wrote down
HEADS = ["7_10", "7_24", "8_16", "9_17", "10_6", "12_3", "12_15", "13_28", "14_5", "14_26", "15_18", "18_6",
         "18_11", "18_15", "19_12", "20_2", "22_27", "26_16", "27_28", "29_26"]
LLAMA_HEAD = "12_15"   # VIBE's Llama (vibe/datasets.py: llama(out_fn, 12, 15))


def downloads(rep):
    f = RAW / FILES["nytimes"] / f"{FILES['nytimes']}.hdf5"
    sha, tag = file_hashes(f)
    rep.check("nytimes: file matches its server's checksum", tag == NYT_TAG and sha == (f.parent / "OK").read_text().strip(),
              f"piece checksum {tag}, sha256 {sha[:16]}... ({f.stat().st_size / 1e9:.2f} GB)")
    f = RAW / FILES["imagenet"] / f"{FILES['imagenet']}.hdf5"
    sha, _ = file_hashes(f)
    rep.check("imagenet: file matches the published checksum",
              sha == PUBLISHED[f.name] == (f.parent / "OK").read_text().strip(),
              f"sha256 {sha[:16]}... ({f.stat().st_size / 1e9:.2f} GB)")
    for head in ("15_18", LLAMA_HEAD):
        d = RAW / f"llama-heads-{head.replace('_', '-')}"
        recorded = dict(line.split() for line in (d / "OK").read_text().splitlines())
        got = {n: file_hashes(d / n)[0] for n in (f"K_{head}.npy", f"Q_{head}.npy")}
        rep.check(f"llama head {head}: both files match the published checksums",
                  all(got[n] == PUBLISHED[n] == recorded.get(n) for n in got),
                  ", ".join(f"{n} {s[:16]}..." for n, s in got.items()))


def made_like_vibe(rep):
    others = [h for h in HEADS if h != LLAMA_HEAD]
    drawn = random.Random(23).choice(others)
    rep.check("llama2: the head is the one the spec's draw names", drawn == "15_18",
              f"random.Random(23).choice over the {len(others)} heads that are not Llama's gives {drawn}")
    # in the experiments' Python (it has h5py): the file's arrays against the rule run again on the raw head, and
    # the rule run on Llama's head against VIBE's own Llama file
    code = ("import sys, json, h5py, numpy as np; sys.path.insert(0, sys.argv[1] + '/scripts')\n"
            "from make_llama2 import make\n"
            "raw = sys.argv[1] + '/data/raw/'\n"
            "out = {}\n"
            "for name, head, f in (('llama2', '15_18', 'llama2-128-ip'), ('vibe', '12_15', 'llama-128-ip')):\n"
            "    K, test, learn = make(__import__('pathlib').Path(raw + 'llama-heads-' + head.replace('_', '-')), head)\n"
            "    with h5py.File(raw + f + '/' + f + '.hdf5', 'r') as h:\n"
            "        out[name] = {k: bool(np.array_equal(np.asarray(h[k]), v)) for k, v in (('train', K), ('test', test), ('learn', learn))}\n"
            "        out[name]['sizes'] = [len(K), len(test), len(learn)]\n"
            "print(json.dumps(out))\n")
    run = subprocess.run([PY, "-I", "-c", code, str(EXP)], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    if run.returncode:
        rep.check("llama2: made by VIBE's rule", False, f"error: {run.stderr.strip()[-300:]}")
        return
    got = json.loads(run.stdout)
    same = lambda g: g["train"] and g["test"] and g["learn"]
    rep.check("llama2: the file is the head made by VIBE's rule", same(got["llama2"]),
              f"items, test and extra questions equal the rule run again on the raw head "
              f"({got['llama2']['sizes'][0]:,} / {got['llama2']['sizes'][1]:,} / {got['llama2']['sizes'][2]:,})")
    rep.check("the rule, run on Llama's own head, gives VIBE's Llama file exactly", same(got["vibe"]),
              f"layer 12, head 15: items, test and extra questions identical ({got['vibe']['sizes'][0]:,} items)")


def prepared(rep):
    # CARD.md is a hand-written summary of the five, not something the preparation makes
    expected = sorted(p.name for p in (DATA / "glove").iterdir() if p.name != "CARD.md")
    for name, metric in HELD_BACK_3.items():
        d = DATA / name
        have = sorted(p.name for p in d.iterdir()) if d.exists() else []
        missing = [x for x in expected if x not in have]
        if missing:
            rep.check(f"{name}: prepared like the five", False, f"missing {', '.join(missing[:5])}")
            continue
        card = json.loads((d / "card.json").read_text())
        q, h = card["questions"], card["health"]
        groups = card["precision_check"]
        # a file answer that ties with our 10th (the same distance) counts as agreeing: NYTimes has 40,793 exact
        # duplicates, and which of several tied items makes a top 10 is arbitrary
        agree = card["agreement_with_vibe_top10_full_counting_ties"]
        plain, lost = card["agreement_with_vibe_top10_full"], card["vibe_top10_items_we_set_aside_as_questions"]
        unexplained = sum(g["of_which_not_explained_by_ties"] for g in groups.values())
        ok = (card["formula"] == metric and card["slice_items"] == 100_000 and q["test"] == 1000
              and q["tune"] == 1000 and q["train"] == 20_000 and agree + lost >= 0.995 and unexplained == 0
              and h["rows_with_broken_values_nan"] == 0 and h["rows_with_broken_values_inf"] == 0
              and (metric != "cos" or h["all_zero_rows"] == 0))
        whose = "our own float64 run's (the file has none of its own)" if name == "llama2" else "the file's"
        rep.check(f"{name}: prepared like the five", ok,
                  f"{card['items_full']:,} items, {h['numbers_per_item']} numbers, formula {card['formula']}; "
                  f"test questions {q['test']:,} (of {q.get('test_in_file', q['test']):,} in the file); "
                  f"questions from {q['train_source']}; our top 10 agree with {whose} on {agree:.2%} counting "
                  f"ties ({plain:.2%} without), +{lost:.2%} set aside as questions; answers differing beyond ties: "
                  f"{unexplained}; all-zero rows {h['all_zero_rows']} (left out: "
                  f"{card.get('rows_without_angle_left_out', {}).get('items', 0)}), exact duplicates "
                  f"{h['exact_duplicate_rows']:,}",
                  {"card": {k: card.get(k) for k in ("items_full", "questions", "agreement_with_vibe_top10_full",
                                                      "agreement_with_vibe_top10_full_counting_ties",
                                                      "vibe_top10_items_we_set_aside_as_questions",
                                                      "rows_without_angle_left_out", "precision_check")}})


def locks(rep):
    names = list(HELD_BACK) + list(HELD_BACK_3)
    refused = []
    for name in names:
        for call in (lambda: load(name, "slice"), lambda: truth(name, "slice_test")):
            try:
                call()
            except PermissionError:
                refused.append(True)
            else:
                refused.append(False)
    rep.check("our loader refuses all seven held-back datasets", all(refused), f"{sum(refused)}/{len(refused)} refused")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from bench.data import prepared\n"
            "prepared('glove')\n"
            "for n in sys.argv[2:]:\n"
            "    try: prepared(n)\n"
            "    except PermissionError: print('refused', n)\n")
    run = subprocess.run([PY, "-I", "-c", code, str(EXP), *names], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"})
    n = run.stdout.count("refused")
    rep.check("the experiments' loader refuses all seven", run.returncode == 0 and n == len(names),
              f"{n}/{len(names)} refused" + (f"; error: {run.stderr.strip()[-200:]}" if run.returncode else ""))


# where step 27's rival runs put their results, in the experiments' results folder
STEP27_PLACES = ("step3/grid/", "step3/final/", "step3/choices-nytimes-imagenet-llama2.json")


def untouched(rep):
    """Before step 27, no result here or in the experiments names the three. After it, only step 27's own runs do:
    ours in validate/results/step27*, the rivals' in the experiments' step 3 folders."""
    pat = re.compile(r"\b(nytimes|imagenet|llama2)\b", re.I)
    after27 = (OUT / "step27.json").exists()
    mine = ("step23.", "step27", *PART4_RESULTS) if after27 else ("step23.",)
    hits = [p.name for p in OUT.glob("*") if p.is_file() and not p.name.startswith(mine)
            and pat.search(p.read_text(errors="ignore"))]
    exp = [str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_file() and pat.search(str(p.relative_to(REF)))]
    exp += [str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_dir() and p.name in HELD_BACK_3]
    if after27:
        exp = [e for e in exp if not e.startswith(STEP27_PLACES + PART4_PLACES)]
    rep.check("nothing has been run on the three" + (" but step 27" if after27 else ""), not hits and not exp,
              "no result names them" + (" outside step 27's runs" if after27 else "") if not hits and not exp
              else f"found: {', '.join((hits + exp)[:5])}")


def main():
    rep = Report("step23")
    downloads(rep)
    made_like_vibe(rep)
    prepared(rep)
    locks(rep)
    untouched(rep)
    rep.finish()


if __name__ == "__main__":
    main()
