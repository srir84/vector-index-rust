"""Step 14: the held-back datasets are downloaded, checked, prepared like the five, and locked.
  - each download matches its published checksum (GIST1M: its server's 8 MiB-piece checksum);
  - each is prepared like the five: the same files, 100,000-item slice, the same question groups, our true
    answers agree with the file's own, and no question's answers differ beyond ties;
  - both loaders refuse them: ours (validate/common.py) and the experiments' (bench/data.py);
  - nothing has been run on them: no result here or in the experiments names them (after step 22, none but its own);
  - the straight-line formula works on real data, checked on Llama's numbers (GIST1M stays untouched).
Nothing here searches a held-back dataset; it reads only the download records and preparation cards.

.venv/bin/python -I validate/step14.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import vindex

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I leaves the script folder out
from common import (DATA, EXP, HELD_BACK, L_SWEEP, OUT, PART4_PLACES, PART4_RESULTS, REF, Report, accuracy,
                    correct_counts, load, sweep_targets, truth)

FINAL = dict(code_bits=8, learned_effort=True, question_links="auto", q_rounds=2, build_space="auto", effort_apart=False, tune="off", renumber=False, question_bits=32, popular_links="off")
FILES = {"arxiv": "arxiv-nomic-768-normalized", "laion": "laion-clip-512-normalized", "yi": "yi-128-ip",
         "gist": "gist-960-euclidean"}
PUBLISHED = {  # Hugging Face's listing of the VIBE files
    "arxiv": "8be0993b978b0d0ef023d21d878251a5ed09e058adb25994553c08388d37d414",
    "laion": "a152088cefc13ac9fdfaa1722ae40a67a3a3b6598d764bf01ab162a61a2d1be9",
    "yi": "23d76fdbb0808909d02be83f95c074c3ccf54018b03a80d54e8e31ad1815071e",
}
GIST_TAG = "34da1d8a80764582ee4b0c0839b7c32a-459"   # ann-benchmarks' server tag: MD5 of the 8 MiB pieces' MD5s
RAW = EXP / "data" / "raw"


def file_hashes(path):
    sha, part, digests = hashlib.sha256(), 8 * 1024 * 1024, []
    with open(path, "rb") as fh:
        while chunk := fh.read(part):
            sha.update(chunk)
            digests.append(hashlib.md5(chunk).digest())
    return sha.hexdigest(), hashlib.md5(b"".join(digests)).hexdigest() + f"-{len(digests)}"


def downloads(rep):
    for name, file in FILES.items():
        f, ok_file = RAW / file / f"{file}.hdf5", RAW / file / "OK"
        if not f.exists() or not ok_file.exists():
            rep.check(f"{name}: downloaded and checked", False, f"missing {f if not f.exists() else ok_file}")
            continue
        sha, tag = file_hashes(f)
        recorded = ok_file.read_text().strip()
        if name in PUBLISHED:
            rep.check(f"{name}: file matches the published checksum", sha == PUBLISHED[name] == recorded,
                      f"sha256 {sha[:16]}... ({f.stat().st_size / 1e9:.2f} GB)")
        else:
            rep.check(f"{name}: file matches its server's checksum", tag == GIST_TAG and sha == recorded,
                      f"piece checksum {tag}, sha256 {sha[:16]}... ({f.stat().st_size / 1e9:.2f} GB)")


def prepared(rep):
    # CARD.md is a hand-written summary of the five, not something the preparation makes
    expected = sorted(p.name for p in (DATA / "glove").iterdir() if p.name != "CARD.md")
    for name, metric in HELD_BACK.items():
        d = DATA / name
        have = sorted(p.name for p in d.iterdir()) if d.exists() else []
        missing = [x for x in expected if x not in have]
        if missing:
            rep.check(f"{name}: prepared like the five", False, f"missing {', '.join(missing[:5])}")
            continue
        card = json.loads((d / "card.json").read_text())
        q, h = card["questions"], card["health"]
        groups = card["precision_check"]
        agree, lost = card["agreement_with_vibe_top10_full"], card["vibe_top10_items_we_set_aside_as_questions"]
        unexplained = sum(g["of_which_not_explained_by_ties"] for g in groups.values())
        # an all-zero row has no angle, so only angle data must be free of them; for the other formulas it is
        # an ordinary point (GIST1M has 10)
        ok = (card["formula"] == metric and card["slice_items"] == 100_000 and q["test"] == 1000
              and q["tune"] == 1000 and q["train"] == 20_000 and agree + lost >= 0.995 and unexplained == 0
              and h["rows_with_broken_values_nan"] == 0 and h["rows_with_broken_values_inf"] == 0
              and (metric != "cos" or h["all_zero_rows"] == 0))
        rep.check(f"{name}: prepared like the five", ok,
                  f"{card['items_full']:,} items, {h['numbers_per_item']} numbers, formula {card['formula']}; "
                  f"questions from {q['train_source']}; our top 10 agree with the file's on {agree:.2%} "
                  f"(+{lost:.2%} set aside as questions); answers differing beyond ties: {unexplained}; "
                  f"all-zero rows {h['all_zero_rows']}, exact duplicates {h['exact_duplicate_rows']:,}",
                  {"card": {k: card[k] for k in ("items_full", "questions", "agreement_with_vibe_top10_full",
                                                  "vibe_top10_items_we_set_aside_as_questions", "precision_check")}})


def locks(rep):
    refused = []
    for name in HELD_BACK:
        for call in (lambda: load(name, "slice"), lambda: truth(name, "slice_test")):
            try:
                call()
            except PermissionError:
                refused.append(True)
            else:
                refused.append(False)
    rep.check("our loader refuses every held-back dataset", all(refused), f"{sum(refused)}/{len(refused)} refused")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from bench.data import prepared\n"
            "prepared('glove')\n"
            "for n in ('arxiv', 'laion', 'yi', 'gist'):\n"
            "    try: prepared(n)\n"
            "    except PermissionError: print('refused', n)\n")
    run = subprocess.run([str(EXP / ".venv" / "bin" / "python"), "-I", "-c", code, str(EXP)],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    n = run.stdout.count("refused")
    rep.check("the experiments' loader refuses every held-back dataset", run.returncode == 0 and n == 4,
              f"{n}/4 refused" + (f"; error: {run.stderr.strip()[-200:]}" if run.returncode else ""))


# where step 22's rival runs put their results, in the experiments' results folder
STEP22_PLACES = ("step3/grid/", "step3/final/", "step3/choices-arxiv-laion-yi-gist.json", "step3/invalid-wrong-distance/")


def untouched(rep):
    """Before step 22, no result here or in the experiments names a held-back dataset. After it, only step 22's own
    runs do: ours in validate/results/step22.*, the rivals' in the experiments' step 3 folders; and after step 27,
    its own runs too (validate/results/step27*)."""
    pat = re.compile(r"\b(arxiv|laion|gist|yi)\b", re.I)
    hits = [p.name for p in OUT.glob("*") if p.is_file() and not p.name.startswith(("step14.", "step22.", "step27", *PART4_RESULTS))
            and pat.search(p.read_text(errors="ignore"))]
    exp = [str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_dir() is False and pat.search(str(p.relative_to(REF)))]
    exp += [str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_dir() and p.name in HELD_BACK]
    after22 = (OUT / "step22.json").exists()
    if after22:
        exp = [e for e in exp if not e.startswith(STEP22_PLACES + PART4_PLACES)]
    rep.check("nothing has been run on a held-back dataset" + (" but step 22" if after22 else ""), not hits and not exp,
              "no result names them" + (" outside step 22's runs" if after22 else "") if not hits and not exp
              else f"found: {', '.join((hits + exp)[:5])}")


def straight_line(rep):
    """Llama's numbers searched by straight-line distance: our data, a formula none of the five use."""
    X, Qt, Qr = load("llama", "slice"), load("llama", "q_tune"), load("llama", "q_train")
    X64 = X.astype(np.float64)
    xx = np.einsum("nd,nd->n", X64, X64)
    td = np.empty((len(Qt), 10))
    for i in range(0, len(Qt), 100):
        q = Qt[i:i + 100].astype(np.float64)
        d = xx[None, :] - 2 * q @ X64.T + np.einsum("nd,nd->n", q, q)[:, None]
        td[i:i + 100] = np.sort(np.partition(d, 9, axis=1)[:, :10], axis=1)
    idx = vindex.Index(X.shape[1], "l2", **FINAL)
    idx.build(X, np.arange(len(X)), train_questions=Qr)
    curve = []
    for L in L_SWEEP:
        ids, _, checks = idx.search(Qt, k=10, list_size=L, threads=16)
        curve.append({"L": L, "accuracy": accuracy(correct_counts(ids, Qt, X, td, "l2")), "checks": float(checks.mean())})
        if curve[-1]["accuracy"] >= 0.99:
            break
    best = max(c["accuracy"] for c in curve)
    rep.check("straight-line distance on Llama's numbers: a fixed list size reaches 99%", best >= 0.99,
              "; ".join(f"list {c['L']}: {c['accuracy']:.2%}" for c in curve[-3:]), {"curve": curve})
    targets = sweep_targets(idx, Qt, X, td, "l2")
    at = {t["target"]: t["accuracy"] for t in targets}
    rep.check("straight-line: the learned effort lands near its target", at[0.95] >= 0.935 and at[0.99] >= 0.98,
              f"target 95% gives {at[0.95]:.2%}, 99% gives {at[0.99]:.2%} (pass: at least 93.5% and 98%)",
              {"targets": targets})
    ids, dists, _ = idx.search(Qt[:200], k=10, list_size=100)
    safe = np.where(ids < 0, 0, ids)
    diff = X64[safe] - Qt[:200, None, :].astype(np.float64)
    exact = np.einsum("nmd,nmd->nm", diff, diff)
    rel = float(np.max(np.abs(dists - exact) / np.maximum(exact, 1e-12)))
    rep.check("straight-line: returned distances equal numpy's", rel <= 1e-4, f"largest gap {rel:.1e} (pass: 1e-4)")
    stats = idx.stats()
    rep.note(f"question links {'on' if stats['question_links'] else 'off'}, build space {stats['build_space']}, "
             f"unlike-items ratio {stats['ood_ratio']:.2f}")


def main():
    rep = Report("step14")
    downloads(rep)
    prepared(rep)
    locks(rep)
    untouched(rep)
    straight_line(rep)
    rep.finish()


if __name__ == "__main__":
    main()
