# Using vindex

Install, try, use from Python or Rust, the settings, and how to check the results. Paths in this page are from the repository's top folder.

## Get started

### 1. Install
Needs a Mac with Apple silicon, [rustup](https://rustup.rs), [uv](https://docs.astral.sh/uv/) (or plain `pip`), and Python 3.10 or newer (the commands below use 3.13, which `uv` downloads if needed).

```sh
git clone https://github.com/srir84/vector-index-rust.git
cd vector-index-rust
uv venv --python 3.13 .venv                          # or: python3 -m venv .venv
uv pip install --python .venv numpy maturin h5py     # or: .venv/bin/pip install numpy maturin h5py
tools/build_py.sh                                    # builds the package into .venv and checks it loads
```
The last line should print `vindex 0.1.0 built and loads`. The first build downloads the Rust version pinned in `rust-toolchain.toml` (a nightly; the file says why). `h5py` is only for the playground below.

For 1–7% more speed, `tools/build_py_pgo.sh` builds it guided by a practice run (needs `rustup component add llvm-tools-preview`).

### 2. Try it on a public dataset
The playground downloads a dataset, builds an index on it, searches it, and checks every answer against the true ones:
```sh
.venv/bin/python playground/play.py list                               # 22 datasets: text, photos, code, chatbot
.venv/bin/python playground/play.py run yi-128-ip                     # quickest: 0.27 GB download, a few minutes in all
.venv/bin/python playground/play.py run yahoo-minilm-384-normalized    # 1 GB download, then about 5 minutes to build
.venv/bin/python playground/play.py ui                                 # or the browser page: http://127.0.0.1:8035/ (Ctrl-C stops it)
```
It shows each build step as it runs. Then it shows the memory, the settings the index picked, and, at each target, the share of the true top 10 found and the searches per second. `playground/README.md` has the steps, the datasets, and how to use your own file.

### 3. Use it in your own code
```python
import numpy as np
import vindex

rng = np.random.default_rng(0)
X = rng.standard_normal((20_000, 64), dtype=np.float32)       # items: one row of numbers each
ids = np.arange(len(X))                                        # their ids: whole numbers, 0 or more
train = rng.standard_normal((2_000, 64), dtype=np.float32)     # sample questions, never your test ones
Q = rng.standard_normal((100, 64), dtype=np.float32)           # questions

idx = vindex.Index(64, metric="cos")                           # "cos" (angle), "ip" (best score) or "l2"
idx.build(X, ids, train_questions=train)                       # about a minute here: it tunes itself; tune="off" skips that

found, dists, checks = idx.search(Q, k=10, target=0.95)        # effort learned per question, aiming at 95%
found, dists, checks = idx.search(Q, k=10, list_size=100)      # or one fixed effort for every question
found, dists, checks = idx.search(Q, k=10, target=0.95, threads=8)   # spread the questions over 8 cores

idx.add(1_000_000, rng.standard_normal(64, dtype=np.float32))  # changes, one at a time
idx.delete(0)
idx.update(1, rng.standard_normal(64, dtype=np.float32))

idx.save("my_index")                                           # a folder
idx = vindex.Index.load("my_index", numbers="ssd")             # full numbers stay on SSD: search only
print(idx.memory())                                            # bytes kept in memory, by part
print(idx.stats())
```

- **Build:** uses all cores (`threads=N` to limit). While it runs, `idx.build_progress()` from another thread gives the phase, step, done and total.
- **Numbers:** `float32` only (convert with `.astype(np.float32)`; `float64` is refused), every row the same length, no NaN or infinity. Questions must come from the same model as the items, or its paired model (for example CLIP's text and image halves).
- **Metric:** the one the model was made for. `cos` suits most text and image models; `ip` is for models scored by a plain dot product; `l2` is straight-line distance.
- **Answers:** `found` holds ids (`-1` where fewer than `k` exist), `dists` their distances (smaller is closer), `checks` the work per question.
- **Training questions** are needed for the learned effort. Without them, pass `learned_effort=False` and search with `list_size=`.
- **Threads:** searches from several Python threads run side by side; a change waits until running searches finish.
- **On SSD:** an index loaded with `numbers="ssd"` searches only; load it with `numbers="ram"` (the default) to change it.

## It fits itself to your data
- **At build** (`tune="auto"`, the default): with at least 10,000 items and 800 sample questions, it tries a few choices of links per item and slack on a trial sample of the items (a fifth of them, at least 20,000 and at most 100,000) and keeps the fastest at 95%. After the build, on the real index, it tries 4-bit copies, fewer re-checks, and, when the sample answers bunch up on a few items, extra links between those popular items. A close call is timed longer, and a new choice replaces the one in use only when clearly faster, so fresh builds mostly pick the same settings. `idx.tuning()` shows what it tried. Builds take 2.6–7× longer than without it on the 100,000-item samples; with it, the full-size builds took under a minute (Llama) to 15 minutes (GIST1M) on all cores.
- **In use:** it keeps usage stats, and sets aside a few searches for spot checks (up to about 2% of the search work). `idx.audit()` checks those against every item; when they show a target missed, it learns its effort table again. `idx.suggest()` suggests rebuilds with their measured gain.
- **A target is aimed at, not promised:** in the playground, Yahoo's target 0.95 found 93.6% of the true top 10. For the benchmark, the index picked each dataset's target on tuning questions, and the test questions then found 94.68% to 96.24%.

```python
print(idx.usage())                  # searches, checks, time per search, how hard and how "unlike the items" questions are
print(idx.audit())                  # spot checks so far: accuracy per target, with a margin of error
idx.relearn_effort()                # learn the effort table from the spot-checked questions (audit() does this when a target is missed)
for s in idx.suggest():             # notes, and a rebuild when a trial shows it is at least 15% faster
    print(s["what"], s["why"])
real = idx.checked_questions()      # the spot-checked questions, to rebuild with when a suggestion says so
```

## Settings
Pass them as keywords, e.g. `vindex.Index(64, "cos", r=48, rerank=100)`. The main ones:

| Setting | Default | What it does |
|---|---|---|
| `r` | 32 | links per item |
| `l_build` | 200 | effort while building: higher makes a better map, more slowly |
| `alpha` | 1.2 | how spread out each item's links must be |
| `rerank` | 50 | how many of the best are re-checked with the full numbers (the tuning also tries 15 and 25) |
| `learned_effort` | true | learn the effort table at build (needs training questions) |
| `question_links` | auto | links between training questions' answers: `auto`, `on` or `off` |
| `build_space` | auto | links for best-score data built on flipped copies: `auto`, `same` or `flipped` |
| `seed` | 7 | the build order's random seed |
| `tune` | auto | pick links, slack and copy size at build: `auto` or `off` |
| `max_bytes_per_item` | 0 | the tuning skips choices that keep more memory than this (0: no limit) |
| `question_bits` | 16 | a question as 16-bit whole numbers against the copies (32: floats, 8: 8-bit) |
| `code_bits` | 8 | copies of 8 bits, or 4 (two numbers per byte; picked by the tuning when faster), or 0 (full numbers) |
| `audit_budget` | 0.02 | spot checks may cost this share of the search work (0: off) |
| `relearn_every` | 500 | after this many new spot checks, relearn the effort table if a target is missed (0: never) |
| `track_usage` | true | keep the usage stats |
| `effort_apart` | true | learn the effort table on sample questions the question links do not use |
| `renumber` | true | put linked items next to each other in memory after the build |
| `popular_links` | auto | extra links between popular items (those in the sample questions' answers): `auto` (when those answers bunch up on a few items, kept if faster), `on` or `off` |
| `repair` | occlude | how a delete re-links its neighbours: `occlude` (take a replacement only where it reaches a new way), `append` (parts 1 to 4) or `prune` |
| `n_replace` | 8 | replacement links a delete offers each neighbour (parts 1 to 4: 3) |

`DESIGN.md` explains every setting and how each part works.

## The bucket search (an experiment)
`idx.build_buckets()` and `idx.search_buckets(Q, k=10, n_probe=8, pool=100)` group the items into buckets and scan the nearest buckets' 4-bit codes, as ScaNN does. They are kept only so steps 39 and 40 can be rerun. `search` never uses them, and they are not saved, changed or tuned with the index (`DESIGN.md` section 23).

## Use from Rust
`crates/vindex/examples/basic.rs` builds, searches, changes, saves and loads:

```sh
cargo run --release -p vindex --example basic
```

## Checking it
```sh
cargo test -p vindex --release           # Rust tests on small random data, including recall@10 against exact search
uv pip install --python .venv pytest    # once, for the Python tests
.venv/bin/python -I -m pytest tests/     # the Python package
REUSE_GRID=1 validate/run_all.sh         # each build step's check, against the experiments' recorded results
validate/run_all.sh 9 10                 # or just some steps
```

- **CI** (`.github/workflows/ci.yml`) runs clippy and both test suites on macOS (Apple silicon, the NEON code) and Linux x86 (the plain-code fallback).

- **`validate/run_all.sh`** needs [vector-index-experiments](https://github.com/srir84/vector-index-experiments) cloned next to this repository's folder (or its path in `VINDEX_EXPERIMENTS`), with its prepared data: its README says how to download and prepare it (about 57 GB). Without it, the results are in `validate/results/`. `cargo test` and the playground need nothing else. With the data, `run_all.sh` takes about an hour and a half.
  - `REUSE_GRID=1` reuses steps 18 and 21's saved grids: every choice of links, slack and copy size, built at full size and timed. Without it, they are made again, which takes about an hour more.
- **Steps that time searches** (11, 12, 18 to 21 and 24 to 26) should run on an otherwise idle machine.
- **The held-back tests and full-size runs are not part of it.** Each is meant to run once:
  - part 2: `validate/step22.py rivals`, then `validate/step22.py`;
  - part 3: `validate/step27.py rivals`, then `validate/step27.py`;
  - part 4: steps 28 to 32 (`validate/step29.py`, `validate/step30.py rivals`, then `validate/step30.py`, and so on);
  - part 5: `validate/step42.py ours`, `validate/step42.py changes` and `validate/step42.py rivals`, then `validate/step42.py`. These take about 6 hours.
  - ScaNN runs in Docker, which must be running for the rivals' runs.
- **The reports' numbers are checked too:** `validate/step33.py` (the results page), `validate/step43.py` (`REPORT.md` part 5 and `WRITE-UP.md`) and `validate/step47_readme.py` (the two tables in `docs/BENCHMARK.md`) check every number against the results files. Steps 44 and 45 changed the code after the freeze, so `step43.py`'s first check ("the code is as frozen") fails. `STEPS.md` steps 44 and 45 show that the changed code builds the same index, byte for byte.
- **`STEPS.md`** says what each step checks and what it found.

## Files
| File | What it holds |
|---|---|
| `WRITE-UP.md` | one page for readers new to it, with a link to the results page |
| `docs/BENCHMARK.md`, `docs/GUIDE.md`, `docs/HOW-IT-WORKS.md` | the benchmark in full, this guide, and how it works in standard terms |
| `DESIGN.md` | how each part works, with the default settings |
| `ARCHITECTURE.md` | the parts and how data sits in memory and on disk |
| `REPORT.md` | results for all five parts, newest first: against the rivals on 13 datasets, met and missed targets |
| `SPEC.md`, `STEPS.md` | the project's history: the targets set before each part, and each build step (0 to 43 in five parts; 44 to 48 after the freeze) with its check and result |
| `playground/` | try it on a public dataset or your own file, from the command line or a browser page |
| `demo/` | ask questions of GooAQ's 1.45 million Google answers on your own machine |
| `crates/vindex` | the index (Rust library) |
| `crates/vindex-py` | the Python package |
| `validate/` | one check per step, plus `run_all.sh` |
| `tests/` | the Python package's tests |
| `tools/` | `build_py.sh` and `build_py_pgo.sh` (build the package), `draw_report.py` (the report's and README's charts, and the benchmark table), `results_page.py` (the results page), `record_playground.py` (the playground recording) |
