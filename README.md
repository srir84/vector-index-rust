# vindex

**A graph vector search index in Rust, with Python bindings, tuned for Apple silicon.**

- **Graph:** a Vamana graph (DiskANN's build), searched with beam search.
- **Compression:** the search compares 8-bit or 4-bit scalar-quantized vectors, then re-ranks the best few with the full float32 vectors. The full vectors can stay on SSD.
- **Search budget:** set per query from a table learned on sample queries. You ask for a recall target, for example 0.95.
- **Out-of-distribution queries:** when queries look unlike the data (text searching photos), it adds RoarGraph-style links built from the sample queries' answers. For maximum inner product, the graph is built on transformed vectors.
- **It tunes itself:** at build time it picks graph degree, pruning alpha, code size and re-rank depth on a sample. In use, it spot-checks a few searches and relearns the budget table if a target is missed.
- **Changes:** add, delete and update in place, then save and load.

Each piece builds on published work; "How it compares to known methods" below says which. It runs on macOS with Apple silicon only: it has NEON kernels and no x86 SIMD yet.

## How fast is it?

![vindex against the fastest other index tried, on 13 public datasets](diagrams/benchmark.png)

**The test:** 13 public datasets, every item of each (187,843 to 1.45 million), 1,000 test queries each, one core of an Apple M3 Max, at 95% recall@10.

- **Faster than the fastest other index tried on 11 of 13,** by 1.08× to 2.19×. Timing a pair again moved its ratio by up to 16% (part 4: Yandex 1.30× then 1.10×, ImageNet 1.60× then 1.86×), so Yandex's 1.08× is a tie.
- **The largest leads are on chatbot attention vectors** (Llama, Yi; maximum inner product): 1.75× to 1.95× RoarGraph, which is built for such queries.
- **Slower on the two GloVe word datasets:** ScaNN is 1.36× and 1.37× as fast. On GloVe-100 Glass is faster too (vindex 0.93×). GloVe-100 is the one dataset kept untouched until the code was frozen.
- **On all 16 cores the lead is smaller,** and Glass wins on Landmark and GIST1M (vindex 0.69× and 0.68×).

<details open><summary><b>The numbers</b></summary>

| Dataset | vindex (queries/s) | Fastest other index | vindex ÷ it | at 90% | all 16 cores | hnswlib |
|---|---|---|---|---|---|---|
| LAION: text to photos | 10,566 | Glass 4,833 | **2.19×** | **1.58×** | **1.07×** | 801 |
| Llama, 2nd head: chatbot attention | 57,607 | RoarGraph 29,507 | **1.95×** | **2.06×** | **1.50×** | never 95% |
| Yi: chatbot attention | 44,284 | RoarGraph 24,371 | **1.82×** | **1.94×** | **2.02×** | never 95% |
| Llama: chatbot attention | 100,112 | RoarGraph 57,226 | **1.75×** | **1.95×** | **1.53×** | never 95% |
| ImageNet: text to photos | 15,379 | Glass 9,068 | **1.70×** | **1.43×** | **1.25×** | 985 |
| GooAQ: Google answers | 39,545 | Glass 26,512 | **1.49×** | **1.50×** | **1.14×** | 3,221 |
| NYTimes: news | 12,169 | Glass 8,523 | **1.43×** | **1.16×** | **1.38×** | 701 |
| arXiv: paper summaries | 29,407 | Glass 21,627 | **1.36×** | **1.15×** | **1.19×** | 2,694 |
| GIST1M: photo features | 3,921 | Glass 2,905 | **1.35×** | **1.27×** | 0.68× | 302 |
| Landmark: photos | 46,560 | Glass 34,762 | **1.34×** | **1.44×** | 0.69× | 5,194 |
| Yandex: text to photos | 14,971 | Glass 13,872 | **1.08×** | **1.04×** | **1.06×** | 2,187 |
| GloVe: words | 2,907 | ScaNN 3,956 | 0.73× | 0.72× | 0.76× | 297 |
| GloVe-100: words | 7,184 | ScaNN 9,828 | 0.73× | 0.93× | 0.90× | 1,012 |

- **vindex ÷ it:** one core, 95% recall@10, read off each index's effort-vs-recall curve at exactly 95%.
- **At 90%:** the same at 90% recall, against the fastest other index there.
- **All 16 cores:** queries/s with every core, at the setting each index picked for 95% on the tuning queries (test recall 94% to 97%). This compares the indexes roughly, not at an exact recall. ScaNN's container had 8 cores.
- **hnswlib:** plain float32 HNSW, which has no ARM SIMD (see below). It is not a fair speed reference on this machine.

Made by `tools/draw_report.py readme`. vindex's numbers come from `validate/results/step42-ours.json`. The others' come from [vector-index-experiments](https://github.com/srir84/vector-index-experiments) (`results/step6/`, and `results/step7/` for GloVe-100).
</details>

**Read this before trusting the numbers**
- **The machine favours vindex.**
  - vindex has hand-written NEON kernels and was built with profile-guided optimization (`tools/build_py_pgo.sh`).
  - The others were written for x86 first. hnswlib has no ARM SIMD: its distance loop compiles to scalar code without fast-math. Glass's ARM kernels are commented out upstream, so it runs plain loops, which the compiler auto-vectorizes. RoarGraph's x86 SIMD goes through the SIMDe translation layer. ScaNN ran in a Linux container on the same Mac.
  - On an x86 server the results may differ.
- **Not every strong index was tried.**
  - SymphonyQG, the fastest on many VIBE datasets, needs AVX-512, which Apple chips lack.
  - NGT-QG, LoRANN and Faiss were not run.
  - "Fastest other index" means the fastest of Glass, ScaNN, RoarGraph, CoreNN and hnswlib. At full size, only the fastest on 100,000-item samples, plus hnswlib, were run.
- **The others got small settings grids.** Each graph index tried 9 settings (32/64/96 links × build effort 100/200/400), and ScaNN tried 6. Glass picked the largest corner on 4 of 8 datasets, so a bigger grid might help it.
- **vindex learns from sample queries.** It uses 20,000 training queries from the same source as the test queries, for its search budget and its extra links. RoarGraph uses them too; Glass, ScaNN and hnswlib use none.
- **The held-back datasets were held back for one round each.**
  - The project was built in five rounds. Eight datasets were each kept out of one round, then measured, then used like the others.
  - Only GloVe-100 was untouched until the code was frozen, and there vindex came third.
  - Test queries were never used to pick vindex's settings, but they measured progress between rounds. Step 12 kept speed-only code changes (answers unchanged) when they were at least 10% faster on test queries.
- **Can be rerun:** every script, setting and result is in this repo and in [vector-index-experiments](https://github.com/srir84/vector-index-experiments), including each index's version, commit and patch (`baselines/VERSIONS.md`).

**Where it is not better**
- **Build time:** slower than the fastest other index on all 12 datasets where both were timed, by 1.5× (GIST1M) to 23× (Yi). Trying settings is part of it: on 100,000-item samples, builds took 2.6–7× longer with it. `tune="off"` skips it, at a cost in speed.
- **Memory:** with everything in memory, about as much as Glass or a little less, and more than RoarGraph and ScaNN. It can leave the full vectors on SSD, using 0.17–0.70× hnswlib's memory, but then answers 2–23× fewer queries per second.
- **Unreachable items:** a few items can't be found by any search: 0 to 1,209 per dataset, out of up to 1.45 million (GIST1M has the most).
- **Scope:** one machine, tested up to 1.45 million items, no filtered search, and macOS on Apple silicon only.

**More detail:**
- `WRITE-UP.md`: one page for newcomers.
- `REPORT.md`: every table, from part 1 to part 5.
- [The results page](https://srir84.github.io/vector-index-rust/page/results.html): every number, with a chart for each dataset (`page/results.html`).

**Demo:** the playground downloads a dataset, builds an index while showing each step, searches it, and checks the answers (step 2 below). Waits are sped up about 7×; the page's own clock shows the real time.

![The playground page: download a dataset, build an index, search it](playground/playground.gif)

## Get started

### 1. Install
Needs a Mac with Apple silicon, [rustup](https://rustup.rs), [uv](https://docs.astral.sh/uv/) (or plain `pip`), and Python 3.10 or newer.

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
.venv/bin/python playground/play.py run yahoo-minilm-384-normalized    # 1 GB; about 5 minutes
.venv/bin/python playground/play.py ui                                 # or use the browser page: http://127.0.0.1:8035/
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
idx.build(X, ids, train_questions=train)                       # uses all cores; threads=N to limit

found, dists, checks = idx.search(Q, k=10, target=0.95)        # effort learned per question, aiming at 95%
found, dists, checks = idx.search(Q, k=10, list_size=100)      # or one fixed effort for every question
found, dists, checks = idx.search(Q, k=10, target=0.95, threads=8)   # spread the questions over 8 cores

idx.add(1_000_000, rng.standard_normal(64, dtype=np.float32))  # changes, one at a time
idx.delete(0)
idx.update(1, rng.standard_normal(64, dtype=np.float32))

print(idx.build_progress())                                    # from another thread while build() runs: phase, step, done, total
idx.save("my_index")                                           # a folder
idx = vindex.Index.load("my_index", numbers="ssd")             # full numbers stay on SSD: search only
print(idx.memory())                                            # bytes kept in memory, by part
print(idx.stats())
```

- **Numbers:** `float32` only (convert with `.astype(np.float32)`; `float64` is refused), every row the same length, no NaN or infinity. Questions must come from the same model as the items, or its paired model (for example CLIP's text and image halves).
- **Metric:** the one the model was made for. `cos` suits most text and image models; `ip` is for models scored by a plain dot product; `l2` is straight-line distance.
- **Answers:** `found` holds ids (`-1` where fewer than `k` exist), `dists` their distances (smaller is closer), `checks` the work per question.
- **Training questions** are needed for the learned effort. Without them, pass `learned_effort=False` and search with `list_size=`.
- **Threads:** searches from several Python threads run side by side; a change waits until running searches finish.
- **On SSD:** an index loaded with `numbers="ssd"` searches only; load it with `numbers="ram"` (the default) to change it.

## How it works
- **Graph:** a Vamana graph. Each item links to at most `r` others, chosen with alpha-pruning (`alpha`, called "slack" in the docs). A search is a beam search from one start item, with a candidate list of size L.
- **Compressed search, exact re-rank:** the beam search compares 8-bit (or 4-bit) per-dimension scalar-quantized codes, with the query as 16-bit integers. The best `rerank` (default 50) are then re-scored with the full float32 vectors.
- **Per-query budget:** each query starts with L = 10. The distance of its 10th candidate puts it in one of 8 groups, and each group has its own L, so hard queries get a longer search. A table of L per group is learned on sample queries for each recall target, so one index serves any target.
- **Out-of-distribution links:** when queries look unlike the items, each sample query's answers are linked to each other, as in RoarGraph.
- **Inner product:** for maximum inner product data, the graph is built on the Möbius-transformed vectors x / |x|², and searched on the real ones.
- **Deletes:** a deleted item's in-neighbours are repaired at once. Each takes up to 8 of the deleted item's links, keeping one only where alpha-pruning says it reaches a new direction. After 210,831 changes to Landmark, recall stayed within 0.63 points of a fresh build.
- **Memory layout:** after the build, items are renumbered in breadth-first order from the start item, so linked items sit near each other in memory.

`DESIGN.md` explains each part and every default. The ideas were first tried in Python in [vector-index-experiments](https://github.com/srir84/vector-index-experiments) (`bench/ours/`).

### How it compares to known methods
None of these parts is new on its own. The work is in combining them, the self-tuning and checking, and a fast ARM implementation.

| Part | Builds on | What vindex does differently |
|---|---|---|
| Graph build and search | Vamana / DiskANN ([Subramanya et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)) | Nothing new. |
| Quantized search, exact re-rank | Glass (SQ8 on HNSW), Faiss scalar quantizer | 4-bit codes when the tuning finds them faster, and the query kept as 16-bit integers. |
| Per-query budget | LAET ([Li et al., SIGMOD 2020](https://dl.acm.org/doi/10.1145/3318464.3380600)), DARTH ([2025](https://arxiv.org/abs/2505.19001)) | One cheap feature (the 10th candidate's distance) and a lookup table, no model. Any recall target at query time. |
| Out-of-distribution links | RoarGraph ([Chen et al., VLDB 2024](https://arxiv.org/abs/2408.08933)), OOD-DiskANN ([Jaiswal et al., 2022](https://arxiv.org/abs/2211.12850)) | Switched on automatically by comparing query-to-item and item-to-item distances, and redone every 3,000 changes. |
| Inner product on a graph | Möbius transformation ([Zhou et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/0fd7e4f42a8b4b4ef33394d35212b13e-Abstract.html)) | Used only to build the graph. |
| Deletes | FreshDiskANN ([Singh et al., 2021](https://arxiv.org/abs/2105.09613)) | Repaired at once, without batch consolidation. |
| Picking settings | Faiss's parameter autotuning | Trial builds on a sample, timed. A new choice wins only when clearly faster, so rebuilds pick the same. |

### Words used in the code and docs
The code and older docs use plain words for standard terms:

| In the code | Standard term |
|---|---|
| links, `r` | graph edges, max degree R |
| slack, `alpha` | pruning parameter α |
| list size, `list_size` | search list size L (ef in HNSW) |
| 1-byte or 4-bit copies, `code_bits` | SQ8 / SQ4 scalar-quantized vectors |
| full numbers | float32 vectors |
| re-check, `rerank` | exact re-ranking |
| questions; training or sample questions | queries; training queries |
| checks | distance computations |
| effort, `target` | search budget; target recall@10 |
| flipped copies | Möbius-transformed vectors |
| question links | query-guided (RoarGraph-style) edges |
| best score, `ip` | maximum inner product |

### It fits itself to your data
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

### Settings
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

### The bucket search (an experiment)
`idx.build_buckets()` and `idx.search_buckets(Q, k=10, n_probe=8, pool=100)` group the items into buckets and scan the nearest buckets' 4-bit codes, as ScaNN does. They are kept only so steps 39 and 40 can be rerun. `search` never uses them, and they are not saved, changed or tuned with the index (`DESIGN.md` section 23).

## Use from Rust
`crates/vindex/examples/basic.rs` builds, searches, changes, saves and loads:

```sh
cargo run --release -p vindex --example basic
```

## Checking it
```sh
cargo test -p vindex --release           # Rust tests on small random data, including recall@10 against exact search
.venv/bin/python -I -m pytest tests/     # the Python package (uv pip install --python .venv pytest)
REUSE_GRID=1 validate/run_all.sh         # each build step's check, against the experiments' recorded results
validate/run_all.sh 9 10                 # or just some steps
```

- **CI** (`.github/workflows/ci.yml`) runs clippy and both test suites on macOS (Apple silicon, the NEON code) and Linux x86 (the plain-code fallback).

- **`validate/run_all.sh`** needs [vector-index-experiments](https://github.com/srir84/vector-index-experiments) cloned next to this folder (or its path in `VINDEX_EXPERIMENTS`), with its prepared data: its README says how to download and prepare it (about 57 GB). Without it, the results are in `validate/results/`. `cargo test` and the playground need nothing else. With the data, `run_all.sh` takes about an hour and a half.
  - `REUSE_GRID=1` reuses steps 18 and 21's saved grids: every choice of links, slack and copy size, built at full size and timed. Without it, they are made again, which takes about an hour more.
- **Steps that time searches** (11, 12, 18 to 21 and 24 to 26) should run on an otherwise idle machine.
- **The held-back tests and full-size runs are not part of it.** Each is meant to run once:
  - part 2: `validate/step22.py rivals`, then `validate/step22.py`;
  - part 3: `validate/step27.py rivals`, then `validate/step27.py`;
  - part 4: steps 28 to 32 (`validate/step29.py`, `validate/step30.py rivals`, then `validate/step30.py`, and so on);
  - part 5: `validate/step42.py ours`, `validate/step42.py changes` and `validate/step42.py rivals`, then `validate/step42.py`. These take about 6 hours.
  - ScaNN runs in Docker, which must be running for the rivals' runs.
- **The reports' numbers are checked too:** `validate/step33.py` (the results page) and `validate/step43.py` (`REPORT.md` part 5 and `WRITE-UP.md`) check every number against the results files. Steps 44 and 45 changed the code after the freeze, so `step43.py`'s first check ("the code is as frozen") fails. `STEPS.md` steps 44 and 45 show that the changed code builds the same index, byte for byte.
- **`STEPS.md`** says what each step checks and what it found.

## Files
| File | What it holds |
|---|---|
| `WRITE-UP.md` | one page for readers new to it, with a link to the results page |
| `DESIGN.md` | how each part works, with the default settings |
| `ARCHITECTURE.md` | the parts and how data sits in memory and on disk |
| `REPORT.md` | results for all five parts, newest first: against the rivals on 13 datasets, met and missed targets |
| `SPEC.md`, `STEPS.md` | the project's history: the targets set before each part, and each build step (0 to 43 in five parts; 44 and 45 after the freeze) with its check and result |
| `playground/` | try it on a public dataset or your own file, from the command line or a browser page |
| `demo/` | ask questions of GooAQ's 1.45 million Google answers on your own machine |
| `crates/vindex` | the index (Rust library) |
| `crates/vindex-py` | the Python package |
| `validate/` | one check per step, plus `run_all.sh` |
| `tests/` | the Python package's tests |
| `tools/` | `build_py.sh` and `build_py_pgo.sh` (build the package), `draw_report.py` (the report's and README's charts), `results_page.py` (the results page), `record_playground.py` (the playground recording) |

## License
MIT (`LICENSE`). The datasets have their own licences: see each source.
