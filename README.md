# vindex

A vector search index in Rust, with a Python package. Give it items (each a list of numbers, such as text embeddings) and it finds the 10 closest items to a question, fast, without checking every item.

It walks a map of links between nearby items (DiskANN's method), plus the fixes that survived the experiments in `../vector-index-experiments`:
- **Effort per question:** each question gets its own effort, from a table learned on sample questions.
- **Links between answers:** where questions look unlike the items, it adds links between each sample question's answers.
- **1-byte copies:** memory holds a 1-byte copy of each number (or a 4-bit one, when the tuning finds it faster); the full numbers can stay on SSD.
- **Repairs on delete:** a deleted item's neighbours are re-linked right away, each new link kept only where it reaches a new way.

It also picks its own settings for each dataset and keeps itself on target in use (see "It fits itself to your data" below).

**How it does** (part 5, the latest; `REPORT.md` has the details and pictures, `WRITE-UP.md` a one-page summary): on 13 full-size public datasets (187,843 to 1.45 million items), at 95% of the true top 10, on one core of an M3 Max, against the fastest rival on each and HNSW:
- **Ahead of the fastest rival on 11 of 13:** 1.08× (Yandex, against Glass) to 2.19× (LAION, against Glass). Eight of the 13 were held back while the index was built.
- **Behind on two:** ScaNN is 1.36× as fast on GloVe and 1.37× on GloVe-100, the dataset held back for part 5. A ScaNN-style bucket search was tried (steps 39 and 40) but reached only 0.88× ScaNN, so it is not part of the index.
- **Against HNSW:** 6.8–17.4× as fast, wherever HNSW reaches 95%. On the three chatbot datasets it never does.
- **Changes:** after adding, deleting and updating 10% of the items each, accuracy stays within 1 point of a fresh build on all 13. No deleted item ever came back.
- **Steady builds, almost:** three fresh builds pick the same settings on 11 of 12 samples. ImageNet's sample still flips between two settings, and Yandex's full-size build can come out up to 17% slower than its luckiest one.
- **Memory:** with everything in memory, about as much as Glass or a little less, and more than RoarGraph and ScaNN. With the full numbers left on SSD, 0.17–0.70× HNSW's memory, but 2–23× fewer searches per second.

## Install
Needs macOS on Apple silicon (tested on an M3 Max with macOS 27), [rustup](https://rustup.rs), and Python 3.10 or newer.

```sh
uv venv --python 3.13 .venv                    # or: python3 -m venv .venv
uv pip install --python .venv numpy maturin    # or: .venv/bin/pip install numpy maturin
tools/build_py.sh                              # builds the package into .venv and checks it loads
```

The first build downloads the Rust version pinned in `rust-toolchain.toml` (a nightly; the file says why).

For 1–7% more speed, `tools/build_py_pgo.sh` builds it guided by a practice run (needs `rustup component add llvm-tools-preview`).

## Use from Python
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

idx.save("my_index")                                           # a folder
idx = vindex.Index.load("my_index", numbers="ssd")             # full numbers stay on SSD: search only
print(idx.memory())                                            # bytes kept in memory, by part
print(idx.stats())
```

- **Answers:** `found` holds ids (`-1` where fewer than `k` exist), `dists` their distances (smaller is closer), `checks` the work per question.
- **Training questions** are needed for the learned effort. Without them, pass `learned_effort=False` and search with `list_size=`.
- **Threads:** searches from several Python threads run side by side; a change waits until running searches finish.
- **On SSD:** an index loaded with `numbers="ssd"` searches only; load it with `numbers="ram"` (the default) to change it.

### It fits itself to your data
- **At build** (`tune="auto"`, the default): with at least 10,000 items and 800 sample questions, it tries a few choices of links per item and slack on a fifth of the items and keeps the fastest at 95%. After the build, on the real index, it tries 4-bit copies, fewer re-checks, and, when the sample answers bunch up on a few items, extra links between those popular items. A close call is timed longer, and a new choice replaces the one in use only when clearly faster, so fresh builds mostly pick the same settings. `idx.tuning()` shows what it tried. Builds take 2.6–7× longer than without it on the 100,000-item samples; with it, the full-size builds took under a minute (Llama) to 15 minutes (GIST1M) on all cores.
- **In use:** it keeps usage stats, and now and then spot-checks a search against every item (about 2% of the search work). It learns its effort table again when the spot checks show a target missed, and suggests rebuilds with their measured gain.

```python
print(idx.usage())                  # searches, checks, time per search, how hard and how "unlike the items" questions are
print(idx.audit())                  # spot checks so far: accuracy per target, with a margin of error
idx.relearn_effort()                # learn the effort table from the spot-checked questions (also happens by itself)
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
cargo test -p vindex --release      # unit tests on small made-up data
REUSE_GRID=1 validate/run_all.sh    # each build step's check, against the experiments' recorded results
validate/run_all.sh 9 10            # or just some steps
```

- **`validate/run_all.sh`** needs `../vector-index-experiments` with its prepared data. It takes about an hour and a half.
  - `REUSE_GRID=1` reuses steps 18 and 21's saved grids: every choice of links, slack and copy size, built at full size and timed. Without it, they are made again, which takes about an hour more.
- **Steps that time searches** (11, 12, 18 to 21 and 24 to 26) should run on an otherwise idle machine.
- **The held-back tests and full-size runs are not part of it.** Each is meant to run once:
  - part 2: `validate/step22.py rivals`, then `validate/step22.py`;
  - part 3: `validate/step27.py rivals`, then `validate/step27.py`;
  - part 4: steps 28 to 32 (`validate/step29.py`, `validate/step30.py rivals`, then `validate/step30.py`, and so on);
  - part 5: `validate/step42.py ours`, `validate/step42.py changes` and `validate/step42.py rivals`, then `validate/step42.py`. These take about 6 hours.
  - ScaNN runs in Docker, which must be running for the rivals' runs.
- **The reports' numbers are checked too:** `validate/step33.py` (the results page) and `validate/step43.py` (`REPORT.md` part 5 and `WRITE-UP.md`) check every number against the results files.
- **`STEPS.md`** says what each step checks and what it found.

## Files
| File | What it holds |
|---|---|
| `SPEC.md` | what the index must do, and the targets |
| `ARCHITECTURE.md` | the parts and how data sits in memory and on disk |
| `DESIGN.md` | how each part works, with the default settings |
| `STEPS.md` | the build steps (0 to 43, in five parts), each one's check and result |
| `REPORT.md` | results for all five parts, newest first: against the rivals on 13 datasets, met and missed targets |
| `WRITE-UP.md` | one page for readers new to it, with a link to the results page |
| `demo/` | ask questions of GooAQ's 1.45 million Google answers on your own machine |
| `crates/vindex` | the index (Rust library) |
| `crates/vindex-py` | the Python package |
| `validate/` | one check per step, plus `run_all.sh` |
| `tools/` | `build_py.sh` and `build_py_pgo.sh` (build the package), `draw_report.py` (the report's pictures), `results_page.py` (the results page) |
