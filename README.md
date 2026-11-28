# vindex

**Fast "find similar" search for AI apps, in Rust and Python, tuned for Apple silicon Macs.**

AI models turn text, photos and other data into lists of numbers (vectors), where similar things get similar numbers. Finding the vectors closest to a question is how a chatbot looks up documents, how a photo app finds "dog on a beach", and how a shop picks "more like this". Doing that in a fraction of a millisecond over millions of items, without comparing against every one, is the job of a *vector index*.

vindex is a vector index that tunes itself. Finding every true match means comparing against everything, which is slow; settling for most of them is what makes search fast. You say how many of the true closest matches you want found, for example 95%, and vindex picks its own settings and search effort to aim for that as fast as it can (in tests, a 95% target found 93.6% to 96.2%).

## Why it's interesting
- **Fast on a Mac:** on one core of an M3 Max, faster than the fastest other index tried on 9 of 13 public datasets (1.12× to 2.08×), level on 2, slower on 2.
- **No settings to guess:** you give an accuracy target, not tuning parameters. It learns how hard to search for each question, and spot-checks itself while in use.
- **Handles hard queries:** on the lookups a chatbot makes inside its own model while reading long text (attention vectors), it is 1.75× to 1.95× as fast as RoarGraph, which was built for them, and Faiss HNSW never finds 95%. On text searching photos it is 2.08× (LAION) and 1.38× (ImageNet) as fast as Faiss IVF-PQ, and level with Glass on Yandex (0.93×).
- **Changes in place:** add, delete and update items without a rebuild. After adding 10% more items, deleting 10% and updating 10%, accuracy stayed within 1 point of a fresh build's on all 13 datasets (`REPORT.md`).
- **Checked in the open:** scripts recompute the benchmark tables, the write-up and the results page from the raw results files, and the whole history, including the misses, is in the repo.

## How fast is it?

![vindex against the fastest other index tried, on 13 public datasets](diagrams/benchmark.png)

Queries per second on one core of an Apple M3 Max, finding 95% of each question's true 10 closest items (recall@10), on 13 public datasets of 187,843 to 1.45 million items:
- **Faster on 9:** 1.12× to 2.08× the fastest other index tried.
- **Level on 2** (within 10%): GIST1M and Yandex.
- **Slower on 2:** the two GloVe word datasets, where ScaNN is 1.36× and 1.37× as fast.
- **On all 16 cores** the lead is smaller: vindex is slower on 5 of the 13 (Glass is faster on Landmark, GIST1M and Yandex).

The others include Google's ScaNN, Meta's Faiss (HNSW, HNSW-SQ8 and IVF-PQ), and the research indexes Glass, RoarGraph, CoreNN and LoRANN.

**The catches,** in short ([docs/BENCHMARK.md](docs/BENCHMARK.md) has them all, with every number):
- **The machine favours vindex.** It has hand-written ARM (NEON) code and a profile-guided build. Glass, the fastest rival on 5 datasets, runs without its ARM kernels (they are switched off upstream). On an Intel or AMD server the results may differ.
- **We set the rivals up.** Most were tuned over a small grid of settings on sample queries, widened where a pick sat at the edge. Glass and RoarGraph ran only on the datasets where they led on 100,000-item samples. Their authors may know better settings.
- **Only one dataset was untouched until the end.** GloVe-100 was kept out until the code was frozen, and vindex came third on it. Eight others were each held back for one round, and test queries measured progress between rounds, though they never picked vindex's settings.

## Try it in 5 minutes
Needs a Mac with Apple silicon, [rustup](https://rustup.rs) and [uv](https://docs.astral.sh/uv/). It builds from source (version 0.1.0, not on PyPI); the first build downloads the Rust version the project pins.

```sh
git clone https://github.com/srir84/vector-index-rust.git
cd vector-index-rust
uv venv --python 3.13 .venv
uv pip install --python .venv numpy maturin h5py
tools/build_py.sh                                    # prints "vindex 0.1.0 built and loads"
.venv/bin/python playground/play.py run yi-128-ip    # downloads 0.27 GB, builds, searches, checks every answer
.venv/bin/python playground/play.py ui               # or the same in a browser: http://127.0.0.1:8035/
```

![The playground page: download a dataset, build an index, search it](playground/playground.gif)

*The playground in a browser. Waits are sped up about 7×; the page's own clock shows the real time.*

## Use it in your code
```python
import numpy as np
import vindex

rng = np.random.default_rng(0)
X = rng.standard_normal((20_000, 64), dtype=np.float32)       # your items, one row each (for example text embeddings)
train = rng.standard_normal((2_000, 64), dtype=np.float32)     # sample questions, from the same model
Q = rng.standard_normal((100, 64), dtype=np.float32)           # the questions to answer

idx = vindex.Index(64, metric="cos")
idx.build(X, np.arange(len(X)), train_questions=train)         # tunes itself (about a minute here)
found, dists, checks = idx.search(Q, k=10, target=0.95)        # the 10 closest to each question, aiming at 95%

idx.add(20_000, rng.standard_normal(64, dtype=np.float32))     # changes in place
idx.save("my_index")
```

[docs/GUIDE.md](docs/GUIDE.md) covers the rest: changes, keeping the full vectors on SSD, threads, the settings, use from Rust, and how to check the results yourself.

## How it works, in plain words
- **A map of neighbours:** each item links to a few similar items (a Vamana graph, as in DiskANN). A search walks the links toward the question.
- **Small copies first:** the walk compares compressed copies of the vectors (8 or 4 bits per number), then re-checks the best few with the full numbers.
- **Effort per question:** easy questions stop early and hard ones search longer, using a table learned from sample questions.
- **Shortcuts for unusual questions:** when questions look unlike the items (text searching photos), it links items that sample questions found together, as RoarGraph does.
- **It fits itself:** while building, it tries a few settings on a sample of your data and keeps the fastest.

None of these parts is new on its own. The work is in combining them, the self-tuning and self-checking, and a fast ARM implementation. [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md) names the published work each part builds on, [docs/BENCHMARK.md](docs/BENCHMARK.md#what-each-piece-adds) measures what each adds, and [DESIGN.md](DESIGN.md) goes deeper.

## Is it for you?
**A good fit:**
- Search over up to about 1.5 million items on an Apple silicon Mac, from Python or Rust.
- Questions that look different from the items (text searching photos), or vectors from inside a chatbot.
- When you would rather set an accuracy target than tune an index.

**Not yet:**
- **Intel and AMD servers:** it builds and passes its tests on Linux x86, but runs plain loops there, without vector code, so it is not fast.
- **Quick builds:** it takes 1.4× to 24× as long to build as the fastest other index (454× on GIST1M); `tune="off"` builds faster, at a cost in speed.
- **Tight memory:** 0.92× to 2.13× the memory per item of the fastest other index, unless the full vectors stay on SSD, which makes searches 2 to 23 times slower.
- **Filtered search** ("only items from 2024"), and more items than were tested (1.45 million).

## Where to go next
| If you want | Read |
|---|---|
| the whole story on one page | [WRITE-UP.md](WRITE-UP.md) |
| every benchmark number and caveat | [docs/BENCHMARK.md](docs/BENCHMARK.md), and [the results page](https://srir84.github.io/vector-index-rust/page/results.html), with a chart per dataset |
| to use it | [docs/GUIDE.md](docs/GUIDE.md), [playground/README.md](playground/README.md) |
| how it works | [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md), [DESIGN.md](DESIGN.md), [ARCHITECTURE.md](ARCHITECTURE.md) |
| how it was built, step by step, misses included | [STEPS.md](STEPS.md), [REPORT.md](REPORT.md), [SPEC.md](SPEC.md) |
| a demo: ask questions of 1.45 million Google answers | [demo/README.md](demo/README.md) |
| the other indexes' scripts and raw results | [vector-index-experiments](https://github.com/srir84/vector-index-experiments) |

## License
MIT ([LICENSE](LICENSE)). The datasets have their own licences: see each source.
