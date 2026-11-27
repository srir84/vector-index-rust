# Steps

**In one line:** the index is built in 13 steps. Each one adds one piece and has its own check against a recorded result of the experiments; a step starts only after the one before it passes.

Reference results come from `../vector-index-experiments/results/step5/` (`runs/<stage>/<dataset>.json` are tuning questions). "Checks at 95%" is read off the effort-vs-accuracy curve at exactly 95%, as in the experiments. Run every check from this folder.

## 0. Set-up
- **Builds:** the Cargo workspace (`crates/vindex`, `crates/vindex-py`), the Python environment (`.venv`), `validate/common.py` (data loading, tie-rule accuracy, reading 95% off a curve).
- **Check:** `cargo test` and `maturin develop` both run; `import vindex` works.
- **Build the Python package:** `tools/build_py.sh` (builds in release mode, then checks that it loads).
- **Status:** passed.
  - Rust is pinned to nightly in `rust-toolchain.toml`. With stable Rust (1.90, 1.97) on macOS 27, the built package refused to load ("mis-aligned LINKEDIT string pool").

## 1. Distances and 1-byte copies
- **Builds:** `metric` and `codes`. The Python package exposes them for checking.
- **Check:** Rust unit tests on made-up numbers; `validate/step1.py` compares with numpy on 1,000 real items and questions from each dataset.
- **Pass:**
  - full-number distances agree within 1e-5, or within twice numpy's own 32-bit error where that is larger;
  - 1-byte distances agree with numpy's version of the same formula within 1e-4;
  - copies decode to within half a step.
- **Status:** passed, 15/15 (`validate/results/step1.json`).
  - The rule changed during the step. Llama is not scaled to length 1, so its dot products can be small results of large terms that cancel, and every 32-bit sum loses relative accuracy there.
  - Our worst Llama error is 2.6e-5, smaller than numpy's own 32-bit 3.9e-5. On the other datasets ours is 2–10× smaller than numpy's 32-bit error.

## 2. The walk on a known map
- **Builds:** `graph`, `search` (fixed list size, check counts, visited stamps), and a way to load a map from arrays.
- **Check:** `validate/step2.py` loads the prototype's own map of GloVe (its plain version, exported once) and searches the same 1,000 tuning questions with both, at list sizes 10, 50 and 200.
- **Pass:**
  - the same top 10 on at least 99.5% of questions;
  - total checks within 0.5%.
  - Not exactly 100%: near-equal distances can tie-break differently in 32-bit sums.
- **Status:** passed, 6/6 (`validate/results/step2.json`), with an exact match.
  - Same top 10 on 100% of questions at every list size.
  - Exactly the same total checks: 555,309, 1,640,115 and 4,911,919.

## How every step from 3 on is checked
Each stage is built twice and compared with the prototype's recorded result:
1. **With the prototype's own insertion order** (its start item, then numpy's shuffle with seed 7). If the port is exact, this must give the prototype's numbers. Pass: checks at 95% within 1%.
2. **With our own random order.** Pass: within 5%; or, where maps swing with the order, the prototype inside our spread over four seeds, or the averages over seeds within 5%.

## 3. Building the plain map
- **Builds:** `prune` and `build` (start item, batches, back-links).
- **Check:** `validate/step3.py` builds all five slices with the starting settings (32 links, effort 200, slack 1.2) and sweeps the list size, like the prototype's `v0`.
- **Pass:** as above; for Llama, which never reaches 95% here, the same best accuracy.
- **Status:** passed, 10/10 (`validate/results/step3.json`).
  - **Exact with the prototype's order:** GooAQ 540, GloVe 7,293 vs 7,292, Yandex 990, Landmark 381; Llama's best accuracy 62.52%, identical.
  - **Our own order:** within 0.8–2.4%.
  - Build checks per item and links per item match within 0.3%. GloVe builds in 10 seconds on all cores.

## 4. Flipped build copies for best-score data
- **Builds:** the build space x / |x|² with straight-line pruning, used when the formula is best score.
- **Check:** `validate/step4.py` on Llama, against `runs/v0+build-mobius/llama.json` (before the flip, Llama never reached 95%).
- **Pass:** as above, plus at most 10 items with no links in.
- **Status:** passed, 4/4 (`validate/results/step4.json`).
  - **Exact with the prototype's order:** 6,845 checks at 95%, identical.
  - **Items with no links in:** 1; before the flip, about 60,000.
  - **Llama's maps swing with the random order**, in both implementations:
    - checks at list size 10 range from about 480 to 650 over seeds;
    - our first four seeds happened to land high, but over 8 more seeds each, numpy's orders and ours give the same spread (averages 547 and 564).
  - **How this was found:** our walk on the prototype's own Llama map reproduced it exactly. Then our build, given the prototype's order, reproduced its map exactly. So the earlier gap was the order, not the code.

## 5. Learned effort
- **Builds:** `effort`: learn the table from 3,000 training questions, the two-stage walk, and a table for any target.
- **Check:** `validate/step5.py` on all five, against `runs/v0+build-mobius+learned-effort`.
- **Pass:** as above, plus the hardest group gets at least as much effort as the easiest.
- **Status:** passed, 15/15 (`validate/results/step5.json`).
  - **Exact with the prototype's order:** GooAQ 507, GloVe 5,622, Yandex 870, Llama 6,820, Landmark 381, all identical. That covers learning the table: the true answers by checking every item, the cut points, the greedy table, and the two-stage walk.
  - **Our own order:** within 0.3–3.4%; Llama inside our seed spread.
  - The tables have the expected shape, e.g. GloVe at 95%: [400, 400, 300, 300, 150, 100, 70, 50].

## 6. Question links
- **Builds:** `qlinks`: the "questions unlike items?" test and one round of question links.
- **Check:** `validate/step6.py`:
  - the test is on for Yandex and Llama only (the prototype's ratios: 0.93, 0.99, 2.08, 24.8, 1.00);
  - Yandex and Llama checks are compared against `runs/...+qlinks-auto`.
- **Pass:** the switch matches on all five; checks at 95% as above on Yandex and Llama.
- **Status:** passed, 9/9 (`validate/results/step6.json`).
  - **The switch:** on for Yandex and Llama only. Our ratios are 0.98, 0.99, 2.05, 24.79 and 1.03; the prototype's are 0.93, 0.99, 2.08, 24.80 and 1.00. Ours sample with a different random generator, hence the small gaps.
  - **Exact with the prototype's order:** Yandex 801, Llama 1,263, identical.
  - **Our own order:** Yandex within 1.1%. Llama's average over four seeds is 4.5% above the prototype's average over its four seeds (1,321 vs 1,265), the same order-luck as in step 4.

## 7. 1-byte copies with the full re-check
- **Builds:** copies made after the build, the walk on copies, the re-check of the best 50; the table is learned on this walk.
- **Check:** `validate/step7.py` on all five, against `runs/...+tiny8`.
- **Pass:** as in "How every step from 3 on is checked", plus memory per item within 1% of the prototype's (944, 376, 376, 304, 944 bytes).
- **Status:** passed, 15/15 (`validate/results/step7.json`).
  - **Exact with the prototype's order:** GooAQ 524, GloVe 5,735 vs 5,734, Yandex 854, Llama 1,318, Landmark 392.
  - **Our own order:** within 0.8–4.8%.
  - **Memory per item, identical:** 944, 376, 376, 304, 944 bytes (links 168, 1-byte copies 1 byte per number, id 8).

## 8. Question links made twice
- **Builds:** the second round at build (Fix 8). This completes the final design.
- **Check:** `validate/step8.py` on Yandex and Llama, against `runs/...+tiny8+qrounds2`.
- **Pass:** as in "How every step from 3 on is checked".
- **Status:** passed, 4/4 (`validate/results/step8.json`). The final design is an exact port.
  - **Exact with the prototype's order:** Yandex 854, Llama 1,223.
  - **Our own order:** Yandex within 1.0%; Llama's average over four seeds is 4.5% above, the same order-luck as in steps 4 and 6.

## 9. Changes
- **Builds:** `changes`: add, delete with repair right away, update in place, slot recycling, question-link refresh every 3,000 changes.
- **Check:** `validate/step9.py` runs the experiments' change script on all five (build with 90%, then 30,000 mixed changes, 10 checkpoints on the test questions, then a fresh build). Each dataset runs twice:
  - with the prototype's insertion orders, against the prototype's own re-run (`results/step5/changes/...+qrounds2-final`);
  - with our own order, against Fix 7's pass rule.
- **Pass:**
  - **Prototype's order:** same target; accuracy before, after and drift within 0.3 points of the prototype's re-run.
  - **Our own order:** drift under 1 point (or better than a fresh build), no deleted or old items returned, memory within 5% of a fresh build.
- **Status:** passed, 10/10 (`validate/results/step9.json`, log in `step9.log`).
  - **Prototype's order, exact:** same target, same accuracy before and after, same drift, and the same checks spent per change on GooAQ, Yandex, Llama and Landmark (e.g. Yandex drift −0.76, 24,678 checks per change).
  - **Prototype's order, GloVe:** within 0.02 points after 30,000 changes (32-bit vs 64-bit rounding adding up).
  - **Our own order:**
    - drift −0.41 to +0.55 points; no deleted or old items returned at any checkpoint; memory 1.01× a fresh build;
    - 30,000 changes take 26–66 seconds (15–43 after the speed tuning in step 12);
    - learning the effort table again after the changes changes accuracy by under a point.
  - **Spare room:** the index reserves 12% spare slots at build (setting `spare`), as the prototype's change test did. This decides when freed slots are recycled, and it was needed to match exactly.

## 10. Save, load, and the full numbers on SSD
- **Builds:** `persist` (the folder format) and the SSD mode for the full numbers.
- **Check:** `validate/step10.py`:
  - save, load, and search again: identical answers and checks;
  - the same with `numbers="ssd"`;
  - load refuses a damaged or wrong-version folder;
  - a reloaded index takes changes and still passes step 9's checks on one dataset;
  - (GloVe) threads: 8 Python threads searching at once, and 1,000 changes made while 4 threads keep searching.
- **Pass:** all identical; damaged files are refused with an error; the threads give the same answers and the same index as without them, with no errors.
- **Status:** passed, 14/14 on GloVe, Yandex and Llama (`validate/results/step10.json`), plus Rust unit tests on all three formulas.
  - **Same results after reloading:** identical answers, distances and checks, with the full numbers in memory or on SSD.
  - **Damaged folders are refused**, with a plain message (e.g. "links.u32 has 7380000 bytes, expected 14760000").
  - **Changes continue after reloading:** the first 3,000 changes on a reloaded copy match the original exactly, including the question-link refresh at change 3,000.
  - **On disk:** a saved slice is 84–122 MB; most of it is the full numbers (`numbers.f32`).
  - An index loaded with its numbers on SSD refuses changes, with a message saying to load it with numbers in memory.
  - **Threads:** 8 Python threads searching at once run 7.3× faster than one, with the same answers. 1,000 changes made while 4 threads kept searching (3,435 searches ran meanwhile) gave no errors and exactly the index made with nobody searching.
  - **The first thread check failed, and found two faults.**
    - A change made during a search raised "Already borrowed". The Python package now holds the index behind a read-write lock: searches run side by side, and a change waits for them.
    - Searches with one thread all went through one shared single-thread pool, so they queued up (1.0×). They now run on the calling thread.
    - The test itself hung on the first failure, because its searcher threads were never told to stop. It now always stops them.

## 11. Speed
- **Builds:** nothing new; measurement only.
- **Check:** `validate/step11.py` on the test questions, measured as the rivals were in Step 3:
  - speed at 95%: one core, 1,000-question batches, middle of 5;
  - all-core speed;
  - typical and slow time per question;
  - build time;
  - measured memory;
  - the SSD mode's speed, with the full numbers read from disk.
- **Pass:** the numbers are recorded and compared with all six rivals. The targets in `SPEC.md` are reported as met or missed.
- **Status before tuning** (first run, kept for comparison):
  - **Faster than HNSW, DiskANN and CoreNN:** GooAQ (9,168 vs DiskANN 8,679 searches/s, one core) and Landmark (19,357 vs 16,567).
  - **Llama:** 22,774/s, where none of the three reaches 95%. ScaNN does 11,891/s and RoarGraph 76,990/s.
  - **Slower than DiskANN:** GloVe (1,983 vs 3,307) and Yandex (11,112 vs 18,032).
  - **Why:** each check took too long: about 87 ns on GloVe (5,807 checks in 504 µs). Step 12 found where that time went.
- **Status after tuning:** passed, 10/10 (`validate/results/step11.json`, log in `step11.log`; `REPORT.md` has the full picture).
  - **Faster than HNSW, DiskANN and CoreNN on all four datasets where they reach 95%**, one core: GooAQ 20,362 (DiskANN 8,679), GloVe 4,941 (3,307), Yandex 28,330 (18,032), Landmark 28,370 (16,567). Llama: 30,867, where none of the three reaches 95%.
  - **Against the fastest rival:** Glass 1.6–2.1× faster where it reaches 95%, ScaNN 2.6× faster on GloVe, RoarGraph 2.5× faster on Llama. The SPEC's aim of getting close to Glass and ScaNN is missed.
  - **SSD mode, file dropped from the cache before each batch:** identical answers; 2,600–9,261 searches/s on one core; one question from a cold cache typically 0.4–1.0 ms.
  - **Memory measured in a fresh process:** 1,003, 411, 571, 441 and 1,003 bytes per item with the full numbers in a file (counted: 944, 376, 376, 304, 944). The difference is room kept for adds, the id table and working memory. Yandex and Llama also keep their training questions for question links.
  - **Other programs were busy during the run:** a macOS storage scan and another job, using 1–3.5 cores (recorded per dataset). One-core figures matched an earlier run within 3%; all-core figures move more.

## 12. Speed tuning
- **Builds:** fetching ahead, less work keeping the list, vector instructions. Tuning may change time only, never results.
- **Check:** `validate/step12.py`:
  - `step12.py record`, run once before any tuning, saved the untuned build's answers, distances and checks at every learned-effort target and at a fixed list size, plus its speed, in `validate/results/step12-before/`;
  - after each change, `step12.py` builds all five again: build work, answers, distances and checks must be identical to the recording, and the speed at 95% (one core) at least 10% higher. The untuned build, timed twice, differed by 6–8%.
- **Pass:** identical results on all five, and faster.
- **Status:** passed, 10/10 (`validate/results/step12.json`). One core at 95%, last run: GooAQ 2.23×, GloVe 2.12×, Yandex 2.27×, Llama 1.39×, Landmark 2.06×. Builds faster too (GooAQ 27 → 19 s, Landmark 19 → 14 s).
  - **What changed**, in the order tried, each re-checked for identical answers:
    1. **Fetching ahead** (1.1–2.0×): before scoring an opened item's new links, the walk asks the chip to start loading all of their copies, so the reads overlap. It also starts loading the link lists of the next two items it is likely to open.
    2. **Less list work** (about 10% on GloVe, Yandex, Llama): each list entry's distance, id and "opened" mark sit together, so making room moves one array instead of three; the place is found without unpredictable branches; and after the learned effort's first stage, the list stops keeping entries beyond what this question's list size and the re-check can use.
    3. **"Seen already?" without branches** (7–21%): links are filtered with arithmetic instead of one branch each, so their memory reads overlap.
    4. **Vector instructions** (search speed unchanged, builds faster): the compiler had turned every distance into one-number-at-a-time instructions. They are now written with the chip's 4-wide instructions, in exactly the same order of operations; unit tests compare them with the old sums bit for bit. Four items are scored side by side.
    5. **One-thread searches on the calling thread, keeping their working memory** (typical time per question down another 10–15%): no hop to a pool thread, and the seen-marks (4 bytes per item) are no longer allocated and cleared on every call.
  - **What the arithmetic costs now**, on data already in the cache (`cargo run --release -p vindex --example kernels`): a 768-number 1-byte copy about 49 ns, full numbers about 38 ns, four at a time. A real check on GooAQ costs about 94 ns: the rest is waiting for each copy to arrive from memory.
  - **Tried and dropped:** putting each opened item's new links into the list in one merge. It moved fewer bytes but was 12% slower on GloVe; the cost was per entry put in, not per byte moved.
  - **Also added:** the index refuses numbers that are not finite (NaN or infinity), which the sorted list relies on; and loading reads each file straight into place instead of holding the full numbers twice.

## 13. Report and hand-over
- **Builds:**
  - `REPORT.md`: results against the rivals, with met and missed targets;
  - `README.md`: install and use;
  - `diagrams/speed.png`, drawn by `tools/draw_report.py`;
  - `crates/vindex/examples/basic.rs`: use from Rust;
  - `validate/run_all.sh`: every check in order.
- **Check:** every command in the README and in this file runs as written.
- **Status:** passed. Every README command was run, as written, in a fresh copy of the project (no build folder, no Python environment, no results):
  - both install routes (`uv`, and `python3 -m venv` with `pip`), then `tools/build_py.sh`;
  - the Python example, taken straight out of the README;
  - the Rust example, `cargo test -p vindex --release`, and `validate/run_all.sh 9 10`;
  - `validate/run_all.sh` with no step numbers: the unit tests and all 12 steps passed, in about 16 minutes.
  - **Found and fixed:** `validate/run_all.sh` with no step numbers ran nothing, because zsh kept the default "1 2 … 12" as one word. Every earlier run had passed explicit steps.
  - The fresh copy's speed run measured 6–14% higher one-core speeds than the official step 11 run, with less else running on the machine.

---

# Part 2: fitting itself to each dataset

**In one line:** nine more steps (`SPEC.md`, part 2). Each adds one piece and has its own check; a step starts only after the one before it passes.

Checks use the tuning questions, and test questions only where a step measures speed for the report. The four held-back datasets are locked until step 22.

## 14. Held-back datasets, locked
- **Builds:**
  - `scripts/download_held_back.sh` in the experiments: arXiv, LAION, Yi and GIST1M, each checked against its published checksum;
  - preparation as for the five (100,000-item slices, three question groups, true answers, change script);
  - a lock: `validate/common.py` refuses to load a held-back dataset unless step 22 asks.
- **Check:** `validate/step14.py`: files and checksums; our true answers agree with each file's own on the full data; the lock refuses a normal load.
- **Pass:** all of the above.
- **Status:** passed, 14/14 (`validate/results/step14.json`).
  - **Downloads:** arXiv, LAION and Yi match VIBE's published checksums. GIST1M matches its server's 8 MiB-piece checksum (ann-benchmarks publishes none).
  - **Prepared like the five:** arXiv 1,323,643 items, LAION 1,000,448, Yi 187,843, GIST1M 979,000 (after setting questions aside).
    - Our true answers agree with each file's own top 10, apart from the items we set aside as questions; on GIST1M, 0.05% more differ.
    - Our own high-precision check finds no difference beyond ties.
  - **The rule changed once:** it banned all-zero rows, which matters only for angle data. GIST1M has 10, an ordinary point for straight-line distance. It also has 17,306 exact duplicate rows (1.7%); they are noted, not removed.
  - **Locked:** both loaders refuse the four, and no result here or in the experiments names them.
  - **Straight-line distance works on our own data** (Llama's numbers): 99.7% at list size 15; the learned effort at 95% gives 96.0%; distances equal numpy's within 2e-7.

## 15. Usage stats
- **Builds:** counters kept by the index (searches, checks, effort used, re-checks, how far questions sit from their answers), with `usage()` and `reset_usage()` in Python.
- **Check:** `validate/step15.py`: the counts equal what the calls returned, also with 8 threads; results identical to step 12's recording; the time it costs.
- **Pass:** exact counts, identical results, under 1% slower.
- **Status:** passed, 22/22 (`validate/results/step15.json`; how it works: `DESIGN.md` section 13).
  - **Exact counts** on GloVe, Yandex and Llama: a batch spread over 16 threads, a batch on one thread, and 800 single questions from 8 Python threads at once all add up to exactly what the calls returned.
  - **Hardness bins:** the 3,000 sample questions the effort table learned from fill its 8 bins with exactly 375 each, on all three.
  - **"Questions unlike items" in use** agrees with the build's test: GloVe 0.98 (build 0.99), Yandex 2.01 (2.05), Llama 25.8 (24.8).
  - **Time counted** is 99–100% of the wall clock.
  - **Results unchanged:** builds and answers on all five equal step 12's recording.
  - **Cost:** too small to measure. One core, taking turns with an identical index without the counters: Llama 30,648 vs 30,371 searches/s, Landmark 27,135 vs 27,102.

## 16. Spot checks
- **Builds:** a share of questions (`audit_rate`) is kept with the answers given; their true answers are found later, in batches, by checking every item; the result is an accuracy estimate with a margin of error.
- **Check:** `validate/step16.py` on the five:
  - with every question spot-checked, the estimate equals the accuracy from the true-answer files exactly;
  - with 1 in 20 spot-checked, over 100 different samples per dataset, how often the real accuracy falls inside the stated margin;
  - the time it costs at the default rate;
  - results unchanged, searches running during a spot check, and the counts kept through save and load.
- **Pass:** exact; over the 500 samples, 92% to 98% inside, and no dataset under 85%; under 2% slower.
  - Why pooled: a single dataset's 100 samples swing by about 2 either way, and checking 50 of 1,000 questions without repeats makes the margin slightly wide on purpose (about 96% expected). One dataset's count alone would fail by chance too often.
- **Status:** passed, 23/23 (`validate/results/step16.json`; how it works: `DESIGN.md` section 14).
  - **Exact:** with every question spot-checked, the estimates equal the true-answer files' accuracy on all five, at the 95% target and at list size 100.
  - **Margins:** the real accuracy was inside the stated margin in 481 of 500 samples (96.2%; 94 to 98 of 100 per dataset). The typical margin from 50 questions is ±2.2% to ±4.3%.
  - **Cost at the default budget** (2% of the checks): 0.5–1.0% of search time on one core.
  - **How often:** rarely on 100,000 items, because one spot check costs a check per item. GooAQ gets about 1 per 10,000 searches, GloVe 1 per 1,000. A margin of ±2–3% takes about 50 checked questions.
  - **Results unchanged** (step 12's recording). Searches from 4 threads ran through 79,713 spot-check runs with the same answers. Counts, margins and checked questions come back after save and load.
  - **Found and fixed, 1 (margins):** the first margins used only the 50 questions' own spread: Landmark 80 of 100 inside, 91.2% over all five.
    - On Landmark, 8 of 1,000 questions carry a third of the errors, and 50 questions usually miss all 8.
    - The margin now uses the spread of every question spot-checked so far (the last 2,000).
    - A simulation on the tuning questions chose it: 96.0–96.9% on all five, against 82–95% before.
  - **Found and fixed, 2 (cost):** the first spot checks read items in hash-table order, jumping around memory: 5–8% of search time for under 2% of the checks. They now read in memory order, four items at a time, 8 questions per pass: 0.5–1.0%.

## 17. Learning from use
- **Builds:**
  - **At build:** the effort table learns from sample questions the question links don't use (setting `effort_apart`, on). Found while planning this step: on Yandex, where the links and the table shared their questions, a 95% target gave 92.1% on the tuning questions, because the links make their own questions look easy. Part 1's checks keep the old way (`effort_apart=False`), so they still compare one to one.
  - **In use:** after every 500 new spot-checked questions (`relearn_every`), if a target's estimate falls below it by more than its margin, the table is learned again from the spot-checked questions (at least 500; their true answers are already known, so it costs only searches). It never relearns just because accuracy is above target: that can be the table's floor, which relearning cannot lower.
- **Check:** `validate/step17.py`:
  - **At build** (Yandex, Llama): accuracy at the 95% target on the tuning questions, with and without `effort_apart`; on GooAQ, GloVe and Landmark (no question links) the answers must not change.
  - **Wrong sample questions** (Yandex, Llama): build with sample questions taken from the items, so the table aims wrong (and question links stay off). Search 2,000 real questions with spot checks on, then measure the tuning questions' accuracy before and after.
  - **Right sample questions** (all five): the same use must not start a relearn. A forced relearn must move accuracy by under 1 point: a table learned from 2,000 questions lands within about ±0.4 points of its target on new questions, and the build's own table is as noisy.
  - Searches running while the table is learned again; the new table kept through save and load.
- **Pass:** at build, within 1 point of the target with `effort_apart`; wrong sample questions: back to within 1 point of the target; right ones: no relearn on its own, and under 1 point when forced.
- **Status:** passed, 14/14 (`validate/results/step17.json`; how it works: `DESIGN.md` section 15).
  - **At build:** with `effort_apart`, the 95% target gives 95.07% on Yandex (92.12% before), at 879 checks per search instead of 678. Honest 95% costs 30% more there. Llama: 94.99% (94.81% before). GooAQ, GloVe and Landmark have no question links, and their answers are identical.
  - **Wrong sample questions** (taken from the items, so question links also stay off):
    - Yandex starts at 82.3%. The spot checks saw 82.3% ± 0.9%, the table was learned again on its own, and accuracy came back to 94.6% (915 checks).
    - Llama: 85.6%, then 95.3% (6,188 checks; with question links it would need about 1,270, which is step 19's suggestion to make).
  - **Right sample questions:** no relearn started on its own on any of the five. A forced one moved accuracy by 0.0 to 0.8 points (GooAQ 95.56% → 94.76%, the largest).
  - **Searches ran** through a relearn (204 batches, no errors). The new table comes back after save and load.
  - **Step 16's margin check** now turns relearning off: Yandex's part-1 build misses its target, so the index rightly relearned in the middle of a check that measures one fixed table.

## 18. Picking its own settings
- **Builds:** `tune="auto"` at build: try a few choices of links per item and slack on a small part of the data, with the build's sample questions; keep the fastest at the target accuracy (optionally within a memory limit); build the full index with it; record what was tried.
- **Check:** `validate/step18.py` on the five: build every choice at full size to find the true best, then compare the self-picked choice with it, and with part 1's single setting.
- **Pass:** within 10% of the best's speed at 95% on all five. Extra build time reported.
- **Status:** passed, 7/7 (`validate/results/step18.json`, the grids in `step18-grid-<dataset>.json`; how it works: `DESIGN.md` section 16). Rerun after step 21 changed how the tuning chooses; the grids were reused, since they do not depend on it.
  - **Self-picked against the best**, timed head-to-head (float questions, as in part 1):
    - GooAQ: kept 32 links, slack 1.2 (100% of the best);
    - GloVe: 96 links, slack 1.3 (100%; 1.30× part 1's setting);
    - Yandex: kept 32 links (97%);
    - Llama: 24 links, slack 1.3 (97%; 1.25×);
    - Landmark: 24 links, slack 1.0 (100%; 1.33×).
  - **Memory per item** moves with the links: GloVe 376 → 708 bytes, Llama 304 → 264, Landmark 944 → 904. A memory limit (`max_bytes_per_item`) kept GloVe at 376.
  - **Build time:** 2.6× to 7.0× longer with tuning (GooAQ 52 s against 19; GloVe 37 s against 5). The 8 trials take longer than the real build when the data builds fast.
  - **Found and fixed:**
    - **First run, 3 checks failed, from timing noise:** the same index (Llama, 24 links, slack 1.3) was timed at 37,426 and 41,424 searches/s, because 1,000 of its searches take only 25 ms. The tuner now times runs of at least 0.1 s (the fastest of 3), and the check times its candidates head-to-head (the pick, part 1's setting and the grid's top 3, in 5 rounds).
    - **Step 21's run, a choosing rule that blocked real gains:** each round's winner had to beat the leader by 5%, so on GloVe 96 links (4.8% ahead of 48 in the trial, 19% at full size) lost. Now each round keeps its fastest trial, and only the final choice must beat the starting setting by 5%, head-to-head.
    - **Saved numbers:** a saved tuning record read back one unit off in a last digit. The JSON reader now reads numbers back exactly; the same fix covers the usage and spot-check files.

## 19. Suggestions
- **Builds:** `suggest()`:
  - **Notes:** many slots holding deleted items; real questions that look unlike the items while question links are off; accuracy below target before learning again can start.
  - **A settings trial:** step 18's search on the whole index (up to 100,000 items), with the spot-checked questions as sample and timing questions. The index itself, timed as it is, is the one to beat. A rebuild is suggested only when the trial is at least 15% faster. The suggestion says what to change, the expected gain, and whether to use the spot-checked questions (`checked_questions()`) as the rebuild's sample questions.
- **Check:** `validate/step19.py`, with 2,000 real questions (each spot-checked) as the use. Every suggested rebuild is carried out as suggested and measured: it must really gain at least 15%, and the trial's gain must be within 15% of the measured one.
  - GloVe built with 16 links must get "more links".
  - Yandex and Llama built with sample questions taken from the items (so question links stay off) must get the note. A rebuild must be suggested exactly when one really gains at least 15%.
  - The five, tuned at build, must get no rebuild that does not really pay.
  - A quarter of the items deleted must get the memory note.
- **Pass:** as above.
- **Status:** 11 of 12 passed (`validate/results/step19.json`; how it works: `DESIGN.md` section 17). One target was missed: a gain predicted within 15%, on Llama.
  - **GloVe with 16 links:** "Rebuild with 64 links per item: about 2.9x faster". Measured 2.78× (6,855 against 2,469 searches/s).
  - **Yandex with item-like sample questions:** the note appears, and no rebuild is suggested. Rightly so: the rebuild the note points to measured 0.97×, since question links barely help Yandex at this size.
  - **Llama with item-like sample questions:** the note, and "rebuild with 64 links and the spot-checked questions, which turn question links on: about 4.4x faster". Measured 3.62×, so the suggestion is right but its prediction is 17% high.
    - Timing on half the spot-checked questions instead of a quarter predicted 3.3× against 4.2× measured.
    - With about 2,000 real questions, a question-links rebuild's gain is predictable to about ±20%; the quarter split is kept, as the build's tuning uses it.
  - **Tuned at build:** no rebuild suggested on any of the five. Empty slots: the memory note at 25%.
  - **Found and fixed:**
    - **A fifth-size trial understated better links:** GloVe at 16 links was predicted 1.8× and gained 2.9×. Suggestions now take the whole index, up to 100,000 items.
    - **A rebuilt baseline overstated gains:** a trial of "the index as it is", rebuilt from the spot-checked questions, had weaker question links than the real index, built from 17,000 sample questions. That led to a false 1.2× suggestion for the tuned Llama. The baseline is now the index itself, timed without counting as use.
    - **The first version of this check expected a rebuild for Yandex.** Measuring showed none pays, so the check now compares every suggestion with a measured rebuild.

## 20. Cheaper checks, same results
- **Builds:**
  - fetch-ahead hints in this chip's 128-byte memory lines (one hint per line, not per 64 bytes);
  - items renumbered after the build (`renumber`, on), in the order a breadth-first walk of the links meets them, so linked items sit near each other in memory;
  - a build guided by a recorded run, if it helps.
- **Check:** `validate/step20.py`:
  - **Results:** builds and answers on all five, renumbered, must equal step 12's recording.
  - **Speed:** one core at 95% on the tuning questions, against the speed recorded with the code just before these changes (`step20.py record`). Each speed is the fastest of 5 runs of at least half a second.
- **Pass:** identical; at least 10% faster on average over the five, and no dataset more than 3% slower.
- Part 1's checks keep `renumber=False`: step 9's exact match with the prototype depends on the order in which deleted items' slots are reused.
- **Status:** 10 of 11 passed (`validate/results/step20.json`). The 10% target was **missed by 1 point**.
  - **Results identical** on all five, renumbered.
  - **Speed:** measured on one core at 95%, tuning questions, built with `tools/build_py_pgo.sh`, against the code just before this step:
    - GooAQ 1.16×, Landmark 1.15×, Llama 1.07×, Yandex 1.06×, GloVe 1.02×;
    - 1.09× on average, against a target of 1.10×. No dataset is slower.
  - **Where it came from**, each measured head-to-head:
    - **Fetch-ahead hints:** one per 128-byte line on long rows gave 13–14% on GooAQ and Landmark, whose copies are 768 bytes. Short regions (up to 256 bytes) keep a hint every 64 bytes, because one per line made GloVe's 200-byte rows 5% slower.
    - **The guided build:** 1–7% (Llama most).
    - **Renumbering:** 0–2% at 100,000 items, where most of the data stays in the chip's caches anyway.
  - **Kept anyway:** all three help and none hurts. The guided build stays optional (`tools/build_py_pgo.sh`); the plain build gives about 3% less.
  - **Found on the way:** a unit test had assumed an item's id is its slot number, which stopped being true once items were renumbered. The test now looks the slot up.

## 21. Cheaper checks, new options
- **Builds:**
  - **Whole-number questions** (`question_bits`): a question held as 16-bit whole numbers against the 1-byte copies, so a check needs about 6 instructions per 16 numbers instead of 14. An 8-bit version uses the chip's dot instruction, 16 multiply-adds at once.
    - Measured first: 16-bit is 1.26× to 1.53× faster with the same checks and accuracy; 8-bit is no faster (each check now waits on memory, not arithmetic) and slightly less accurate.
    - So 16-bit becomes the default rather than a tuning choice. Part 1's checks keep floats (`question_bits=32`).
  - **4-bit copies** (`code_bits=4`, with 16-bit questions): two numbers per byte.
    - Measured first: GooAQ 1.11× faster, Landmark and GloVe about the same, Yandex 0.93×, and Llama never reaches 95%. Memory per item drops by 23–41%.
    - So it is a tuning choice: the tuning tries it last, at the chosen links and slack, and keeps it only on a clear win. A memory limit can also make it the only choice that fits.
- **Check:** `validate/step21.py`, on the tuning questions:
  - 16-bit questions on the five: accuracy at the 90%, 95% and 99% targets within half a point of float questions, and at least 10% faster on one core (head-to-head);
  - 4-bit copies: where they reach 95%, the learned effort lands within 1 point of it;
  - step 18's check again, the grid now links × slack × copy size (40 builds per dataset): the self-picked choice within 10% of the best.
- **Pass:** as above.
- **Status:** 18 of 20 passed (`validate/results/step21.json`; how it works: `DESIGN.md` section 19).
  - **16-bit questions, all five:** 1.48× (GooAQ), 1.33× (GloVe), 1.30× (Yandex), 1.26× (Llama), 1.53× (Landmark) faster on one core. Checks per search are identical, and accuracy at the 90%, 95% and 99% targets is within 0.25 points.
  - **4-bit copies:** GloVe lands at 95.1% and Yandex at 95.7% at the 95% target; Llama never reaches 95% (91.3% at best), so the tuning must not pick them there, and it does not.
  - **Two checks failed as written, both on the high side:** at the 95% target, GooAQ gives 96.0% and Landmark 97.3% with 4-bit copies. That is more accurate than asked, from the table's floor (Landmark gives 98.1% with 1-byte copies too). The rule should have been one-sided, as step 17's relearning is.
  - **Self-picked, with 4-bit copies allowed** (the grid now links × slack × copy size, 40 per dataset), timed head-to-head:
    - GooAQ: 4-bit copies (99% of the best; 1.11×; 944 → 560 bytes per item);
    - GloVe: 96 links, slack 1.3, 4-bit (95%; 1.35×; 620 bytes);
    - Yandex: 48 links (100%; 1.06×);
    - Llama: 24 links, slack 1.3 (93%; 1.23×);
    - Landmark: 24 links, slack 1.0 (100%; 1.28×).
  - **Found and fixed (the first run failed GooAQ at 90% and GloVe at 79% of the best):**
    - **Copy size can't be judged in a trial:** a 20,000-item trial sits in the chip's caches, so 4-bit copies looked 2% slower there and are 11% faster at full size. The copy size is now chosen on the real index after the build, head-to-head (DESIGN.md section 19).
    - **The choosing rule** blocked GloVe's 96 links (see step 18).

## 22. The held-back test and report
- **Builds:** nothing new.
  - `step22.py rivals`: each rival gets its settings search on the held-back datasets' tuning questions, as in part 1 (the experiments' step 3 script).
  - `step22.py`: part 2's index on the five and the four held-back datasets, with every part 2 default (it picks its own settings from its sample questions), built with `tools/build_py_pgo.sh`. It is measured exactly as step 11 and the rivals were: test questions, one core, middle of 5 timed runs, read off at 95%; all cores; time per question; build time; memory; the SSD mode.
- **Check:** `validate/step22.py`; `REPORT.md`, part 2.
- **Pass:** every number recorded and reported as it comes, losses included. Targets from `SPEC.md`: on each of the five, closer to the fastest rival than part 1; faster than HNSW, DiskANN and CoreNN wherever they reach 95%.
- **Expected, written before any held-back run** (from the tuning-question numbers of steps 18 and 21; if a result contradicts these, the claim changes, not the result):
  - **GooAQ:** about level with Glass. **Landmark:** faster than Glass, about 1.15×. **Yandex:** about level with Glass.
  - **GloVe:** faster than Glass, but ScaNN still about 1.4× faster.
  - **Llama:** RoarGraph still about 1.6× faster.
  - **Held back:** arXiv like GooAQ (around Glass); LAION like Yandex; Yi like Llama (RoarGraph ahead); GIST1M unknown, the first straight-line dataset.
  - **Faster than HNSW, DiskANN and CoreNN** on all nine, wherever they reach 95%.
- **Status:** passed, 14/14 (`validate/results/step22.json` and `step22.log`; the write-up is `REPORT.md`, part 2).
  - **Closer to the fastest rival than part 1, on all five:** GooAQ 0.51 → 1.05× Glass; GloVe 0.39 → 0.75× ScaNN; Yandex 0.62 → 0.91× Glass; Llama 0.40 → 0.59× RoarGraph; Landmark 0.47 → 1.16× Glass.
  - **Held back, fastest of seven on all four:** arXiv 1.05× Glass, LAION 1.10× Glass, Yi 1.11× RoarGraph, GIST1M 1.25× Glass. The tuning kept its starting setting on arXiv, LAION and GIST1M, and took slack 1.3 on Yi.
  - **Faster than HNSW, DiskANN and CoreNN** wherever they reach 95% (2.3–4.8× DiskANN). On Llama and Yi none of them does.
  - **Against the expectations:** they held, except Yandex (slightly behind Glass, 0.91×), LAION (better, 1.10×) and Yi (wrong: we are ahead of RoarGraph, not behind).
  - **Glass timed again** on the five, the same day as ours: 0.2–5.3% below part 1's records, so the records stand (`../vector-index-experiments/results/step3/retime-2026-11-05/`).
  - **Found and fixed, in the experiments' test set-up:**
    - **The wrong distance for GIST1M:** RoarGraph and ScaNN were always run with inner product, which is right only for angle and best-score data. RoarGraph's first GIST1M run found 0.3% of the true answers. It was fixed (`bench/runners/roargraph.py`, `scann.py`, `scann_job.py`) and the run redone; the bad run is kept in `results/step3/invalid-wrong-distance/`. ScaNN was fixed before its GIST1M runs.
    - **Docker:** ScaNN runs in Docker, which was not running, so ScaNN's first held-back runs failed at once. They ran once Docker was started.

## Final check
Every step's check was run again on the final code, in a fresh copy of the project (as in step 13): guided build, then `REUSE_GRID=1 validate/run_all.sh`, in 57 minutes. The official results above were left as they are.
- **Passed:** the 31 unit tests, steps 1 to 12, and 15 to 18.
- **Missed the same checks as before, by about as much:**
  - **Step 19:** Llama's rebuild gain predicted 4.35×, measured 3.66× (before: 4.37× and 3.62×).
  - **Step 20:** 1.07× faster on average, against the 1.10× target (before: 1.09×).
  - **Step 21:** with 4-bit copies, GooAQ and Landmark again give 96.02% and 97.26% at the 95% target.
- **Step 14, one new fail, fixed:** its check that nothing had run on a held-back dataset failed, because step 22 had now run. The check now allows step 22's own runs (in the experiments' step 3 folders) and nothing else. Rerun: 14/14.

# Part 3: closing the last three gaps

**In one line:** five more steps (`SPEC.md`, part 3), checked the same way: each adds one piece and has its own check, and a step starts only after the one before it passes.

Checks use the tuning questions. The three new held-back datasets, and part 2's four, stay locked until step 27.

## 23. New held-back datasets, locked
- **Builds:**
  - `scripts/download_part3.sh` in the experiments: NYTimes (ann-benchmarks, checked against its server's 8 MiB-piece checksum), ImageNet (VIBE's published checksum), and Llama's second head (layer 15, head 18: two files from `ejaasaari/llama_heads`, checked against Hugging Face's listed checksums);
  - Llama's second head made into a file laid out like VIBE's, the way VIBE made Llama;
  - preparation as for the five, with 1,000 test questions on each;
  - the lock: both loaders refuse the three, as they do the four.
- **Check:** `validate/step23.py`: files and checksums; our true answers agree with each file's own (Llama's second head has none of its own: our answers are checked against a second, high-precision run instead); both loaders refuse all seven; nothing has been run on the three.
- **Pass:** all of the above.
- **Status:** passed, 13/13 (`validate/results/step23.json`).
  - **Downloads:** ImageNet and the two head files match Hugging Face's published checksums; NYTimes matches its server's 8 MiB-piece checksum.
  - **Llama's second head is made as VIBE made Llama:** our rule, run on the head VIBE's Llama came from (layer 12, head 15), gives VIBE's Llama file exactly: items, test questions and extra questions identical.
  - **Prepared like the five:** NYTimes 268,761 items (after setting questions aside), ImageNet 1,281,167, Llama's second head 256,921.
  - **Found and fixed, on NYTimes:**
    - **Rows with no angle:** 239 items and 9 of the file's test questions are all zero. Angle data now leaves such rows out (as items, questions and test questions). No other dataset has any, so nothing else changes.
    - **Ties:** 0.96% of the file's top-10 answers were not in ours, and all 96 have exactly our 10th answer's distance (NYTimes holds 40,793 exact duplicates). The agreement now counts a tie as agreeing: 92.5%, plus 7.5% that we set aside as questions.
  - **Locked:** both loaders refuse all seven held-back datasets, and no result names the three.

## 24. The re-check count, picked
- **Builds:** after the build, with the copy size, the tuning tries re-check counts and keeps the fastest at the target, head-to-head on the real index. Each count gets its own effort table.
- **Check:** `validate/step24.py` on the five: every count, timed head-to-head, to find the best; the picked count against it; accuracy at the 95% target.
- **Pass:** within 3% of the best count's speed on each of the five; at least 94% at the 95% target.
- **Status:** passed, 15/15 (`validate/results/step24.json`; how it works: `DESIGN.md` section 20).
  - **Picked:** 15 on GooAQ, Yandex and Llama (1.03×, 1.05× and 1.05× faster than 50); 50 on GloVe and Landmark, where nothing is faster. Each pick is within 1.2% of the best count from 10 to 100, timed head-to-head.
  - **Accuracy at the 95% target:** 95.05% to 96.00% on the tuning questions.
  - **The map does not depend on the count:** an index built without tuning, with the picked count, gives the tuned index's answers exactly, on all five.
  - **Smaller than expected:** re-checks are cheap, because their memory reads overlap. Cutting 50 to 15 saves only a few percent, not the 10% the count of full-number reads suggested.
  - **Cost:** builds take about 5–11 s longer (Llama 24 s, GooAQ 66 s).

## 25. Cheaper checks, same results (2)
- **Builds:** faster code for the walk, each change measured on its own and kept only if it helps.
- **Check:** `validate/step25.py`:
  - **Results:** builds and answers on all five equal a recording made just before this step (`step25.py record`);
  - **Speed:** one core at 95% on the tuning questions, against the recording's speed, fastest of 5 runs of at least half a second.
- **Pass:** identical; at least 5% faster on average over the five, none more than 2% slower.
- **Status:** 15 of 16 passed (`validate/results/step25.json`; how it works: `DESIGN.md` section 21). The 5% target was **missed**.
  - **Results identical** on all five: part 3's settings against this step's recording, and part 1's design against step 12's.
  - **Speed:** 1.006× (GooAQ), 1.030× (GloVe), 1.024× (Yandex), 1.019× (Llama), 1.012× (Landmark): 1.018× on average. Timed head-to-head: the code before this step and the code now load the same saved index, taking turns in separate processes, 5 rounds each.
  - **Kept:** with no deleted item in the index, the walk skips the "deleted?" lookup for every link.
  - **Tried and dropped**, each timed the same way:
    - "Seen" marks as one bit per item (14 KB instead of 448 KB): 0.98–1.01×.
    - Copy rows padded to the chip's 128-byte memory lines (a 200-byte row takes 256): no gain beyond the kept change.
    - Starting to load the likely next item's links early: 0.85–0.95×, so a wrong guess costs more than a right one saves.
  - **What it says:** each check waits on its copy's first memory read and on its arithmetic, not on bookkeeping. More speed has to come from making fewer checks.

## 26. More links for popular items
- **Builds:** first, measure where Llama's checks go (popular items or not). Then: items that show up in many sample questions' answers get more links, the rest fewer, with the popular ones together in memory. The tuning tries it on the real index and keeps it only when faster.
- **Check:** `validate/step26.py` on the five: speed at 95% on the tuning questions with and without it, head-to-head; what the tuning chose; accuracy at the 95% target.
- **Pass:** Llama at least 1.3× faster than at the end of part 2; the other four not more than 2% slower.
- **Status:** passed, 10/10 (`validate/results/step26.json`, rerun after the fix below with the guided build; how it works: `DESIGN.md` section 22).
  - **Found first, on Llama** (tuning questions, the map walked with full numbers):
    - 2,554 items (2.6%) hold 99.5% of the sample answers; 73 hold half.
    - The map gave popular items no more links than the rest, and only half of their links led to other popular items: with a list of 10, a search found 69% of the true top 10 (RoarGraph: 95.5%).
    - Starting from a popular item saved 9% of the checks. Fewer links for the other items saved none, so they keep theirs.
    - Up to 64 extra links between popular items that share answers: 912 checks at 95% fell to 528.
  - **Llama:** 120,046 searches/s at 95% (567 checks), against 55,188/s (959) for part 2's index: **2.18×**. 3,321 popular items, 104,191 extra links; 273 bytes per item (264 before). The 95% target gives 95.29%. (The first run, with a plain build: 109,385 against 52,119/s, 2.10×.)
  - **The other four:** their answers do not bunch up (67–77% of the items appear in some sample answer, against Llama's 3.6%), so the extra links are not tried, and the speed is unchanged.
  - **Part 3 so far against part 2's index**, the same timing: GooAQ 1.04×, GloVe 1.01×, Yandex 1.05×, Landmark 1.00×.
  - **Found and fixed (in the final check):** the check first compared the index with a second tuned build made with `popular_links="off"`. On GloVe, where no extra links are even tried, that second build picked slightly differently by timing noise alone and was 6% faster, failing the check; it had passed the first time by luck. The index without extra links is now built with the settings the tuning picked, without tuning, so only the extra links differ.

## 27. The held-back test and report
- **Builds:** nothing new. `step27.py rivals` gives each rival its settings search on the three new datasets, as in parts 1 and 2. `step27.py` measures part 3's index on all twelve datasets, as in step 22.
- **Check:** `validate/step27.py`; `REPORT.md`, part 3.
- **Pass:** every number recorded and reported as it comes, losses included.
- **Expected, written before any held-back run** (from the tuning-question gains of steps 24 to 26 on top of step 22's results; if a result contradicts these, the claim changes, not the result):
  - **Llama:** ahead of RoarGraph, about 1.2× (0.59× in part 2, times about 2.1).
  - **Yandex:** still slightly behind Glass, about 0.95× (0.91×, times 1.05). The extra links do not apply there.
  - **GloVe:** ScaNN still about 1.3× faster (0.75×, times 1.01).
  - **GooAQ, Landmark:** about as in part 2 (1.05–1.1× and 1.15× Glass).
  - **Part 2's held back, run again:** arXiv, LAION and GIST1M about as in part 2; Yi faster than in part 2 if its answers bunch up as Llama's do (attention data from another chatbot), so further ahead of RoarGraph.
  - **New, held back:**
    - **NYTimes** like GloVe: ScaNN ahead of us, perhaps 1.3–1.5×.
    - **ImageNet** like Yandex and LAION: around Glass.
    - **Llama's second head:** ahead of RoarGraph if its answers bunch up like Llama's; heads differ, so this is the least certain.
  - **Faster than HNSW, DiskANN and CoreNN** on all twelve, wherever they reach 95%.
- **Status:** passed, 18/18 (`validate/results/step27.json` and `step27.log`; the rivals' runs: `step27-rivals.log`; the write-up is `REPORT.md`, part 3).
  - **Against the fastest rival, one core at 95%, test questions:**
    - **Ahead on eight:** Yi 2.09× and Llama's second head 1.84× (RoarGraph); Llama 1.49× (RoarGraph); GIST1M 1.35×, LAION 1.32×, NYTimes 1.32×, Landmark 1.18×, GooAQ 1.11× (Glass).
    - **Level on three:** Yandex 1.01× and arXiv 1.01× Glass, ImageNet 0.97× Glass, all within the 5% that timing the same index again moves it.
    - **Behind on one:** GloVe, 0.80× ScaNN.
  - **The extra links switched on by themselves exactly on the three attention datasets** (Llama, Yi, Llama's second head) and stayed off on the other nine.
  - **Against the expectations:**
    - **Llama:** better than expected (1.49×, expected 1.2×). **Yandex:** better (level, expected 0.95×).
    - **GloVe:** as expected (ScaNN 1.25× faster). **GooAQ, Landmark, arXiv:** as expected.
    - **LAION and GIST1M:** better than in part 2 (1.32× and 1.35×, against 1.10× and 1.25×). **Yi:** ahead by much more, as expected if its answers bunch up (they do: 9,347 popular items).
    - **NYTimes: wrong.** ScaNN is not ahead here; Glass is the fastest rival, and we are 1.32× faster than it. **ImageNet:** as expected, around Glass. **Llama's second head:** ahead, as hoped (1.84×).
  - **Faster than HNSW, DiskANN and CoreNN** on all twelve, wherever they reach 95%; on Llama, Yi and Llama's second head none of them does.
  - **RoarGraph's numbers checked** after step 22's wrong-distance bug: it is slow on ImageNet (4,472/s), as on LAION (3,023/s) and GooAQ (6,796/s), because it compares full 4-byte numbers on long rows; its checks per search are ordinary there (882). Nothing pointed to a set-up problem.
  - **The guided build's practice run** gained a best-score case with question links and extra links on, so the build is guided through their code too.

## Final check (part 3)
Every step's check was run again on the final code, in a fresh copy of the project (guided build, then `REUSE_GRID=1 validate/run_all.sh`, now with steps 23 to 26), in 81 minutes. The official results above were left as they are, except step 26, rerun after the fix below.
- **Passed:** the 32 unit tests, steps 1 to 18, 20, 23, 24 and 25 (steps 14 and 23: nothing but steps 22 and 27 has run on the held-back datasets).
- **Missed the same checks as before:**
  - **Step 19:** Llama's rebuild gain predicted 4.44×, measured 3.77×.
  - **Step 21:** with 4-bit copies, GooAQ and Landmark again give 96.02% and 97.26% at the 95% target.
- **Passed now, but not a fair pass:**
  - **Step 20** (10% faster than the code before it): 11/11, because part 3's later gains count too.
  - **Step 25** (5% faster than the code before it): 16/16 at 1.064×, because the fresh copy's code is a guided build and the recorded code before step 25 a plain one. The fair figure is the official 1.018×.
- **Step 26, one new fail, fixed:** the GloVe noise described in step 26. With the fix: 10/10, in the fresh copy and in the official rerun.

# Part 4: the full-size proof, and sharing

**In one line:** eight more steps (`SPEC.md`, part 4). Steps 28 to 32 measure on the full datasets; steps 33 to 35 share the results. The index's code does not change.

## 28. Locked: code, rivals and expectations; full-size data
- **Builds:**
  - `validate/step28.py freeze`: a fingerprint (SHA-256) of every source file of the index, its build scripts and the installed package, and of the rivals' builds and versions, in `validate/results/step28-frozen.json`. Every later step checks it.
  - `scripts/step6_prepare_full.py` in the experiments, for all twelve datasets:
    - true answers on the full items for the 1,000 tuning questions, in two precisions, as for the test questions;
    - true answers for the 20,000 training questions on Llama, Yi and Llama's second head, which RoarGraph builds from;
    - the full-size change script: 5% of the items set aside for full replacements; build with 90% of the rest, then add, delete and update 10% each in one mixed order (half small edits, half full replacements), with true answers at 10 checkpoints.
  - The experiments' runners take full-size jobs; the default stays the sample.
- **Check:** `validate/step28.py`:
  - the samples' items are rows of the full items;
  - on the full items, no question's 10th true distance is further than on the sample (more items can only bring closer ones);
  - the two precisions agree, up to ties and rounding;
  - the change script has the right counts, and its spares are not among its items;
  - the fingerprint is recorded.
- **Pass:** all of the above.
- **Expected, written before any full-size run** (from the samples' results and how each kind of index grows; if a result contradicts these, the claim changes, not the result):
  - **Our speed** at 95% falls from the samples': by 1.5–3× on the eight datasets with 700,000 items or more, and by less on the four smaller ones (Llama, Yi, Llama's second head, NYTimes: 188,000–269,000 items).
  - **Glass falls about as much as ours,** since it is a link map too, so our ratio to it stays within about 15% of the samples':
    - still ahead on GooAQ, Landmark, LAION, GIST1M and NYTimes (1.11–1.35× in the samples);
    - level on Yandex, arXiv and ImageNet (0.97–1.01×), where it could tip either way.
  - **RoarGraph:** still behind us on the three attention datasets, by 1.3× or more (1.49–2.09× in the samples).
  - **ScaNN:** still ahead on GloVe, but by less than in the samples (1.25×), because a bucket scan's work grows faster with the items than a link map's walk. On NYTimes, still slower than Glass.
  - **HNSW:** we are at least 5× faster wherever it reaches 95% (9–18× in the samples). On the three attention datasets it still never reaches 95%.
  - **GIST1M, against CoreNN on its author's settings:** we are at least 10× faster, and use less memory.
  - **Memory per item:** within 10% of the samples', because each item's share is fixed; 0.15–0.6× HNSW's with the full numbers in a file.
  - **Picks:** they may differ from the samples' (the trial build is now 100,000 items, against 20,000). The extra links switch on only on the three attention datasets.
  - **Builds:** 10–25× the samples' time on the big datasets, 5–25 minutes on all cores.
  - **Changes:** the rule passes on every dataset, as in the samples.
- **Status:** passed, 28/28 (`validate/results/step28.json`, log in `step28.log`; the fingerprint is `step28-frozen.json`, recorded before any full-size run).
  - **True answers at full size:** for the tuning questions on all twelve, with no difference between the two precisions except ties; for the training questions on the three attention datasets.
  - **The samples are part of the full data:** on every question, the 10th true answer among the full items is at least as close as the sample's.
  - **Change scripts:** for example GooAQ: 1,381,323 items and 72,701 spares; 1,243,190 built first, then 138,133 adds, 138,133 deletes and 138,133 updates.
  - **Time:** 1 to 15 minutes per dataset, four at a time. The slow part is finding each small edit's 10 closest items.
  - **The experiments' runners** now take full-size jobs, and ScaNN's container reads the full items; sample jobs run as before. This was done before the lock.

## 29. Our index at full size
- **Builds:** nothing new.
- **Check:** `validate/step29.py`: part 3's index, with every default, built on each dataset's full items with its 20,000 training questions, and measured as in step 27: test questions, one core, middle of 5 timed runs, read off at exactly 95%; all cores; time per question; build time; memory, counted and measured; the SSD mode.
- **Pass:** every number recorded; the SSD mode gives the same answers while reading the full numbers from the SSD; the target picked on the tuning questions finds at least 94% on the test questions.
- **Status:** passed, 25/25 (`validate/results/step29.json`, log in `step29.log`), in 69 minutes.
  - **Accuracy:** on all twelve, the target picked on the tuning questions finds 95.0–96.3% on the test questions. The SSD mode gives the same answers everywhere.
  - **Speed at 95%, one core** (in brackets, how much slower than the sample): GooAQ 39,116/s (1.14×), GloVe 2,936 (3.46×), Yandex 18,057 (2.55×), Llama 104,715 (1.10×), Landmark 48,733 (1.47×), arXiv 30,005 (1.32×), LAION 10,628 (2.27×), Yi 44,290 (1.48×), GIST1M 3,980 (3.21×), NYTimes 12,702 (0.93×, so faster), ImageNet 14,499 (2.25×), Llama's second head 54,188 (1.45×).
  - **Why it falls:** mostly more checks per question (GloVe 12,662 against 5,655; GIST1M 3,805 against 1,451). GooAQ makes about the same number (523 against 518), so its fall is in the cost of each check.
  - **Picks:** the same as the samples' on Llama, Landmark and Llama's second head; different on the other nine (for example GIST1M 64 links instead of 32, ImageNet 8-bit copies instead of 4-bit, arXiv 4-bit instead of 8-bit). The extra links switched on only on the three attention datasets.
  - **Memory per item** moved with the picks: within 2% of the samples' on six datasets; ImageNet 816 bytes (496 in the sample), arXiv 560 (944), GIST1M 1,304 (1,136), NYTimes 636 (764), GloVe 708 (620), LAION 772 (688).
  - **Builds:** 29 seconds (Llama) to 13 minutes (GIST1M), on all cores.
  - **Two of step 28's expectations are already wrong** (step 30 marks them all): our speed fell 1.14–3.46× on the big datasets, not 1.5–3×; memory per item moved more than 10% on six datasets, because the picks changed.
  - **Machine:** the macOS storage scan (not ours) used 0.8–3.1 other cores during the runs; in part 1, one-core numbers held within 3% under such load.

## 30. The rivals at full size
- **Builds:** `scripts/step6_baselines.py` in the experiments: the settings search of the experiments' step 3, on the full items, then the test questions, written to `results/step6/`:
  - HNSW on all twelve;
  - Glass on GooAQ, Yandex, Landmark, arXiv, LAION, GIST1M, NYTimes and ImageNet;
  - ScaNN on GloVe and NYTimes, with the bucket counts scaled to the items;
  - RoarGraph on Llama, Yi and Llama's second head.
- **Check:** `validate/step30.py`: ours (step 29) against each rival at 95%, one core, test questions; then each expectation of step 28 is marked right or wrong.
- **Pass:** faster than HNSW wherever it reaches 95%.
- **Status:** passed, 13/13 (`validate/results/step30.json`; the rivals' runs are in the experiments' `results/step6/`, logged in its `grid.log` and `final.log`). The runs took 14 hours (2026-11-06 13:52 to 11-07 03:39), with no failures.
  - **Against the fastest rival, at 95% on one core:** ahead on 11 of 12.
    - Glass: GooAQ 1.48×, Yandex 1.30×, Landmark 1.40×, arXiv 1.39×, LAION 2.20×, GIST1M 1.37×, NYTimes 1.49×, ImageNet 1.60×.
    - RoarGraph: Llama 1.83×, Yi 1.82×, Llama's second head 1.84×.
    - Behind on GloVe: ScaNN is 1.35× as fast as we are.
  - **HNSW:** we are 8.3–18.1× as fast wherever it reaches 95%. It never does on the three attention datasets (best 26.9–49.0%).
  - **Glass on Landmark:** its quickest setting (search effort 10) already finds 96.72%, so its speed is that setting's.
  - **Timed again, head to head** (added after the runs: `validate/step30b.py`, `validate/results/step30b.json`). Our runs and the rivals' were hours apart, so the two biggest moves from the samples were timed again, Glass then ours, one right after the other.
    - ImageNet came out at 1.86× (ours 16,133/s, Glass 8,673/s), against 1.60× the first time.
    - Yandex came out at 1.10× (16,013/s against 14,526/s), against 1.30×.
    - **Both missed the check's own rule** (the ratio within 10% of step 30's): 1/3 passed.
    - Glass moved by up to 4.7%. It reran its recorded settings.
    - Ours moved by up to 11.3%. It was built again from scratch, and its setting search picks by timing trial builds, so it can pick differently: on Yandex, its checks per search went from 1,733 to 2,117.
    - Every win held both times. The results page keeps the first runs and reports both.
  - **Timing, picks or layout?** (added after the runs: `validate/step30c.py`, `validate/results/step30c.json`; passed, 5/5). GooAQ and Yandex were built twice more, one right after the other, with step 29's picks fixed and the setting search off.
    - **The same picks give the same index:** the same accuracy and checks per search as step 29 at every target, and its speed within 1.0% (GooAQ 39,342/s against 39,116; Yandex 17,880 against 18,057). So step 30b's 11% came from fresh builds picking differently, not from the timing.
    - **The memory layout is a small part of the lead:** with renumbering off (same answers, same work), our speed fell by 4.3% on GooAQ and 3.6% on Yandex on one core (1.13× on all cores). DESIGN.md had guessed it "should matter more beyond" the chip's caches; it does, but not by much. Why Glass's checks slow down more at full size is not pinned down.
  - **Where Glass lost ground at full size:**
    - Each check got slower for it than for us. ImageNet: 35 to 59 ns for Glass, 46 to 58 ns for ours. GooAQ: 45 to 77 ns, against 44 to 49 ns.
    - On Yandex, LAION and ImageNet it also needed 2.1–2.3× the samples' checks. We needed 1.8–1.9×.
    - It is not mainly our memory layout (step 30c, below); the rest is not pinned down.
  - **Step 28's expectations:** 5 right, 6 wrong (`validate/results/step30-expectations.json`).
    - Right:
      - still ahead of Glass on its five;
      - ahead of RoarGraph by 1.3× or more;
      - ScaNN slower than Glass on NYTimes;
      - at least 5× HNSW, which still never reaches 95% on the attention datasets;
      - extra links only on the attention datasets.
    - Wrong:
      - our speed fell by 0.93–3.46×, not 1.5–3× on the big datasets and less on the small ones;
      - the ratio to Glass grew instead of staying within 15% (Yandex 1.01× to 1.30×, arXiv 1.01× to 1.39×, ImageNet 0.97× to 1.60×);
      - so those three are not level;
      - ScaNN's lead on GloVe grew (1.25× to 1.35×) instead of shrinking;
      - memory per item moved more than 10% on six datasets, because the picks changed;
      - builds took 5–10× the samples' time (2.8–13.2 minutes), not 10–25×.

## 31. GIST1M against CoreNN
- **Builds:** CoreNN on GIST1M's full items with its author's own settings (64 links, build list 128, slack 1.1: libcorenn's defaults), with no settings search. As for every rival, its search effort is the smallest that reaches 95% on the tuning questions.
- **Check:** `validate/step31.py`: speed at 95% (one core, test questions) and memory per item, ours (step 29) against CoreNN, and the one sentence.
- **Pass:** the sentence's numbers come straight from the two results files.
- **Status:** passed, 2/2 (`validate/results/step31.json`; the sentence is in `step31-sentence.txt`):
  - "On GIST1M, the dataset CoreNN's author tested on, our index answers 17.1x as many questions per second as CoreNN at 95% of the true top 10 (one core), and uses 76% less memory with the full numbers on SSD (9% less with them in memory too)."
  - CoreNN: 233/s at search effort 50, with a 41-minute build. Ours: 3,980/s, with a 13-minute build.
  - Step 28's expectation (at least 10× as fast, with less memory) is right.

## 32. Changes at full size
- **Builds:** nothing new.
- **Check:** `validate/step32.py` on all twelve, the full-size change script as step 9 runs it:
  - the first build picks its own settings, and the target is picked on the tuning questions right after it;
  - the effort table is kept through every change; 10 checkpoints on the test questions;
  - then a fresh build on the final items with the first build's picks, so the comparison is not between two separate tunings (step 26's lesson).
- **Pass:** drift under 1 point (or better than the fresh build), no deleted or old items returned at any checkpoint, memory within 5% of the fresh build.
- **Status:** 12/13: passed on 11 datasets, missed on Landmark (`validate/results/step32.json`, log in `step32.log`), in 3 hours.
  - **Every dataset:** no deleted item and no old numbers returned at any checkpoint; memory 1.001–1.006× a fresh build's; the changes took under 1 minute (Yi) to 14 minutes (ImageNet).
  - **Drift against a fresh build,** in points: GooAQ −0.17, GloVe −0.05, Yandex −0.45, Llama −0.35, Landmark −1.08, arXiv +0.32, LAION −0.48, Yi −0.44, GIST1M +1.14, NYTimes −0.06, ImageNet −0.81, Llama's second head −0.68.
  - **Landmark's miss:** after 210,831 changes it finds 94.13%, against 95.21% for the fresh build, with the same work (335 checks per question against 336). Its target, 0.88, is the lowest the test tries (even that reached 95% on the tuning questions), so it searches with little to spare, where damage from changes shows most. Learning the effort table again gave exactly the same answers. In the samples (part 1's index), Landmark's drift was −0.41 at the same target, with 98.7% found.
  - **GIST1M gained** because its changed index searched 9% more (3,281 checks per question against 3,007).
  - **Step 28's expectation "the rule passes on every dataset" is wrong:** Landmark misses by 0.08 points. The code stays as frozen; the miss is reported as it is.

## 33. The results page
- **Builds:** `tools/results_page.py`: one web page, made from the results files only:
  - for everyone, at the top: one sentence per dataset (what it is, and whether we won) and one picture;
  - for technical readers: speed against accuracy for every method on every dataset; memory; the held-back datasets marked; settings and versions;
  - the honesty box: what we did not beat, where the rivals' settings search may be smaller than their authors would use, and what needed a Linux container or a patch.
- **Check:** `validate/step33.py`: every number on the page is recomputed from the results files and matches; the page shows no errors and fits a phone screen.
- **Pass:** all of the above.
- **Status:** passed, 8/8 (`validate/results/step33.json`). Published as a private page; the file itself is `page/results.html`.
  - `tools/results_page.py` writes `page/results.html` from the results files only: 358 different numbers, each recomputed independently by `validate/step33.py`, which also finds no number outside them except the method's constants and axis ticks.
  - **In Chrome,** at 390 px and 1280 px, light and dark: no errors, no sideways scroll, and the chart readout works. Headless Chrome keeps its window at least 500 px wide, so the check runs the page in a frame of exactly the width tested.
  - **Added after the rival runs:** Glass's quickest setting on Landmark (it already finds 96.72%; its dot sits there), the re-checks of steps 30b and 30c, and a short "why the lead grew".

## 34. The GooAQ demo
- **Builds:** `demo/`: a page with one text box. A typed question is turned into numbers by the model that made GooAQ's numbers, searched in our index of the full GooAQ, and the 10 closest Google answers are shown with the time taken. A switch shows HNSW's answers beside ours. Plus a short recording of it in use.
- **Check:** `validate/step34.py`:
  - each text shown is the one its numbers were made from (the model, run on the text, gives the numbers that were searched);
  - for 100 questions, the demo's answers are the index's own;
  - GooAQ's licence allows showing its text.
- **Pass:** all of the above.
- **Status:** passed, 5/5 (`validate/results/step34.json`, log in `step34.log`). How to run it: `demo/README.md`; the recording is `demo/demo.gif`.
  - **Its own environment** (`demo/.venv`: Python 3.13, torch on the Apple GPU, sentence-transformers, hnswlib 0.8.0). vindex is the package frozen in step 28, linked in, not rebuilt.
  - **Preparing the data took about 70 minutes:**
    - 44 minutes to turn GooAQ's 1,476,024 unique answers into numbers;
    - then matching each item to its answer: 1,453,978 of 1,454,024 items (99.997%) have their text, at a similarity of 0.99901 or more (the middle one 1.000000);
    - then our index (644 s, as step 29) and HNSW (484 s, with step 30's settings).
    - It writes 14 GB to `demo/work/`.
  - **For 100 of GooAQ's own questions,** the demo showed the same 10 answers in the same order as the indexes' own searches, for vindex and for HNSW. The model, run on each of the 1,082 texts shown, gives that item's numbers.
  - **Times shown** (the middle over the 100 questions): vindex 0.018 ms, HNSW 0.225 ms, question to numbers 5.7 ms. These are shorter than the benchmark's, since one search repeated keeps what it reads in the chip's caches.
  - **Licence:** GooAQ is Apache 2.0, with the request that it not be used for any commercial purpose. The page credits it and runs on this machine only.
  - **Fixed on the way:** the helper index that matches items to answers needed `learned_effort=False` (it is searched with a fixed list size); the server now answers the browser's icon request, so the page logs no error.

## 35. Report and the one-page write-up
- **Builds:** `REPORT.md`, with part 4 at the top; `WRITE-UP.md`, one page for everyone: the goal, the datasets, what we built, what we found and what we did not achieve.
- **Check:** every number in them is in a results file, and the fingerprint is unchanged since step 28.
- **Status:** passed, 6/6 (`validate/step35.py`, `validate/results/step35.json`).
  - **REPORT.md part 4's three tables** (speed, all cores, memory): every cell of their 36 rows, recomputed from the results files.
  - **Every other number** in REPORT.md part 4 and in `WRITE-UP.md` comes from the results files (recomputed by the check, or one of the results page's), or is a constant of the method or a step's number.
  - **The CoreNN sentence** is word for word step 31's.
  - **Nothing left unfinished:** no placeholders in either, or in `demo/README.md`.
  - **Pictures:** the write-up's diagram is baked to a picture, and every picture shown exists.
  - **It caught three slips in the first draft:** a build time 0.1 minute off; the page's number count from before its last passages; and a sentence saying Glass needed more extra checks, which holds on three of its eight datasets only.

# Part 5: steady builds, the change test, and a second way to search

Plan: `SPEC.md`, part 5. The code changes again in this part. Each step is checked on the samples first; the full-size runs come once, at the end, with the code frozen again (step 42).

## 36. Locked: a new held-back dataset, and expectations
- **Builds:**
  - GloVe-100 (ann-benchmarks' glove-100-angular: 1.18 million words, 100 numbers each), downloaded into its own folder and checked against the server's checksum, then prepared as part 3's NYTimes: sample, questions, true answers on the sample and the full items, the full-size change script.
  - It is locked like the earlier held-back datasets: nothing runs on it until step 42.
- **Check:** `validate/step36.py`: the file's checksum; the prepared data's own checks (as step 23); the lock holds; the expected results are written below before any part 5 code changes.
- **Pass:** all of the above.
- **Expected, written before any part 5 code change** (if a result contradicts these, the claim changes, not the result):
  - **Which choices flip** (step 37): mostly the re-check count, whose bar is only 2%, and links and slack on Yandex, whose full-size speeds jump between neighbouring settings.
  - **Landmark's changes** (step 38): the drift comes from links that the repair after a delete leaves weak, not from the effort table (learning it again changed nothing in step 32).
  - **The bucket search** is picked on GloVe and GloVe-100 and on none of the other eleven. On NYTimes, the other dataset where a link walk needs thousands of checks, links still win: ScaNN was half Glass's speed there in part 4.
  - **GloVe:** 1.25–1.5× part 4's speed, and so within 10% of ScaNN (0.9–1.1×).
  - **GloVe-100 (held back):** ScaNN is the fastest rival, then Glass. Ours, with the bucket search, within 10% of ScaNN; at least 5× HNSW; the change test passes.
  - **The other eleven:** still ahead of the fastest rival, but steadier picks may cost a few percent where part 4's builds were lucky (Yandex).
- **Status:** passed, 5/5 (`validate/results/step36.json`).
  - **The download** (`scripts/download_part5.sh` in the experiments) matches ann-benchmarks' server tag.
  - **Prepared like the others** (`scripts/step2_prepare.py glove100`, then `scripts/step6_prepare_full.py glove100` with only this run unlocked, 9 minutes): 1,162,514 items of 100 numbers; 1,000 test questions (a seeded draw of the file's 10,000), 1,000 tuning and 20,000 training questions, set aside from the items. Our true top 10 agree with the file's on 98.31%; the other 1.69% are exactly the items set aside as questions. No difference between the two precisions.
  - **Its own lock:** `validate/common.py` `HELD_BACK_5` and `unlock_held_back_5()`, and in the experiments `bench/paths.py` `HELD_BACK_5` with the variable `BENCH_UNLOCK_HELD_BACK_5`. Both refuse it even with the earlier held-back datasets unlocked.
  - **Related to GloVe:** the same Twitter words, from a separately trained model with 100 numbers per word. So it tests whether the bucket search carries over to another set of word numbers, not to a new kind of data.

## 37. Steady picks
- **Builds:**
  - First, measure: three fresh builds of each dataset's sample, keeping every trial's timing, to see which choices flip.
  - Then the rule: a choice replaces the one in use only when it is clearly faster in a head-to-head with interleaved runs; otherwise the plainer choice stays (fewer bytes per item, then the default). The same rule for links, slack, copy size, the re-check count and the extra links.
- **Check:** `validate/step37.py`: three fresh builds of each sample pick the same settings; their speed at 95% (tuning questions) within 3%. At full size, Yandex and ImageNet, built three times each, within 3%.
- **Pass:** all of the above, and no dataset more than 3% slower than part 3's sample speed (tuning questions).
- **Status:** failed, and kept as it stands: samples 33/36 (`validate/results/step37.json`), full size 4/6 (`step37-full.json`).
  - **Before the fix** (`step37-before.json`): 5 of 12 samples made the same picks in all three builds, and the speeds of the three builds were up to 13.5% apart.
  - **The first rule** (ties within 2% go to the plainer choice; `step37-first-rule.json`, `step37-full-first-rule.json`): 8 of 12. The flips were all close calls sitting right on the 2% or 5% bar. Three copies of one index, timed against each other for longer, came within 0.3%, so the flips were timing noise.
  - **The rule now** (`DESIGN.md` section 16): close calls (within 1.5% of a bar) are timed for 20 more rounds; when new links lose the final test, slack is tried again on the old links.
    - **Samples:** 11 of 12 make the same picks, so the same index, every time; their three builds are within 0.2–1.0% of each other, head to head, and never more than 1.1% slower than the fastest pick made before the fix.
    - **ImageNet's sample still flips:** slack 1.3 against the start's 1.2 measured 5.4%, 6.1% and 3.4% faster in the three builds, against a 5% bar. The true gap sits on the bar, so no amount of timing settles it. The build that kept 1.2 then went on to 4-bit copies and ended 7–8% slower.
  - **Full size:**
    - **Yandex:** all three builds made the same picks, so the same index (1,770.19 checks per search each). Yet the head-to-head put them 3.2% apart, so at full size the check's own noise is about 3%.
    - **ImageNet:** the copy size and re-check count differ (4-bit, 50 against 8-bit, 15), but the three builds are within 0.6% of each other and 14.5% faster than step 29's pick.
  - **Against the expectation:** half right. The re-check count flipped under the first rule, but under this rule it no longer does. What still flips is slack, on ImageNet, not Yandex.
  - **Not done:** carrying both sides of a close call at the 5% bar through to the full-size build would settle it, at roughly half again ImageNet's build time. That is left for later.

## 38. The change test on Landmark
- **Builds:** first find why Landmark's accuracy falls 1.08 points further than a fresh build's after its changes; then fix that in the change code.
- **Check:** `validate/step38.py`: Landmark's full-size change test again, as step 32 runs it; the samples' change tests on all twelve.
- **Pass:** Landmark's drift under 1 point at full size; on the samples, no dataset's drift worse than before by more than 0.2 points; nothing deleted or outdated returned; memory within 5% of a fresh build.
- **Diagnosis** (`validate/step38.py diagnose` and `try`, `validate/results/step38-diagnose.json`): Landmark's full-size change test, one thing changed at a time.

  | Run | Drift, points | Links per item after (fresh 16.3) | Items no walk can reach (fresh 241) |
  |---|---|---|---|
  | As it is | -1.08 | 17.5 | 562 |
  | No cleanup of dead links | -1.01 (memory 1.11×) | 17.6 | 568 |
  | 8 replacements, appended | -1.68 | 16.7 | 564 |
  | No spot checks or learning from use | -1.08 | 17.5 | 562 |
  | Repaired lists pruned again | -4.96 | 10.2 | 1,047 |
  | Only replacements that reach a new way, from 3 | -1.24 | 15.3 | 470 |
  | Only replacements that reach a new way, from 8 | **-0.63** | 15.7 | 383 |

  - The start item never moved, and the twelve samples' own change tests all pass as they are (drift -0.42 to +1.12): the miss needs full size and Landmark's very cheap search (its target 0.88 is the lowest the test tries).
  - **The fix** (`DESIGN.md` section 9): deletes repair with `repair="occlude"` and `n_replace=8`, now the defaults. Parts 1 to 4's checks keep the old repair (`validate/common.py` `PART1`).
- **Status:** 11 of 13 passed (`validate/results/step38.json`, log in `step38.log`).
  - **Landmark at full size passes:** drift -0.63 (94.58% after 210,831 changes, against 95.21% for a fresh build), nothing deleted or outdated returned, memory 1.002× a fresh build's.
  - **The samples:** all twelve stay well inside the 1-point rule (drift -0.35 to +1.35). Ten are no worse than before by more than 0.2 points; Llama got better (-0.42 to -0.03).
  - **Two missed the 0.2-point bar:** Landmark's sample, +0.22 against +0.42 (exactly at the bar, and still better than a fresh build), and NYTimes, -0.25 against +0.04. NYTimes's earlier record came from a build whose setting search still sent slack ties toward the setting in use, so its picks may differ too.

## 39. Buckets and their fast codes
- **Builds:** `buckets.rs` (`DESIGN.md` section 23): the index groups its items into buckets around centres (k-means, about 300 items per bucket), and gives each item a short code, 4 bits per pair of numbers, laid out 32 items to a block so that one vector table lookup scores 16 of them from a small table made for each question. A search scores the centres, scans the nearest buckets' codes, and re-checks the best with the full numbers. For now it is reached on its own (`Index.build_buckets`, `Index.search_buckets`). (Planned as two steps, a plain scan first; the fast codes were written straight away, so the two became one.)
- **Check:** Rust unit tests (the vector sums equal the one-at-a-time sums; every live item in exactly one bucket; scanning everything and re-checking everything gives the true top 10); `validate/step39.py`, on the samples of GloVe and NYTimes (tuning questions): the same exact case on real data; speed against accuracy for buckets scanned and re-check pools, and the link walk on the same index, timed the same way.
- **Pass:** exact when nothing is left out; the speeds recorded.
- **Status:** passed, 4/4 (`validate/results/step39.json`), with plain codes and 300 items per bucket.
  - **Exact:** scanning all 333 buckets and re-checking every item gives the true top 10 on GloVe and NYTimes (200 tuning questions each). The buckets take 0.4 s to make on a 100,000-item sample.
  - **Slower than the link walk:** at 95% on the tuning questions, GloVe 8,055/s (pool 100) against the link walk's 11,454/s (0.70×); NYTimes 4,219/s against 11,994/s (0.35×).
  - **Why:** the scan costs about 4 ns per item, about ScaNN's own cost, but needs 29,434 items per question on GloVe, where ScaNN needs about 15,000 (part 2's records). Its codes are less exact for their size. So step 40 tries ScaNN's score-aware codes, which count errors along each item more, and smaller buckets.

## 40. Buckets: tuned
- **Builds:** whatever step 39's numbers ask for to reach the link walk's speed on GloVe: the number of items per bucket, the re-check pool, the codes themselves.
- **Check:** `validate/step39.py` again, on the tuning questions only.
- **Pass:** on GloVe's sample, the bucket search at least as fast as the link walk at 95% (tuning questions).
- **Status:** passed on the sample (`validate/results/step40.json`, GloVe only), but at full size the bucket search does not beat ScaNN, so it is not built into the index (the rule agreed with the user: build it in only if it beats ScaNN).
  - **Sample (tuning questions):** score-aware codes, 100 items per bucket, re-checking the best 100, and whole blocks passed over when none of their 32 sums can enter the best 100: 11,445/s at 95%, 1.02× the link walk (11,171/s). NYTimes: 0.36×.
  - **Full GloVe, first try** (`step40-full.json`): 2,902/s at 95% with 300 items per bucket, 0.97× the link walk (2,988/s); ScaNN does 3,956/s (part 4).
  - **Where the answers were lost:** not in the codes. Re-checking the best 100 by their codes finds what re-checking every item in the opened buckets finds (88.91% against 89.12% at 64 buckets on the sample). They were in buckets left closed: ScaNN reaches 95% opening 7% of its buckets at full size; ours needed about 10%.
  - **The buckets trained harder** (`step40-grouping.json`): with 20 k-means rounds on 256 items per centre, 95.56% of the true answers sit in the nearest 7% of buckets (95.12% before), and the search does 3,458/s at 95% scoring 80,551 items per question (ScaNN about 82,000). With 30 rounds on every item: 3,482/s. That is 1.17× our link walk but 0.88× ScaNN: what remains is per question (scoring 3,900 centres, the table, the re-checks).
  - **Kept as a recorded experiment:** `buckets.rs` and `Index.build_buckets` / `search_buckets` stay, so steps 39 and 40 can be rerun; the index's own search does not use them.

## 41. Buckets in the index: picked, changed, saved (not done)
- **Not done:** step 40 found the bucket search at 0.88× ScaNN on full GloVe at best, and the agreed rule was to build it in only if it beat ScaNN. Building it in would have taken GloVe from 0.74× ScaNN to about 0.87×, still behind.
- **Was planned:** the setting search to time the bucket search against the link walk and keep the faster; an effort table for buckets; adds, deletes and updates; saving and loading; the SSD mode.
- **Status:** not done, by decision (above).

## 42. Frozen again; full size; the held-back test
- **Builds:** a new fingerprint (as step 28); our index at full size on the twelve datasets and GloVe-100 (as steps 29 and 32); the rivals at full size on GloVe-100 only: ScaNN, Glass and HNSW, each with its settings search (as step 30).
- **Check:** `validate/step42.py`: everything steps 29, 30 and 32 measured, against part 4; SPEC part 5's targets; step 36's expectations marked right or wrong.
- **Pass:** the targets in SPEC.md, part 5; any miss is reported as it is.
- **Status:** 3/4 (`validate/results/step42.json`). Our full-size runs 14/14 (`step42-ours.json`, 90 minutes), the change test 14/14 (`step42-changes.json`, 3.3 hours), GloVe-100's rivals in 47 minutes with no failures (the experiments' `results/step7/`). The code was frozen first (`step42-frozen.json`).
  - **One core, at 95% (test questions), against part 4:** GooAQ 39,545/s (1.01×), GloVe 2,907 (0.99×), Yandex 14,971 (**0.83×**), Llama 100,112 (0.96×), Landmark 46,560 (0.96×), arXiv 29,407 (0.98×), LAION 10,566 (0.99×), Yi 44,284 (1.00×), GIST1M 3,921 (0.99×), NYTimes 12,169 (0.96×), ImageNet 15,379 (1.06×), Llama second head 57,607 (1.06×).
  - **Still ahead of the fastest rival on all eleven won in part 4**: 1.08× (Yandex) to 2.19× (LAION). GloVe is still behind ScaNN (0.73×).
  - **GloVe-100, the held-back dataset:** ours 7,184/s; ScaNN about 9,830, Glass about 7,750, HNSW about 1,010. So 0.73× ScaNN, 0.93× Glass and 7.1× HNSW.
  - **The change test passes on all thirteen**, Landmark now -0.63 (part 4: -1.08). The worst are GIST1M at +1.02, which improved after its changes, and ImageNet at -0.90.
  - **The miss:** the GloVe target, because the bucket search is not built in (step 41).
  - **Yandex flipped again.** This build picked 96 links and slack 1.2. All three of step 37's full-size builds picked 48 links and slack 1.3, at 16,320–16,547/s on step 29's measure; this one is 8.3% slower than the slowest of them (14,971/s). In the links trial, 48 and 96 ended 0.14% apart. But this build's first round put 96 more than 3.5% ahead, outside the close-call window, so it was never timed longer. A wider window would have caught it. Part 4's 18,057/s was a lucky build: even the steady picks are 0.90–0.92× it.
  - **Step 36's expectations:** 4 of 8 right, 1 half right, 3 wrong.
    - **Which choices flip:** half right (step 37).
    - **Landmark's cause:** right. The repair left weak links, and the effort table was not to blame.
    - **The bucket search picked on GloVe and GloVe-100:** wrong, since it is not built in.
    - **GloVe within 10% of ScaNN:** wrong, 0.73×.
    - **GloVe-100, ScaNN then Glass:** right.
    - **GloVe-100, ours within 10% of ScaNN:** wrong, 0.73×.
    - **GloVe-100, at least 5× HNSW:** right, 7.1×. Its change test passes.
    - **The other eleven still ahead:** right. "A few percent" lower on Yandex was too mild: it fell 17%.

## 43. The page and the reports again
- **Builds:** the results page with part 5's numbers (part 4's kept beside them), `REPORT.md` part 5, `WRITE-UP.md` brought up to date.
- **Check:** steps 33 and 35's checks, extended to part 5.
- **Pass:** all of the above.
- **Status:** passed. `validate/step33.py` 8/8; `validate/step43.py` 5/5 (part 5's version of step 35).
  - **The results page** (`tools/results_page.py`, republished at the same link):
    - our numbers are part 5's (step 42), with part 4's build beside them: a thin mark on the ratio chart, a row in the table, and a drift column for the change test;
    - GloVe-100 is added, with its rivals from part 5;
    - the "timed again" and "why the lead grew" passages (part 4's) gave way to how steady fresh builds are, and to why the bucket search was left out;
    - the expectations are step 36's.
  - **`REPORT.md` part 5** comes first; the picture is `diagrams/speed-part5.png` (`tools/draw_report.py part5`). **`WRITE-UP.md`** is brought up to part 5.
  - **`validate/step43.py`** recomputes both of part 5's tables cell by cell, and accounts for every other number in part 5 and the write-up: from the results files, or from the tuning constants in `tune.rs`. On its first run it caught two numbers copied wrong from the logs, 0.15% for 0.14% and 6.0% for 6.1%. Both are fixed here, and in step 37's and step 42's notes.
  - **Part 4's checks:** `validate/step35.py` is not rerun. It checks part 4's report against part 4's page and step 28's freeze, both since moved on, and its results file stays as part 4's record. `validate/results/step33.json` now holds part 5's page check; part 4's was 8/8 with 358 numbers (`REPORT.md` part 4).

## 44. After the freeze: the playground and build progress
- **Builds:** `playground/` (download, build and search a public dataset from the command line or a browser page, with every answer checked), and `idx.build_progress()`, which the page reads while a build runs. A README chart and table of part 5's speeds against the other indexes (`tools/draw_report.py readme`).
- **Check:** build progress only reports. With the same data and settings, the old code (step 42's freeze) and the new code save the same index, byte for byte.
- **Pass:** identical saved files for `cos`, `ip` and `l2`.
- **Status:** passed. `validate/step44_same_build.py OUT_DIR` builds 40,000 random items of 48 numbers with `tune="off"` (the settings trial picks by timing, so two builds with it on can differ anyway) and saves each index. Run under step 42's package and under the new one, `diff -r` finds no difference.
  - **Part 5's results stand** for the frozen code. `validate/step43.py`'s first check, "the code is as frozen in step 42", now fails, as it should: `index.rs`, `lib.rs`, `suggest.rs`, `tune.rs` and the Python package changed, by the progress lines only.

## 45. After the freeze: review fixes, tests and CI
- **Builds:** fixes from an outside review of the code:
  - `search` (and `search_buckets`) refuse `k = 0`; before, it panicked;
  - `build_in_order` refuses an order that lists a row twice; before, it checked only the length and range;
  - `save` writes each file to `<name>.tmp` and puts them in place only once all are written, so a failed save leaves the previous index whole; before, it overwrote each file in place;
  - the switches `learned_effort`, `track_usage`, `effort_apart` and `renumber` read `on`/`off` and `yes`/`no` as the others do, and refuse anything else; before, any other word meant false.
  - New tests: recall@10 against exact search on random data (cos, ip, l2), the refused inputs, the failed save, the switch words, and a Python test suite (`tests/`). Clippy's warnings are cleared. CI runs both test suites on macOS and Linux x86.
- **Check:** with the same data and settings, the code before and after these fixes saves the same index, byte for byte (`validate/step44_same_build.py`, as in step 44).
- **Pass:** identical saved files for `cos`, `ip` and `l2`.
- **Status:** passed: `diff -r` finds no difference. `cargo test --release -p vindex`: 43 passed. `pytest tests/`: 7 passed. The recall test finds 93.5% (cos), 95.4% (ip) and 94.3% (l2) at target 0.95; it requires 90%.
  - **A second review** found three more: a save that replaced an index of another kind could leave the old index's extra files (such as `checked_q.f32`) beside the new `meta.bin`; a save without tuning kept the old `tuning.json`; and `debug_search_graph` panicked on `k = 0`. Now `save` removes `meta.bin` and the old extra files before the new files go in, removes a stale `tuning.json`, and reports any error removing them; `debug_search_graph` refuses `k = 0`. A save stopped partway leaves no `meta.bin`, so `load` refuses the folder instead of reading a mix. Step 44's check, run again on this code under the profile-guided build: identical saved files for `cos`, `ip` and `l2`. CI: both jobs pass.

## 46. After the freeze: what each piece adds
- **Builds:** `validate/step46_pieces.py`. Each of GooAQ, GloVe, LAION and Llama is built at full size with the settings its tuned build picked in step 42, tuning off, then again with one piece changed: 8-bit codes where 4-bit was picked, no codes (the walk on full vectors), no question links, no popular links, the graph on x instead of the Möbius-transformed vectors, and the default settings. "Fixed effort" searches the first build with one list size for every query, read off at 95%. Measured as step 42 measured.
- **Check:** each variant's speed at 95% recall@10, as a share of the first row.
- **Status:** done, 20 rows in 47 minutes. The first row is 1% to 8% faster than step 42's tuned build of the same settings (40,019 against 39,545 on GooAQ; 108,604 against 100,112 on Llama): a new build, timed again.

| Change | GooAQ | GloVe | LAION | Llama |
|---|---|---|---|---|
| As the tuning picked (tuning off, same settings) | 40,019/s | 3,056/s | 10,858/s | 108,604/s |
| One list size for every query (no per-query budget) | 0.94× | 0.78× | 0.92× | 1.00× |
| 8-bit codes instead of 4-bit | 0.90× | – | – | – |
| No codes: the walk compares full float32 vectors | 0.38× | 0.58× | 0.44× | 0.84× |
| No question links | – | – | 0.94× | 0.03× |
| No popular links | – | – | – | 0.42× |
| Graph built on x, not x / ‖x‖² | – | – | – | never 95% (best 56%) |
| Default settings (r 32, alpha 1.2, 8-bit, re-check 50) | 0.90× (the 8-bit row) | 0.49× | 0.90× | 0.88× |

  - **The codes** matter most on wide data (0.38× to 0.58× without them on GooAQ, GloVe and LAION; 0.84× on Llama).
  - **The per-query budget** makes it 1.06× to 1.28× as fast, and changes nothing on Llama (1.00×).
  - **Llama needs its question links and the Möbius graph:** 0.03× without the links; with the graph on x the search never passes 56%. The popular links make it 2.4× as fast (0.42× without).
  - **4-bit over 8-bit on GooAQ:** 1.11× (0.90× with 8-bit).

## 47. After the freeze: Faiss and LoRANN
- **Builds:** four more rivals, each with its own ARM NEON code, on all 13 datasets at full size: Faiss HNSW (`IndexHNSWFlat`), Faiss HNSW-SQ8 (8-bit HNSW, then the best 20 re-checked on the full numbers), Faiss IVF-PQ (buckets and 4-bit fast-scan codes, then a re-check) and LoRANN (buckets and a small learned model per bucket, then a re-check). The experiments' step 8: `scripts/step8_more_rivals.py`, `bench/runners/faiss_idx.py`, `bench/runners/lorann_idx.py`. Reviewers had pointed out that hnswlib has no ARM vector code and that Faiss and LoRANN were never tried.
- **How they were run:** as steps 6 and 7 ran the others: a settings search on the tuning queries, then the best setting on the test queries, one core, median of 5 runs, read off at exactly 95%.
  - Faiss HNSW used the links and build effort hnswlib picked; Faiss HNSW-SQ8 Glass's (hnswlib's where Glass did not run).
  - Faiss IVF-PQ tried one bucket per 1,000 and per 250 items; LoRANN sqrt(items) and 2 × sqrt(items) buckets, with and without its reduced 128 or 256 numbers. LoRANN got the training queries on the three chatbot datasets, as its documentation advises.
  - Buckets searched and candidates re-checked were swept together. The first sweep stopped at 3,200 re-checks and 0.5% of the buckets: LoRANN on GloVe and Faiss IVF-PQ on Llama and Yi were still gaining at 3,200 with every bucket searched, and Faiss IVF-PQ on GooAQ reached 95% at 0.5%. The sweep was widened to 25,600 and 0.1%, and those four rerun; all four then reach 95%.
  - NGT-QG was not run: its quantized-graph search is x86-only.
- **Check:** each rival's result per dataset; vindex's part 5 numbers (step 42) against the fastest other index, now counting these four and no longer hnswlib.
- **Status:** done. Test queries per second, one core, at 95% recall@10:

| Dataset | vindex | Faiss HNSW | Faiss HNSW-SQ8 | Faiss IVF-PQ | LoRANN | hnswlib (step 6/7) |
|---|---|---|---|---|---|---|
| GooAQ | 39,545 | 10,923 | ≥11,014 | 12,503 | 6,351 | 3,221 |
| GloVe | 2,907 | 518 | 672 | 3,773 | 690 | 297 |
| Yandex | 14,971 | 4,153 | 4,771 | 7,603 | 5,829 | 2,187 |
| Llama | 100,112 | never (26%) | never (26%) | 1,190 | 6,162 | never (30%) |
| Landmark | 46,560 | 18,565 | ≥13,350 | 16,129 | 22,407 | 5,194 |
| arXiv | 29,407 | 8,926 | 9,619 | 11,089 | 6,992 | 2,694 |
| LAION | 10,566 | 1,410 | 2,457 | 5,088 | 1,213 | 801 |
| Yi | 44,284 | never (40%) | never (40%) | 870 | 5,166 | never (49%) |
| GIST1M | 3,921 | 1,347 | 1,575 | 845 | 3,866 | 302 |
| NYTimes | 12,169 | 1,414 | 2,546 | 4,319 | 1,152 | 701 |
| ImageNet | 15,379 | 2,416 | 3,547 | 11,120 | 4,814 | 985 |
| Llama, 2nd head | 57,607 | never (28%) | never (28%) | 2,454 | 5,440 | never (27%) |
| GloVe-100 | 7,184 | 1,971 | 1,071 | 6,027 | 2,680 | 1,012 |

  - **The fastest other index changed on three datasets:** LAION and ImageNet (Faiss IVF-PQ), and GIST1M (LoRANN). vindex's lead there fell: LAION 2.19× to 2.08×, ImageNet 1.70× to 1.38×, GIST1M 1.35× to 1.01× (level).
  - **vindex is now faster on 10 of 13** (1.08× to 2.08×; Yandex's 1.08× is within timing noise), level on GIST1M (1.01×), and slower on GloVe (ScaNN 1.36×, Faiss IVF-PQ 1.30× as fast) and GloVe-100 (ScaNN 1.37×).
  - **Faiss HNSW is 1.7× to 4.5× as fast as hnswlib** at the same settings, and still never reaches 95% on the three chatbot datasets. Faiss HNSW-SQ8 reached 95% at its smallest search effort on GooAQ and Landmark, so its speed there is a lower bound.
  - **LoRANN builds fast:** 2 seconds on GIST1M against vindex's 903.
  - The README's table and chart (`tools/draw_report.py readme`), the write-up and the results page now count these rivals. `validate/step43.py` checks the write-up's new numbers against the step 8 files, and `validate/step33.py` the page's (774 numbers, 0 wrong). In both, the first check, "the code is as frozen in step 42", fails, as expected since step 44: the code changed after the freeze.

