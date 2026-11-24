# Design

**In one line:** exactly the prototype's kept design (`../vector-index-experiments/bench/ours/`), written so it can be built and checked piece by piece. Every setting below is the prototype's final value.

## 1. Settings (defaults)
- **Links:** R = 32 links per item, 41 slots (1.3 × R) so a list can grow before it is pruned back to R.
- **Build:** build effort L_build = 200; slack alpha = 1.2; random order seed 7.
- **Best-score data:** links built on flipped copies (x / |x|²) by straight-line distance.
- **1-byte copies:** on; the best 50 are re-checked with full numbers.
- **Question links:**
  - on automatically when questions look 1.5× farther from items than items are from each other;
  - 32 answers per training question; at most 16 question links kept first per item;
  - made twice at build.
- **Learned effort:** first list 10; 8 groups; 3,000 training questions; targets 88% to 99.5%.
- **Changes:** a deleted item's neighbours take up to 3 of its links; freed slots are recycled in batches of 1,000; question links are redone every 3,000 changes.
- **Part 3:** the tuning also picks the re-check count (15, 25 or 50; section 20), and, when sample answers bunch up on few items, whether popular items keep up to 64 extra links (section 22).

## 2. Distances (smaller = closer)
- **Angle:** `1 − dot(q, x)`. The index scales items and questions to length 1 when they come in.
- **Best score:** `−dot(q, x)`.
- **Straight-line:** squared distance `Σ (q − x)²`.
- **A check** is one distance worked out during a search. Checks on 1-byte copies and on full numbers are both counted, the same way as the prototype.

## 3. The 1-byte copies
- Made after the build. For every number position j over all items: `low_j = min`, `step_j = (max − min) / 255` (1 if max = min).
- Each number is stored as `round((x_j − low_j) / step_j)`, clipped to 0..255. Items added later use the same low and step (clipped).
- **Distance from a question:** prepare once per question `a_j = q_j × step_j` and `c = Σ q_j × low_j`; then dot ≈ `c + Σ a_j × code_j`.
  - Angle: `1 − dot`; best score: `−dot`.
  - Straight-line: `|q|² + |copy|² − 2 dot`, with `|copy|²` stored per item.

## 4. The walk (search with a fixed list size L)
1. A list sorted by distance, holding at most L entries, each marked "opened" or not. Start with the start item.
2. Repeat: open the closest entry not yet opened. For each of its links to an alive item not seen before:
   - mark it seen;
   - work out its distance (on 1-byte copies when they exist, else full numbers);
   - insert it into the list (dropped if the list is full and it is farther than the last).
3. Stop when every entry in the list is opened.
4. Answer:
   - **with 1-byte copies:** take the first max(50, k) alive entries, re-check each with the full numbers, and sort;
   - **without:** the first k alive entries.

Dead items are never inserted, so they are never returned. The insert keeps equal distances in arrival order (the new entry goes before existing equal ones), as in the prototype.

## 5. The learned effort (Fix 1, better version)
**Search with a table.** The table holds 7 cut points and 8 list sizes.
1. Run the walk with list size 10, but let the list hold up to the largest size in the table, so nothing is thrown away.
2. When every entry among the first 10 is opened:
   - take z = minus the distance of the 10th alive entry (of the last alive one if fewer than 10);
   - count how many cut points z is above: that picks the group;
   - grow the list to that group's size.
3. Continue the same walk (no work repeated) until every entry within the grown size is opened. Answer as in section 4.

**Learning the table** (at build, from training questions only):
1. Take the first 3,000 training questions and find their true top 10 among the current items, by checking every item.
2. Run each at list size 10 (one group); the 8 groups are split at the 1/8, 2/8, … 7/8 points of their z values.
3. For each list size in 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000:
   - run all 3,000 with that size for every group;
   - record per question the correct answers (tie rule: an item counts if its distance ≤ the true 10th + 0.001) and the checks;
   - stop after the first size where the average reaches 99.5%.
4. **A table for a target accuracy T:**
   - start every group at the smallest size;
   - repeatedly give one group a bigger size, choosing the upgrade with the most extra correct answers per extra check;
   - stop when the training questions' correct answers reach T × 10 × 3,000.

At search time the target picks the table, so one built index serves any target. The learned numbers are stored with the index.

## 6. Picking links (prune)
Given an item v and candidates with their distances to v, pick at most R that point in different directions (DiskANN's rule):
1. Optional "keep first" links go in first (used by question links).
2. Go through candidates closest first, in rounds with a = 1.0, then a × 1.2, up to alpha.
3. A candidate p is skipped in a round when an already-kept link j "occludes" it:
   - **distance formulas** (angle, straight-line): when `d(v, p) / d(j, p) > a`;
   - **best score:** when `d(j, p) < a × d(v, p)`.
   - The prototype tracks each candidate's worst ratio so far, so it is never re-checked against the same kept link.
4. Stop at R kept links.

All building and pruning runs in the **build space**:
- for best-score data, the flipped copies (x / |x|²) by straight-line distance;
- otherwise, the real numbers and formula.

## 7. Building the map
1. **Start item:** the item closest to the average of all items (in build space).
2. **Order:** a shuffle with seed 7, with the start item first.
3. **Batches:** items join in batches of 1, 2, 4, … doubling, up to 2% of the items. Only items already in the map count as alive.
4. **For each new item in a batch, in parallel, against the map as it was before the batch:**
   - walk from the start item with list size L_build, in build space, recording every opened item;
   - the opened items (minus itself) are the candidates;
   - prune to R links.
5. **Then:**
   - write each new item's links;
   - give each linked item a back-link. If a list is over 41 slots, prune it back to R, using its current links plus the new ones as candidates.
6. **After all items:**
   - the "questions unlike items?" test (section 8), then question links if on;
   - learn the effort table (section 5);
   - make the 1-byte copies.

   The prototype learns the effort table after the copies are made, so it is learned on the walk that searches will use. We do the same: copies first, then the table.

## 8. Question links (Fix 3 automatic, Fix 8 twice)
**The test:**
- Pick 256 training questions and 256 items at random (seed 5), and scale everything to length 1.
- For each picked question, the angle distance to its nearest item; for each picked item, to its nearest other item.
- **On** if the median for questions is more than 1.5× the median for items.

**The links** (done twice when on):
1. Search every training question for its top 32, with list size 200 and the current search (full numbers during the build; 1-byte copies once they exist).
2. For each question: its closest answer (the "hub") gets the other 31 as candidates, and each of them gets the hub.
3. For each item with candidates:
   - pick at most 16 spread-out ones with the prune rule;
   - merge them with its current links;
   - prune back to R, with those up-to-16 kept first.
4. Remember the training questions, so the links can be redone later.

## 9. Changes (Fix 7)
**Add (id, numbers):**
1. **Take a slot:**
   - a free one if there is one;
   - otherwise, if 1,000 freed slots are waiting (or the index is full), first remove every link that points at a dead slot (one pass over all lists), then reuse them;
   - otherwise a new slot.
2. **Link it:** the same as one new item in a build batch (section 7, steps 4–5), against the current map.
3. Make its 1-byte copy, then count the change.

**Delete (id):**
1. If it is the start item, the start moves to its first alive neighbour.
2. Walk to the item's own numbers (build space, list size L_build), recording the opened items; then mark it dead.
3. Each opened item that links to it:
   - drops that link;
   - looks at up to 8 of the deleted item's alive links, closest to it first and not already linked, and takes each one only where none of its own links already reaches that way (a build's rule: alpha × d(link, candidate) > d(item, candidate)), while it has room (part 5, `repair="occlude"`).
   - Parts 1 to 4 took up to 3 of them whatever their direction, pruned to R only when over the slots (`repair="append"`, `n_replace=3`).
4. Its slot waits to be recycled; count the change.
- **Why part 5 changed it** (step 38): on Landmark at full size, 210,831 changes left accuracy 1.08 points below a fresh build's, against a limit of 1. Appended links crowd a list with near-duplicates, which a cheap search's short candidate list then spends its places on:
  - appending more of them (8) made it worse: -1.68;
  - pruning the whole list again, as a build does, cut lists from 16.6 to 10.2 links: -4.96;
  - taking only the ones that reach a new way, from 8 candidates: -0.63.
- **What it was not:** the start item never moved; leaving dead links in place changed little (-1.01); spot checks and learning from use changed nothing (-1.08).

**Update (id, numbers):** delete without freeing the slot, then link the new numbers into the same slot. One change.

**Every 3,000 changes:** redo the question links with the remembered training questions (when they are on). The learned effort table is kept; the change test showed it stays good.

## 10. Memory (what is counted)
- **Kept in memory:**
  - links: 41 slots × 4 bytes, plus a 4-byte degree;
  - 1-byte copies: dim bytes;
  - id: 8 bytes.
- **Counted separately:**
  - the full numbers (on SSD when saved and loaded with `numbers="ssd"`);
  - the build copies (only needed for changes);
  - the training questions (only when question links are on).

This matches the prototype's own count, so memory can be compared one to one.

## 11. Where we differ from the prototype on purpose
- **Random numbers:** our own small generator (SplitMix64), not numpy's. Build order and samples differ, so maps differ slightly. That is why quality is matched within a tolerance, not exactly.
- **Float sums:** 32-bit, in SIMD-friendly order. The prototype adds in 64-bit.
- **No allocation per question:** visited marks use a per-thread stamp array; the prototype allocates per question.
- **Speed tuning (step 12), none of which changes a result:** the walk asks the chip to start loading all of an opened item's new links before scoring any of them, scores them four side by side, then puts them in the list in arrival order; list entries keep distance, id and "opened" mark together; the distances use vector instructions with the same order of operations. `validate/step12.py` checks answers, distances and checks against a recording of the untuned build.
- **Angle data:** scaled to length 1 inside the index; the prototype was given pre-scaled data.
- **Learned-effort truth:** the training questions' true answers are found inside the index by checking every item (the prototype's change test did the same).

## 12. Order of building and checking
`STEPS.md` builds this in the same order the experiments added the fixes, so every step can be checked against a recorded stage of the prototype:
1. plain map (start);
2. flipped build copies (Fix 4);
3. learned effort (Fix 1);
4. question links (Fix 3);
5. 1-byte copies (Fix 6);
6. question links twice (Fix 8);
7. changes (Fix 7).

---

# Part 2: fitting itself to each dataset

`SPEC.md` part 2 says what and why; these sections say how. Each is written when its step is built.

## 13. Usage stats (step 15)
- **What is counted**, for every search call since the build or the last `reset_usage()`:
  - calls and questions;
  - checks, and how many were re-checks with the full numbers;
  - time per question (a histogram);
  - fixed-list questions, and learned-effort questions per hardness bin;
  - each question's angle distance to its best answer (a histogram).
- **Searches made inside the build** (learning the effort, question links) are not counted.
- **Hardness bins:** the learned effort sorts each question into one of 8 bins by how far the 10th best item is after a short first look (DESIGN.md section 5). The bins' edges are the sample questions' eighths, so sample-like questions fill them equally. Real questions piling into the hardest bins means they are harder than the sample.
- **"Questions unlike items" in use:** the median angle distance from a question to its best answer, over the items' own median distance to their nearest other item. That is the build's test (section 8), measured on real questions. The items' median comes from the build's test when it ran; otherwise it is worked out the first time it is needed, on 256 sampled items.
- **How it stays cheap:** each search call adds up its own tally (one per thread in a spread-out batch) and merges it into the index's counters once, under a lock. Per question that costs a clock read and a few additions. Histograms have buckets 2^(1/4) apart, so a middle or 99th-percentile value read off them is within about 9%.
- **Switch:** setting `track_usage` (default on).
- **Saved:** `usage.json` in the index folder (readable as it is), and `item_gap` and part 2's settings as text pairs at the end of `meta.bin` (format version 2; version 1 folders still load).

## 14. Spot checks (step 16)
- **What:** now and then, a searched question is kept with the answers it got. `audit()` later finds its true answers by checking every item with the full numbers, and counts how many of its answers were right (ties count, as everywhere else).
- **How often:** each search call earns credit: `audit_budget` (default 0.02) times the checks it made. A spot check costs one check per item; a question is kept whenever the credit covers that.
  - So spot checks cost about 2% of the checks searches make, at any index size.
  - On 100,000 items that is about 1 in 10,000 searches on GooAQ and 1 in 1,000 on GloVe.
- **Which question:** a position in the call that moves on by a fixed step each time, so the choice never depends on the question's content or difficulty.
- **Waiting list:** at most 16 MB of question numbers. Past that, questions are not kept and the credit stops growing.
- **`audit()`:** reads the items in memory order, four at a time with the search's own arithmetic, 8 questions per pass, in parallel over groups of questions. Each question keeps only its closest k (at least 10) distances. Searches run meanwhile: the lock is held only to take the waiting list and to add the results.
- **Estimates:** one per kind of search (a target, or a list size): the mean of the per-question accuracies since the last reset, ± a 95% margin.
  - The margin's spread comes from every question of that kind spot-checked so far (the last 2,000). A few very wrong questions can carry much of the error, and a small group often misses them.
  - It is never taken as smaller than if each answer were right or wrong on its own, so a run of perfect questions still gets a margin. Small groups use Student's t.
- **Kept:** the last 2,000 checked questions with their true k-th and 10th distances (step 17 learns the effort table again from them). Saved in `usage.json` and `checked_q.f32`.

## 15. Learning from use (step 17)
- **At build (`effort_apart`, on):** with question links on, the links use only the sample questions after the effort table's own (the first `effort_questions`, at most half of them).
  - **Why:** question links join each of their questions' answers, so those questions become easy on the map. A table learned on them under-estimates the effort new questions need. Yandex at a 95% target gave 92.1% that way, and gives 95.1% now.
  - With question links off, nothing changes.
- **In use:** after every `relearn_every` (500) new spot-checked questions, the index checks each target's estimate. If one falls below its target by more than its margin, the table is learned again from the spot-checked questions (the last 2,000, at least 500).
  - It is learned the same way as at build: cut points from the questions' first-stage hardness, then correct answers and checks per group at each list size.
  - Their true 10th distances are already known from the spot checks, so this costs only searches.
  - The new table replaces the old one at once; a search call in progress keeps the table it started with.
  - Only below target: accuracy above target can be the table's floor (every group already at the smallest list), which learning again cannot lower.
  - The targets' counts start again, since they measured the old table; the checked questions are kept.
  - `relearn_effort()` does it on request.
- **Why real questions only:** they are what the table must serve. Mixing in the build's sample would hold the table near its old aim when that sample was wrong.

## 16. Picking its own settings (step 18)
- **What it picks:** links per item (24, 32, 48, 64 or 96) and slack (1.0, 1.1, 1.2 or 1.3).
  - All 20 pairs, built at full size on the five datasets, gave up to 1.36× over part 1's single setting, in opposite directions:
    - GloVe wants more links (96: 1.36×);
    - Landmark wants fewer links and less slack (24 and 1.0: 1.33×);
    - Llama wants more slack (1.24×);
    - GooAQ's best is part 1's setting.
- **How:** a trial on a random fifth of the items (at least 20,000, at most 100,000). It is built with the build's sample questions except the last ones (at most 1,000, and at most a quarter), which are kept back to time it.
  - **Each choice:** build the trial, time the kept-back questions on one core at the two targets around 95%, and read off the time at exactly 95% from the trial's own accuracy. A round's choices are timed together: 5 rounds, taking turns, each run at least 0.2 s, each choice keeping its fastest (part 5).
  - **Order:** links first at slack 1.2, then slack at the fastest links: 8 trials.
  - **The choice:** each round keeps the plainest of its choices within 2% of the fastest: the one in use first, then fewer links; for slack, the largest, since it keeps more long links, which a trial cannot see paying off (LAION's sample: 1.2 and 1.3 tied in the trial, and 1.3 was 23% faster built in full). A clearer lead wins by any margin above that. At the end, the pick replaces the starting setting only when it is more than 5% faster, timed again head-to-head.
    - Part 4's rule kept each round's fastest by any margin, so timing noise decided near-ties: three fresh builds of a sample picked differently on 7 of 12 datasets, and on LAION and ImageNet the picks were 9-13% apart once built (step 37). The same picks always give the same index.
    - A first version required each step to beat the current leader by 5%. That blocked real gains the trial understates: on GloVe, 96 links led 48 by 4.8% in the trial and by 19% at full size.
    - **Close calls are timed longer:** a choice within 1.5% of a bar (the 2% tie or the 5% clear win) is timed for 20 more rounds, with the fastest, before it is decided. The same goes for copy size, the extra links and the re-check count.
      - Ties alone still flipped on 4 of the 12 samples (step 37). Timed for 5 rounds of 0.2 s, one choice moved by up to 3% between fresh builds. Three identical indexes, timed for 12 rounds of 0.5 s, came within 0.3% of each other.
    - **When new links lose the final test,** slack is tried again at the starting links, and that pick faces the same test. On ImageNet's sample, 48 links passed the tie bar by 0.04%, then lost the final test. Slack 1.3 at 32 links, 6% ahead, was never tried, so that build was 8.5% slower than the other two.
  - **Copy size is not chosen in the trial** (section 19): a trial small enough to sit in the chip's caches cannot see what smaller copies save in memory reads.
- **Why time, not checks:** on GloVe, more links barely cut the checks but make each one cheaper (more reads in flight per step). Counting checks would miss its 1.36×.
- **Not picked:** build effort (`l_build`). The trial over-rates a lower one: on Yandex, effort 100 looked 1.01× in the trial and was 0.88× at full size.
- **Known limit:** Yandex's full-size best (48 links, slack 1.3: 1.10×) does not show in its trial, which keeps part 1's setting. Its full-size numbers jump between neighbours (32/1.3: 0.88×, 48/1.3: 1.10×), as much as a build's random order alone can move them. Scaling the trial's question-link questions down with its items changed nothing.
- **Memory limit:** `max_bytes_per_item` skips choices that keep more.
- **When:** `tune="auto"` (the default) with at least 10,000 items and 800 sample questions; otherwise the settings stay as given.
- **Saved:** `tuning.json`: each choice tried, with its trial speed, checks and memory, and the choice.

## 17. Suggestions (step 19)
- `suggest()` returns notes and, when it is worth it, one rebuild with the evidence for it.
- **Notes:**
  - when slots holding deleted items reach 20% of the used slots: rebuild to free memory (adds reuse freed slots only partly);
  - when real questions look unlike the items (section 13's measure above the 1.5 switch) while question links are off: a rebuild with the spot-checked questions as sample questions would turn them on;
  - when accuracy is below a target before learning again can start (fewer than 500 spot-checked questions).
- **The settings trial:** step 18's search, with the spot-checked questions as sample questions and, the last quarter of them, timing questions.
  - **Items:** the whole index, up to 100,000 items. A fifth of it understated what better links gain at full size: GloVe at 16 links was 1.8× in a fifth-size trial and 2.9× for real. Above 100,000 items the trial is smaller than the index and tends to understate gains.
  - **The one to beat:** the index itself, timed on the timing questions without counting as use. A trial rebuilt from the spot-checked questions would have weaker question links than the real index, and would overstate gains.
  - **The choices** start from the index's settings, with question links decided by the spot-checked questions.
  - **A rebuild is suggested** only when its trial is at least 15% faster. The suggestion names the settings, the expected gain, the memory per item, and whether to rebuild with the spot-checked questions as sample questions (when they change whether question links turn on).
- **Accuracy of the prediction:** within 5% for a change of links (GloVe). For a change of question links it is about ±20% (Llama: said 4.4×, gave 3.6×), because 2,000 real questions must both build and time the trial.
- **Cost:** about nine builds of the index (up to 100,000 items), so minutes. Searches continue meanwhile.

## 18. Cheaper checks, same results (step 20)
- **Fetch-ahead hints:** a region up to 256 bytes gets a hint every 64 bytes. A longer one gets one hint per cache line (128 bytes on Apple's M chips), from the line holding its first byte to the line holding its last.
  - Measured head-to-head on the M3 Max: one hint per 128-byte line made 768-byte rows (GooAQ, Landmark) 14% faster, with half the hints and so less queueing, but made 200-byte rows (GloVe) 5% slower.
  - The mixed rule kept both: within 2% of the better of the two on all five.
- **Renumbering (`renumber`, on):** after the build, items take the order in which a breadth-first walk of the links from the start item meets them.
  - Answers and checks do not change: the walk depends on the links and the distances, never on slot numbers. A unit test and step 12's recording confirm it.
  - It gained 0–2% at 100,000 items, where most of the data sits in the chip's caches anyway; it should matter more beyond them.
  - After changes, a renumbered index is as good as one that is not, but not bit-identical to it: freed slots are reused in another order. Part 1's change test keeps `renumber=False` to compare exactly with the prototype.
- **Guided build (`tools/build_py_pgo.sh`):** a practice run on made-up data (`tools/pgo_workload.py`) records which paths the code takes, and the compiler then lays the code out for them.
  - 1–7% faster (Llama most), results identical.
  - Optional: it needs `rustup component add llvm-tools-preview` and a few more minutes to build.

## 19. Whole-number questions and 4-bit copies (step 21)
- **16-bit questions (`question_bits`, default 16):** the prepared question `a = q × step` is scaled so its largest value is the largest a 32-bit sum allows (at most 32,767), and rounded.
  - A check then multiplies 16-bit numbers by the copy's bytes and adds them into 32-bit sums: 6 instructions per 16 numbers, against 14 with floats. The sums are exact, so the order of adding cannot change them.
  - The rounding is far below the copies' own: on all five, the same checks per search and accuracy within 0.25 points at every target. 1.26× to 1.53× faster on one core.
- **8-bit questions (`question_bits=8`):** the chip's signed dot instruction does 16 multiply-adds at once (each byte shifted by 128, the question's sum added back). No faster than 16-bit, since each check now waits on memory rather than arithmetic, and slightly less accurate. Kept as an option, not the default.
- **4-bit copies (`code_bits=4`, needing 16-bit questions):** 16 levels per number, two numbers per byte. In each block of 32 numbers, byte j holds number j in its low half and number j + 16 in its high half, so two vector steps unpack a block in order.
  - **Memory:** 23–41% less per item (GooAQ 944 → 560 bytes).
  - **Speed:** GooAQ 1.11× faster; Landmark and GloVe about the same; Yandex 0.93× (14% more checks); Llama never reaches 95%, its numbers spanning too wide a range for 16 levels.
- **Choosing the copy size, after the build:** the copies do not change the link map, so the choice is made on the real index.
  - It is timed at the tuning target with its 1-byte copies, then with 4-bit copies, the effort table learned again for them from the same questions. The two take turns, 5 rounds (3 before part 5).
  - It keeps the 4-bit copies only when they are more than 5% faster. That costs one more effort table and a few seconds of timing, not a rebuild.
  - A 20,000-item trial could not tell: it sits in the chip's caches, so on GooAQ the 4-bit copies looked 2% slower there and are 11% faster at full size.

# Part 3: closing the last three gaps

## 20. The re-check count, picked (step 24)
- **What:** how many of the best entries the search re-checks with the full numbers (`rerank`, 50 in parts 1 and 2).
- **How:** after the copy size, on the real index, the tuning tries 15, 25 and 50. Each count gets its own effort table, learned from the same sample questions; all are timed at the tuning target, taking turns, 5 rounds. The plainest count within 2% of the fastest is kept: 50 first, then 25, then 15 (part 5; before, the fastest replaced 50 when more than 2% faster, so timing noise decided near-ties).
- **The map does not depend on it:** question links always re-check 50 to find their answers (`QLINK_RERANK`), so an index built with any count has the same links.
- **Measured first** (tuned builds of the five): re-checks are cheap, since their memory reads overlap. 15 instead of 50 gained 2–4% on GooAQ, Yandex and Llama; GloVe needs 50 (with 15, its 4-bit copies never reach 95%); Landmark is flat. 10 and 100 were never the fastest, so they are not tried.
- **Cost:** two more effort tables and a few seconds of timing per build.

## 21. Cheaper checks, same results (step 25)
- **Kept:** when no slot in use holds a deleted item (the usual case after a build), the walk skips the "deleted?" lookup for each link: one lookup per link instead of two (`walk_with`). 1.01–1.03× faster, the same answers.
- **Tried and dropped**, each timed head-to-head (the same saved index, the two codes taking turns in separate processes, 5 rounds):
  - **One bit per item for "seen"**, cleared each search (14 KB instead of 448 KB of 4-byte stamps): 0.98–1.01×. The stamps already sit in the chip's second-level cache, and their reads overlap.
  - **Copy rows padded to the chip's 128-byte memory lines** (a 200-byte row takes 256, so it never spans 3 lines): no gain beyond the kept change, for up to 28% more memory.
  - **Loading the likely next item's links' copies early**, while the current ones go into the list: 0.85–0.95×.
- **What it says:** a check waits on its copy's first memory read and on its arithmetic, not on the walk's bookkeeping.

## 22. More links for popular items (step 26)
- **Found first** (Llama, tuning questions; `validate/step26.py` records the timing, the rest is in `STEPS.md`):
  - 2,554 items (2.6%) hold 99.5% of the sample questions' answers, and 73 hold half.
  - The map gave them no more links than the rest (24 on average), and only half of their links led to other popular items. A search that reached one answer still had to walk to the others: with a list of 10 it found 69% of the true top 10.
  - Starting from a popular item saved 9% of the checks; fewer links for the other items saved none.
  - Extra links between popular items that share answers, tried on the exported map: 912 checks at 95% fell to 656 with up to 16 each, 589 with 32, 528 with 64. With 128, a list of 10 already gives 97.5%, so each step costs more than it needs to.
- **What it does:** in the last round of question links, every item in some sample question's first 10 answers is popular. It gets up to 64 extra links, to the popular items it most often shares a question's first 10 with, leaving out its own links.
- **When:** only when the answers bunch up: the distinct items among them are at most a quarter of the most there could be (the alive items, or 10 per question if fewer). Llama 3.6%; GooAQ, GloVe, Yandex and Landmark 67–77%.
- **Kept only when faster:** after the build, the tuning times the real index with and without them (each with its own effort table), taking turns, 5 rounds, and keeps them only when more than 5% faster. The setting (`popular_links`, `auto`) then becomes `on` or `off`.
- **Stored** beside the link slots: an offset and a count per slot, then the extra links themselves (Llama: about 10 bytes per item).
- **The walk** reads an item's extra links after its links. Renumbering follows them too, so popular items end up near each other in memory.
- **Changes:** a reused slot starts with no extra links; recycling freed slots drops extra links that lead to deleted items; each refresh of the question links (every 3,000 changes) makes them again.

## 23. Buckets: a second way to search (part 5, steps 39-40; an experiment, not built in)
- **Why:** on GloVe, a link walk needs about 12,600 checks per question at full size, each a jump to a far-off place in memory. ScaNN, 1.35× as fast there, groups the items into buckets and reads straight through the closest buckets' short codes: more items scored, each far cheaper.
- **Not built in (step 41 not done):** the plan was one index with two ways to search, the setting search picking the faster for each dataset. On full GloVe the bucket search reached only 0.88× ScaNN (step 40), and the agreed rule was to build it in only if it beat ScaNN. So the buckets are reached only through `build_buckets` and `search_buckets`: `search` never uses them, and they are not saved, kept up to date through changes, or tuned. They stay so steps 39 and 40 can be rerun.
- **The buckets** (`buckets.rs`): k-means on the items (angle data: centres of length 1, items assigned by the largest dot product; otherwise by straight line), about 300 items per bucket (ScaNN's settings search found about 330 best on GloVe). Ten rounds on a sample of at most 64 items per centre, then every item is assigned once.
- **The short codes:** each pair of numbers is coded in 4 bits, one of 16 points learned from the items (k-means within each pair). For angle and best-score data the code holds the item's offset from its bucket's centre, so an estimate is the centre's own score plus the offset's.
- **The scan:** for each question, a table of one byte per point and pair: that point's share of the distance, scaled so a block's sums fit 16 bits. Codes sit 32 items to a block (byte t of a pair's 16 bytes: item t in the low half, item t + 16 in the high half), so one vector table lookup scores 16 items, and a block costs two lookups and four adds per pair.
- **Re-check:** the `pool` best by their codes are re-checked with the full numbers, as the link walk's best are.
- **Effort:** how many buckets to scan (`n_probe`) and how many to re-check (`pool`), given by hand. Step 41 would have learned a table from the sample questions (target accuracy to buckets); it was not done.
