# How vindex works

The README says it in plain words; this page uses the standard terms. `DESIGN.md` explains each part and every default in depth. Paths in this page are from the repository's top folder.

## The parts
- **Graph:** a Vamana graph. Each item links to at most `r` others, chosen with alpha-pruning (`alpha`, called "slack" in the docs). A search is a beam search from one start item, with a candidate list of size L.
- **Compressed search, exact re-rank:** the beam search compares 8-bit (or 4-bit) per-dimension scalar-quantized codes, with the query as 16-bit integers. The best `rerank` (default 50) are then re-scored with the full float32 vectors.
- **Per-query budget:** each query starts with L = 10. The distance of its 10th candidate puts it in one of 8 groups, and each group has its own L, so hard queries get a longer search. A table of L per group is learned on sample queries for each recall target, so one index serves any target.
- **Out-of-distribution links:** when queries look unlike the items, each sample query's answers are linked to each other, as in RoarGraph.
- **Inner product:** for maximum inner product data, the graph is built on the Möbius-transformed vectors x / |x|², and searched on the real ones.
- **Deletes:** a deleted item's in-neighbours are repaired at once. Each takes up to 8 of the deleted item's links, keeping one only where alpha-pruning says it reaches a new direction. After 210,831 changes to Landmark, recall stayed within 0.63 points of a fresh build.
- **Memory layout:** after the build, items are renumbered in breadth-first order from the start item, so linked items sit near each other in memory.

`DESIGN.md` explains each part and every default. The ideas were first tried in Python in [vector-index-experiments](https://github.com/srir84/vector-index-experiments) (`bench/ours/`).

## How it compares to known methods
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

## Words used in the code and docs
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
