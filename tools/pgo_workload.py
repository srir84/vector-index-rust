"""A practice run for the profile-guided build (tools/build_py_pgo.sh): builds, searches and changes on made-up
clustered data in all three closeness formulas, so the compiler sees the paths real use takes. It needs no dataset."""
import numpy as np

import vindex


def clustered(rng, n, dim, centres):
    c = rng.standard_normal((centres, dim)).astype(np.float32)
    return (c[rng.integers(0, centres, n)] + 0.35 * rng.standard_normal((n, dim))).astype(np.float32)


rng = np.random.default_rng(1)
for metric, n, dim in (("cos", 40_000, 192), ("l2", 30_000, 384), ("ip", 30_000, 96)):
    X = clustered(rng, n, dim, 200)
    train = clustered(rng, 4_000, dim, 200)
    Q = clustered(rng, 2_000, dim, 200)
    idx = vindex.Index(dim, metric, tune="off", audit_budget=0.05)
    idx.build(X, np.arange(n), train_questions=train)
    for target in (0.9, 0.95, 0.99):
        idx.search(Q, k=10, target=target, threads=1)
    idx.search(Q, k=10, list_size=100, threads=1)
    idx.audit()
    for i in range(500):
        idx.delete(i)
        idx.add(n + i, X[i])
    idx.search(Q, k=10, target=0.95, threads=1)
# best score with questions unlike the items, where a few long items are most of the answers: question links and
# popular items' extra links (part 3), both forced on, so the practice run takes their paths
X = rng.standard_normal((30_000, 64)).astype(np.float32)
X[:300] *= 5
train = (rng.standard_normal((4_000, 64)) + 0.5).astype(np.float32)
Q = (rng.standard_normal((2_000, 64)) + 0.5).astype(np.float32)
idx = vindex.Index(64, "ip", tune="off", audit_budget=0.05, question_links="on", popular_links="on")
idx.build(X, np.arange(30_000), train_questions=train)
for target in (0.9, 0.95, 0.99):
    idx.search(Q, k=10, target=target, threads=1)
print("practice run done; popular items in the last one:", idx.stats()["popular_items"])
