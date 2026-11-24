//! Python package for vindex: numpy in, numpy out. A thin wrapper; all logic lives in the `vindex` crate.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ::vindex::{Codes, Graph, Metric, Prepared, Store};

fn metric_of(name: &str) -> PyResult<Metric> {
    Metric::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown metric {name:?}: use cos, ip or l2")))
}

/// Search a supplied map (`nbrs`: n x slots, `deg`: n) with full numbers and a fixed list size, from `entry`.
/// Returns (ids, dists, checks per question) — a checking helper for step 2.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn debug_search_graph<'py>(
    py: Python<'py>,
    metric: &str,
    x: PyReadonlyArray2<'py, f32>,
    nbrs: PyReadonlyArray2<'py, u32>,
    deg: PyReadonlyArray1<'py, u32>,
    entry: u32,
    q: PyReadonlyArray2<'py, f32>,
    list_size: usize,
    k: usize,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u64>>)> {
    let m = metric_of(metric)?;
    let (x, nbrs, deg, q) = (x.as_array(), nbrs.as_array(), deg.as_array(), q.as_array());
    let (n, d, slots) = (x.nrows(), x.ncols(), nbrs.ncols());
    if nbrs.nrows() != n || deg.len() != n || q.ncols() != d {
        return Err(PyValueError::new_err("x, nbrs, deg and q do not fit together"));
    }
    let mut store = Store::ram(d, n);
    for (i, r) in x.rows().into_iter().enumerate() {
        store.set(i, &r.to_vec());
    }
    let g = Graph::from_parts(nbrs.iter().copied().collect(), deg.to_vec(), slots);
    let qv: Vec<f32> = q.iter().copied().collect();
    let nq = q.nrows();
    let (ids, dists, checks) =
        py.detach(|| ::vindex::debug::search_full_fixed(&g, &store, m, entry, &qv, list_size, k));
    Ok((
        Array2::from_shape_vec((nq, k), ids).expect("shape").into_pyarray(py),
        Array2::from_shape_vec((nq, k), dists).expect("shape").into_pyarray(py),
        checks.into_pyarray(py),
    ))
}

/// Distances from every row of `q` to every row of `x` (debug and checking helper).
#[pyfunction]
fn debug_distances<'py>(
    py: Python<'py>,
    metric: &str,
    q: PyReadonlyArray2<'py, f32>,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let m = metric_of(metric)?;
    let (q, x) = (q.as_array(), x.as_array());
    if q.ncols() != x.ncols() {
        return Err(PyValueError::new_err("q and x need the same number of columns"));
    }
    let mut out = Array2::<f32>::zeros((q.nrows(), x.nrows()));
    for (i, qr) in q.rows().into_iter().enumerate() {
        let qs = qr.to_vec();
        for (j, xr) in x.rows().into_iter().enumerate() {
            out[[i, j]] = m.distance(&qs, &xr.to_vec());
        }
    }
    Ok(out.into_pyarray(py))
}

/// Train 1-byte copies on `x`, encode `x`, and return (low, step, codes, distances from each row of `q`
/// to each copy) — a checking helper for step 1.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn debug_codes<'py>(
    py: Python<'py>,
    metric: &str,
    x: PyReadonlyArray2<'py, f32>,
    q: PyReadonlyArray2<'py, f32>,
) -> PyResult<(
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let m = metric_of(metric)?;
    let (x, q) = (x.as_array(), q.as_array());
    let d = x.ncols();
    let rows: Vec<Vec<f32>> = x.rows().into_iter().map(|r| r.to_vec()).collect();
    let mut c = Codes::train(rows.iter().map(|r| r.as_slice()), d, m, rows.len());
    for (i, r) in rows.iter().enumerate() {
        c.set(i, r);
    }
    let mut dist = Array2::<f32>::zeros((q.nrows(), rows.len()));
    let mut p = Prepared::default();
    for (i, qr) in q.rows().into_iter().enumerate() {
        c.prepare(&qr.to_vec(), &mut p);
        for j in 0..rows.len() {
            dist[[i, j]] = c.distance(&p, j);
        }
    }
    let codes = Array2::from_shape_vec((rows.len(), d), c.data.clone()).expect("shape");
    Ok((
        c.low.clone().into_pyarray(py),
        c.step.clone().into_pyarray(py),
        codes.into_pyarray(py),
        dist.into_pyarray(py),
    ))
}

#[pymodule]
#[pyo3(name = "vindex")]
fn vindex_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(debug_distances, m)?)?;
    m.add_function(wrap_pyfunction!(debug_codes, m)?)?;
    m.add_function(wrap_pyfunction!(debug_search_graph, m)?)?;
    m.add_class::<PyIndex>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

// ---------------------------------------------------------------- the index

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use numpy::PyReadonlyArray1 as Arr1;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyList};
use rayon::ThreadPool;

fn to_py(e: ::vindex::Error) -> PyErr {
    PyValueError::new_err(e.0)
}

fn rows_of(a: &PyReadonlyArray2<'_, f32>) -> Vec<f32> {
    a.as_array().iter().copied().collect()
}

fn id_of(id: i64) -> PyResult<u64> {
    u64::try_from(id).map_err(|_| PyValueError::new_err("ids must be non-negative"))
}

/// The vector index. `Index(dim, metric="cos", **settings)`; settings as in DESIGN.md (e.g. r=32, code_bits=8).
///
/// Safe to share between Python threads: searches run side by side; a change (build, add, delete, update,
/// learn_effort) waits until running searches finish, and searches wait for a running change. The lock is only
/// ever taken while Python's own lock is released, so the two cannot deadlock.
#[pyclass(name = "Index", frozen)]
struct PyIndex {
    dim: usize,
    inner: RwLock<::vindex::Index>,
    pools: Mutex<HashMap<usize, Arc<ThreadPool>>>,
}

/// The spot checks' report as a dict: one accuracy estimate per kind of search, plus what they cost.
fn spot_dict<'py>(py: Python<'py>, r: &::vindex::SpotReport) -> PyResult<Bound<'py, PyDict>> {
    let acc = PyDict::new(py);
    for e in &r.estimates {
        let g = PyDict::new(py);
        g.set_item("questions", e.questions)?;
        g.set_item("accuracy", e.accuracy)?;
        g.set_item("margin", e.margin)?;
        acc.set_item(&e.label, g)?;
    }
    let d = PyDict::new(py);
    d.set_item("accuracy", acc)?;
    d.set_item("pending", r.pending)?;
    d.set_item("dropped", r.dropped)?;
    d.set_item("checks", r.checks)?;
    d.set_item("search_checks", r.search_checks)?;
    d.set_item("share_of_search_checks", (r.search_checks > 0).then(|| r.checks as f64 / r.search_checks as f64))?;
    d.set_item("relearned", r.relearned)?;
    Ok(d)
}

/// A tuning record as a dict.
fn tuning_dict<'py>(py: Python<'py>, t: &::vindex::Tuning) -> PyResult<Bound<'py, PyDict>> {
    let trials = PyList::empty(py);
    for tr in &t.trials {
        let d = PyDict::new(py);
        d.set_item("r", tr.r)?;
        d.set_item("alpha", tr.alpha)?;
        d.set_item("code_bits", tr.code_bits)?;
        d.set_item("rerank", tr.rerank)?;
        d.set_item("links", &tr.links)?;
        d.set_item("question_links", tr.question_links)?;
        d.set_item("searches_per_s", tr.searches_per_s)?;
        d.set_item("checks", tr.checks)?;
        d.set_item("bytes_per_item", tr.bytes_per_item)?;
        d.set_item("seconds", tr.seconds)?;
        trials.append(d)?;
    }
    let d = PyDict::new(py);
    d.set_item("trial_items", t.trial_items)?;
    d.set_item("timing_questions", t.timing_questions)?;
    d.set_item("target", t.target)?;
    d.set_item("trials", trials)?;
    d.set_item("chosen_r", t.chosen_r)?;
    d.set_item("chosen_alpha", t.chosen_alpha)?;
    d.set_item("chosen_code_bits", t.chosen_code_bits)?;
    d.set_item("chosen_rerank", t.chosen_rerank)?;
    d.set_item("chosen", t.chosen)?;
    d.set_item("seconds", t.seconds)?;
    Ok(d)
}

fn broken() -> PyErr {
    PyRuntimeError::new_err("an earlier internal error left this index in an unknown state; load or build it again")
}

impl PyIndex {
    fn wrap(inner: ::vindex::Index) -> Self {
        PyIndex { dim: inner.dim, inner: RwLock::new(inner), pools: Mutex::new(HashMap::new()) }
    }

    fn read(&self) -> PyResult<RwLockReadGuard<'_, ::vindex::Index>> {
        self.inner.read().map_err(|_| broken())
    }

    fn write(&self) -> PyResult<RwLockWriteGuard<'_, ::vindex::Index>> {
        self.inner.write().map_err(|_| broken())
    }

    fn pool(&self, threads: usize) -> PyResult<Arc<ThreadPool>> {
        let mut pools = self.pools.lock().map_err(|_| PyValueError::new_err("thread pool lock poisoned"))?;
        if let Some(p) = pools.get(&threads) {
            return Ok(p.clone());
        }
        let p = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
        pools.insert(threads, p.clone());
        Ok(p)
    }

    fn check_cols(&self, cols: usize, what: &str) -> PyResult<()> {
        if cols != self.dim {
            return Err(PyValueError::new_err(format!("{what} has {cols} columns; this index has {}", self.dim)));
        }
        Ok(())
    }
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (dim, metric="cos", **settings))]
    fn new(dim: usize, metric: &str, settings: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut s = ::vindex::Settings::default();
        if let Some(kw) = settings {
            for (k, v) in kw.iter() {
                let key: String = k.extract()?;
                let val = v.str()?.to_string();
                s.set(&key, &val).map_err(to_py)?;
            }
        }
        Ok(PyIndex::wrap(::vindex::Index::new(dim, metric_of(metric)?, s).map_err(to_py)?))
    }

    /// Build from items `x` (n x dim, float32) with ids `ids` (n, non-negative). Uses all cores unless `threads` is set.
    #[pyo3(signature = (x, ids, train_questions=None, threads=None, order=None))]
    fn build(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ids: Arr1<'_, i64>,
        train_questions: Option<PyReadonlyArray2<'_, f32>>,
        threads: Option<usize>,
        order: Option<Arr1<'_, i64>>,
    ) -> PyResult<()> {
        self.check_cols(x.as_array().ncols(), "x")?;
        if let Some(t) = &train_questions {
            self.check_cols(t.as_array().ncols(), "train_questions")?;
        }
        let xv = rows_of(&x);
        let idv: Vec<u64> = ids.as_array().iter().map(|&i| id_of(i)).collect::<PyResult<_>>()?;
        let tq = train_questions.as_ref().map(rows_of);
        let ord: Option<Vec<u32>> = order.map(|o| o.as_array().iter().map(|&v| v as u32).collect());
        let pool = match threads {
            Some(t) => Some(self.pool(t)?),
            None => None,
        };
        py.detach(|| {
            let mut g = self.write()?;
            let inner: &mut ::vindex::Index = &mut g;
            match pool {
                Some(p) => p.install(|| inner.build_in_order(&xv, &idv, tq.as_deref(), ord.as_deref())),
                None => inner.build_in_order(&xv, &idv, tq.as_deref(), ord.as_deref()),
            }
            .map_err(to_py)
        })
    }

    /// Search: returns (ids, distances, checks per question). Give `list_size` (fixed effort) or `target` (learned effort).
    #[pyo3(signature = (q, k=10, list_size=None, target=None, threads=1))]
    #[allow(clippy::type_complexity)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f32>,
        k: usize,
        list_size: Option<usize>,
        target: Option<f64>,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u64>>)> {
        let effort = match (list_size, target) {
            (Some(l), None) => ::vindex::Effort::List(l),
            (None, Some(t)) => ::vindex::Effort::Target(t),
            (None, None) => ::vindex::Effort::Target(0.95),
            _ => return Err(PyValueError::new_err("give list_size or target, not both")),
        };
        self.check_cols(q.as_array().ncols(), "q")?;
        let nq = q.as_array().nrows();
        let qv = rows_of(&q);
        // one thread: on the calling thread itself, so searches from several Python threads run side by side
        let pool = if threads > 1 { Some(self.pool(threads)?) } else { None };
        let out = py.detach(|| {
            let g = self.read()?;
            let inner: &::vindex::Index = &g;
            match pool {
                Some(p) => p.install(|| inner.search(&qv, k, effort)),
                None => inner.search_here(&qv, k, effort),
            }
            .map_err(to_py)
        })?;
        Ok((
            Array2::from_shape_vec((nq, k), out.ids).expect("shape").into_pyarray(py),
            Array2::from_shape_vec((nq, k), out.dists).expect("shape").into_pyarray(py),
            out.checks.into_pyarray(py),
        ))
    }

    /// Group the items into buckets of about `per_bucket` items, for `search_buckets` (part 5). `eta`: the weight on
    /// coding errors along each item (1: plain; None: ScaNN's default for the item size).
    #[pyo3(signature = (per_bucket=300, eta=None, threads=None, rounds=10, per_centre=64))]
    fn build_buckets(
        &self,
        py: Python<'_>,
        per_bucket: usize,
        eta: Option<f32>,
        threads: Option<usize>,
        rounds: usize,
        per_centre: usize,
    ) -> PyResult<()> {
        let eta = eta.unwrap_or_else(|| ::vindex::buckets::default_eta(self.dim));
        let pool = match threads {
            Some(t) => Some(self.pool(t)?),
            None => None,
        };
        py.detach(|| {
            let mut g = self.write()?;
            let inner: &mut ::vindex::Index = &mut g;
            match pool {
                Some(p) => p.install(|| inner.build_buckets_with(per_bucket, eta, rounds, per_centre)),
                None => inner.build_buckets_with(per_bucket, eta, rounds, per_centre),
            }
            .map_err(to_py)
        })
    }

    /// Search with the buckets: the `n_probe` nearest buckets scanned by their codes, the `pool` best re-checked with
    /// the full numbers. Returns (ids, distances, checks per question), as `search`.
    #[pyo3(signature = (q, k=10, n_probe=8, pool=100, threads=1))]
    #[allow(clippy::type_complexity)]
    fn search_buckets<'py>(
        &self,
        py: Python<'py>,
        q: PyReadonlyArray2<'py, f32>,
        k: usize,
        n_probe: usize,
        pool: usize,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u64>>)> {
        self.check_cols(q.as_array().ncols(), "q")?;
        let nq = q.as_array().nrows();
        let qv = rows_of(&q);
        let pool_t = if threads > 1 { Some(self.pool(threads)?) } else { None };
        let out = py.detach(|| {
            let g = self.read()?;
            let inner: &::vindex::Index = &g;
            match pool_t {
                Some(p) => p.install(|| inner.search_buckets(&qv, k, n_probe, pool, true)),
                None => inner.search_buckets(&qv, k, n_probe, pool, false),
            }
            .map_err(to_py)
        })?;
        Ok((
            Array2::from_shape_vec((nq, k), out.ids).expect("shape").into_pyarray(py),
            Array2::from_shape_vec((nq, k), out.dists).expect("shape").into_pyarray(py),
            out.checks.into_pyarray(py),
        ))
    }

    /// Save to a folder (created if needed).
    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        py.detach(|| self.read()?.save(&path).map_err(to_py))
    }

    /// Load an index saved with `save`. numbers="ram" (search and change) or "ssd" (search only; the full numbers
    /// stay in the file and are read when needed).
    #[staticmethod]
    #[pyo3(signature = (path, numbers="ram"))]
    fn load(py: Python<'_>, path: std::path::PathBuf, numbers: &str) -> PyResult<Self> {
        let mode = match numbers {
            "ram" => ::vindex::Numbers::Ram,
            "ssd" => ::vindex::Numbers::Ssd,
            _ => return Err(PyValueError::new_err("numbers must be \"ram\" or \"ssd\"")),
        };
        let inner = py.detach(|| ::vindex::Index::load(&path, mode)).map_err(to_py)?;
        Ok(PyIndex::wrap(inner))
    }

    /// Add one item (its id must be new).
    fn add(&self, py: Python<'_>, id: i64, x: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let v: Vec<f32> = x.as_array().to_vec();
        let id = id_of(id)?;
        py.detach(|| self.write()?.add(id, &v).map_err(to_py))
    }

    /// Delete one item; its neighbours are repaired right away.
    fn delete(&self, py: Python<'_>, id: i64) -> PyResult<()> {
        let id = id_of(id)?;
        py.detach(|| self.write()?.delete(id).map_err(to_py))
    }

    /// Replace one item's numbers (same id).
    fn update(&self, py: Python<'_>, id: i64, x: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let v: Vec<f32> = x.as_array().to_vec();
        let id = id_of(id)?;
        py.detach(|| self.write()?.update(id, &v).map_err(to_py))
    }

    /// Learn the effort table again from training questions, on the items the index holds now.
    fn learn_effort(&self, py: Python<'_>, train_questions: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        self.check_cols(train_questions.as_array().ncols(), "train_questions")?;
        let tq = rows_of(&train_questions);
        py.detach(|| {
            self.write()?.learn_effort(&tq);
            Ok(())
        })
    }

    /// The learned effort's list size per group for a target accuracy (hardest group first), and the group cut points.
    fn effort_table(&self, py: Python<'_>, target: f64) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let t = py.detach(|| self.read().map(|g| g.learned().map(|e| (e.table(target), e.cuts.clone()))))?;
        t.ok_or_else(|| PyValueError::new_err("this index has no learned effort"))
    }

    /// Bytes kept in memory, by part (links, numbers searched on, other, total), plus the full numbers that can stay on SSD.
    fn memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = py.detach(|| self.read().map(|g| g.memory()))?;
        let d = PyDict::new(py);
        d.set_item("items", m.items)?;
        d.set_item("links_bytes", m.links)?;
        d.set_item("numbers_bytes", m.numbers)?;
        d.set_item("other_bytes", m.other)?;
        d.set_item("total_bytes", m.total)?;
        d.set_item("full_numbers_on_ssd_bytes", m.full_numbers_on_ssd)?;
        Ok(d)
    }

    /// Facts about the index (items, start, build checks, links per item, items with no links in, ...).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (s, changes, slots_used) = py.detach(|| self.read().map(|g| (g.stats(), g.changes, g.n_used)))?;
        let d = PyDict::new(py);
        d.set_item("items", s.items)?;
        d.set_item("start", s.start)?;
        d.set_item("build_checks", s.build_checks)?;
        d.set_item("change_checks", s.change_checks)?;
        d.set_item("mean_links", s.mean_links)?;
        d.set_item("no_links_in", s.no_links_in)?;
        d.set_item("unreachable", s.unreachable)?;
        d.set_item("build_space", s.build_space)?;
        d.set_item("ood_ratio", s.ood_ratio)?;
        d.set_item("question_links", s.question_links)?;
        d.set_item("question_link_items", s.question_link_items)?;
        d.set_item("popular_items", s.popular_items)?;
        d.set_item("extra_links", s.extra_links)?;
        d.set_item("changes", changes)?;
        d.set_item("slots_used", slots_used)?;
        Ok(d)
    }

    /// What searches have done since the build or the last reset: counts, time, effort, and how unlike the items
    /// the questions look (DESIGN.md section 13).
    fn usage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (r, bins, ood, spot) =
            py.detach(|| self.read().map(|g| (g.usage_report(), g.usage().bins, g.ood_ratio, g.spot_report())))?;
        let d = PyDict::new(py);
        d.set_item("calls", r.calls)?;
        d.set_item("questions", r.questions)?;
        d.set_item("checks", r.checks)?;
        d.set_item("rechecks", r.rechecks)?;
        d.set_item("checks_per_question", r.checks_per_question)?;
        d.set_item("rechecks_per_question", r.rechecks_per_question)?;
        d.set_item("seconds", r.seconds)?;
        d.set_item("typical_ms", r.typical_ms)?;
        d.set_item("slow_ms", r.slow_ms)?;
        d.set_item("learned_questions", r.learned_questions)?;
        d.set_item("fixed_questions", r.fixed_questions)?;
        d.set_item("hardness", r.hardness)?;
        d.set_item("hardness_counts", bins.to_vec())?;
        d.set_item("hard_share", r.hard_share)?;
        d.set_item("unlike_items", r.unlike_items)?;
        d.set_item("unlike_items_at_build", ood)?;
        d.set_item("spot_checks", spot_dict(py, &spot)?)?;
        Ok(d)
    }

    /// Find the true answers of the questions the spot checks kept, by checking every item, and return the
    /// accuracy estimates (DESIGN.md section 14). Searches from other threads continue meanwhile.
    #[pyo3(signature = (threads=None))]
    fn audit<'py>(&self, py: Python<'py>, threads: Option<usize>) -> PyResult<Bound<'py, PyDict>> {
        let pool = match threads {
            Some(t) => Some(self.pool(t)?),
            None => None,
        };
        let r = py.detach(|| {
            let g = self.read()?;
            let inner: &::vindex::Index = &g;
            Ok::<_, PyErr>(match pool {
                Some(p) => p.install(|| inner.audit()),
                None => inner.audit(),
            })
        })?;
        spot_dict(py, &r)
    }

    /// Learn the effort table again from the spot-checked questions (at least 500); returns how many were used,
    /// or None when there are too few. Searches continue meanwhile (DESIGN.md section 15).
    #[pyo3(signature = (threads=None))]
    fn relearn_effort(&self, py: Python<'_>, threads: Option<usize>) -> PyResult<Option<usize>> {
        let pool = match threads {
            Some(t) => Some(self.pool(t)?),
            None => None,
        };
        py.detach(|| {
            let g = self.read()?;
            let inner: &::vindex::Index = &g;
            Ok(match pool {
                Some(p) => p.install(|| inner.relearn_effort()),
                None => inner.relearn_effort(),
            })
        })
    }

    /// What the build's tuning tried and chose (DESIGN.md section 16), or None when it did not run.
    fn tuning<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let t = py.detach(|| self.read().map(|g| g.tuning.clone()))?;
        t.map(|t| tuning_dict(py, &t)).transpose()
    }

    /// Rebuilds worth doing, and notes, from how the index is used (DESIGN.md section 17): a list of dicts with
    /// "kind" ("rebuild" or "note"), "what", "why", "settings", "use_checked_questions", "now" and "after". The
    /// settings trial needs 800 spot-checked questions; it builds several small indexes, so it takes a while.
    #[pyo3(signature = (threads=None))]
    fn suggest<'py>(&self, py: Python<'py>, threads: Option<usize>) -> PyResult<Bound<'py, PyList>> {
        let pool = match threads {
            Some(t) => Some(self.pool(t)?),
            None => None,
        };
        let (list, trial) = py.detach(|| {
            let g = self.read()?;
            let inner: &::vindex::Index = &g;
            Ok::<_, PyErr>(match pool {
                Some(p) => p.install(|| inner.suggest()),
                None => inner.suggest(),
            })
        })?;
        let out = PyList::empty(py);
        for s in &list {
            let d = PyDict::new(py);
            d.set_item("kind", s.kind)?;
            d.set_item("what", &s.what)?;
            d.set_item("why", &s.why)?;
            let st = PyDict::new(py);
            for (k, v) in &s.settings {
                st.set_item(k, v)?;
            }
            d.set_item("settings", st)?;
            d.set_item("use_checked_questions", s.use_checked_questions)?;
            d.set_item("now", s.now)?;
            d.set_item("after", s.after)?;
            if s.kind == "rebuild" {
                if let Some(t) = &trial {
                    d.set_item("trial", tuning_dict(py, t)?)?;
                }
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// The spot-checked questions (the last 2,000), one row each, as they were searched: sample questions that
    /// look like real use, for a rebuild.
    fn checked_questions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (rows, flat) = py.detach(|| {
            self.read().map(|g| {
                let c = g.checked_questions();
                (c.len(), c.iter().flat_map(|x| x.q.iter().copied()).collect::<Vec<f32>>())
            })
        })?;
        Ok(Array2::from_shape_vec((rows, self.dim), flat).expect("shape").into_pyarray(py))
    }

    /// Start the usage counters again from zero.
    fn reset_usage(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.read().map(|g| g.reset_usage()))
    }
}
