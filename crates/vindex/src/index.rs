//! The public index: settings, build, search, statistics.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

use rayon::prelude::*;

use crate::build::{back_links, out_links_batch};
use crate::effort::{self, Learned, L1};
use crate::graph::Graph;
use crate::metric::{dot, normalize, Metric};
use crate::prune::BuildSpace;
use crate::qlinks;
use crate::rng::Rng;
use crate::codes::Codes;
use crate::prefetch::prefetch_bytes;
use crate::score::{CodeScore, FullScore, Score};
use crate::search::{finish_plain, finish_rerank, seed, walk, walk_with, Scratch, SortedList};
use crate::store::Store;
use crate::audit::{Checked, Estimate, Sample, SpotChecks, RELEARN_MIN};
use crate::tune::Tuning;
use crate::usage::{Usage, UsageReport};

/// An error with a plain message.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}

/// Refuse numbers that are not finite: a NaN distance has no place in a sorted list.
pub(crate) fn check_finite(x: &[f32], what: &str) -> Result<()> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        err(format!("{what} hold a number that is not finite (NaN or infinity)"))
    }
}

/// Where links are built: the real numbers, or flipped copies x / |x|^2 (best-score data).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildSpaceKind {
    /// Flipped for best score, the real numbers otherwise.
    Auto,
    Same,
    Flipped,
}

/// Whether the build picks links per item and slack itself (part 2, DESIGN.md section 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tune {
    Auto,
    Off,
}

/// A switch that can also decide for itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    Auto,
    On,
    Off,
}

/// How a delete repairs the links of the items that linked to the deleted one (part 5, step 38): `Append` its
/// replacements while there is room (parts 1 to 4, with 3 replacements), `Prune` the whole list again as a build
/// would, or `Occlude` (part 5's default, with 8): add a replacement only where the item's own links do not already
/// reach that way (a build's rule, kept links first). On Landmark at full size, 210,831 changes left a drift of
/// -1.08 points with `Append`, -4.96 with `Prune` (lists fell to 10 links), and -0.63 with `Occlude` and 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repair {
    Append,
    Prune,
    Occlude,
}

/// Every setting, with the final design's defaults (DESIGN.md section 1).
#[derive(Clone, Debug)]
pub struct Settings {
    pub r: usize,
    pub l_build: usize,
    pub alpha: f32,
    pub degree_slack: f32,
    pub build_space: BuildSpaceKind,
    pub seed: u64,
    pub max_batch_frac: f32,
    pub code_bits: u8,
    pub rerank: usize,
    pub question_links: Switch,
    pub ood_ratio: f32,
    pub q_topk: usize,
    pub q_links_max: usize,
    pub q_rounds: usize,
    pub learned_effort: bool,
    pub effort_questions: usize,
    pub n_replace: usize,
    pub qlinks_refresh_every: u64,
    pub cleanup_batch: usize,
    /// Spare slots reserved at build, as a share of the items (room for adds before the arrays grow).
    pub spare: f32,
    /// Keep usage stats on searches (part 2, DESIGN.md section 13).
    pub track_usage: bool,
    /// Spot checks may cost this share of the checks searches make (DESIGN.md section 14); 0 turns them off.
    pub audit_budget: f64,
    /// Learn the effort table on sample questions the question links do not use (DESIGN.md section 15).
    pub effort_apart: bool,
    /// After this many new spot-checked questions, learn the effort table again from them if the spot checks
    /// show a target missed (DESIGN.md section 15); 0 never does it on its own.
    pub relearn_every: u64,
    /// Pick links per item and slack at build, on a trial (DESIGN.md section 16).
    pub tune: Tune,
    /// The accuracy the tuning aims at.
    pub tune_target: f64,
    /// The tuning skips choices that keep more bytes per item in memory than this (0: no limit).
    pub max_bytes_per_item: usize,
    /// After the build, renumber items so linked items sit near each other in memory (DESIGN.md section 18).
    pub renumber: bool,
    /// How a question is held against the 1-byte copies: 32 (floats), or 16 or 8 (whole numbers; DESIGN.md
    /// section 19). Changes the walk's distances slightly; the re-check with the full numbers is unchanged.
    pub question_bits: u8,
    /// Extra links for popular items (part 3, DESIGN.md section 22): `Auto` makes them when the sample questions'
    /// answers bunch up on few items and, when the tuning runs, keeps them only if faster; `On` always keeps them
    /// (with question links); `Off` never makes them.
    pub popular_links: Switch,
    /// How a delete repairs the links around it (part 5, step 38).
    pub repair: Repair,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            r: 32,
            l_build: 200,
            alpha: 1.2,
            degree_slack: 1.3,
            build_space: BuildSpaceKind::Auto,
            seed: 7,
            max_batch_frac: 0.02,
            code_bits: 8,
            rerank: 50,
            question_links: Switch::Auto,
            ood_ratio: 1.5,
            q_topk: 32,
            q_links_max: 16,
            q_rounds: 2,
            learned_effort: true,
            effort_questions: 3000,
            n_replace: 8,
            qlinks_refresh_every: 3000,
            cleanup_batch: 1000,
            spare: 0.12,
            track_usage: true,
            audit_budget: 0.02,
            effort_apart: true,
            relearn_every: 500,
            tune: Tune::Auto,
            tune_target: 0.95,
            max_bytes_per_item: 0,
            renumber: true,
            question_bits: 16,
            popular_links: Switch::Auto,
            repair: Repair::Occlude,
        }
    }
}

impl Settings {
    /// Link slots per item: room to grow before a list is pruned back to `r` (DiskANN uses 1.3 x r).
    pub fn slots(&self) -> usize {
        self.r.max((self.degree_slack * self.r as f32) as usize)
    }

    /// Set one setting from text (used by the Python package).
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        fn num<T: std::str::FromStr>(k: &str, v: &str) -> Result<T> {
            v.parse::<T>().map_err(|_| Error(format!("setting {k}: cannot read {v:?}")))
        }
        match key {
            "r" => self.r = num(key, value)?,
            "l_build" => self.l_build = num(key, value)?,
            "alpha" => self.alpha = num(key, value)?,
            "degree_slack" => self.degree_slack = num(key, value)?,
            "seed" => self.seed = num(key, value)?,
            "max_batch_frac" => self.max_batch_frac = num(key, value)?,
            "code_bits" => self.code_bits = num(key, value)?,
            "rerank" => self.rerank = num(key, value)?,
            "ood_ratio" => self.ood_ratio = num(key, value)?,
            "q_topk" => self.q_topk = num(key, value)?,
            "q_links_max" => self.q_links_max = num(key, value)?,
            "q_rounds" => self.q_rounds = num(key, value)?,
            "effort_questions" => self.effort_questions = num(key, value)?,
            "n_replace" => self.n_replace = num(key, value)?,
            "qlinks_refresh_every" => self.qlinks_refresh_every = num(key, value)?,
            "cleanup_batch" => self.cleanup_batch = num(key, value)?,
            "spare" => self.spare = num(key, value)?,
            "audit_budget" => self.audit_budget = num(key, value)?,
            "relearn_every" => self.relearn_every = num(key, value)?,
            "tune_target" => self.tune_target = num(key, value)?,
            "max_bytes_per_item" => self.max_bytes_per_item = num(key, value)?,
            "tune" => {
                self.tune = match value {
                    "auto" | "on" | "true" | "True" => Tune::Auto,
                    "off" | "false" | "False" => Tune::Off,
                    _ => return err(format!("tune: use auto or off, not {value:?}")),
                }
            }
            "learned_effort" => self.learned_effort = matches!(value, "true" | "True" | "1"),
            "track_usage" => self.track_usage = matches!(value, "true" | "True" | "1"),
            "effort_apart" => self.effort_apart = matches!(value, "true" | "True" | "1"),
            "renumber" => self.renumber = matches!(value, "true" | "True" | "1"),
            "question_bits" => self.question_bits = num(key, value)?,
            "build_space" => {
                self.build_space = match value {
                    "auto" => BuildSpaceKind::Auto,
                    "same" => BuildSpaceKind::Same,
                    "flipped" | "mobius" => BuildSpaceKind::Flipped,
                    _ => return err(format!("build_space: use auto, same or flipped, not {value:?}")),
                }
            }
            "question_links" | "popular_links" => {
                let v = match value {
                    "auto" => Switch::Auto,
                    "on" | "true" | "True" => Switch::On,
                    "off" | "false" | "False" => Switch::Off,
                    _ => return err(format!("{key}: use auto, on or off, not {value:?}")),
                };
                if key == "question_links" {
                    self.question_links = v;
                } else {
                    self.popular_links = v;
                }
            }
            "repair" => {
                self.repair = match value {
                    "append" => Repair::Append,
                    "prune" => Repair::Prune,
                    "occlude" => Repair::Occlude,
                    _ => return err(format!("repair: use append, prune or occlude, not {value:?}")),
                }
            }
            _ => return err(format!("unknown setting {key:?}")),
        }
        Ok(())
    }
}

/// How hard a search looks: a fixed list size, or a target accuracy (learned effort).
#[derive(Clone, Copy, Debug)]
pub enum Effort {
    List(usize),
    Target(f64),
}

/// Results of a batch search: k per question (-1 / infinity where fewer were found), and checks per question.
pub struct SearchOut {
    pub k: usize,
    pub ids: Vec<i64>,
    pub dists: Vec<f32>,
    pub checks: Vec<u64>,
}

/// Facts about the index, for checking.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub items: usize,
    pub start: u32,
    pub build_checks: u64,
    pub change_checks: u64,
    pub mean_links: f64,
    pub no_links_in: usize,
    pub unreachable: usize,
    pub build_space: &'static str,
    pub ood_ratio: Option<f32>,
    pub question_links: bool,
    pub question_link_items: usize,
    /// Items with extra links (part 3), and how many extra links there are.
    pub popular_items: usize,
    pub extra_links: usize,
}

/// The index.
pub struct Index {
    pub dim: usize,
    pub metric: Metric,
    pub s: Settings,
    /// Full numbers (search space), slot-major.
    pub store: Store,
    /// Build copies (flipped), only when they differ from the full numbers.
    pub(crate) flipped: Option<Vec<f32>>,
    pub graph: Graph,
    pub ext: Vec<u64>,
    pub slot_of: HashMap<u64, u32>,
    pub n_used: usize,
    pub start: u32,
    pub build_checks: u64,
    pub change_checks: u64,
    /// The learned effort (Fix 1, better version), when on. Behind a lock so it can be learned again from use
    /// while searches run; each search call keeps one table from start to end.
    pub(crate) effort: RwLock<Option<Arc<Learned>>>,
    /// Training questions, kept when question links are on (to redo them after changes).
    pub train_q: Option<Vec<f32>>,
    /// The "questions unlike items?" ratio, when the test ran.
    pub ood_ratio: Option<f32>,
    /// Items whose links were changed by the last round of question links.
    pub question_link_items: usize,
    /// The 1-byte copies (Fix 6), when on.
    pub codes: Option<Codes>,
    /// Freed slots ready for reuse, and deleted slots waiting to be recycled in a batch.
    pub(crate) free: Vec<u32>,
    pub(crate) pending_free: Vec<u32>,
    /// Changes since the build (for the question-link refresh).
    pub changes: u64,
    /// What searches have done, and the spot checks (part 2): merged once per search call.
    pub(crate) monitor: Mutex<Monitor>,
    /// The items' own median angle distance to their nearest other item: from the "questions unlike items" test
    /// when it ran, else worked out the first time the usage report needs it.
    pub(crate) item_gap: OnceLock<f32>,
    /// What the build's tuning tried and chose, when it ran.
    pub tuning: Option<Tuning>,
    /// The items grouped into buckets, for the bucket search (part 5, DESIGN.md section 23), when built.
    pub buckets: Option<crate::buckets::Buckets>,
}

/// Bytes kept in memory, by part (DESIGN.md section 10).
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub items: usize,
    pub links: usize,
    pub numbers: usize,
    pub other: usize,
    pub total: usize,
    /// Full numbers that can stay on SSD when the 1-byte copies are on.
    pub full_numbers_on_ssd: usize,
}

/// The space links are built in, from the fields that hold it (so the map can be changed at the same time).
pub(crate) fn space_of<'a>(flipped: &'a Option<Vec<f32>>, store: &'a Store, dim: usize, metric: Metric) -> BuildSpace<'a> {
    match flipped {
        Some(b) => BuildSpace { data: b, dim, metric: Metric::L2 },
        None => match store {
            Store::Ram { data, .. } => BuildSpace { data, dim, metric },
            Store::Mapped { .. } => panic!("linking needs the full numbers in memory"),
        },
    }
}

pub(crate) const NO_ID: u64 = u64::MAX;
/// Question links find their answers re-checking this many with the full numbers, whatever count searches use, so
/// the map does not depend on the re-check count (part 3: the tuning picks that after the build).
const QLINK_RERANK: usize = 50;
/// Popular items (DESIGN.md section 22): those in some sample question's first `POPULAR_TOP` answers; each gets up
/// to `POPULAR_MAX` extra links, when the distinct items among the answers are at most `POPULAR_SHARE` of the most
/// there could be.
const POPULAR_TOP: usize = 10;
pub const POPULAR_MAX: usize = 64;
const POPULAR_SHARE: f64 = 0.25;

thread_local! {
    /// Each thread's working memory for searches (list, seen marks, the scaled question), kept between calls so a
    /// search allocates nothing.
    static WORK: std::cell::RefCell<(Scratch, Vec<f32>)> = std::cell::RefCell::new((Scratch::default(), Vec::new()));
}

/// How one search decides its list size.
enum Plan<'a> {
    Fixed(usize),
    Learned { cuts: &'a [f32], table: &'a [usize], lmax: usize },
}

/// Usage counters and spot checks, under one lock (part 2).
#[derive(Default)]
pub(crate) struct Monitor {
    pub usage: Usage,
    pub spot: SpotChecks,
}

/// What the spot checks have found so far.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotReport {
    /// One estimate per kind of search (a target or a list size).
    pub estimates: Vec<Estimate>,
    /// Questions kept and waiting for `audit`, and questions not kept because that list was full.
    pub pending: usize,
    pub dropped: u64,
    /// Checks spent on spot checks, and checks searches made (when usage is tracked).
    pub checks: u64,
    pub search_checks: u64,
    /// Whether this `audit` learned the effort table again.
    pub relearned: bool,
}

/// What one search found, before ids are mapped back.
struct One {
    n: usize,
    checks: u64,
    /// Of the checks, re-checks with the full numbers.
    rechecks: u64,
    z: f32,
}

impl Index {
    pub fn new(dim: usize, metric: Metric, s: Settings) -> Result<Self> {
        if dim == 0 {
            return err("dim must be at least 1");
        }
        if s.r < 2 || s.l_build < 1 || s.alpha < 1.0 {
            return err("need r >= 2, l_build >= 1 and alpha >= 1");
        }
        if ![0, 4, 8].contains(&s.code_bits) {
            return err("code_bits must be 0 (off), 4 or 8");
        }
        if s.code_bits == 4 && s.question_bits != 16 {
            return err("4-bit copies need question_bits=16");
        }
        if ![8, 16, 32].contains(&s.question_bits) {
            return err("question_bits must be 8, 16 or 32");
        }
        Ok(Index {
            dim,
            metric,
            graph: Graph::new(0, s.slots()),
            s,
            store: Store::ram(dim, 0),
            flipped: None,
            ext: Vec::new(),
            slot_of: HashMap::new(),
            n_used: 0,
            start: 0,
            build_checks: 0,
            change_checks: 0,
            effort: RwLock::new(None),
            train_q: None,
            ood_ratio: None,
            question_link_items: 0,
            codes: None,
            free: Vec::new(),
            pending_free: Vec::new(),
            changes: 0,
            monitor: Mutex::new(Monitor::default()),
            item_gap: OnceLock::new(),
            tuning: None,
            buckets: None,
        })
    }

    fn uses_flipped(&self) -> bool {
        match self.s.build_space {
            BuildSpaceKind::Auto => self.metric == Metric::Ip,
            BuildSpaceKind::Same => false,
            BuildSpaceKind::Flipped => self.metric != Metric::Cos,
        }
    }

    /// The space links are built in.
    pub fn build_space(&self) -> BuildSpace<'_> {
        space_of(&self.flipped, &self.store, self.dim, self.metric)
    }

    /// Rebuild the flipped build copies from the full numbers (after loading). Only with the numbers in memory:
    /// an index loaded with its numbers on SSD only searches, and searching does not need them.
    pub(crate) fn recompute_flipped(&mut self) {
        if !self.uses_flipped() || self.store.is_mapped() {
            self.flipped = None;
            return;
        }
        let (d, cap) = (self.dim, self.graph.capacity());
        let mut b = vec![0f32; cap * d];
        for u in 0..self.n_used {
            Self::flip_into(self.store.row(u), &mut b[u * d..(u + 1) * d]);
        }
        self.flipped = Some(b);
    }

    /// The flipped copy x / |x|^2 of one item.
    pub(crate) fn flip_into(x: &[f32], out: &mut [f32]) {
        let n2: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let inv = 1.0 / n2.max(1e-12);
        for (o, v) in out.iter_mut().zip(x) {
            *o = (*v as f64 * inv) as f32;
        }
    }

    /// Build from `x` (n rows of `dim`) with ids `ids`. `train_q`: training questions, rows of `dim` (optional).
    pub fn build(&mut self, x: &[f32], ids: &[u64], train_q: Option<&[f32]>) -> Result<()> {
        self.build_in_order(x, ids, train_q, None)
    }

    /// `build`, with the insertion order given (row numbers, start item first) instead of shuffled. For checking
    /// against the prototype with its exact order.
    pub fn build_in_order(&mut self, x: &[f32], ids: &[u64], train_q: Option<&[f32]>, order: Option<&[u32]>) -> Result<()> {
        let d = self.dim;
        if x.len() % d != 0 || x.len() / d != ids.len() {
            return err(format!("x has {} numbers, which is not {} rows of {d}", x.len(), ids.len()));
        }
        if self.n_used > 0 {
            return err("this index is already built; make a new one");
        }
        check_finite(x, "the items")?;
        if let Some(tq) = train_q {
            check_finite(tq, "the training questions")?;
        }
        if self.s.question_links == Switch::On && train_q.is_none_or(|t| t.is_empty()) {
            return err("question links need training questions (train_questions=...)");
        }
        if let Some(tq) = train_q {
            if tq.len() % d != 0 {
                return err("training questions are not whole rows");
            }
        }
        if self.s.learned_effort && train_q.is_none_or(|t| t.is_empty()) {
            return err("the learned effort needs training questions (train_questions=...), or set learned_effort=False");
        }
        let n = ids.len();
        if n == 0 {
            return err("nothing to build");
        }
        // links per item and slack, picked on a trial (part 2); too little data keeps the settings as they are
        if self.s.tune == Tune::Auto {
            if let Some(t) = train_q.and_then(|tq| crate::tune::pick(&self.s, self.metric, d, x, tq)) {
                self.s.r = t.chosen_r;
                self.s.alpha = t.chosen_alpha;
                self.s.code_bits = t.chosen_code_bits;
                self.tuning = Some(t);
            }
        }
        // full numbers (cosine data scaled to length 1), ids; spare slots for adds
        let cap = if self.s.spare > 0.0 { (n as f64 * (1.0 + self.s.spare as f64)) as usize + 16 } else { n };
        self.store = Store::ram(d, cap);
        self.ext = vec![NO_ID; cap];
        self.slot_of = HashMap::with_capacity(n);
        let mut row = vec![0f32; d];
        for i in 0..n {
            row.copy_from_slice(&x[i * d..(i + 1) * d]);
            if self.metric == Metric::Cos {
                normalize(&mut row);
            }
            self.store.set(i, &row);
            if self.slot_of.insert(ids[i], i as u32).is_some() {
                return err(format!("id {} appears twice", ids[i]));
            }
            self.ext[i] = ids[i];
        }
        self.n_used = n;
        // build copies
        self.flipped = if self.uses_flipped() {
            let mut b = vec![0f32; cap * d];
            for i in 0..n {
                Self::flip_into(self.store.row(i), &mut b[i * d..(i + 1) * d]);
            }
            Some(b)
        } else {
            None
        };
        self.graph = Graph::new(cap, self.s.slots());
        if let Some(o) = order {
            if o.len() != n || o.iter().any(|&i| i as usize >= n) {
                return err("order must list every row once");
            }
        }
        self.link_all(order);
        // The effort table learns from the first `effort_questions` sample questions. With question links on and
        // `effort_apart`, the links use only the questions after those (and the table at most half), so the table
        // is learned on questions the links never saw (DESIGN.md section 15).
        let mut effort_q = train_q;
        if let Some(tq) = train_q.filter(|t| !t.is_empty()) {
            if self.question_links_wanted(tq) {
                let n_train = tq.len() / d;
                let links_q = if self.s.effort_apart && n_train >= 2 {
                    let ne = self.s.effort_questions.min(n_train / 2);
                    effort_q = Some(&tq[..ne * d]);
                    &tq[ne * d..]
                } else {
                    tq
                };
                self.question_links(links_q);
            }
        }
        if self.s.code_bits != 0 {
            self.make_codes();
        }
        let mut effort_truth = None;
        if self.s.learned_effort {
            let (qs, tenth) = self.scaled_with_tenth(effort_q.expect("checked above"));
            self.set_learned(Some(self.learn_from(&qs, &tenth)));
            effort_truth = Some((qs, tenth));
        }
        if self.s.renumber {
            self.renumber();
        }
        // popular items' extra links (part 3), then the copy size (part 2), then the re-check count (part 3), each
        // on the real index, only when the tuning ran
        if let (Some((qs, tenth)), Some(t)) = (&effort_truth, self.tuning.as_ref()) {
            if self.graph.has_extra() && self.s.popular_links == Switch::Auto {
                let tq = train_q.expect("the tuning ran on them");
                let timing = &tq[tq.len() - t.timing_questions * d..];
                self.choose_popular_links(qs, tenth, timing);
            }
        }
        if let (Some((qs, tenth)), Some(t)) = (effort_truth, self.tuning.as_ref()) {
            if self.s.code_bits == 8 && self.s.question_bits == 16 {
                let tq = train_q.expect("the tuning ran on them");
                let timing = &tq[tq.len() - t.timing_questions * d..];
                self.choose_copy_size(&qs, &tenth, timing);
            }
            // then the re-check count (part 3), at the chosen copy size
            if self.codes.is_some() {
                let tq = train_q.expect("the tuning ran on them");
                let nt = self.tuning.as_ref().map_or(0, |t| t.timing_questions);
                self.choose_rerank(&qs, &tenth, &tq[tq.len() - nt * d..]);
            }
        }
        Ok(())
    }

    /// The first `effort_questions` rows, scaled as searches scale them, with their true 10th distances.
    fn scaled_with_tenth(&self, train_q: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let d = self.dim;
        let nq = (train_q.len() / d).min(self.s.effort_questions);
        let mut qs = train_q[..nq * d].to_vec();
        if self.metric == Metric::Cos {
            qs.chunks_mut(d).for_each(normalize);
        }
        let tenth = self.tenth_distances(&qs);
        (qs, tenth)
    }

    /// Popular items' extra links, kept or dropped on the real index (DESIGN.md section 22): timed at the tuning
    /// target with them, then without (the effort table learned again for that, from the same questions), taking
    /// turns (longer when close, part 5); kept only when more than 5% faster. Either way the setting becomes `On` or `Off`, so later rounds of
    /// question links (after changes) make them again only if they were kept.
    fn choose_popular_links(&mut self, qs: &[f32], tenth: &[f32], timing_q: &[f32]) {
        let d = self.dim;
        let target = self.s.tune_target;
        let mut scaled = timing_q.to_vec();
        if self.metric == Metric::Cos {
            scaled.chunks_mut(d).for_each(normalize);
        }
        let timing_tenth = self.tenth_distances(&scaled);
        let p_on = crate::tune::plan(self, timing_q, &timing_tenth, target);
        let on_table = self.learned();
        let take = |g: &mut Graph| (std::mem::take(&mut g.extra_at), std::mem::take(&mut g.extra_n), std::mem::take(&mut g.extra));
        let mut ex = take(&mut self.graph);
        self.set_learned(Some(self.learn_from(qs, tenth)));
        let off_table = self.learned();
        let p_off = crate::tune::plan(self, timing_q, &timing_tenth, target);
        let swap = |g: &mut Graph, ex: &mut (Vec<u32>, Vec<u8>, Vec<u32>)| {
            std::mem::swap(&mut g.extra_at, &mut ex.0);
            std::mem::swap(&mut g.extra_n, &mut ex.1);
            std::mem::swap(&mut g.extra, &mut ex.2);
        };
        let (mut t_on, mut t_off) = (f64::INFINITY, f64::INFINITY);
        for rounds in [crate::tune::ROUNDS, crate::tune::MORE_ROUNDS] {
            for _ in 0..rounds {
                if let Some(p) = p_off {
                    t_off = t_off.min(crate::tune::time_plan(self, timing_q, &p));
                }
                if let Some(p) = p_on {
                    swap(&mut self.graph, &mut ex);
                    self.put_learned(on_table.clone());
                    t_on = t_on.min(crate::tune::time_plan(self, timing_q, &p));
                    swap(&mut self.graph, &mut ex);
                    self.put_learned(off_table.clone());
                }
            }
            if !crate::tune::close(t_off / (crate::tune::CLEAR_WIN * t_on)) {
                break; // not a close call (those are timed longer, part 5)
            }
        }
        let keep = t_on.is_finite() && (!t_off.is_finite() || t_off > crate::tune::CLEAR_WIN * t_on);
        if keep {
            swap(&mut self.graph, &mut ex);
            self.put_learned(on_table);
        }
        self.s.popular_links = if keep { Switch::On } else { Switch::Off };
        if let Some(t) = self.tuning.as_mut() {
            for (label, sec, p) in [("full size, popular links on", t_on, p_on), ("full size, popular links off", t_off, p_off)] {
                t.trials.push(crate::tune::Trial {
                    r: self.s.r,
                    alpha: self.s.alpha,
                    code_bits: self.s.code_bits,
                    rerank: self.s.rerank,
                    links: label.to_string(),
                    question_links: self.train_q.is_some(),
                    searches_per_s: sec.is_finite().then(|| 1.0 / sec),
                    checks: p.map(|p| p.checks),
                    bytes_per_item: 0.0,
                    seconds: 0.0,
                });
            }
        }
    }

    /// Copy size, chosen on the real index (DESIGN.md section 19): time it at the tuning target with its 1-byte
    /// copies, then with 4-bit copies (the effort table learned again for them, from the same questions), taking
    /// turns (longer when close, part 5); keep the 4-bit copies only when more than 5% faster. A trial small enough to sit in the chip's caches
    /// cannot see what smaller copies save in memory reads.
    fn choose_copy_size(&mut self, qs: &[f32], tenth: &[f32], timing_q: &[f32]) {
        let d = self.dim;
        let target = self.s.tune_target;
        let mut scaled = timing_q.to_vec();
        if self.metric == Metric::Cos {
            scaled.chunks_mut(d).for_each(normalize);
        }
        let timing_tenth = self.tenth_distances(&scaled);
        let Some(p8) = crate::tune::plan(self, timing_q, &timing_tenth, target) else { return };
        let mut other = (self.codes.take(), self.learned());
        self.s.code_bits = 4;
        self.make_codes();
        self.set_learned(Some(self.learn_from(qs, tenth)));
        let p4 = crate::tune::plan(self, timing_q, &timing_tenth, target);
        let swap = |idx: &mut Index, other: &mut (Option<Codes>, Option<Arc<Learned>>)| {
            std::mem::swap(&mut idx.codes, &mut other.0);
            let e = idx.learned();
            *idx.effort.write().unwrap_or_else(|e| e.into_inner()) = other.1.take();
            other.1 = e;
        };
        let (mut t4, mut t8) = (f64::INFINITY, f64::INFINITY);
        if let Some(p4) = p4 {
            for rounds in [crate::tune::ROUNDS, crate::tune::MORE_ROUNDS] {
                for _ in 0..rounds {
                    t4 = t4.min(crate::tune::time_plan(self, timing_q, &p4));
                    swap(self, &mut other);
                    t8 = t8.min(crate::tune::time_plan(self, timing_q, &p8));
                    swap(self, &mut other);
                }
                if !crate::tune::close(t8 / (crate::tune::CLEAR_WIN * t4)) {
                    break; // not a close call (those are timed longer, part 5)
                }
            }
        }
        let keep4 = t4.is_finite() && t8 > crate::tune::CLEAR_WIN * t4;
        if !keep4 {
            swap(self, &mut other); // back to the 1-byte copies and their table
            self.s.code_bits = 8;
        }
        if let Some(t) = self.tuning.as_mut() {
            for (bits, sec, p) in [(8u8, t8, Some(p8)), (4, t4, p4)] {
                t.trials.push(crate::tune::Trial {
                    r: self.s.r,
                    alpha: self.s.alpha,
                    code_bits: bits,
                    rerank: self.s.rerank,
                    links: "full size".to_string(),
                    question_links: self.train_q.is_some(),
                    searches_per_s: sec.is_finite().then(|| 1.0 / sec),
                    checks: p.map(|p| p.checks),
                    bytes_per_item: crate::tune::bytes_per_item(&Settings { code_bits: bits, ..self.s.clone() }, d),
                    seconds: 0.0,
                });
            }
            t.chosen_code_bits = self.s.code_bits;
        }
    }

    /// Re-check count, chosen on the real index after the copy size (DESIGN.md section 20): each count in
    /// `RERANK_CHOICES` gets its own effort table, learned from the same questions (the map does not depend on the
    /// count), and all are timed at the tuning target, taking turns. The plainest count within 2% of the fastest is
    /// kept: the count in use first, then the nearest to it (part 5: a 2% bar alone let timing noise decide). Counts
    /// close to that bar are timed longer, with the fastest.
    fn choose_rerank(&mut self, qs: &[f32], tenth: &[f32], timing_q: &[f32]) {
        let d = self.dim;
        let target = self.s.tune_target;
        let mut scaled = timing_q.to_vec();
        if self.metric == Metric::Cos {
            scaled.chunks_mut(d).for_each(normalize);
        }
        let timing_tenth = self.tenth_distances(&scaled);
        let start = self.s.rerank;
        let mut counts: Vec<usize> = crate::tune::RERANK_CHOICES.to_vec();
        if !counts.contains(&start) {
            counts.push(start);
        }
        counts.sort_by_key(|&c| (c.abs_diff(start), std::cmp::Reverse(c))); // plainest first: the count in use
        // each count with its table and where it reaches the target; the count in use keeps its own table
        let start_table = self.learned();
        let mut cands: Vec<(usize, Option<Arc<Learned>>, Option<crate::tune::Plan>)> = Vec::new();
        for &c in &counts {
            self.s.rerank = c;
            let table = if c == start { start_table.clone() } else { Some(Arc::new(self.learn_from(qs, tenth))) };
            self.put_learned(table.clone());
            let p = crate::tune::plan(self, timing_q, &timing_tenth, target);
            cands.push((c, table, p));
        }
        let mut best = vec![f64::INFINITY; cands.len()];
        let mut who: Vec<usize> = (0..cands.len()).collect();
        for rounds in [crate::tune::ROUNDS, crate::tune::MORE_ROUNDS] {
            for _ in 0..rounds {
                for &i in &who {
                    let (c, table, p) = &cands[i];
                    if let Some(p) = p {
                        self.s.rerank = *c;
                        self.put_learned(table.clone());
                        best[i] = best[i].min(crate::tune::time_plan(self, timing_q, p));
                    }
                }
            }
            // the counts close to the tie bar are timed longer, with the fastest (part 5)
            let fastest = best.iter().copied().fold(f64::INFINITY, f64::min);
            who = (0..cands.len()).filter(|&i| crate::tune::close(crate::tune::TIE * fastest / best[i])).collect();
            if who.is_empty() {
                break;
            }
            who.extend(best.iter().position(|&b| b == fastest));
        }
        let fastest = best.iter().copied().fold(f64::INFINITY, f64::min);
        let pick = (0..cands.len()).find(|&i| best[i] <= crate::tune::TIE * fastest).unwrap_or(0);
        self.s.rerank = cands[pick].0;
        self.put_learned(cands[pick].1.clone());
        if let Some(t) = self.tuning.as_mut() {
            for ((c, _, p), sec) in cands.iter().zip(&best) {
                t.trials.push(crate::tune::Trial {
                    r: self.s.r,
                    alpha: self.s.alpha,
                    code_bits: self.s.code_bits,
                    rerank: *c,
                    links: "full size, re-check".to_string(),
                    question_links: self.train_q.is_some(),
                    searches_per_s: sec.is_finite().then(|| 1.0 / sec),
                    checks: p.map(|p| p.checks),
                    bytes_per_item: crate::tune::bytes_per_item(&self.s, d),
                    seconds: 0.0,
                });
            }
            t.chosen_rerank = self.s.rerank;
        }
    }

    /// Put this table in use (it may be shared with a table kept elsewhere).
    fn put_learned(&self, l: Option<Arc<Learned>>) {
        *self.effort.write().unwrap_or_else(|e| e.into_inner()) = l;
    }

    /// The "questions unlike items?" test (or the switch): are question links on? (DESIGN.md section 8)
    fn question_links_wanted(&mut self, train_q: &[f32]) -> bool {
        match self.s.question_links {
            Switch::Off => false,
            Switch::On => true,
            Switch::Auto => {
                let alive: Vec<u32> = (0..self.n_used as u32).filter(|&u| self.graph.is_alive(u as usize)).collect();
                let (ratio, gap, checks) = qlinks::questions_unlike_items(&self.store, &alive, train_q, self.dim, 5);
                self.build_checks += checks;
                self.ood_ratio = Some(ratio);
                let _ = self.item_gap.set(gap);
                ratio > self.s.ood_ratio
            }
        }
    }

    /// Question links from these training questions, made `q_rounds` times; the questions are kept for the
    /// refresh after changes.
    fn question_links(&mut self, train_q: &[f32]) {
        let mut tq = train_q.to_vec();
        if self.metric == Metric::Cos {
            tq.chunks_mut(self.dim).for_each(normalize);
        }
        self.train_q = Some(tq);
        for _ in 0..self.s.q_rounds {
            let checks = self.question_links_round();
            self.build_checks += checks;
        }
    }

    /// One round of question links with the remembered training questions: find each one's top answers with
    /// the current search, then link them. Returns the checks made.
    pub fn question_links_round(&mut self) -> u64 {
        let Some(tq) = self.train_q.as_ref() else { return 0 };
        let (d, k) = (self.dim, self.s.q_topk);
        let plan_l = self.s.l_build.max(k);
        let runs: Vec<(Vec<i64>, u64)> = tq
            .par_chunks(d)
            .map_init(Scratch::default, |s, q| {
                let (mut ids, mut ds) = (vec![-1i64; k], vec![f32::INFINITY; k]);
                let one = self.search_one_rr(q, k, QLINK_RERANK, &Plan::Fixed(plan_l), s, &mut ids, &mut ds);
                (ids, one.checks)
            })
            .collect();
        let checks = runs.iter().map(|r| r.1).sum();
        let answers: Vec<Vec<i64>> = runs.into_iter().map(|r| r.0).collect();
        let sp = space_of(&self.flipped, &self.store, self.dim, self.metric);
        self.question_link_items = qlinks::link_answers(&mut self.graph, &sp, &answers, self.s.r, self.s.alpha, self.s.q_links_max);
        // part 3: popular items' extra links, from the same answers
        if self.s.popular_links != Switch::Off {
            let share = if self.s.popular_links == Switch::On { f64::INFINITY } else { POPULAR_SHARE };
            match qlinks::popular_links(&self.graph, &answers, self.slot_of.len(), POPULAR_TOP, POPULAR_MAX, share) {
                Some(lists) => self.graph.set_extra(&lists),
                None => self.graph.clear_extra(),
            }
        }
        checks
    }

    /// The 1-byte copies of every item (DESIGN.md section 3); their scale comes from the items in the index now.
    fn make_codes(&mut self) {
        let alive: Vec<usize> = (0..self.n_used).filter(|&u| self.graph.is_alive(u)).collect();
        let mut c = Codes::train_bits(alive.iter().map(|&u| self.store.row(u)), self.dim, self.metric, self.graph.capacity(),
                                      self.s.code_bits);
        c.qbits = self.s.question_bits;
        for &u in &alive {
            c.set(u, self.store.row(u));
        }
        self.codes = Some(c);
    }

    /// Bytes kept in memory, counted like the prototype: link slots and a degree per used slot, the numbers
    /// searched on (1-byte copies, or the full numbers when the copies are off), and an 8-byte id.
    pub fn memory(&self) -> Memory {
        let n = self.n_used;
        // the link slots and a degree per used slot, and the popular items' extra links with their offset and count
        let extra = if self.graph.has_extra() { n * 5 + self.graph.extra.len() * 4 } else { 0 };
        let links = n * (self.graph.slots * 4 + 4) + extra;
        let (numbers, ssd) = match &self.codes {
            Some(c) => (n * c.bytes_per_item(), n * self.dim * 4),
            None => (n * self.dim * 4, 0),
        };
        let other = n * 8 + 8;
        Memory { items: n, links, numbers, other, total: links + numbers + other, full_numbers_on_ssd: ssd }
    }

    /// The training questions' true 10th distance among the alive items, found by checking every item.
    pub(crate) fn tenth_distances(&self, qs: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let alive: Vec<u32> = (0..self.n_used as u32).filter(|&u| self.graph.is_alive(u as usize)).collect();
        qs.par_chunks(d)
            .map_init(Vec::<f32>::new, |all, q| {
                all.clear();
                all.extend(alive.iter().map(|&u| self.metric.distance(q, self.store.row(u as usize))));
                let k = 9.min(all.len().saturating_sub(1));
                *all.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
            })
            .collect()
    }

    /// The effort table in use, if any.
    pub fn learned(&self) -> Option<Arc<Learned>> {
        self.effort.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub(crate) fn set_learned(&self, l: Option<Learned>) {
        *self.effort.write().unwrap_or_else(|e| e.into_inner()) = l.map(Arc::new);
    }

    /// Learn the effort table from the first `effort_questions` training questions (DESIGN.md section 5).
    pub fn learn_effort(&self, train_q: &[f32]) {
        let d = self.dim;
        let nq = (train_q.len() / d).min(self.s.effort_questions);
        let mut qs = train_q[..nq * d].to_vec();
        if self.metric == Metric::Cos {
            qs.chunks_mut(d).for_each(normalize);
        }
        let tenth = self.tenth_distances(&qs);
        self.set_learned(Some(self.learn_from(&qs, &tenth)));
    }

    /// Learn the effort table again from the spot-checked questions (DESIGN.md section 15). Their true 10th
    /// distances are already known, so this costs only searches. Needs `RELEARN_MIN` of them; returns how many
    /// were used. The counts for targets start again, since they measured the old table.
    pub fn relearn_effort(&self) -> Option<usize> {
        if self.learned().is_none() {
            return None;
        }
        let checked: Vec<Checked> = self.checked_questions().into_iter().filter(|c| c.tenth.is_finite()).collect();
        if checked.len() < RELEARN_MIN {
            return None;
        }
        let qs: Vec<f32> = checked.iter().flat_map(|c| c.q.iter().copied()).collect();
        let tenth: Vec<f32> = checked.iter().map(|c| c.tenth).collect();
        self.set_learned(Some(self.learn_from(&qs, &tenth)));
        let mut m = self.monitor();
        m.spot.since_learn = 0;
        m.spot.sums.retain(|l, _| !l.starts_with("target"));
        Some(checked.len())
    }

    /// The table from already-scaled questions and their true 10th distances: cut points from the first stage's
    /// hardness, then correct answers and checks per group at each list size (DESIGN.md section 5).
    fn learn_from(&self, qs: &[f32], tenth: &[f32]) -> Learned {
        let nq = tenth.len();
        // first stage alone: each question's hardness z
        let z: Vec<f32> = self.run_learned(&qs, &[], &[L1]).into_iter().map(|r| r.2).collect();
        let mut sorted = z.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let cuts: Vec<f32> = (1..effort::N_BINS).map(|i| effort::quantile(&sorted, i as f64 / effort::N_BINS as f64)).collect();
        let bins: Vec<usize> = z.iter().map(|&v| effort::bin_of(&cuts, v)).collect();
        let (mut levels, mut correct, mut checks) = (Vec::new(), Vec::new(), Vec::new());
        for &l in effort::L_GRID.iter() {
            let runs = self.run_learned(&qs, &[], &[l]);
            let c: Vec<u32> = runs
                .iter()
                .zip(tenth)
                .map(|(r, &t)| r.0.iter().filter(|&&dd| dd <= t + effort::TIE_EPS).count().min(10) as u32)
                .collect();
            let mean = c.iter().map(|&v| v as f64).sum::<f64>() / (10.0 * nq.max(1) as f64);
            levels.push(l);
            correct.push(c);
            checks.push(runs.iter().map(|r| r.1).collect::<Vec<u64>>());
            if mean >= effort::ENOUGH {
                break;
            }
        }
        Learned::from_runs(cuts, &bins, levels, &correct, &checks)
    }

    /// Run learned-effort searches for already-scaled questions; per question: (found distances, checks, z).
    fn run_learned(&self, qs: &[f32], cuts: &[f32], table: &[usize]) -> Vec<(Vec<f32>, u64, f32)> {
        let d = self.dim;
        let lmax = table.iter().copied().max().unwrap_or(L1).max(L1);
        qs.par_chunks(d)
            .map_init(Scratch::default, |s, q| {
                let plan = Plan::Learned { cuts, table, lmax };
                let (mut ids, mut ds) = (vec![-1i64; 10], vec![f32::INFINITY; 10]);
                let one = self.search_one(q, 10, &plan, s, &mut ids, &mut ds);
                (ds[..one.n].to_vec(), one.checks, one.z)
            })
            .collect()
    }

    /// One search for an already-scaled question. Writes slots (not ids) into `ids`.
    fn search_one(&self, q: &[f32], k: usize, plan: &Plan, s: &mut Scratch, ids: &mut [i64], dists: &mut [f32]) -> One {
        self.search_one_rr(q, k, self.s.rerank, plan, s, ids, dists)
    }

    /// `search_one` re-checking the first max(`rr`, k) entries with the full numbers (when the copies are on).
    #[allow(clippy::too_many_arguments)]
    fn search_one_rr(&self, q: &[f32], k: usize, rr: usize, plan: &Plan, s: &mut Scratch, ids: &mut [i64], dists: &mut [f32]) -> One {
        match &self.codes {
            Some(c) => {
                let Scratch { list, visited, prep, rerank, .. } = s;
                c.prepare(q, prep);
                let mut sc = CodeScore { codes: c, prep: &*prep };
                let (mut checks, z) = self.walk_plan(k, rr, plan, list, visited, &mut sc);
                self.prefetch_rerank(list, k, rr);
                let mut full = FullScore { metric: self.metric, q, store: &self.store };
                let (n, rc) = finish_rerank(&self.graph, list, k, rr, &mut full, rerank, ids, dists);
                checks += rc;
                One { n, checks, rechecks: rc, z }
            }
            None => {
                let Scratch { list, visited, .. } = s;
                let mut sc = FullScore { metric: self.metric, q, store: &self.store };
                let (checks, z) = self.walk_plan(k, rr, plan, list, visited, &mut sc);
                let n = finish_plain(&self.graph, list, k, ids, dists);
                One { n, checks, rechecks: 0, z }
            }
        }
    }

    /// Start loading the full numbers of the entries the re-check will read (memory or SSD).
    fn prefetch_rerank(&self, list: &SortedList, k: usize, rr: usize) {
        let rr = list.size.min(rr.max(k));
        let mut picked = [0u32; 256];
        let mut m = 0;
        for j in 0..list.size {
            if m == rr || m == picked.len() {
                break;
            }
            let i = list.id(j);
            if self.graph.is_alive(i as usize) {
                picked[m] = i;
                m += 1;
            }
        }
        if self.store.is_mapped() {
            self.store.prefetch_rows(&picked[..m]);
        } else {
            for &i in &picked[..m] {
                let r = self.store.row(i as usize);
                prefetch_bytes(r.as_ptr() as *const u8, r.len() * 4);
            }
        }
    }

    /// The walk for one search plan: fixed list size, or the learned effort's two stages; `rr` is the re-check
    /// count the answer will read. Returns (checks, z).
    fn walk_plan<S: Score>(
        &self,
        k: usize,
        rr: usize,
        plan: &Plan,
        list: &mut SortedList,
        visited: &mut crate::search::Visited,
        sc: &mut S,
    ) -> (u64, f32) {
        visited.begin(self.graph.capacity());
        let (cap, first) = match plan {
            Plan::Fixed(l) => (*l + 1, *l + 1),
            Plan::Learned { lmax, .. } => (*lmax + 1, L1 + 1),
        };
        list.reset(cap);
        let mut checks = seed(&self.graph, list, visited, &[self.start], sc);
        // no slot in use is deleted (the usual case after a build): no link can lead to a deleted item
        let any_dead = self.slot_of.len() < self.n_used;
        let go = |list: &mut SortedList, visited: &mut crate::search::Visited, sc: &mut S, limit: usize| {
            if any_dead { walk(&self.graph, list, visited, sc, limit, None).0 } else { walk_with::<S, false>(&self.graph, list, visited, sc, limit, None).0 }
        };
        checks += go(list, visited, sc, first);
        let mut z = 0.0;
        if let Plan::Learned { cuts, table, .. } = plan {
            z = -self.kth_alive(list, k);
            let b = effort::bin_of(cuts, z);
            let grow = table[b.min(table.len() - 1)].max(L1) + 1;
            // entries past this question's limit and past what the answer reads can never be used: stop keeping
            // them (one extra for a start item that may have been deleted)
            let reads = if self.codes.is_some() { rr.max(k) } else { k };
            list.shrink_cap(grow.max(reads + 1));
            checks += go(list, visited, sc, grow);
        }
        (checks, z)
    }

    /// Distance of the k-th alive entry in the list (of the last alive one if there are fewer).
    fn kth_alive(&self, list: &SortedList, k: usize) -> f32 {
        let mut m = 0;
        let mut dk = f32::INFINITY;
        for j in 0..list.size {
            if !self.graph.is_alive(list.id(j) as usize) {
                continue;
            }
            m += 1;
            dk = list.dist(j);
            if m == k {
                break;
            }
        }
        dk
    }

    /// The map: start item, shuffled order, growing batches (DESIGN.md section 7).
    fn link_all(&mut self, given: Option<&[u32]>) {
        let n = self.n_used;
        let d = self.dim;
        let sp = self.build_space();
        // start: the item closest to the average, in build space
        let mut mean = vec![0f64; d];
        for i in 0..n {
            for (m, v) in mean.iter_mut().zip(sp.row(i as u32)) {
                *m += *v as f64;
            }
        }
        let meanf: Vec<f32> = mean.iter().map(|m| (*m / n as f64) as f32).collect();
        let mut start = 0u32;
        let mut best = f32::INFINITY;
        for i in 0..n {
            let dd = crate::metric::l2sq(sp.row(i as u32), &meanf);
            if dd < best {
                best = dd;
                start = i as u32;
            }
        }
        let order: Vec<u32> = match given {
            Some(o) => {
                start = o[0];
                o[1..].to_vec()
            }
            None => {
                let mut o: Vec<u32> = (0..n as u32).filter(|&i| i != start).collect();
                Rng::new(self.s.seed).shuffle(&mut o);
                o
            }
        };
        let (r, l, alpha) = (self.s.r, self.s.l_build, self.s.alpha);
        let max_batch = ((self.s.max_batch_frac * n as f32) as usize).max(1);
        let flipped = self.flipped.take();
        let mut graph = std::mem::replace(&mut self.graph, Graph::new(0, 1));
        graph.alive[start as usize] = 1;
        let mut checks = 0u64;
        {
            let sp = match &flipped {
                Some(b) => BuildSpace { data: b, dim: d, metric: Metric::L2 },
                None => self.build_space(),
            };
            let (mut pos, mut bsz) = (0usize, 1usize);
            while pos < order.len() {
                let batch = &order[pos..(pos + bsz).min(order.len())];
                let (outs, c) = out_links_batch(&graph, &sp, start, batch, l, r, alpha);
                checks += c;
                for (&u, out) in batch.iter().zip(&outs) {
                    graph.set_links(u as usize, out);
                    graph.alive[u as usize] = 1;
                }
                back_links(&mut graph, &sp, batch, r, alpha);
                pos += batch.len();
                bsz = (bsz * 2).min(max_batch);
            }
        }
        self.graph = graph;
        self.flipped = flipped;
        self.start = start;
        self.build_checks += checks;
    }

    /// Group the live items into buckets of about `per_bucket` items, for the bucket search (part 5, DESIGN.md
    /// section 23), each item coded with weight `eta` on errors along it (1: plain). Needs the full numbers in memory.
    pub fn build_buckets(&mut self, per_bucket: usize, eta: f32) -> Result<()> {
        self.build_buckets_with(per_bucket, eta, crate::buckets::KMEANS_ROUNDS, crate::buckets::KMEANS_ROWS_PER_CENTRE)
    }

    /// `build_buckets` with the k-means rounds and sample rows per centre given (step 40's measurements).
    pub fn build_buckets_with(&mut self, per_bucket: usize, eta: f32, rounds: usize, per_centre: usize) -> Result<()> {
        let Store::Ram { data, .. } = &self.store else {
            return err("the full numbers are on SSD; load with numbers=\"ram\" to build buckets");
        };
        let alive: Vec<bool> = (0..self.n_used).map(|u| self.graph.is_alive(u)).collect();
        let rows = &data[..self.n_used * self.dim];
        self.buckets = Some(crate::buckets::Buckets::build_with(rows, &alive, self.dim, self.metric, per_bucket, eta, rounds, per_centre, self.s.seed));
        Ok(())
    }

    /// Search with the buckets: score the centres, scan the `n_probe` nearest buckets' codes, and re-check the `pool`
    /// best with the full numbers. Checks per question count both the codes scanned and the full re-checks.
    pub fn search_buckets(&self, q: &[f32], k: usize, n_probe: usize, pool: usize, parallel: bool) -> Result<SearchOut> {
        let Some(b) = &self.buckets else { return err("this index has no buckets; build them first") };
        let d = self.dim;
        if q.len() % d != 0 {
            return err("questions are not whole rows");
        }
        check_finite(q, "the questions")?;
        let nq = q.len() / d;
        let mut out = SearchOut { k, ids: vec![-1; nq * k], dists: vec![f32::INFINITY; nq * k], checks: vec![0; nq] };
        let one = |i: usize, oi: &mut [i64], od: &mut [f32], oc: &mut u64| {
            let mut qb = q[i * d..(i + 1) * d].to_vec();
            if self.metric == Metric::Cos {
                normalize(&mut qb);
            }
            let mut w = crate::buckets::Work::default();
            let cands = b.scan(&qb, n_probe, pool.max(k), |u| self.graph.is_alive(u as usize), &mut w);
            let slots: Vec<u32> = cands.iter().map(|c| c.1).collect();
            self.store.prefetch_rows(&slots);
            let mut exact: Vec<(f32, u32)> = slots.iter().map(|&u| (self.metric.distance(&qb, self.store.row(u as usize)), u)).collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (j, &(dd, u)) in exact.iter().take(k).enumerate() {
                oi[j] = self.ext[u as usize] as i64;
                od[j] = dd;
            }
            *oc = w.scanned + exact.len() as u64;
        };
        let rows = out.ids.chunks_mut(k).zip(out.dists.chunks_mut(k)).zip(out.checks.iter_mut());
        if parallel {
            rows.collect::<Vec<_>>().into_par_iter().enumerate().for_each(|(i, ((oi, od), oc))| one(i, oi, od, oc));
        } else {
            rows.enumerate().for_each(|(i, ((oi, od), oc))| one(i, oi, od, oc));
        }
        Ok(out)
    }

    /// Search a batch of questions (rows of `dim`), spread over the current rayon pool.
    pub fn search(&self, q: &[f32], k: usize, effort: Effort) -> Result<SearchOut> {
        self.search_batch(q, k, effort, true, false)
    }

    /// `search` on the calling thread only, with no thread pool: searches started from several threads at once
    /// each run on their own thread, side by side.
    pub fn search_here(&self, q: &[f32], k: usize, effort: Effort) -> Result<SearchOut> {
        self.search_batch(q, k, effort, false, false)
    }

    /// `search` (`parallel`) or `search_here`, left out of the usage stats and the spot checks: the index's own
    /// timing of itself (suggestions) is not use.
    pub(crate) fn search_quiet(&self, q: &[f32], k: usize, effort: Effort, parallel: bool) -> Result<SearchOut> {
        self.search_batch(q, k, effort, parallel, true)
    }

    fn search_batch(&self, q: &[f32], k: usize, effort: Effort, parallel: bool, quiet: bool) -> Result<SearchOut> {
        let d = self.dim;
        if q.len() % d != 0 {
            return err("questions are not whole rows");
        }
        check_finite(q, "the questions")?;
        if self.n_used == 0 {
            return err("the index is empty; build it first");
        }
        let learned = self.learned();
        let (table, lmax) = match effort {
            Effort::List(_) => (Vec::new(), 0),
            Effort::Target(t) => match &learned {
                Some(le) => {
                    let tab = le.table(t);
                    let lmax = tab.iter().copied().max().unwrap_or(L1).max(L1);
                    (tab, lmax)
                }
                None => return err("this index has no learned effort; search with list_size=..."),
            },
        };
        let cuts: &[f32] = learned.as_ref().map(|e| e.cuts.as_slice()).unwrap_or(&[]);
        let nq = q.len() / d;
        let mut out = SearchOut { k, ids: vec![-1; nq * k], dists: vec![f32::INFINITY; nq * k], checks: vec![0; nq] };
        let cos = self.metric == Metric::Cos;
        let track = self.s.track_usage && !quiet;
        let one = |i: usize, oi: &mut [i64], od: &mut [f32], oc: &mut u64, t: &mut Usage| {
            WORK.with(|w| {
                let t0 = track.then(Instant::now);
                let mut w = w.borrow_mut();
                let (s, qb) = &mut *w;
                qb.clear();
                qb.extend_from_slice(&q[i * d..(i + 1) * d]);
                if cos {
                    normalize(qb);
                }
                let plan = match effort {
                    Effort::List(l) => Plan::Fixed(l.max(k)),
                    Effort::Target(_) => Plan::Learned { cuts, table: &table, lmax },
                };
                let r = self.search_one(qb, k, &plan, s, oi, od);
                if let Some(t0) = t0 {
                    let gap = (r.n > 0).then(|| self.angle_gap(qb, oi[0] as usize, od[0]));
                    let bin = matches!(effort, Effort::Target(_)).then(|| effort::bin_of(cuts, r.z));
                    t.question(r.checks, r.rechecks, t0.elapsed().as_nanos() as u64, bin, gap);
                }
                for v in oi[..r.n].iter_mut() {
                    *v = self.ext[*v as usize] as i64;
                }
                *oc = r.checks;
            })
        };
        // each part of the batch adds up its own tally; the index's is locked once per call
        let mut tally = if parallel {
            out.ids
                .par_chunks_mut(k)
                .zip(out.dists.par_chunks_mut(k))
                .zip(out.checks.par_iter_mut())
                .enumerate()
                .fold(Usage::default, |mut t, (i, ((oi, od), oc))| {
                    one(i, oi, od, oc, &mut t);
                    t
                })
                .reduce(Usage::default, |mut a, b| {
                    a.merge(&b);
                    a
                })
        } else {
            let mut t = Usage::default();
            for (i, ((oi, od), oc)) in out.ids.chunks_mut(k).zip(out.dists.chunks_mut(k)).zip(out.checks.iter_mut()).enumerate() {
                one(i, oi, od, oc, &mut t);
            }
            t
        };
        let budget = if quiet { 0.0 } else { self.s.audit_budget };
        if track || budget > 0.0 {
            let mut m = self.monitor();
            if track {
                tally.calls = 1;
                m.usage.merge(&tally);
            }
            if budget > 0.0 {
                let checks: u64 = out.checks.iter().sum();
                let question = |j: usize| {
                    let mut v = q[j * d..(j + 1) * d].to_vec();
                    if cos {
                        normalize(&mut v);
                    }
                    v
                };
                m.spot.offer(budget, checks, self.slot_of.len(), d, k, effort, nq, question, &out.ids);
            }
        }
        Ok(out)
    }

    /// Angle distance (1 - cosine) from a question to an item, given the search's own distance to it.
    fn angle_gap(&self, q: &[f32], slot: usize, dist: f32) -> f32 {
        match self.metric {
            Metric::Cos => dist, // both scaled to length 1: the distance is already 1 - cosine
            _ => {
                let x = self.store.row(slot);
                let n = (dot(q, q) * dot(x, x)).sqrt();
                if n > 0.0 { 1.0 - dot(q, x) / n } else { 1.0 }
            }
        }
    }

    pub(crate) fn monitor(&self) -> std::sync::MutexGuard<'_, Monitor> {
        self.monitor.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The usage counters so far (part 2, DESIGN.md section 13).
    pub fn usage(&self) -> Usage {
        self.monitor().usage.clone()
    }

    /// Start the usage counters and the spot checks' counts again from zero (the checked questions are kept).
    pub fn reset_usage(&self) {
        let mut m = self.monitor();
        m.usage = Usage::default();
        m.spot.reset();
    }

    /// Put back saved counters (used when loading).
    pub(crate) fn set_monitor(&mut self, usage: Usage, spot: SpotChecks) {
        let m = self.monitor.get_mut().unwrap_or_else(|e| e.into_inner());
        m.usage = usage;
        m.spot = spot;
    }

    /// Find the true answers of the questions the spot checks kept, by checking every item with the full
    /// numbers, and add them to the estimates (DESIGN.md section 14). Searches can run meanwhile.
    pub fn audit(&self) -> SpotReport {
        let pending = std::mem::take(&mut self.monitor().spot.pending);
        if !pending.is_empty() {
            // items in memory order, so the reads stream; 8 questions share each pass over the items
            let alive: Vec<u32> = (0..self.n_used as u32).filter(|&u| self.graph.is_alive(u as usize)).collect();
            let done: Vec<Checked> = pending.par_chunks(8).flat_map_iter(|b| self.check_block(b, &alive)).collect();
            let mut m = self.monitor();
            m.spot.checks += (done.len() * alive.len()) as u64;
            m.spot.since_learn += done.len() as u64;
            for c in done {
                m.spot.add(c);
            }
            let due = self.s.relearn_every > 0 && m.spot.since_learn >= self.s.relearn_every && m.spot.target_missed();
            drop(m);
            if due {
                // report what the spot checks saw, before learning again starts the targets' counts afresh
                let mut r = self.spot_report();
                if self.relearn_effort().is_some() {
                    r.relearned = true;
                    return r;
                }
            }
        }
        self.spot_report()
    }

    /// What the spot checks have found so far, without checking the waiting questions.
    pub fn spot_report(&self) -> SpotReport {
        let m = self.monitor();
        SpotReport {
            estimates: m.spot.estimates(),
            pending: m.spot.pending.len(),
            dropped: m.spot.dropped,
            checks: m.spot.checks,
            search_checks: m.usage.checks,
            relearned: false,
        }
    }

    /// The questions spot-checked most recently, with their true answers' distances, newest last.
    pub fn checked_questions(&self) -> Vec<Checked> {
        self.monitor().spot.window.iter().cloned().collect()
    }

    /// Spot checks for a few questions in one pass over the items: each question's true k-th and 10th distances
    /// among the alive items, and how many of the answers it got are within the k-th (ties count, as in the
    /// learned effort and the validation). Items are scored four at a time with the same arithmetic as one at a
    /// time, and each question keeps only its closest few distances.
    fn check_block(&self, block: &[Sample], alive: &[u32]) -> Vec<Checked> {
        let keep: Vec<usize> = block.iter().map(|s| s.k.max(10)).collect();
        let mut best: Vec<Vec<f32>> = keep.iter().map(|&n| Vec::with_capacity(n + 1)).collect();
        let add = |b: &mut Vec<f32>, n: usize, d: f32| {
            if b.len() == n && d >= b[n - 1] {
                return;
            }
            let at = b.partition_point(|&x| x <= d);
            b.insert(at, d);
            b.truncate(n);
        };
        let mut quads = alive.chunks_exact(4);
        for c in &mut quads {
            let rows = [0, 1, 2, 3].map(|j| self.store.row(c[j] as usize));
            for (i, s) in block.iter().enumerate() {
                for d in self.metric.distance_n(&s.q, rows) {
                    add(&mut best[i], keep[i], d);
                }
            }
        }
        for &u in quads.remainder() {
            for (i, s) in block.iter().enumerate() {
                add(&mut best[i], keep[i], self.metric.distance(&s.q, self.store.row(u as usize)));
            }
        }
        let nth = |b: &[f32], n: usize| if b.is_empty() { f32::INFINITY } else { b[n.clamp(1, b.len()) - 1] };
        block
            .iter()
            .zip(&best)
            .map(|(s, b)| {
                let (kth, tenth) = (nth(b, s.k), nth(b, 10));
                let mut seen: Vec<u32> = Vec::with_capacity(s.k);
                let mut correct = 0u32;
                for &id in s.found.iter().take(s.k) {
                    let Some(&u) = (id >= 0).then(|| self.slot_of.get(&(id as u64))).flatten() else { continue };
                    if seen.contains(&u) {
                        continue;
                    }
                    seen.push(u);
                    if self.metric.distance(&s.q, self.store.row(u as usize)) <= kth + effort::TIE_EPS {
                        correct += 1;
                    }
                }
                Checked { q: s.q.clone(), effort: s.effort, k: s.k, correct, kth, tenth }
            })
            .collect()
    }

    /// The usage counters with the numbers worth reading worked out.
    pub fn usage_report(&self) -> UsageReport {
        self.usage().report(self.item_gap())
    }

    /// The items' own median angle distance to their nearest other item (see `item_gap`); worked out once,
    /// on a sample of 256 items, when the build did not already find it.
    pub fn item_gap(&self) -> Option<f32> {
        if self.n_used < 2 {
            return None;
        }
        Some(*self.item_gap.get_or_init(|| {
            let alive: Vec<u32> = (0..self.n_used as u32).filter(|&u| self.graph.is_alive(u as usize)).collect();
            qlinks::items_gap(&self.store, &alive, 5).0
        }))
    }

    pub fn stats(&self) -> Stats {
        let n = self.n_used;
        let alive: Vec<usize> = (0..n).filter(|&u| self.graph.is_alive(u)).collect();
        let links: usize = alive.iter().map(|&u| self.graph.links(u).len()).sum();
        let indeg = self.graph.in_degrees(n);
        let no_in = alive.iter().filter(|&&u| indeg[u] == 0 && u as u32 != self.start).count();
        // reachable from the start
        let mut seen = vec![false; self.graph.capacity()];
        let mut stack = vec![self.start];
        if n > 0 {
            seen[self.start as usize] = true;
        }
        while let Some(u) = stack.pop() {
            for &v in self.graph.links(u as usize) {
                if !seen[v as usize] && self.graph.is_alive(v as usize) {
                    seen[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        Stats {
            items: alive.len(),
            start: self.start,
            build_checks: self.build_checks,
            change_checks: self.change_checks,
            mean_links: links as f64 / alive.len().max(1) as f64,
            no_links_in: no_in,
            unreachable: alive.iter().filter(|&&u| !seen[u]).count(),
            build_space: if self.flipped.is_some() { "flipped" } else { "same" },
            ood_ratio: self.ood_ratio,
            question_links: self.train_q.is_some(),
            question_link_items: self.question_link_items,
            popular_items: self.graph.extra_n.iter().filter(|&&k| k > 0).count(),
            extra_links: self.graph.extra_n.iter().map(|&k| k as usize).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    #[test]
    fn usage_counts_what_searches_returned() {
        let d = 16;
        let (x, tq, q) = (data(3000, d, 1), data(500, d, 2), data(70, d, 3));
        for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
            let s = Settings { r: 12, l_build: 40, effort_questions: 400, ..Settings::default() };
            let mut a = Index::new(d, metric, s).unwrap();
            a.build(&x, &(0..3000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
            assert_eq!(a.usage(), Usage::default(), "the build's own searches are not counted");
            let r1 = a.search(&q, 10, Effort::Target(0.95)).unwrap(); // spread over the pool
            let r2 = a.search_here(&q[..d * 9], 10, Effort::List(80)).unwrap(); // one thread
            let u = a.usage();
            assert_eq!((u.calls, u.questions, u.fixed), (2, 79, 9));
            assert_eq!(u.checks, r1.checks.iter().chain(&r2.checks).sum::<u64>());
            // a list of 80 holds at least 50 to re-check; the learned effort's shorter lists may hold fewer
            assert!(u.rechecks > 9 * 50 && u.rechecks <= 79 * 50);
            assert_eq!(u.bins.iter().sum::<u64>(), 70);
            assert_eq!(u.time.iter().sum::<u64>(), 79);
            assert_eq!(u.gap.iter().sum::<u64>(), 79);
            let r = a.usage_report();
            assert!(r.unlike_items.unwrap() > 0.0 && r.typical_ms.unwrap() > 0.0);
            a.reset_usage();
            assert_eq!(a.usage(), Usage::default());
            a.search_here(&q[..d * 9], 10, Effort::List(80)).unwrap();
            assert_eq!(a.usage().rechecks, 9 * 50, "a list of 80 re-checks its best 50");
            a.reset_usage();
            a.s.track_usage = false;
            a.search(&q, 10, Effort::List(30)).unwrap();
            assert_eq!(a.usage(), Usage::default(), "nothing is counted with track_usage off");
        }
    }

    #[test]
    fn audit_matches_a_brute_force_count() {
        let d = 16;
        let (x, tq, q) = (data(3000, d, 4), data(500, d, 5), data(60, d, 6));
        for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
            // a budget this large keeps every question of every call
            let s = Settings { r: 8, l_build: 20, effort_questions: 400, audit_budget: 1e12, ..Settings::default() };
            let mut a = Index::new(d, metric, s).unwrap();
            a.build(&x, &(0..3000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
            let r = a.search(&q, 10, Effort::List(12)).unwrap();
            assert_eq!(a.spot_report().pending, 60);
            let rep = a.audit();
            assert_eq!(rep.pending, 0);
            assert_eq!(rep.checks, 60 * 3000);
            // the same count, done here by sorting every item's distance
            let mut want = 0usize;
            for i in 0..60 {
                let mut qi = q[i * d..(i + 1) * d].to_vec();
                if metric == Metric::Cos {
                    normalize(&mut qi);
                }
                let mut all: Vec<f32> = (0..3000).map(|u| metric.distance(&qi, a.store.row(u))).collect();
                all.sort_by(|a, b| a.total_cmp(b));
                let tenth = all[9];
                want += r.ids[i * 10..(i + 1) * 10]
                    .iter()
                    .filter(|&&id| id >= 0 && metric.distance(&qi, a.store.row(a.slot_of[&(id as u64)] as usize)) <= tenth + effort::TIE_EPS)
                    .count();
            }
            let e = &rep.estimates[0];
            assert_eq!((e.label.as_str(), e.questions), ("list 12", 60));
            assert!((e.accuracy - want as f64 / 600.0).abs() < 1e-12, "{metric:?}: {} vs {}", e.accuracy, want as f64 / 600.0);
            assert_eq!(a.checked_questions().len(), 60);
        }
        // the default budget: spot checks cost about 2% of the checks searches make
        let mut b = Index::new(d, Metric::Cos, Settings { r: 8, l_build: 20, effort_questions: 400, ..Settings::default() }).unwrap();
        b.build(&x, &(0..3000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
        for _ in 0..200 {
            b.search(&q, 10, Effort::List(40)).unwrap();
        }
        let rep = b.audit();
        let share = rep.checks as f64 / rep.search_checks as f64;
        assert!(share <= 0.02 && share > 0.015, "spot checks used {share:.4} of the search checks");
    }

    #[test]
    fn effort_is_learned_again_from_checked_questions() {
        let d = 16;
        let (x, tq) = (data(4000, d, 7), data(1200, d, 8));
        let s = Settings { r: 10, l_build: 30, effort_questions: 600, audit_budget: 1e12, relearn_every: 0, ..Settings::default() };
        let a = {
            let mut a = Index::new(d, Metric::L2, s).unwrap();
            a.build(&x, &(0..4000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
            a
        };
        let before = a.learned().unwrap();
        // too few checked questions: nothing changes
        a.search(&data(100, d, 9), 10, Effort::Target(0.95)).unwrap();
        a.audit();
        assert_eq!(a.relearn_effort(), None);
        assert_eq!(a.learned().unwrap(), before);
        // enough: the table is learned from them, and the target's counts start again
        a.search(&data(500, d, 10), 10, Effort::Target(0.95)).unwrap();
        a.audit();
        assert_eq!(a.relearn_effort(), Some(600));
        let after = a.learned().unwrap();
        assert_eq!(after.questions, 600);
        assert_ne!(after.cuts, before.cuts);
        assert!(a.spot_report().estimates.iter().all(|e| !e.label.starts_with("target")));
        assert_eq!(a.checked_questions().len(), 600, "the checked questions are kept");
    }

    #[test]
    fn effort_apart_keeps_the_tables_questions_away_from_question_links() {
        let d = 16;
        let (x, tq) = (data(3000, d, 11), data(1000, d, 12));
        let mut tables = Vec::new();
        for apart in [false, true] {
            let s = Settings { r: 10, l_build: 30, effort_questions: 300, question_links: Switch::On, effort_apart: apart, ..Settings::default() };
            let mut a = Index::new(d, Metric::Cos, s).unwrap();
            a.build(&x, &(0..3000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
            // the links keep the questions after the table's 300 when apart, all 1,000 otherwise
            assert_eq!(a.train_q.as_ref().unwrap().len() / d, if apart { 700 } else { 1000 });
            tables.push(a.learned().unwrap().questions);
        }
        assert_eq!(tables, vec![300, 300]);
    }

    #[test]
    fn popular_items_get_extra_links_that_survive_save_renumbering_and_deletes() {
        // best score, with questions leaning one way: a few long items are everyone's answers
        let d = 12;
        let n = 3000;
        let mut x = data(n, d, 21);
        for u in 0..40 {
            for j in 0..d {
                x[u * d + j] *= 6.0;
            }
        }
        let mut tq = data(1200, d, 22);
        for v in tq.iter_mut() {
            *v += 0.3;
        }
        let ids: Vec<u64> = (0..n as u64).collect();
        let base = Settings { r: 12, l_build: 40, effort_questions: 300, question_links: Switch::On, tune: Tune::Off, ..Settings::default() };
        let mut on = Index::new(d, Metric::Ip, Settings { popular_links: Switch::Auto, ..base.clone() }).unwrap();
        on.build(&x, &ids, Some(&tq)).unwrap();
        let st = on.stats();
        assert!(st.popular_items > 0 && st.popular_items < n / 4, "answers bunch up: {} popular items", st.popular_items);
        for u in 0..n {
            let ex = on.graph.extra_links(u);
            assert!(ex.len() <= POPULAR_MAX && ex.iter().all(|&v| (v as usize) < n && v as usize != u));
            assert!(ex.iter().all(|v| !on.graph.links(u).contains(v)), "extra links leave out the item's own links");
        }
        let mut off = Index::new(d, Metric::Ip, Settings { popular_links: Switch::Off, ..base.clone() }).unwrap();
        off.build(&x, &ids, Some(&tq)).unwrap();
        assert_eq!(off.stats().popular_items, 0);
        // items that do not bunch up get none
        let mut flat = Index::new(d, Metric::Cos, Settings { popular_links: Switch::Auto, ..base.clone() }).unwrap();
        flat.build(&data(n, d, 23), &ids, Some(&data(1200, d, 24))).unwrap();
        assert_eq!(flat.stats().popular_items, 0);
        // the same answers after save and load, and after renumbering (the slots differ, the answers do not)
        let q = data(60, d, 25);
        let before = on.search(&q, 10, Effort::List(30)).unwrap();
        let dir = std::env::temp_dir().join(format!("vindex-test-popular-{}", std::process::id()));
        on.save(&dir).unwrap();
        let back = Index::load(&dir, crate::persist::Numbers::Ram).unwrap();
        let after = back.search(&q, 10, Effort::List(30)).unwrap();
        assert_eq!((before.ids, before.dists, before.checks), (after.ids, after.dists, after.checks));
        assert_eq!(back.stats().popular_items, st.popular_items);
        std::fs::remove_dir_all(&dir).ok();
        let mut plain = Index::new(d, Metric::Ip, Settings { renumber: false, ..base.clone() }).unwrap();
        plain.build(&x, &ids, Some(&tq)).unwrap();
        let p = plain.search(&q, 10, Effort::List(30)).unwrap();
        let r = on.search(&q, 10, Effort::List(30)).unwrap();
        assert_eq!((p.ids, p.dists, p.checks), (r.ids, r.dists, r.checks));
        // deletes: once freed slots are recycled, no extra link leads to a deleted item, and a reused slot has none
        for id in 0..200u64 {
            on.delete(id).unwrap();
        }
        on.s.cleanup_batch = 1;
        on.add(1_000_000, &x[..d]).unwrap();
        for u in 0..on.n_used {
            assert!(on.graph.extra_links(u).iter().all(|&v| on.graph.is_alive(v as usize)));
        }
        let u = on.slot_of[&1_000_000] as usize;
        assert!(on.graph.extra_links(u).is_empty());
    }
}
