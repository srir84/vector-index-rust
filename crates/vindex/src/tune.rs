//! Picking settings at build (part 2, DESIGN.md sections 16 and 19): try a few choices of links per item, slack and
//! copy size on a small part of the data, keep the fastest at the target accuracy, then build the whole index with it.
//!
//! The trial: a random fifth of the items (at least 20,000, at most 100,000), built with the build's own sample
//! questions except the last ones (at most 1,000, and at most a quarter), which are kept back to time it. Each
//! choice is timed on one core at the target, read off its learned effort's accuracy curve. Links per item are
//! tried first, then slack at the chosen links; each round's choices are timed together, and the plainest of those
//! within 2% of the fastest is kept (fewer links; the larger slack), so fresh builds pick alike (part 5). The pick replaces the current settings
//! only when more than 5% faster: a 20% trial over-rates small changes. Close calls are timed longer, and when new
//! links lose that last test, slack is tried again at the links in use.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::effort::TIE_EPS;
use crate::index::{Effort, Index, Settings, Switch, Tune};
use crate::metric::{normalize, Metric};
use crate::progress::Progress;
use crate::rng::Rng;

/// Links per item tried, and slack (alpha) tried at the chosen links.
pub const R_CHOICES: [usize; 5] = [24, 32, 48, 64, 96];
pub const ALPHA_CHOICES: [f32; 4] = [1.0, 1.1, 1.2, 1.3];
/// A choice must be this much faster in its trial to replace the current one.
pub const CLEAR_WIN: f64 = 1.05;
/// Tuning runs only with at least this many items and sample questions.
pub const MIN_ITEMS: usize = 10_000;
pub const MIN_QUESTIONS: usize = 800;
/// Targets searched to find where the trial reaches the tuning target.
const TARGETS: [f64; 11] = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995];

/// One choice tried.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Trial {
    pub r: usize,
    pub alpha: f32,
    /// Bits per number in the copies (8, or 4 with 16-bit questions).
    #[serde(default = "eight")]
    pub code_bits: u8,
    /// How many of the best the search re-checks with the full numbers.
    #[serde(default = "fifty")]
    pub rerank: usize,
    /// The question-link switch it was built with ("auto", "on" or "off"), and whether its links ended up on.
    pub links: String,
    pub question_links: bool,
    /// Searches per second on one core at the target (None: the trial never reached it).
    pub searches_per_s: Option<f64>,
    /// Checks per search at the target.
    pub checks: Option<f64>,
    /// Bytes per item kept in memory with this choice (full numbers not counted when the 1-byte copies are on).
    pub bytes_per_item: f64,
    /// Seconds to build and time this trial.
    pub seconds: f64,
}

/// What a trial search did: every choice tried, and the one kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tuning {
    pub trial_items: usize,
    pub timing_questions: usize,
    pub target: f64,
    pub trials: Vec<Trial>,
    pub chosen_r: usize,
    pub chosen_alpha: f32,
    #[serde(default = "eight")]
    pub chosen_code_bits: u8,
    #[serde(default = "fifty")]
    pub chosen_rerank: usize,
    /// The kept choice's place in `trials` (None: no trial reached the target, so nothing was changed).
    pub chosen: Option<usize>,
    /// Seconds spent, before the real build.
    pub seconds: f64,
}

fn eight() -> u8 {
    8
}

fn fifty() -> usize {
    50
}

/// Memory per item for a choice, counted as `Index::memory` does.
pub fn bytes_per_item(s: &Settings, dim: usize) -> f64 {
    let links = s.slots() * 4 + 4;
    let numbers = if s.code_bits == 0 { dim * 4 } else { crate::codes::Codes::row_bytes_for(dim, s.code_bits) };
    (links + numbers + 8) as f64
}

/// How many items a trial takes: a fifth, at least 20,000 (all of them when fewer), at most 100,000.
pub fn trial_size(n: usize) -> usize {
    n.min((n / 5).max(20_000)).min(100_000)
}

/// Pick links per item and slack for these items and sample questions; None when there is too little data.
pub fn pick(base: &Settings, metric: Metric, dim: usize, x: &[f32], train_q: &[f32], progress: &Arc<Progress>) -> Option<Tuning> {
    let (n, nt) = (x.len() / dim, train_q.len() / dim);
    if n < MIN_ITEMS || nt < MIN_QUESTIONS {
        return None;
    }
    let mut rows = Rng::new(base.seed ^ 0x7475_6e65).sample(n, trial_size(n));
    rows.sort_unstable();
    let xs: Vec<f32> = rows.iter().flat_map(|&i| x[i * dim..(i + 1) * dim].iter().copied()).collect();
    let nm = (nt / 4).min(1000);
    Some(search(base, First::Settings(base), metric, dim, &xs, train_q, nm, progress))
}

fn switch_name(s: Switch) -> &'static str {
    match s {
        Switch::Auto => "auto",
        Switch::On => "on",
        Switch::Off => "off",
    }
}

/// Every choice is decided on the choices timed together (part 5, DESIGN.md section 16): this many rounds, the
/// choices taking turns within each round, each keeping its fastest round, so a stall or a slow drift on a busy
/// machine hits them all alike.
pub(crate) const ROUNDS: usize = 5;
/// Re-check counts tried after the build, on the real index (part 3, DESIGN.md section 20).
pub const RERANK_CHOICES: [usize; 3] = [15, 25, 50];
/// Choices within this of the fastest count as tied, and the plainest of them is kept: the one in use first. Ties
/// broken by timing noise made fresh builds pick differently, and some of those picks were 9-13% apart once
/// built in full (part 5, step 37).
pub(crate) const TIE: f64 = 1.02;
/// Each timed run lasts at least this long (the timing questions are repeated to fill it).
const MIN_RUN_S: f64 = 0.2;
const RUNS: usize = 1;
/// A close call, a choice within this of a bar (the 2% tie or the 5% clear win), is timed `MORE_ROUNDS` rounds more
/// before it is decided. Timed as a round times them, the same choice moved by up to 3% between fresh builds; three
/// identical indexes timed for 12 rounds of 0.5 s came within 0.3% of each other (part 5, step 37).
pub(crate) const CLOSE: f64 = 0.015;
pub(crate) const MORE_ROUNDS: usize = 20;

/// Whether a speed against its bar (1.0: exactly on it) is a close call.
pub(crate) fn close(ratio: f64) -> bool {
    (ratio - 1.0).abs() < CLOSE
}

/// Where a trial reaches the target: the two targets around it, the weight between them, and checks per search.
#[derive(Clone, Copy)]
pub(crate) struct Plan {
    lo: f64,
    hi: f64,
    w: f64,
    pub(crate) checks: f64,
}

/// What a trial search starts from and must beat.
pub(crate) enum First<'a> {
    /// A trial built with these settings (the build's own settings).
    Settings(&'a Settings),
    /// The index itself, timed as it is (a suggestion, when the trial items are all of its items).
    Index(&'a Index),
}

/// A trial kept for a head-to-head: one built here, or the index itself.
#[allow(clippy::large_enum_variant)] // at most two are alive at a time
enum Held<'a> {
    Own(Index),
    Borrowed(&'a Index),
}

impl Held<'_> {
    fn get(&self) -> &Index {
        match self {
            Held::Own(i) => i,
            Held::Borrowed(i) => i,
        }
    }
}

/// The trial items, questions, the trials so far, and the built trials still needed for a head-to-head.
struct Ctx<'a> {
    base: &'a Settings,
    metric: Metric,
    dim: usize,
    xs: &'a [f32],
    ids: Vec<u64>,
    build_q: &'a [f32],
    timing_q: &'a [f32],
    scaled: Vec<f32>,
    tenth: Option<Vec<f32>>,
    trials: Vec<Trial>,
    kept: Vec<(usize, Held<'a>, Plan)>,
    /// The build's progress, shared with each trial build.
    progress: Arc<Progress>,
}

impl<'a> Ctx<'a> {
    /// Time the index itself as the first trial: its items must be the trial items.
    fn try_existing(&mut self, idx: &'a Index) -> Option<usize> {
        let t0 = Instant::now();
        let td = self.tenth.get_or_insert_with(|| idx.tenth_distances(&self.scaled));
        let plan = plan(idx, self.timing_q, td, self.base.tune_target);
        self.trials.push(Trial {
            r: idx.s.r,
            alpha: idx.s.alpha,
            code_bits: idx.s.code_bits,
            rerank: idx.s.rerank,
            links: "as built".to_string(),
            question_links: idx.train_q.is_some(),
            searches_per_s: plan.map(|p| 1.0 / time_plan(idx, self.timing_q, &p)),
            checks: plan.map(|p| p.checks),
            bytes_per_item: idx.memory().total as f64 / idx.n_used.max(1) as f64,
            seconds: t0.elapsed().as_secs_f64(),
        });
        let i = self.trials.len() - 1;
        if let Some(p) = plan {
            self.kept.push((i, Held::Borrowed(idx), p));
        }
        Some(i)
    }

    /// Build and time one choice (once: a choice tried before is looked up); returns its place in `trials`, or None
    /// when it is over the memory limit or fails to build.
    fn try_one(&mut self, s: &Settings) -> Option<usize> {
        let links = switch_name(s.question_links);
        if let Some(i) = self.trials.iter().position(|t| t.r == s.r && t.alpha == s.alpha && t.code_bits == s.code_bits && t.links == links) {
            return Some(i);
        }
        let t0 = Instant::now();
        let s = Settings { tune: Tune::Off, track_usage: false, audit_budget: 0.0, relearn_every: 0, ..s.clone() };
        let bytes = bytes_per_item(&s, self.dim);
        if self.base.max_bytes_per_item > 0 && bytes > self.base.max_bytes_per_item as f64 {
            return None;
        }
        let mut idx = Index::new(self.dim, self.metric, s.clone()).ok()?;
        self.progress.phase(format!("trial {}: {} links per item, slack {}", self.trials.len() + 1, s.r, s.alpha));
        idx.progress = self.progress.clone();
        idx.build(self.xs, &self.ids, Some(self.build_q)).ok()?;
        self.progress.step("timing the trial", 0);
        let td = self.tenth.get_or_insert_with(|| idx.tenth_distances(&self.scaled));
        let plan = plan(&idx, self.timing_q, td, self.base.tune_target);
        let sps = plan.map(|p| 1.0 / time_plan(&idx, self.timing_q, &p));
        self.trials.push(Trial {
            r: s.r,
            alpha: s.alpha,
            code_bits: s.code_bits,
            rerank: s.rerank,
            links: links.to_string(),
            question_links: idx.train_q.is_some(),
            searches_per_s: sps,
            checks: plan.map(|p| p.checks),
            bytes_per_item: bytes,
            seconds: t0.elapsed().as_secs_f64(),
        });
        let i = self.trials.len() - 1;
        if let Some(p) = plan {
            self.kept.push((i, Held::Own(idx), p));
        }
        Some(i)
    }

    fn speed(&self, i: Option<usize>) -> Option<f64> {
        i.and_then(|i| self.trials[i].searches_per_s)
    }

    /// Time these trials again, `ROUNDS` rounds taking turns, and record the fastest round of each.
    fn time_together(&mut self, who: &[usize]) {
        self.time_rounds(who, ROUNDS, false);
    }

    /// Time these trials `rounds` rounds taking turns, and record the fastest round of each; with `more`, counting
    /// their rounds before too (a close call timed longer).
    fn time_rounds(&mut self, who: &[usize], rounds: usize, more: bool) {
        let mut t: Vec<f64> = who
            .iter()
            .map(|&i| match self.trials[i].searches_per_s {
                Some(s) if more => 1.0 / s,
                _ => f64::INFINITY,
            })
            .collect();
        for _ in 0..rounds {
            for (j, &i) in who.iter().enumerate() {
                if let Some((_, held, p)) = self.kept.iter().find(|k| k.0 == i) {
                    t[j] = t[j].min(time_plan(held.get(), self.timing_q, p));
                }
            }
        }
        for (&i, &tj) in who.iter().zip(&t) {
            if tj.is_finite() {
                self.trials[i].searches_per_s = Some(1.0 / tj);
            }
        }
    }

    /// The plainest of `round` (listed plainest first) within `TIE` of the fastest; None when none reaches the target.
    /// The choices close to the tie bar are timed longer first, with the fastest.
    fn plainest(&mut self, round: &[usize]) -> Option<usize> {
        let fastest = |c: &Self| round.iter().filter_map(|&i| c.speed(Some(i))).reduce(f64::max);
        let f = fastest(self)?;
        let mut near: Vec<usize> = round.iter().copied().filter(|&i| self.speed(Some(i)).is_some_and(|s| close(s * TIE / f))).collect();
        if !near.is_empty() {
            if let Some(top) = round.iter().copied().find(|&i| self.speed(Some(i)) == Some(f)) {
                near.push(top);
            }
            near.retain(|i| self.kept.iter().any(|k| k.0 == *i));
            self.time_rounds(&near, MORE_ROUNDS, true);
        }
        let f = fastest(self)?;
        round.iter().copied().find(|&i| self.speed(Some(i)).is_some_and(|s| s * TIE >= f))
    }

    /// Whether `top` replaces `start`: more than `CLEAR_WIN` faster, timed again head-to-head (longer when close).
    fn clear_win(&mut self, start: Option<usize>, top: usize) -> bool {
        if let Some(a) = start {
            self.time_together(&[a, top]);
            if let (Some(x), Some(y)) = (self.speed(Some(top)), self.speed(start)) {
                if close(x / (CLEAR_WIN * y)) {
                    self.time_rounds(&[a, top], MORE_ROUNDS, true);
                }
            }
        }
        match (self.speed(Some(top)), self.speed(start)) {
            (Some(x), Some(y)) => x > CLEAR_WIN * y,
            (Some(_), None) => true,
            _ => false,
        }
    }

    /// A round: try each choice (listed plainest first: the one in use, then the rest), time them together, and keep
    /// the plainest of the tied fastest. Only the start and the round's pick stay built.
    fn round(&mut self, start: Option<usize>, first: Option<usize>, choices: &[Settings]) -> Option<usize> {
        let mut round: Vec<usize> = first.into_iter().collect();
        for s in choices {
            if let Some(i) = self.try_one(s) {
                if !round.contains(&i) {
                    round.push(i);
                }
            }
        }
        let timed: Vec<usize> = round.iter().copied().filter(|i| self.kept.iter().any(|k| k.0 == *i)).collect();
        self.time_together(&timed);
        let pick = self.plainest(&round).or(first);
        self.kept.retain(|k| Some(k.0) == start || Some(k.0) == pick);
        pick
    }
}

/// The trial search on trial items `xs`, with sample questions `train_q` whose last `nm` are kept back for timing
/// (the build: at most 1,000, and at most a quarter). `first` is timed first and is the one to beat: the build's own
/// settings, or for a suggestion the index itself (or, when it is bigger than the trial, a trial of it as it is).
/// Then, starting from `base`: links per item at `base`'s slack, then slack at the fastest links. The fastest trial
/// of all replaces the start only when more than 5% faster, timed again head-to-head; when new links lose that,
/// slack is tried again at the start's links, and that pick faces the same test. (Copy size is chosen after
/// the build, on the real index: a trial small enough to sit in the chip's caches cannot see what smaller copies
/// save in memory reads.)
#[allow(clippy::too_many_arguments)]
pub(crate) fn search<'a>(
    base: &'a Settings,
    first: First<'a>,
    metric: Metric,
    dim: usize,
    xs: &'a [f32],
    train_q: &'a [f32],
    nm: usize,
    progress: &Arc<Progress>,
) -> Tuning {
    let started = Instant::now();
    let nt = train_q.len() / dim;
    let (build_q, timing_q) = train_q.split_at((nt - nm) * dim);
    let m = xs.len() / dim;
    let mut scaled = timing_q.to_vec();
    if metric == Metric::Cos {
        scaled.chunks_mut(dim).for_each(normalize);
    }
    let mut c = Ctx {
        base,
        metric,
        dim,
        xs,
        ids: (0..m as u64).collect(),
        build_q,
        timing_q,
        scaled,
        tenth: None,
        trials: Vec::new(),
        kept: Vec::new(),
        progress: progress.clone(),
    };
    let start = match first {
        First::Settings(s) => c.try_one(s),
        First::Index(idx) => c.try_existing(idx),
    };
    // each round times its choices together and keeps the plainest of those tied with the fastest (within 2%), so
    // timing noise cannot decide; a clearer lead wins its round by any margin above that, since the trial
    // understates differences that grow at full size (GloVe: 96 links 5% ahead of 48 in the trial, 19% at full size)
    // round 1: links per item, at `base`'s slack: the trial of `base` first, then fewer links before more
    let first_r = c.try_one(base);
    let links: Vec<Settings> = R_CHOICES.iter().map(|&r| Settings { r, ..base.clone() }).collect();
    let lead = c.round(start, first_r, &links);
    // round 2: slack at the chosen links, the largest first among tied ones: it keeps more long links, which the
    // trial cannot see paying off (LAION's sample: 1.2 and 1.3 tied in the trial, 1.3 23% faster built in full)
    let r = lead.map_or(base.r, |i| c.trials[i].r);
    let mut alphas = ALPHA_CHOICES.to_vec();
    if !alphas.contains(&base.alpha) {
        alphas.push(base.alpha);
    }
    alphas.sort_by(|a, b| b.total_cmp(a));
    let slack = |r: usize| -> Vec<Settings> { alphas.iter().map(|&alpha| Settings { r, alpha, ..base.clone() }).collect() };
    let mut lead = c.round(start, None, &slack(r));
    // the pick replaces the start only if it is more than 5% faster, timed again head-to-head. When new links lose
    // there, slack was tried at links not kept, so it is tried again at the start's (ImageNet's sample: 48 links
    // passed the 2% tie by 0.04%, then lost to the start, and slack 1.3 at 32 links, 6% ahead, was never tried)
    let start_r = start.map_or(base.r, |i| c.trials[i].r);
    let mut best = start;
    for again in [false, true] {
        let Some(top) = lead.filter(|&t| Some(t) != start) else { break };
        if c.clear_win(start, top) {
            best = Some(top);
            break;
        }
        if again || c.trials[top].r == start_r {
            break;
        }
        lead = c.round(start, None, &slack(start_r));
    }
    let (chosen_r, chosen_alpha, chosen_code_bits) =
        best.map_or((base.r, base.alpha, base.code_bits), |i| (c.trials[i].r, c.trials[i].alpha, c.trials[i].code_bits));
    Tuning {
        trial_items: m,
        timing_questions: nm,
        target: base.tune_target,
        trials: c.trials,
        chosen_r,
        chosen_alpha,
        chosen_code_bits,
        chosen_rerank: base.rerank,
        chosen: best,
        seconds: started.elapsed().as_secs_f64(),
    }
}

/// Where the trial reaches `target` accuracy on the timing questions: its learned effort's accuracy at each target
/// (read as never going down, as the validation reads curves), and the two targets around the crossing. None when
/// it never gets there.
pub(crate) fn plan(idx: &Index, q: &[f32], tenth: &[f32], target: f64) -> Option<Plan> {
    let nq = tenth.len();
    let mut acc: Vec<f64> = Vec::with_capacity(TARGETS.len());
    let mut chk = Vec::with_capacity(TARGETS.len());
    for &t in TARGETS.iter() {
        let out = idx.search_quiet(q, 10, Effort::Target(t), true).ok()?;
        let correct: usize = out
            .dists
            .chunks(10)
            .zip(tenth)
            .map(|(ds, &tt)| ds.iter().filter(|&&d| d <= tt + TIE_EPS).count().min(10))
            .sum();
        let a = correct as f64 / (10 * nq) as f64;
        acc.push(acc.last().map_or(a, |&p| p.max(a)));
        chk.push(out.checks.iter().sum::<u64>() as f64 / nq as f64);
    }
    let hit = acc.iter().position(|&a| a >= target)?;
    if hit == 0 {
        return Some(Plan { lo: TARGETS[0], hi: TARGETS[0], w: 0.0, checks: chk[0] });
    }
    let (a0, a1) = (acc[hit - 1], acc[hit]);
    let w = if a1 > a0 { (target - a0) / (a1 - a0) } else { 1.0 };
    Some(Plan { lo: TARGETS[hit - 1], hi: TARGETS[hit], w, checks: chk[hit - 1] + w * (chk[hit] - chk[hit - 1]) })
}

/// One core: seconds per search at the plan's target, between the times at its two targets. Each is the fastest of
/// `RUNS` runs of at least `MIN_RUN_S`.
pub(crate) fn time_plan(idx: &Index, q: &[f32], p: &Plan) -> f64 {
    let nq = q.len() / idx.dim;
    let sec = |t: f64| -> f64 {
        let t0 = Instant::now();
        let _ = idx.search_quiet(q, 10, Effort::Target(t), false); // warm-up, and how long one pass takes
        let repeat = ((MIN_RUN_S / t0.elapsed().as_secs_f64().max(1e-6)).ceil() as usize).max(1);
        (0..RUNS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..repeat {
                    let _ = idx.search_quiet(q, 10, Effort::Target(t), false);
                }
                t0.elapsed().as_secs_f64() / repeat as f64
            })
            .fold(f64::INFINITY, f64::min)
            / nq as f64
    };
    let s0 = sec(p.lo);
    if p.hi == p.lo {
        return s0;
    }
    s0 + p.w * (sec(p.hi) - s0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    #[test]
    fn picks_from_its_choices_within_the_memory_limit() {
        let d = 8;
        let (x, tq) = (data(12_000, d, 1), data(1_000, d, 2));
        let s = Settings { l_build: 40, effort_questions: 400, ..Settings::default() };
        let t = pick(&s, Metric::L2, d, &x, &tq, &Arc::default()).unwrap();
        assert_eq!((t.trial_items, t.timing_questions), (12_000, 250));
        assert_eq!((t.trials[0].r, t.trials[0].alpha), (32, 1.2), "the current settings are timed first");
        // links first, then slack at the fastest links (and again at the start's when new links lose at the end)
        assert!([1 + 4 + 3, 1 + 4 + 3 + 3].contains(&t.trials.len()), "{} trials", t.trials.len());
        assert!(R_CHOICES.contains(&t.chosen_r) && ALPHA_CHOICES.contains(&t.chosen_alpha));
        assert!(t.trials.iter().all(|tr| tr.searches_per_s.is_some()));
        // a memory limit leaves out the choices that keep more
        let lim = Settings { max_bytes_per_item: bytes_per_item(&Settings { r: 48, ..s.clone() }, d) as usize, ..s.clone() };
        let t = pick(&lim, Metric::L2, d, &x, &tq, &Arc::default()).unwrap();
        assert!(t.trials.iter().all(|tr| tr.r <= 48));
        // too little data: nothing is tried
        assert!(pick(&s, Metric::L2, d, &x[..9_000 * d], &tq, &Arc::default()).is_none());
    }

    #[test]
    fn the_build_uses_what_the_tuning_chose() {
        let d = 8;
        let (x, tq) = (data(12_000, d, 3), data(1_000, d, 4));
        let mut a = Index::new(d, Metric::Cos, Settings { l_build: 40, effort_questions: 400, ..Settings::default() }).unwrap();
        a.build(&x, &(0..12_000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
        let t = a.tuning.clone().unwrap();
        assert_eq!((a.s.r, a.s.alpha), (t.chosen_r, t.chosen_alpha));
        assert_eq!(a.graph.slots, a.s.slots());
        // copy size is timed on the real index after the build, and the copies in use are the chosen ones
        let full: Vec<u8> = t.trials.iter().filter(|x| x.links == "full size").map(|x| x.code_bits).collect();
        assert_eq!(full, vec![8, 4]);
        assert_eq!((a.s.code_bits, a.codes.as_ref().unwrap().bits), (t.chosen_code_bits, t.chosen_code_bits));
        // then the re-check count, at the chosen copy size, and the search uses the chosen one
        let rr: Vec<usize> = t.trials.iter().filter(|x| x.links == "full size, re-check").map(|x| x.rerank).collect();
        assert_eq!(rr, vec![50, 25, 15], "the count in use first, then the nearest: ties go to the plainest");
        assert!(t.trials.iter().filter(|x| x.links == "full size, re-check").all(|x| x.code_bits == t.chosen_code_bits));
        assert_eq!(a.s.rerank, t.chosen_rerank);
        let off = Settings { tune: Tune::Off, l_build: 40, effort_questions: 400, ..Settings::default() };
        let mut b = Index::new(d, Metric::Cos, off).unwrap();
        b.build(&x, &(0..12_000u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
        assert!(b.tuning.is_none() && b.s.r == 32);
    }
}
