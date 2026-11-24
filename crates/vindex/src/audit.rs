//! Spot checks (part 2, DESIGN.md section 14): now and then a searched question is kept with the answers it
//! got; `Index::audit` later finds its true answers by checking every item, and the results add up to an
//! accuracy estimate with a margin of error.
//!
//! How often: each search call earns credit, `audit_budget` times the checks it made; a spot check costs one
//! check per item in the index. So spot checks cost about `audit_budget` of the checks searches make, whatever
//! the size of the index.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::index::Effort;

/// Checked questions kept, newest last, with their true answers (the effort table can be learned again from them).
pub const WINDOW: usize = 2000;
/// Learning the effort table again needs at least this many spot-checked questions.
pub const RELEARN_MIN: usize = 500;
/// Questions waiting for `audit` are capped at this many bytes of numbers; more are not kept.
const PENDING_BYTES: usize = 16 << 20;

/// A searched question kept for a spot check: the question as searched, and the ids it got.
pub struct Sample {
    pub q: Vec<f32>,
    pub k: usize,
    pub effort: Effort,
    pub found: Vec<i64>,
}

/// A spot-checked question.
#[derive(Clone, Debug)]
pub struct Checked {
    pub q: Vec<f32>,
    pub effort: Effort,
    pub k: usize,
    /// How many of the k answers were really among the k closest (ties count).
    pub correct: u32,
    /// The true k-th and 10th distances.
    pub kth: f32,
    pub tenth: f32,
}

/// Sums for one kind of search (a target or a list size).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Sums {
    pub questions: u64,
    /// Of each question's accuracy (correct / k), and of its square.
    pub sum: f64,
    pub sum_sq: f64,
    /// Answers asked for, and how many were right.
    pub answers: u64,
    pub correct: u64,
}

/// One kind of search's estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    pub label: String,
    pub questions: u64,
    pub accuracy: f64,
    /// Half-width of a 95% interval: the real accuracy of these searches is in `accuracy ± margin` about 19 times
    /// in 20. None below two questions.
    pub margin: Option<f64>,
}

/// The spot checks' state; it sits next to the usage counters, under the same lock.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SpotChecks {
    pub credit: f64,
    pub pick: u64,
    pub sums: BTreeMap<String, Sums>,
    /// Checks spent on spot checks.
    pub checks: u64,
    /// Questions not kept because the waiting list was full.
    pub dropped: u64,
    /// Questions checked since the effort table was last learned.
    #[serde(default)]
    pub since_learn: u64,
    #[serde(skip)]
    pub pending: Vec<Sample>,
    #[serde(skip)]
    pub window: VecDeque<Checked>,
}

impl std::fmt::Debug for Sample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sample({:?}, k {})", self.effort, self.k)
    }
}

pub fn label(e: Effort) -> String {
    match e {
        Effort::Target(t) => format!("target {t}"),
        Effort::List(l) => format!("list {l}"),
    }
}

impl SpotChecks {
    /// After a search call of `nq` questions that made `checks` checks: earn credit, and while there is enough for
    /// a spot check (one check per item), keep one of the call's questions, each at most once.
    #[allow(clippy::too_many_arguments)]
    pub fn offer(&mut self, budget: f64, checks: u64, items: usize, dim: usize, k: usize, effort: Effort,
                 nq: usize, question: impl Fn(usize) -> Vec<f32>, found: &[i64]) {
        let cost = items.max(1) as f64;
        let cap = (PENDING_BYTES / (dim * 4)).max(1);
        self.credit += budget * checks as f64;
        let mut taken: Vec<usize> = Vec::new();
        while self.credit >= cost && taken.len() < nq {
            if self.pending.len() >= cap {
                self.dropped += 1;
                self.credit = self.credit.min(cost); // nobody is running audit(): do not hoard credit
                break;
            }
            let mut j = (self.pick % nq as u64) as usize;
            while taken.contains(&j) {
                j = (j + 1) % nq;
            }
            taken.push(j);
            self.pick = self.pick.wrapping_add(7919);
            self.pending.push(Sample { q: question(j), k, effort, found: found[j * k..(j + 1) * k].to_vec() });
            self.credit -= cost;
        }
    }

    pub fn add(&mut self, c: Checked) {
        let s = self.sums.entry(label(c.effort)).or_default();
        let a = c.correct as f64 / c.k.max(1) as f64;
        s.questions += 1;
        s.sum += a;
        s.sum_sq += a * a;
        s.answers += c.k as u64;
        s.correct += c.correct as u64;
        self.window.push_back(c);
        if self.window.len() > WINDOW {
            self.window.pop_front();
        }
    }

    /// One estimate per kind of search. The margin uses the spread of every question of that kind spot-checked so
    /// far (the window), not just those counted since the last reset: a few very wrong questions carry much of the
    /// error, and a small group often misses them.
    pub fn estimates(&self) -> Vec<Estimate> {
        self.sums
            .iter()
            .map(|(l, s)| {
                let seen: Vec<f64> = self
                    .window
                    .iter()
                    .filter(|c| label(c.effort) == *l)
                    .map(|c| c.correct as f64 / c.k.max(1) as f64)
                    .collect();
                estimate(l, s, &seen)
            })
            .collect()
    }

    /// Whether a target's estimate falls below it by more than its margin. Above target is left alone: it can be
    /// the table's floor (every group already at the smallest list), which learning again cannot lower.
    pub fn target_missed(&self) -> bool {
        self.estimates().iter().any(|e| {
            let target = e.label.strip_prefix("target ").and_then(|t| t.parse::<f64>().ok());
            matches!((target, e.margin), (Some(t), Some(m)) if t - e.accuracy > m)
        })
    }

    /// Forget the counts (the checked questions in the window are kept: they are data, not counts).
    pub fn reset(&mut self) {
        self.credit = 0.0;
        self.sums.clear();
        self.checks = 0;
        self.dropped = 0;
        self.pending.clear();
    }
}

/// Mean of the per-question accuracies, with a 95% margin from their spread: the spread of `seen` (every question
/// of this kind checked so far) when there are two or more, else of the counted questions themselves. It is never
/// taken as smaller than if each answer were right or wrong on its own, so a run of perfect questions still gets a
/// margin.
pub fn estimate(label: &str, s: &Sums, seen: &[f64]) -> Estimate {
    let m = s.questions as f64;
    let mean = s.sum / m.max(1.0);
    let margin = (s.questions >= 2).then(|| {
        let spread = if seen.len() >= 2 {
            let n = seen.len() as f64;
            let avg = seen.iter().sum::<f64>() / n;
            seen.iter().map(|v| (v - avg) * (v - avg)).sum::<f64>() / (n - 1.0)
        } else {
            ((s.sum_sq - s.sum * s.sum / m) / (m - 1.0)).max(0.0)
        };
        let p = (s.correct as f64 + 1.0) / (s.answers as f64 + 2.0);
        let per_answer = p * (1.0 - p) / (s.answers as f64 / m);
        let t = 1.96 + 2.5 / (m - 1.0); // Student's t at 97.5%, closely enough for m >= 2
        t * (spread.max(per_answer) / m).sqrt()
    });
    Estimate { label: label.to_string(), questions: s.questions, accuracy: mean, margin }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checked(correct: u32) -> Checked {
        Checked { q: vec![0.0], effort: Effort::Target(0.95), k: 10, correct, kth: 1.0, tenth: 1.0 }
    }

    #[test]
    fn credit_keeps_about_budget_times_checks() {
        let mut s = SpotChecks::default();
        let found = vec![0i64; 10 * 100];
        // 1,000 calls of 100 questions, 500 checks each; 10,000 items; budget 2% -> 1,000,000 credit -> 100 kept
        for _ in 0..1000 {
            s.offer(0.02, 100 * 500, 10_000, 4, 10, Effort::Target(0.95), 100, |_| vec![0.0; 4], &found);
        }
        assert_eq!(s.pending.len(), 100);
        // a huge budget keeps every question of a call, each once
        let mut all = SpotChecks::default();
        all.offer(1e9, 1, 10, 4, 10, Effort::List(50), 100, |j| vec![j as f32; 4], &found);
        let mut seen: Vec<usize> = all.pending.iter().map(|p| p.q[0] as usize).collect();
        seen.sort();
        assert_eq!(seen, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn estimates_and_margins() {
        let mut s = SpotChecks::default();
        for i in 0..100 {
            s.add(checked(if i % 4 == 0 { 8 } else { 10 }));
        }
        let e = &s.estimates()[0];
        assert_eq!((e.label.as_str(), e.questions), ("target 0.95", 100));
        assert!((e.accuracy - 0.95).abs() < 1e-12);
        // per-question spread: values 0.8 and 1.0, a quarter at 0.8 -> sd 0.0870; t ~ 1.985 -> margin ~ 0.0173
        let m = e.margin.unwrap();
        assert!((m - 0.0173).abs() < 0.001, "margin {m}");
        // all perfect still has a margin
        let mut p = SpotChecks::default();
        (0..50).for_each(|_| p.add(checked(10)));
        assert!(p.estimates()[0].margin.unwrap() > 0.0);
        assert_eq!(s.window.len(), 100);
    }
}
