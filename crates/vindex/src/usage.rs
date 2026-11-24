//! Usage stats (part 2, DESIGN.md section 13): what searches have done since the build, or since the last reset.
//!
//! Each search call adds up its own tally and merges it into the index's once, under a lock, so the cost per
//! question is a clock read and a few additions. Times and distances go into histograms whose buckets are
//! 2^(1/4) apart, so a median or a 99th percentile read off them is within about 9%.

use serde::{Deserialize, Serialize};

use crate::effort::N_BINS;

/// Buckets per doubling, for both histograms.
const STEPS: f64 = 4.0;
/// Time per question: bucket 0 holds under 1 microsecond, then 2^(1/4) apart (the last one holds the rest).
pub const TIME_BUCKETS: usize = 96;
const TIME_FIRST_NS: f64 = 1_000.0;
/// A question's angle distance to its best answer (1 - cosine): bucket 0 holds under 1e-6, then 2^(1/4) apart.
pub const GAP_BUCKETS: usize = 96;
const GAP_FIRST: f64 = 1e-6;

/// The counters. Searches made inside the build (learning the effort, question links) are not counted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Search calls, and the questions they held.
    pub calls: u64,
    pub questions: u64,
    /// Checks made, and how many of them were re-checks with the full numbers.
    pub checks: u64,
    pub rechecks: u64,
    /// Time spent searching, summed over questions.
    pub nanos: u64,
    /// Questions searched with a fixed list size; the rest used the learned effort.
    pub fixed: u64,
    /// Learned-effort questions per hardness bin, hardest first. The build's sample questions fill each bin equally.
    pub bins: [u64; N_BINS],
    #[serde(with = "buckets")]
    pub time: [u64; TIME_BUCKETS],
    #[serde(with = "buckets")]
    pub gap: [u64; GAP_BUCKETS],
}

impl Default for Usage {
    fn default() -> Self {
        Usage {
            calls: 0,
            questions: 0,
            checks: 0,
            rechecks: 0,
            nanos: 0,
            fixed: 0,
            bins: [0; N_BINS],
            time: [0; TIME_BUCKETS],
            gap: [0; GAP_BUCKETS],
        }
    }
}

fn bucket(v: f64, first: f64, n: usize) -> usize {
    if !(v >= first) {
        return 0; // below the first edge, zero, or not a number
    }
    (((v / first).log2() * STEPS) as usize + 1).min(n - 1)
}

/// The value below which a share `q` of the counts lies, reading the histogram as even on a log scale inside
/// each bucket.
fn quantile(h: &[u64], first: f64, q: f64) -> Option<f64> {
    let total: u64 = h.iter().sum();
    if total == 0 {
        return None;
    }
    let want = q * total as f64;
    let mut acc = 0.0;
    for (i, &c) in h.iter().enumerate() {
        let next = acc + c as f64;
        if c > 0 && next >= want {
            let frac = ((want - acc) / c as f64).clamp(0.0, 1.0);
            if i == 0 {
                return Some(first * frac);
            }
            let low = first * 2f64.powf((i - 1) as f64 / STEPS);
            return Some(low * 2f64.powf(frac / STEPS));
        }
        acc = next;
    }
    None
}

impl Usage {
    /// Count one question: its checks, re-checks, time, hardness bin (learned effort only) and angle distance
    /// to its best answer (when it found one).
    #[inline]
    pub fn question(&mut self, checks: u64, rechecks: u64, nanos: u64, bin: Option<usize>, gap: Option<f32>) {
        self.questions += 1;
        self.checks += checks;
        self.rechecks += rechecks;
        self.nanos += nanos;
        match bin {
            Some(b) => self.bins[b.min(N_BINS - 1)] += 1,
            None => self.fixed += 1,
        }
        self.time[bucket(nanos as f64, TIME_FIRST_NS, TIME_BUCKETS)] += 1;
        if let Some(g) = gap {
            self.gap[bucket(g as f64, GAP_FIRST, GAP_BUCKETS)] += 1;
        }
    }

    pub fn merge(&mut self, o: &Usage) {
        self.calls += o.calls;
        self.questions += o.questions;
        self.checks += o.checks;
        self.rechecks += o.rechecks;
        self.nanos += o.nanos;
        self.fixed += o.fixed;
        self.bins.iter_mut().zip(&o.bins).for_each(|(a, b)| *a += b);
        self.time.iter_mut().zip(&o.time).for_each(|(a, b)| *a += b);
        self.gap.iter_mut().zip(&o.gap).for_each(|(a, b)| *a += b);
    }

    /// The numbers worth reading. `item_gap` is the items' own median angle distance to their nearest other item.
    pub fn report(&self, item_gap: Option<f32>) -> UsageReport {
        let q = self.questions.max(1) as f64;
        let learned = self.questions - self.fixed;
        let hardness: Vec<f64> = self.bins.iter().map(|&b| b as f64 / learned.max(1) as f64).collect();
        let median_gap = quantile(&self.gap, GAP_FIRST, 0.5);
        UsageReport {
            calls: self.calls,
            questions: self.questions,
            checks: self.checks,
            rechecks: self.rechecks,
            checks_per_question: self.checks as f64 / q,
            rechecks_per_question: self.rechecks as f64 / q,
            seconds: self.nanos as f64 * 1e-9,
            typical_ms: quantile(&self.time, TIME_FIRST_NS, 0.5).map(|ns| ns * 1e-6),
            slow_ms: quantile(&self.time, TIME_FIRST_NS, 0.99).map(|ns| ns * 1e-6),
            learned_questions: learned,
            fixed_questions: self.fixed,
            hard_share: (learned > 0).then(|| hardness[..2].iter().sum()),
            hardness,
            unlike_items: match (median_gap, item_gap) {
                (Some(m), Some(g)) if g > 0.0 => Some(m / g as f64),
                _ => None,
            },
        }
    }
}

/// What `Usage::report` gives.
#[derive(Clone, Debug, PartialEq)]
pub struct UsageReport {
    pub calls: u64,
    pub questions: u64,
    pub checks: u64,
    pub rechecks: u64,
    pub checks_per_question: f64,
    pub rechecks_per_question: f64,
    pub seconds: f64,
    /// Middle and 99th-percentile time per question.
    pub typical_ms: Option<f64>,
    pub slow_ms: Option<f64>,
    pub learned_questions: u64,
    pub fixed_questions: u64,
    /// Share of learned-effort questions per hardness bin, hardest first (the sample questions: 1/8 each).
    pub hardness: Vec<f64>,
    /// Share in the two hardest bins (the sample questions: 0.25).
    pub hard_share: Option<f64>,
    /// Median angle distance from a question to its best answer, over the items' own (the build's
    /// "questions unlike items" ratio, measured on real questions).
    pub unlike_items: Option<f64>,
}

/// Fixed-size bucket arrays as plain lists (serde reads arrays of up to 32 only).
mod buckets {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer, const N: usize>(a: &[u64; N], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(a.iter())
    }

    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<[u64; N], D::Error> {
        let v = Vec::<u64>::deserialize(d)?;
        <[u64; N]>::try_from(v.as_slice())
            .map_err(|_| serde::de::Error::custom(format!("expected {N} buckets, found {}", v.len())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_quantiles() {
        let mut u = Usage::default();
        for i in 0..1000u64 {
            u.question(100 + i % 3, 50, 10_000 + i * 100, Some((i % 8) as usize), Some(0.1 + (i as f32) * 1e-4));
        }
        u.question(7, 0, 5_000_000, None, None);
        let r = u.report(Some(0.1));
        assert_eq!((r.questions, r.learned_questions, r.fixed_questions), (1001, 1000, 1));
        assert_eq!(r.checks, (0..1000u64).map(|i| 100 + i % 3).sum::<u64>() + 7);
        assert!(r.hardness.iter().all(|&h| h == 0.125));
        // the middle time is about 60 microseconds; buckets are 19% wide
        let t = r.typical_ms.unwrap();
        assert!((t - 0.060).abs() < 0.06 * 0.2, "typical {t}");
        // the middle gap is about 0.15 against an item gap of 0.1
        let ui = r.unlike_items.unwrap();
        assert!((ui - 1.5).abs() < 1.5 * 0.2, "unlike_items {ui}");
    }

    #[test]
    fn merge_and_save() {
        let mut a = Usage::default();
        let mut b = Usage::default();
        a.question(10, 5, 2_000, Some(0), Some(0.5));
        b.question(20, 5, 3_000, None, Some(0.25));
        b.calls = 1;
        a.merge(&b);
        assert_eq!((a.questions, a.checks, a.calls, a.fixed), (2, 30, 1, 1));
        let text = serde_json::to_string(&a).unwrap();
        let back: Usage = serde_json::from_str(&text).unwrap();
        assert_eq!(a, back);
        assert!(serde_json::from_str::<Usage>(&text.replace("\"time\":[", "\"time\":[1,")).is_err());
    }
}
