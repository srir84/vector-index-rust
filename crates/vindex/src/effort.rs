//! The learned effort (Fix 1, better version; DESIGN.md section 5).
//!
//! Every search starts with a list of 10. When those are all opened, z = minus the distance of the 10th
//! alive item found so far tells how hard the question is; its group (one of 8, split at cut points learned
//! from training questions) picks how far the list grows. Per group and list size, the training questions'
//! correct answers and checks are kept, so a table for any target accuracy can be worked out at search time.

/// The first list size of every learned-effort search.
pub const L1: usize = 10;
/// Number of groups.
pub const N_BINS: usize = 8;
/// List sizes tried while learning.
pub const L_GRID: [usize; 18] = [10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000];
/// Learning stops after the first list size where the training questions reach this accuracy.
pub const ENOUGH: f64 = 0.995;
/// An answer counts if its distance is at most the true 10th distance plus this (the experiments' tie rule).
pub const TIE_EPS: f32 = 1e-3;

/// What was learned from the training questions.
#[derive(Clone, Debug, PartialEq)]
pub struct Learned {
    /// N_BINS - 1 cut points on z, ascending.
    pub cuts: Vec<f32>,
    /// The list sizes tried.
    pub levels: Vec<usize>,
    /// Total correct answers of the training questions in each group, per list size: `[level * N_BINS + bin]`.
    pub correct: Vec<f64>,
    /// Total checks, same layout.
    pub checks: Vec<f64>,
    /// How many training questions were used.
    pub questions: usize,
}

/// The group of a question with hardness `z`: how many cut points z is above.
#[inline]
pub fn bin_of(cuts: &[f32], z: f32) -> usize {
    let mut b = 0;
    while b < cuts.len() && z > cuts[b] {
        b += 1;
    }
    b
}

/// numpy's default (linear) quantile of sorted values.
pub fn quantile(sorted: &[f32], q: f64) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let w = pos - lo as f64;
    (sorted[lo] as f64 * (1.0 - w) + sorted[hi] as f64 * w) as f32
}

impl Learned {
    /// Start from per-question results: `z` and `bins` from the first stage, then for each list size,
    /// the correct answers and checks of every question.
    pub fn from_runs(cuts: Vec<f32>, bins: &[usize], levels: Vec<usize>, correct: &[Vec<u32>], checks: &[Vec<u64>]) -> Self {
        let mut c = vec![0f64; levels.len() * N_BINS];
        let mut k = vec![0f64; levels.len() * N_BINS];
        for a in 0..levels.len() {
            for (qi, &b) in bins.iter().enumerate() {
                c[a * N_BINS + b] += correct[a][qi] as f64;
                k[a * N_BINS + b] += checks[a][qi] as f64;
            }
        }
        Learned { cuts, levels, correct: c, checks: k, questions: bins.len() }
    }

    /// The cheapest list size per group so the training questions reach `target` accuracy: repeatedly give one
    /// group a bigger size, choosing the upgrade with the most extra correct answers per extra check.
    pub fn table(&self, target: f64) -> Vec<usize> {
        let nl = self.levels.len();
        let gc = |a: usize, b: usize| self.correct[a * N_BINS + b];
        let gk = |a: usize, b: usize| self.checks[a * N_BINS + b];
        let mut lvl = [0usize; N_BINS];
        let need = target * 10.0 * self.questions as f64;
        loop {
            let have: f64 = (0..N_BINS).map(|b| gc(lvl[b], b)).sum();
            if have >= need {
                break;
            }
            let mut best: Option<(f64, usize, usize)> = None;
            for b in 0..N_BINS {
                for a in lvl[b] + 1..nl {
                    let gain = gc(a, b) - gc(lvl[b], b);
                    let cost = gk(a, b) - gk(lvl[b], b);
                    if gain > 0.0 && cost > 0.0 && best.is_none_or(|x| gain / cost > x.0) {
                        best = Some((gain / cost, b, a));
                    }
                }
            }
            match best {
                Some((_, b, a)) => lvl[b] = a,
                None => break,
            }
        }
        lvl.iter().map(|&a| self.levels[a]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_and_quantiles_follow_numpy() {
        let cuts = [-0.5f32, 0.0, 0.5];
        assert_eq!(bin_of(&cuts, -1.0), 0);
        assert_eq!(bin_of(&cuts, -0.5), 0); // equal to a cut stays below it (searchsorted, side="left")
        assert_eq!(bin_of(&cuts, 0.1), 2);
        assert_eq!(bin_of(&cuts, 9.0), 3);
        let v = [1.0f32, 2.0, 3.0, 4.0];
        assert!((quantile(&v, 0.5) - 2.5).abs() < 1e-6);
        assert!((quantile(&v, 0.125) - 1.375).abs() < 1e-6);
    }

    #[test]
    fn the_table_upgrades_the_group_that_gains_most_per_check() {
        // two questions, both in group 0 / group 1; group 1 needs a bigger list to get its answers
        let bins = [0usize, 1];
        let levels = vec![10, 20, 40];
        let correct = vec![vec![10, 2], vec![10, 6], vec![10, 10]];
        let checks = vec![vec![100, 100], vec![200, 200], vec![400, 400]];
        let l = Learned::from_runs(vec![0.0; N_BINS - 1], &bins, levels, &correct, &checks);
        assert_eq!(l.table(0.6)[..2], [10, 10]);
        assert_eq!(l.table(0.8)[..2], [10, 20]);
        assert_eq!(l.table(1.0)[..2], [10, 40]);
    }
}
