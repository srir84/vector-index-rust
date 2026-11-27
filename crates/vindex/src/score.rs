//! Scoring items for the walk and the re-check: one at a time, or four side by side.
//!
//! A distance is a long chain of multiply-adds into 16 running sums, and each step waits for the step before it
//! in the same sum. Scoring four items together gives the chip four independent chains to work on. Each item's
//! sums are still done in exactly the order used when it is scored alone, so the numbers are identical to the
//! last bit and results do not change.

use crate::codes::{Codes, Prepared};
use crate::metric::Metric;
use crate::prefetch::prefetch_bytes;
use crate::prune::BuildSpace;
use crate::store::Store;

/// How the walk scores items.
pub trait Score {
    /// Distance to item `i`.
    fn one(&mut self, i: u32) -> f32;

    /// `out[t] = self.one(ids[t])` for every t: the same numbers, possibly computed several at a time.
    #[inline]
    fn many(&mut self, ids: &[u32], out: &mut [f32]) {
        for (o, &i) in out.iter_mut().zip(ids) {
            *o = self.one(i);
        }
    }

    /// Start loading item `i`'s numbers (a hint; it never changes results).
    #[inline]
    fn fetch(&self, _i: u32) {}
}

/// A closure scores one item at a time, with no loading hints.
impl<F: FnMut(u32) -> f32> Score for F {
    #[inline]
    fn one(&mut self, i: u32) -> f32 {
        self(i)
    }
}

/// `four` over each group of four ids, `one` over the rest.
#[inline]
fn by_fours(ids: &[u32], out: &mut [f32], four: impl Fn([u32; 4]) -> [f32; 4], one: impl Fn(u32) -> f32) {
    let mut t = 0;
    while t + 4 <= ids.len() {
        out[t..t + 4].copy_from_slice(&four([ids[t], ids[t + 1], ids[t + 2], ids[t + 3]]));
        t += 4;
    }
    for k in t..ids.len() {
        out[k] = one(ids[k]);
    }
}

#[inline]
fn fetch_floats(row: &[f32]) {
    prefetch_bytes(row.as_ptr() as *const u8, std::mem::size_of_val(row));
}

/// A prepared question against the 1-byte copies.
pub struct CodeScore<'a> {
    pub codes: &'a Codes,
    pub prep: &'a Prepared,
}

impl Score for CodeScore<'_> {
    #[inline]
    fn one(&mut self, i: u32) -> f32 {
        self.codes.distance(self.prep, i as usize)
    }

    #[inline]
    fn many(&mut self, ids: &[u32], out: &mut [f32]) {
        let (c, p) = (self.codes, self.prep);
        by_fours(ids, out, |g| c.distance_n(p, g.map(|i| i as usize)), |i| c.distance(p, i as usize));
    }

    #[inline]
    fn fetch(&self, i: u32) {
        let row = self.codes.row(i as usize);
        prefetch_bytes(row.as_ptr(), row.len());
    }
}

/// A question against the full numbers (in memory or on SSD).
pub struct FullScore<'a> {
    pub metric: Metric,
    pub q: &'a [f32],
    pub store: &'a Store,
}

impl Score for FullScore<'_> {
    #[inline]
    fn one(&mut self, i: u32) -> f32 {
        self.metric.distance(self.q, self.store.row(i as usize))
    }

    #[inline]
    fn many(&mut self, ids: &[u32], out: &mut [f32]) {
        let (m, q, st) = (self.metric, self.q, self.store);
        by_fours(ids, out, |g| m.distance_n(q, g.map(|i| st.row(i as usize))), |i| m.distance(q, st.row(i as usize)));
    }

    #[inline]
    fn fetch(&self, i: u32) {
        fetch_floats(self.store.row(i as usize));
    }
}

/// One item's numbers `v` against the others, in build space.
pub struct BuildScore<'a> {
    pub sp: BuildSpace<'a>,
    pub v: &'a [f32],
}

impl Score for BuildScore<'_> {
    #[inline]
    fn one(&mut self, i: u32) -> f32 {
        self.sp.metric.distance(self.v, self.sp.row(i))
    }

    #[inline]
    fn many(&mut self, ids: &[u32], out: &mut [f32]) {
        let (sp, v) = (self.sp, self.v);
        by_fours(ids, out, |g| sp.metric.distance_n(v, g.map(|i| sp.row(i))), |i| sp.metric.distance(v, sp.row(i)));
    }

    #[inline]
    fn fetch(&self, i: u32) {
        fetch_floats(self.sp.row(i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    #[test]
    fn four_side_by_side_give_exactly_the_numbers_of_one_at_a_time() {
        for d in [1usize, 7, 16, 100, 200, 768] {
            let x = data(9 * d, d as u64);
            let q = data(d, 99);
            let ids: Vec<u32> = (0..9).collect();
            for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
                let mut store = Store::ram(d, 9);
                for (i, row) in x.chunks(d).enumerate() {
                    store.set(i, row);
                }
                let mut full = FullScore { metric, q: &q, store: &store };
                let mut one = [0f32; 9];
                let mut many = vec![0f32; 9];
                for (o, &i) in one.iter_mut().zip(&ids) {
                    *o = full.one(i);
                }
                full.many(&ids, &mut many);
                assert_eq!(one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), many.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                let mut codes = Codes::train(x.chunks(d), d, metric, 9);
                for (i, row) in x.chunks(d).enumerate() {
                    codes.set(i, row);
                }
                let mut p = Prepared::default();
                codes.prepare(&q, &mut p);
                let mut cs = CodeScore { codes: &codes, prep: &p };
                for (o, &i) in one.iter_mut().zip(&ids) {
                    *o = cs.one(i);
                }
                cs.many(&ids, &mut many);
                assert_eq!(one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), many.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
    }
}
