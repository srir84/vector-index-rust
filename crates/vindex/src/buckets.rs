//! Buckets (part 5, DESIGN.md section 23): a second way to search the same items. The items are grouped around
//! centres (k-means); a search scores the centres, scans the closest buckets' short codes, and re-checks the best
//! with the full numbers.
//!
//! The short codes: each pair of numbers gets 4 bits, one of 16 points learned from the items (for angle and best
//! score, from each item's offset to its bucket's centre). They are laid out 32 items to a block, so the chip's
//! vector table lookup scores 32 items at once from a small table made for each question (ScaNN's trick).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::metric::{dot, dot_n, l2sq, l2sq_n, normalize, Metric};
use crate::prefetch::prefetch_bytes;
use crate::rng::Rng;

/// Items per bucket aimed for: ScaNN's settings search found about 330 best on GloVe (part 4).
pub const ITEMS_PER_BUCKET: usize = 300;
/// Numbers covered by one 4-bit code.
pub const SUB_DIMS: usize = 2;
/// Items scored together.
pub const BLOCK: usize = 32;
/// A padded place in a block.
pub const NONE: u32 = u32::MAX;
/// k-means rounds, and sample rows per centre (step 40: at full size the buckets need more than the samples do).
pub const KMEANS_ROUNDS: usize = 10;
pub const KMEANS_ROWS_PER_CENTRE: usize = 64;
const CODE_ROUNDS: usize = 20;
const CODE_ROWS: usize = 65_536;

/// The buckets: centres, each bucket's items and their codes, and the 16 points of each code.
#[derive(Clone, Debug)]
pub struct Buckets {
    pub dim: usize,
    pub metric: Metric,
    /// k × dim; for angle data scaled to length 1.
    pub centres: Vec<f32>,
    /// Each bucket's slots in scan order, padded to whole blocks with `NONE`.
    pub slots: Vec<Vec<u32>>,
    /// Each bucket's codes: per block, `n_sub` groups of 16 bytes (byte t of a group: item t's code in the low half,
    /// item t + 16's in the high half).
    pub codes: Vec<Vec<u8>>,
    /// Codes per item: one per `SUB_DIMS` numbers (the last padded with zeros).
    pub n_sub: usize,
    /// n_sub × 16 × SUB_DIMS: the numbers each code stands for.
    pub book: Vec<f32>,
    /// Codes hold offsets from the bucket's centre (angle and best score) rather than the numbers themselves.
    pub residual: bool,
    /// How much more a code's error along the item's own direction counts than one across it, when an item is
    /// coded (ScaNN's score-aware loss; 1 codes each pair to its nearest point).
    pub eta: f32,
}

/// ScaNN's default weight for coding errors along the item: threshold 0.2 gives (d - 1) * 0.04 / 0.96 for d numbers.
pub fn default_eta(dim: usize) -> f32 {
    ((dim as f32 - 1.0) * 0.04 / 0.96).max(1.0)
}

/// What one bucket search did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Work {
    pub scanned: u64,
    pub rechecked: u64,
}

fn sub(x: &[f32], m: usize) -> [f32; SUB_DIMS] {
    let mut out = [0.0; SUB_DIMS];
    for (j, o) in out.iter_mut().enumerate() {
        if let Some(&v) = x.get(m * SUB_DIMS + j) {
            *o = v;
        }
    }
    out
}

/// The nearest centre of each row (largest dot product with `spherical`, else the nearest by straight line).
pub fn assign(rows: &[f32], dim: usize, centres: &[f32], spherical: bool) -> Vec<u32> {
    let k = centres.len() / dim;
    rows.par_chunks(dim)
        .map(|x| {
            let (mut best, mut bd) = (0u32, f32::INFINITY);
            let mut c = 0;
            while c + 4 <= k {
                let cs = [0, 1, 2, 3].map(|j| &centres[(c + j) * dim..(c + j + 1) * dim]);
                let d = if spherical { dot_n(x, cs).map(|v| -v) } else { cs.map(|cc| l2sq(x, cc)) };
                for (j, &dj) in d.iter().enumerate() {
                    if dj < bd {
                        (best, bd) = ((c + j) as u32, dj);
                    }
                }
                c += 4;
            }
            for c in c..k {
                let cc = &centres[c * dim..(c + 1) * dim];
                let d = if spherical { -dot(x, cc) } else { l2sq(x, cc) };
                if d < bd {
                    (best, bd) = (c as u32, d);
                }
            }
            best
        })
        .collect()
}

/// `k` centres for `rows` (n × dim): a seeded draw of starting rows, then `rounds` Lloyd rounds on a sample of at
/// most `per_centre` rows per centre. With `spherical`, centres are scaled to length 1 and rows are assigned by the
/// largest dot product; an empty bucket takes a random row.
pub fn kmeans(rows: &[f32], dim: usize, k: usize, spherical: bool, rounds: usize, per_centre: usize, seed: u64) -> Vec<f32> {
    let n = rows.len() / dim;
    let mut rng = Rng::new(seed);
    let m = n.min(k * per_centre);
    let pick = rng.sample(n, m);
    let sample: Vec<f32> = pick.iter().flat_map(|&i| rows[i * dim..(i + 1) * dim].iter().copied()).collect();
    let mut centres: Vec<f32> = rng.sample(m, k.min(m)).iter().flat_map(|&i| sample[i * dim..(i + 1) * dim].iter().copied()).collect();
    for _ in 0..rounds {
        let a = assign(&sample, dim, &centres, spherical);
        let mut sum = vec![0f64; k * dim];
        let mut count = vec![0usize; k];
        for (i, &c) in a.iter().enumerate() {
            count[c as usize] += 1;
            for (s, &v) in sum[c as usize * dim..(c as usize + 1) * dim].iter_mut().zip(&sample[i * dim..(i + 1) * dim]) {
                *s += v as f64;
            }
        }
        for c in 0..k {
            let row = &mut centres[c * dim..(c + 1) * dim];
            if count[c] == 0 {
                let r = rng.below(m as u64) as usize;
                row.copy_from_slice(&sample[r * dim..(r + 1) * dim]);
                continue;
            }
            for (v, s) in row.iter_mut().zip(&sum[c * dim..(c + 1) * dim]) {
                *v = (*s / count[c] as f64) as f32;
            }
            if spherical {
                normalize(row);
            }
        }
    }
    centres
}

/// 16 points for each group of `SUB_DIMS` numbers, from `rows` (k-means within each group, by straight line).
fn train_book(rows: &[f32], dim: usize, n_sub: usize, seed: u64) -> Vec<f32> {
    let n = rows.len() / dim;
    (0..n_sub)
        .into_par_iter()
        .flat_map_iter(|m| {
            let pts: Vec<[f32; SUB_DIMS]> = (0..n).map(|i| sub(&rows[i * dim..(i + 1) * dim], m)).collect();
            let mut rng = Rng::new(seed ^ (m as u64).wrapping_mul(0x9e37_79b9));
            let mut c: Vec<[f32; SUB_DIMS]> = rng.sample(n, 16.min(n)).iter().map(|&i| pts[i]).collect();
            c.resize(16, [0.0; SUB_DIMS]);
            for _ in 0..CODE_ROUNDS {
                let mut sum = [[0f64; SUB_DIMS]; 16];
                let mut count = [0usize; 16];
                for p in &pts {
                    let j = nearest(&c, p);
                    count[j] += 1;
                    for (s, v) in sum[j].iter_mut().zip(p) {
                        *s += *v as f64;
                    }
                }
                for j in 0..16 {
                    if count[j] > 0 {
                        for (v, s) in c[j].iter_mut().zip(&sum[j]) {
                            *v = (*s / count[j] as f64) as f32;
                        }
                    } else {
                        c[j] = pts[rng.below(n as u64) as usize];
                    }
                }
            }
            c.into_iter().flatten().collect::<Vec<f32>>()
        })
        .collect()
}

fn nearest(points: &[[f32; SUB_DIMS]], p: &[f32; SUB_DIMS]) -> usize {
    let mut best = (0, f32::INFINITY);
    for (j, c) in points.iter().enumerate() {
        let d: f32 = c.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum();
        if d < best.1 {
            best = (j, d);
        }
    }
    best.0
}

/// A question's table: one byte per code point, per group, and how to turn a block's byte sums back into distances.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub bytes: Vec<u8>,
    /// distance ~ sum * step + base + the bucket's own part
    pub step: f32,
    pub base: f32,
}

impl Buckets {
    /// Group `rows` (one per slot, `alive` marking the slots in use) into about `rows / per_bucket` buckets, coding
    /// each item with weight `eta` on errors along it (1: plain).
    pub fn build(rows: &[f32], alive: &[bool], dim: usize, metric: Metric, per_bucket: usize, eta: f32, seed: u64) -> Self {
        Self::build_with(rows, alive, dim, metric, per_bucket, eta, KMEANS_ROUNDS, KMEANS_ROWS_PER_CENTRE, seed)
    }

    /// `build`, with the k-means rounds and sample rows per centre given.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with(
        rows: &[f32],
        alive: &[bool],
        dim: usize,
        metric: Metric,
        per_bucket: usize,
        eta: f32,
        rounds: usize,
        per_centre: usize,
        seed: u64,
    ) -> Self {
        let live: Vec<u32> = (0..alive.len() as u32).filter(|&u| alive[u as usize]).collect();
        let k = (live.len() / per_bucket.max(1)).max(1);
        let xs: Vec<f32> = live.iter().flat_map(|&u| rows[u as usize * dim..(u as usize + 1) * dim].iter().copied()).collect();
        let spherical = metric == Metric::Cos;
        let centres = kmeans(&xs, dim, k, spherical, rounds, per_centre, seed);
        let of = assign(&xs, dim, &centres, spherical);
        let residual = metric != Metric::L2;
        let n_sub = dim.div_ceil(SUB_DIMS);
        let eta = if residual { eta.max(1.0) } else { 1.0 };
        let mut b = Buckets { dim, metric, centres, slots: vec![Vec::new(); k], codes: vec![Vec::new(); k], n_sub, book: Vec::new(), residual, eta };
        let offset = |b: &Buckets, i: usize| -> Vec<f32> {
            let x = &xs[i * dim..(i + 1) * dim];
            if residual {
                let c = &b.centres[of[i] as usize * dim..(of[i] as usize + 1) * dim];
                x.iter().zip(c).map(|(a, b)| a - b).collect()
            } else {
                x.to_vec()
            }
        };
        let train: Vec<f32> =
            Rng::new(seed ^ 0x636f_6465).sample(live.len(), live.len().min(CODE_ROWS)).iter().flat_map(|&i| offset(&b, i)).collect();
        b.book = train_book(&train, dim, n_sub, seed);
        let codes: Vec<Vec<u8>> = (0..live.len()).into_par_iter().map(|i| b.code_of(&xs[i * dim..(i + 1) * dim], &offset(&b, i))).collect();
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &c) in of.iter().enumerate() {
            members[c as usize].push(i);
        }
        for (c, ms) in members.iter().enumerate() {
            for &i in ms {
                b.push(c, live[i], &codes[i]);
            }
        }
        b
    }

    pub fn k(&self) -> usize {
        self.slots.len()
    }

    /// The code of an offset (or of the numbers themselves, straight-line data): one 4-bit point per group.
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        (0..self.n_sub).map(|m| nearest(&self.points(m), &sub(x, m)) as u8).collect()
    }

    fn points(&self, m: usize) -> [[f32; SUB_DIMS]; 16] {
        let mut pts = [[0.0; SUB_DIMS]; 16];
        for (j, p) in pts.iter_mut().enumerate() {
            p.copy_from_slice(&self.book[(m * 16 + j) * SUB_DIMS..(m * 16 + j + 1) * SUB_DIMS]);
        }
        pts
    }

    /// The code of item `x` whose offset is `r`: with `eta` above 1, ScaNN's score-aware coding: start from the
    /// nearest points, then, pair by pair, take the point that best keeps down |error|^2 + (eta - 1) (error . u)^2,
    /// u the item's direction, so the error along the item, which moves its scores most, counts eta times.
    pub fn code_of(&self, x: &[f32], r: &[f32]) -> Vec<u8> {
        let mut code = self.encode(r);
        if self.eta <= 1.0 {
            return code;
        }
        let mut u = x.to_vec();
        normalize(&mut u);
        let pts: Vec<[[f32; SUB_DIMS]; 16]> = (0..self.n_sub).map(|m| self.points(m)).collect();
        let err = |m: usize, j: usize| -> [f32; SUB_DIMS] {
            let (rm, p) = (sub(r, m), pts[m][j]);
            let mut e = [0.0; SUB_DIMS];
            for t in 0..SUB_DIMS {
                e[t] = rm[t] - p[t];
            }
            e
        };
        let along = |m: usize, e: &[f32; SUB_DIMS]| -> f32 { e.iter().zip(sub(&u, m)).map(|(a, b)| a * b).sum() };
        let mut s: f32 = (0..self.n_sub).map(|m| along(m, &err(m, code[m] as usize))).sum();
        for _ in 0..3 {
            for m in 0..self.n_sub {
                let rest = s - along(m, &err(m, code[m] as usize));
                let (mut best, mut cost) = (code[m], f32::INFINITY);
                for j in 0..16 {
                    let e = err(m, j);
                    let a = rest + along(m, &e);
                    let c = e.iter().map(|v| v * v).sum::<f32>() + (self.eta - 1.0) * a * a;
                    if c < cost {
                        (best, cost) = (j as u8, c);
                    }
                }
                code[m] = best;
                s = rest + along(m, &err(m, best as usize));
            }
        }
        code
    }

    /// Put slot `u` with its code in bucket `c`, in the first padded place or a new block.
    pub fn push(&mut self, c: usize, u: u32, code: &[u8]) {
        let slots = &mut self.slots[c];
        let at = match slots.iter().position(|&s| s == NONE) {
            Some(p) => p,
            None => {
                let p = slots.len();
                slots.resize(p + BLOCK, NONE);
                let grown = self.codes[c].len() + self.n_sub * 16;
                self.codes[c].resize(grown, 0);
                p
            }
        };
        slots[at] = u;
        let (blk, t) = (at / BLOCK, at % BLOCK);
        let base = blk * self.n_sub * 16;
        for (m, &v) in code.iter().enumerate() {
            let byte = &mut self.codes[c][base + m * 16 + t % 16];
            *byte = if t < 16 { (*byte & 0xf0) | v } else { (*byte & 0x0f) | (v << 4) };
        }
    }

    /// The table for question `q` (scaled as the search scales it): each code point's share of the distance, as one
    /// byte, so that a block's byte sums fit 16 bits.
    pub fn table(&self, q: &[f32], out: &mut Table) {
        let levels = (u16::MAX as usize / self.n_sub).min(255) as f32;
        let mut share = vec![0f32; self.n_sub * 16];
        for m in 0..self.n_sub {
            let qm = sub(q, m);
            for j in 0..16 {
                let p = &self.book[(m * 16 + j) * SUB_DIMS..(m * 16 + j + 1) * SUB_DIMS];
                share[m * 16 + j] = match self.metric {
                    Metric::L2 => p.iter().zip(&qm).map(|(a, b)| (a - b) * (a - b)).sum(),
                    _ => -p.iter().zip(&qm).map(|(a, b)| a * b).sum::<f32>(),
                };
            }
        }
        let (mut base, mut widest) = (0f32, 0f32);
        let lows: Vec<f32> = (0..self.n_sub).map(|m| share[m * 16..(m + 1) * 16].iter().copied().fold(f32::INFINITY, f32::min)).collect();
        for m in 0..self.n_sub {
            base += lows[m];
            let hi = share[m * 16..(m + 1) * 16].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            widest = widest.max(hi - lows[m]);
        }
        let step = if widest > 0.0 { widest / levels } else { 1.0 };
        out.bytes.clear();
        out.bytes.extend((0..self.n_sub * 16).map(|i| ((share[i] - lows[i / 16]) / step).round().min(levels) as u8));
        out.step = step;
        out.base = base;
    }

    /// The bucket's own part of the distance: minus the dot product with its centre (offsets), else nothing.
    fn bucket_part(&self, q: &[f32], c: usize) -> f32 {
        if self.residual { -dot(q, &self.centres[c * self.dim..(c + 1) * self.dim]) } else { 0.0 }
    }

    /// The `n_probe` buckets whose centres are nearest `q`, nearest first.
    pub fn nearest_buckets(&self, q: &[f32], n_probe: usize) -> Vec<u32> {
        let (k, dim) = (self.k(), self.dim);
        let row = |c: usize| &self.centres[c * dim..(c + 1) * dim];
        let score = |rs: [&[f32]; 4]| match self.metric {
            Metric::L2 => l2sq_n(q, rs),
            _ => dot_n(q, rs).map(|v| -v),
        };
        let mut d: Vec<(f32, u32)> = Vec::with_capacity(k);
        let mut c = 0;
        while c + 4 <= k {
            for (j, v) in score([row(c), row(c + 1), row(c + 2), row(c + 3)]).into_iter().enumerate() {
                d.push((v, (c + j) as u32));
            }
            c += 4;
        }
        for c in c..k {
            d.push((match self.metric { Metric::L2 => l2sq(q, row(c)), _ => -dot(q, row(c)) }, c as u32));
        }
        let n = n_probe.min(k);
        if n < k {
            d.select_nth_unstable_by(n, |a, b| a.0.total_cmp(&b.0));
            d.truncate(n);
        }
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        d.into_iter().map(|x| x.1).collect()
    }

    /// Scan the `n_probe` nearest buckets and keep the `pool` best by their codes (live slots only), best first,
    /// with their estimated distances.
    pub fn scan(&self, q: &[f32], n_probe: usize, pool: usize, alive: impl Fn(u32) -> bool, work: &mut Work) -> Vec<(f32, u32)> {
        let mut t = Table::default();
        self.table(q, &mut t);
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(pool + 1);
        let mut sums = [0u16; BLOCK];
        let order = self.nearest_buckets(q, n_probe);
        for (i, &c) in order.iter().enumerate() {
            // each bucket sits in its own place in memory: start loading the next one's first blocks now
            if let Some(&next) = order.get(i + 1) {
                let codes = &self.codes[next as usize];
                prefetch_bytes(codes.as_ptr(), codes.len().min(2 * self.n_sub * 16));
            }
            let c = c as usize;
            let bias = t.base + self.bucket_part(q, c);
            let slots = &self.slots[c];
            // a sum below the limit may beat the pool's worst, (worst - bias) / step in the sums' own units; a block
            // whose smallest sum cannot is passed over whole
            let limit = |heap: &BinaryHeap<Cand>| -> u32 {
                if heap.len() < pool {
                    return u32::MAX;
                }
                let l = ((heap.peek().map_or(f32::INFINITY, |w| w.0) - bias) / t.step).ceil();
                if l <= 0.0 { 0 } else if l > u16::MAX as f32 { u32::MAX } else { l as u32 }
            };
            let mut lim = limit(&heap);
            for (blk, chunk) in self.codes[c].chunks_exact(self.n_sub * 16).enumerate() {
                work.scanned += BLOCK as u64;
                if scan_block(chunk, &t.bytes, self.n_sub, &mut sums) as u32 >= lim {
                    continue;
                }
                for (j, &s) in sums.iter().enumerate() {
                    let u = slots[blk * BLOCK + j];
                    if s as u32 >= lim || u == NONE || !alive(u) {
                        continue;
                    }
                    heap.push(Cand(s as f32 * t.step + bias, u));
                    if heap.len() > pool {
                        heap.pop();
                    }
                    lim = limit(&heap);
                }
            }
        }
        let mut out: Vec<(f32, u32)> = heap.into_iter().map(|c| (c.0, c.1)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Cand(f32, u32);

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// The 32 byte sums of one block: for each group m, item t's table entry `lut[m * 16 + code]` is added in. Returns
/// the smallest.
#[inline]
pub fn scan_block(codes: &[u8], lut: &[u8], n_sub: usize, out: &mut [u16; BLOCK]) -> u16 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `codes` and `lut` hold n_sub * 16 bytes each (checked below), and `out` holds 32 sums.
    unsafe {
        use std::arch::aarch64::*;
        assert!(codes.len() >= n_sub * 16 && lut.len() >= n_sub * 16);
        let mask = vdupq_n_u8(0x0f);
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0));
        for m in 0..n_sub {
            let c = vld1q_u8(codes.as_ptr().add(m * 16));
            let t = vld1q_u8(lut.as_ptr().add(m * 16));
            let lo = vqtbl1q_u8(t, vandq_u8(c, mask));
            let hi = vqtbl1q_u8(t, vshrq_n_u8::<4>(c));
            a0 = vaddw_u8(a0, vget_low_u8(lo));
            a1 = vaddw_high_u8(a1, lo);
            a2 = vaddw_u8(a2, vget_low_u8(hi));
            a3 = vaddw_high_u8(a3, hi);
        }
        vst1q_u16(out.as_mut_ptr(), a0);
        vst1q_u16(out.as_mut_ptr().add(8), a1);
        vst1q_u16(out.as_mut_ptr().add(16), a2);
        vst1q_u16(out.as_mut_ptr().add(24), a3);
        vminvq_u16(vminq_u16(vminq_u16(a0, a1), vminq_u16(a2, a3)))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scan_block_plain(codes, lut, n_sub, out);
        out.iter().copied().min().unwrap_or(0)
    }
}

/// `scan_block` one entry at a time.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub fn scan_block_plain(codes: &[u8], lut: &[u8], n_sub: usize, out: &mut [u16; BLOCK]) {
    *out = [0; BLOCK];
    for m in 0..n_sub {
        for t in 0..16 {
            let b = codes[m * 16 + t];
            out[t] += lut[m * 16 + (b & 0x0f) as usize] as u16;
            out[t + 16] += lut[m * 16 + (b >> 4) as usize] as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        let mut x: Vec<f32> = (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
        x.chunks_mut(d).for_each(normalize);
        x
    }

    #[test]
    fn vector_block_sums_match_the_one_at_a_time_sums() {
        let mut r = Rng::new(5);
        for n_sub in [1, 7, 100, 384] {
            let codes: Vec<u8> = (0..n_sub * 16).map(|_| r.next_u64() as u8).collect();
            let levels = (u16::MAX as usize / n_sub).min(255) as u64;
            let lut: Vec<u8> = (0..n_sub * 16).map(|_| (r.next_u64() % (levels + 1)) as u8).collect();
            let (mut a, mut b) = ([0u16; BLOCK], [0u16; BLOCK]);
            let low = scan_block(&codes, &lut, n_sub, &mut a);
            scan_block_plain(&codes, &lut, n_sub, &mut b);
            assert_eq!(a, b, "{n_sub} groups");
            assert_eq!(low, *b.iter().min().unwrap());
        }
    }

    #[test]
    fn every_live_item_is_in_one_bucket_and_codes_round_trip() {
        let (n, d) = (3_000, 16);
        let x = data(n, d, 1);
        let alive: Vec<bool> = (0..n).map(|i| i % 10 != 3).collect();
        let b = Buckets::build(&x, &alive, d, Metric::Cos, 100, 1.0, 7);
        assert_eq!(b.k(), alive.iter().filter(|&&a| a).count() / 100);
        let mut seen: Vec<u32> = b.slots.iter().flatten().copied().filter(|&u| u != NONE).collect();
        seen.sort_unstable();
        let want: Vec<u32> = (0..n as u32).filter(|&u| alive[u as usize]).collect();
        assert_eq!(seen, want);
        for (c, slots) in b.slots.iter().enumerate() {
            assert_eq!(slots.len() % BLOCK, 0);
            assert_eq!(b.codes[c].len(), slots.len() / BLOCK * b.n_sub * 16);
        }
    }

    #[test]
    fn scanning_every_bucket_with_a_big_pool_and_rechecking_finds_the_true_top_10() {
        let (n, d) = (4_000, 24);
        let x = data(n, d, 2);
        let alive = vec![true; n];
        let b = Buckets::build(&x, &alive, d, Metric::Cos, 100, default_eta(d), 3);
        let q = data(20, d, 9);
        for qi in q.chunks(d) {
            let mut w = Work::default();
            let pool = b.scan(qi, b.k(), n, |_| true, &mut w);
            assert_eq!(w.scanned, b.slots.iter().map(|s| s.len() as u64).sum::<u64>(), "every place of every block");
            let mut exact: Vec<(f32, u32)> = pool.iter().map(|&(_, u)| (1.0 - dot(qi, &x[u as usize * d..(u as usize + 1) * d]), u)).collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut truth: Vec<(f32, u32)> = (0..n as u32).map(|u| (1.0 - dot(qi, &x[u as usize * d..(u as usize + 1) * d]), u)).collect();
            truth.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(exact[..10].iter().map(|e| e.1).collect::<Vec<_>>(), truth[..10].iter().map(|e| e.1).collect::<Vec<_>>());
        }
    }

    #[test]
    fn code_estimates_follow_the_true_distances() {
        let (n, d) = (4_000, 32);
        let x = data(n, d, 4);
        let b = Buckets::build(&x, &vec![true; n], d, Metric::Cos, 200, 1.0, 5);
        let q = &data(1, d, 11)[..d];
        let mut w = Work::default();
        let est = b.scan(q, b.k(), n, |_| true, &mut w);
        // the estimates' order agrees with the true order far better than chance (rank correlation)
        let truth: Vec<f32> = est.iter().map(|&(_, u)| 1.0 - dot(q, &x[u as usize * d..(u as usize + 1) * d])).collect();
        let (mut agree, mut pairs) = (0u64, 0u64);
        for i in (0..est.len()).step_by(7) {
            for j in (i + 1..est.len()).step_by(11) {
                pairs += 1;
                agree += ((est[i].0 < est[j].0) == (truth[i] < truth[j])) as u64;
            }
        }
        assert!(agree as f64 / pairs as f64 > 0.75, "{agree} of {pairs}");
    }

    #[test]
    fn score_aware_codes_leave_less_error_along_the_item() {
        let (n, d) = (3_000, 32);
        let x = data(n, d, 6);
        let plain = Buckets::build(&x, &vec![true; n], d, Metric::Cos, 100, 1.0, 8);
        let aware = Buckets { eta: default_eta(d).max(4.0), ..plain.clone() };
        let (mut along_p, mut along_a, mut all_p, mut all_a) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..n {
            let xi = &x[i * d..(i + 1) * d];
            let c = assign(xi, d, &plain.centres, true)[0] as usize;
            let r: Vec<f32> = xi.iter().zip(&plain.centres[c * d..(c + 1) * d]).map(|(a, b)| a - b).collect();
            for (b, along, all) in [(&plain, &mut along_p, &mut all_p), (&aware, &mut along_a, &mut all_a)] {
                let code = b.code_of(xi, &r);
                let e: Vec<f32> = (0..d).map(|j| r[j] - b.book[(j / SUB_DIMS * 16 + code[j / SUB_DIMS] as usize) * SUB_DIMS + j % SUB_DIMS]).collect();
                *along += (e.iter().zip(xi).map(|(a, b)| a * b).sum::<f32>() as f64).powi(2);
                *all += e.iter().map(|v| (v * v) as f64).sum::<f64>();
            }
        }
        assert!(along_a < 0.5 * along_p, "error along the items: {along_a} against {along_p}");
        assert!(all_a > all_p, "the plain codes are the nearest points, so their whole error is the smallest");
    }
}
