//! Distances between full numbers. Smaller = closer.
//!
//! - `Cos`: `1 - dot(q, x)`; the index scales cosine data to length 1 when it comes in.
//! - `Ip` ("best score"): `-dot(q, x)`.
//! - `L2`: squared straight-line distance.

/// The closeness formula.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Cos,
    Ip,
    L2,
}

impl Metric {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cos" | "cosine" | "angular" => Some(Metric::Cos),
            "ip" | "dot" | "inner_product" => Some(Metric::Ip),
            "l2" | "euclidean" => Some(Metric::L2),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Metric::Cos => "cos",
            Metric::Ip => "ip",
            Metric::L2 => "l2",
        }
    }

    /// `distance(q, x)` for several rows at once, each to the last bit as `distance` gives it.
    #[inline]
    pub fn distance_n<const N: usize>(self, q: &[f32], xs: [&[f32]; N]) -> [f32; N] {
        match self {
            Metric::Cos => dot_n(q, xs).map(|s| 1.0 - s),
            Metric::Ip => dot_n(q, xs).map(|s| -s),
            Metric::L2 => l2sq_n(q, xs),
        }
    }

    /// Distance from `q` to `x` in this formula.
    #[inline]
    pub fn distance(self, q: &[f32], x: &[f32]) -> f32 {
        match self {
            Metric::Cos => 1.0 - dot(q, x),
            Metric::Ip => -dot(q, x),
            Metric::L2 => l2sq(q, x),
        }
    }
}

/// Lanes per accumulator block: 16 running sums = four 4-lane vector registers on Apple chips.
pub(crate) const LANES: usize = 16;

/// Dot product, in 16 running sums (see `dot_n`).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_n(a, [b])[0]
}

/// Squared straight-line distance, in 16 running sums (see `l2sq_n`).
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    l2sq_n(a, [b])[0]
}

/// `dot(q, x)` for several rows side by side. Each row keeps 16 running sums: number i goes into sum i mod 16 by
/// a fused multiply-add, the sums are added pairwise (sum j + sum j+8, then + j+4, j+2, j+1), and the numbers past
/// the last full block of 16 are multiply-added onto the result one by one. Several rows at once give the chip
/// independent work while each multiply-add finishes; every row's numbers are the same to the last bit.
#[inline]
pub fn dot_n<const N: usize>(q: &[f32], xs: [&[f32]; N]) -> [f32; N] {
    let n = q.len();
    assert!(xs.iter().all(|x| x.len() == n), "rows must have the question's length");
    let blocks = n / LANES;
    let mut out = blocks_n(q, xs, blocks, false);
    for r in 0..N {
        for i in blocks * LANES..n {
            out[r] = q[i].mul_add(xs[r][i], out[r]);
        }
    }
    out
}

/// `l2sq(q, x)` for several rows side by side, summed exactly as `dot_n` sums (with `(q_i - x_i)^2` terms).
#[inline]
pub fn l2sq_n<const N: usize>(q: &[f32], xs: [&[f32]; N]) -> [f32; N] {
    let n = q.len();
    assert!(xs.iter().all(|x| x.len() == n), "rows must have the question's length");
    let blocks = n / LANES;
    let mut out = blocks_n(q, xs, blocks, true);
    for r in 0..N {
        for i in blocks * LANES..n {
            let t = q[i] - xs[r][i];
            out[r] = t.mul_add(t, out[r]);
        }
    }
    out
}

/// The full blocks of 16 for `dot_n` (`sq` false) or `l2sq_n` (`sq` true), with the pairwise sum of the lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn blocks_n<const N: usize>(q: &[f32], xs: [&[f32]; N], blocks: usize, sq: bool) -> [f32; N] {
    use std::arch::aarch64::*;
    let mut out = [0f32; N];
    // SAFETY: every row has q.len() numbers (checked by the callers) and loads stay below blocks * 16 <= q.len().
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 4]; N];
        let qp = q.as_ptr();
        for c in 0..blocks {
            let o = c * LANES;
            let x = [vld1q_f32(qp.add(o)), vld1q_f32(qp.add(o + 4)), vld1q_f32(qp.add(o + 8)), vld1q_f32(qp.add(o + 12))];
            for r in 0..N {
                let yp = xs[r].as_ptr().add(o);
                for k in 0..4 {
                    let y = vld1q_f32(yp.add(4 * k));
                    acc[r][k] = if sq {
                        let t = vsubq_f32(x[k], y);
                        vfmaq_f32(acc[r][k], t, t)
                    } else {
                        vfmaq_f32(acc[r][k], x[k], y)
                    };
                }
            }
        }
        for r in 0..N {
            // lanes 0-3, 4-7, 8-11, 12-15 are registers 0..3: j + (j+8), then + (j+4), then + (j+2), then 0 + 1
            let v = vaddq_f32(vaddq_f32(acc[r][0], acc[r][2]), vaddq_f32(acc[r][1], acc[r][3]));
            let p = vadd_f32(vget_low_f32(v), vget_high_f32(v));
            out[r] = vget_lane_f32::<0>(p) + vget_lane_f32::<1>(p);
        }
    }
    out
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn blocks_n<const N: usize>(q: &[f32], xs: [&[f32]; N], blocks: usize, sq: bool) -> [f32; N] {
    let mut out = [0f32; N];
    for r in 0..N {
        out[r] = portable::blocks(q, xs[r], blocks, sq);
    }
    out
}

/// The same sums written one number at a time: the fallback on other chips, and the reference the tests compare
/// the vector version with, bit for bit. Some of it is used only by the tests, or only on one kind of chip.
#[allow(dead_code)]
pub(crate) mod portable {
    use super::LANES;

    pub fn blocks(q: &[f32], x: &[f32], blocks: usize, sq: bool) -> f32 {
        let mut acc = [0f32; LANES];
        for c in 0..blocks {
            for j in 0..LANES {
                let (a, b) = (q[c * LANES + j], x[c * LANES + j]);
                acc[j] = if sq {
                    let t = a - b;
                    t.mul_add(t, acc[j])
                } else {
                    a.mul_add(b, acc[j])
                };
            }
        }
        sum_lanes(&acc)
    }

    pub fn sum_lanes(acc: &[f32; LANES]) -> f32 {
        // pairwise, so the rounding stays small
        let mut v = *acc;
        let mut w = LANES;
        while w > 1 {
            w /= 2;
            for j in 0..w {
                v[j] += v[j + w];
            }
        }
        v[0]
    }

    pub fn dot(q: &[f32], x: &[f32]) -> f32 {
        let nb = q.len() / LANES;
        let mut s = blocks(q, x, nb, false);
        for i in nb * LANES..q.len() {
            s = q[i].mul_add(x[i], s);
        }
        s
    }

    pub fn l2sq(q: &[f32], x: &[f32]) -> f32 {
        let nb = q.len() / LANES;
        let mut s = blocks(q, x, nb, true);
        for i in nb * LANES..q.len() {
            let t = q[i] - x[i];
            s = t.mul_add(t, s);
        }
        s
    }
}

/// Scale a vector to length 1 in place (left as is if it is all zeros).
pub fn normalize(x: &mut [f32]) {
    let n = dot(x, x).sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_sums_match_the_one_at_a_time_sums_bit_for_bit() {
        let mut r = crate::rng::Rng::new(11);
        for n in [0usize, 1, 5, 15, 16, 17, 31, 32, 100, 128, 200, 768, 1000] {
            let mut v = |k: usize| (0..k).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect::<Vec<f32>>();
            let (q, a, b, c, d) = (v(n), v(n), v(n), v(n), v(n));
            let four = dot_n(&q, [&a, &b, &c, &d]);
            let four_sq = l2sq_n(&q, [&a, &b, &c, &d]);
            for (j, x) in [&a, &b, &c, &d].into_iter().enumerate() {
                assert_eq!(dot(&q, x).to_bits(), portable::dot(&q, x).to_bits(), "dot, n = {n}");
                assert_eq!(l2sq(&q, x).to_bits(), portable::l2sq(&q, x).to_bits(), "l2sq, n = {n}");
                assert_eq!(four[j].to_bits(), portable::dot(&q, x).to_bits());
                assert_eq!(four_sq[j].to_bits(), portable::l2sq(&q, x).to_bits());
            }
        }
    }

    fn naive_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    fn sample(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn dot_and_l2_match_a_plain_sum_for_every_length() {
        for n in [1, 3, 15, 16, 17, 31, 128, 200, 768] {
            let a = sample(n, 1);
            let b = sample(n, 2);
            let d = dot(&a, &b) as f64;
            assert!((d - naive_dot(&a, &b)).abs() < 1e-5 * (1.0 + n as f64), "dot n={n}");
            let l: f64 = a.iter().zip(&b).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum();
            assert!((l2sq(&a, &b) as f64 - l).abs() < 1e-5 * (1.0 + n as f64), "l2 n={n}");
        }
    }

    #[test]
    fn formulas_follow_the_convention_smaller_is_closer() {
        let mut a = vec![1.0, 2.0, 2.0];
        normalize(&mut a);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-6);
        assert!(Metric::Cos.distance(&a, &a).abs() < 1e-6);
        assert_eq!(Metric::Ip.distance(&[1.0, 2.0], &[3.0, 4.0]), -11.0);
        assert_eq!(Metric::L2.distance(&[1.0, 2.0], &[3.0, 5.0]), 13.0);
        assert_eq!(Metric::from_name("ip"), Some(Metric::Ip));
        assert_eq!(Metric::from_name("nope"), None);
    }
}
