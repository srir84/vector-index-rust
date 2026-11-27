//! The 1-byte copies (Fix 6): every number stored as one byte, `round((x - low) / step)`.
//!
//! `low` and `step` are set per number position from the items at build time; items added later
//! reuse them (values outside the range are clipped). A search walks on these copies and
//! re-checks its best candidates with the full numbers.

use crate::metric::{dot, Metric};

/// Per-position scale of the copies, plus the copies themselves (one row of `dim` bytes per slot).
#[derive(Clone, Debug)]
pub struct Codes {
    pub dim: usize,
    pub metric: Metric,
    pub low: Vec<f32>,
    pub step: Vec<f32>,
    /// `dim` bytes per slot, slot-major.
    pub data: Vec<u8>,
    /// Straight-line formula only: squared length of each decoded copy.
    pub norm2: Vec<f32>,
    /// How the question is held when scored against the copies: 32 (floats, as in part 1), or 16 or 8 (whole
    /// numbers, part 2's step 21: DESIGN.md section 19).
    pub qbits: u8,
    /// Bits per number: 8 (one byte each), or 4 (two per byte: in each block of 32 numbers, byte j holds number j
    /// in its low half and number j + 16 in its high half, so two vector steps unpack them in order).
    pub bits: u8,
}

/// A question prepared for distances to copies: `a = q * step`, `c = sum(q * low)`, `qn2 = |q|^2`. With whole-number
/// questions, `a` is held as `a16` or `a8` with `a ~ scale * a16`; `a8_sum` is the sum of `a8`.
#[derive(Clone, Debug, Default)]
pub struct Prepared {
    pub a: Vec<f32>,
    pub c: f32,
    pub qn2: f32,
    pub a16: Vec<i16>,
    pub a8: Vec<i8>,
    pub scale: f32,
    pub a8_sum: i32,
}

const LEVELS: f32 = 255.0;
const LEVELS4: f32 = 15.0;
use crate::metric::LANES;

impl Codes {
    /// Work out `low` and `step` from `rows` (each `dim` long) and make room for `capacity` slots.
    pub fn train<'a>(rows: impl Iterator<Item = &'a [f32]>, dim: usize, metric: Metric, capacity: usize) -> Self {
        Self::train_bits(rows, dim, metric, capacity, 8)
    }

    /// `train` for copies of `bits` bits per number (8 or 4).
    pub fn train_bits<'a>(rows: impl Iterator<Item = &'a [f32]>, dim: usize, metric: Metric, capacity: usize, bits: u8) -> Self {
        let levels = if bits == 4 { LEVELS4 } else { LEVELS };
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for r in rows {
            for j in 0..dim {
                lo[j] = lo[j].min(r[j]);
                hi[j] = hi[j].max(r[j]);
            }
        }
        let step = lo
            .iter()
            .zip(&hi)
            .map(|(l, h)| if h > l { (h - l) / levels } else { 1.0 })
            .collect();
        let low = lo.iter().map(|l| if l.is_finite() { *l } else { 0.0 }).collect();
        Codes {
            dim,
            metric,
            low,
            step,
            data: vec![0; capacity * Self::row_bytes_for(dim, bits)],
            norm2: if metric == Metric::L2 { vec![0.0; capacity] } else { Vec::new() },
            qbits: 32,
            bits,
        }
    }

    /// Bytes per copy: `dim` at 8 bits; at 4 bits, 16 per block of 32 numbers (the last block padded with zeros).
    pub fn row_bytes_for(dim: usize, bits: u8) -> usize {
        if bits == 4 { dim.div_ceil(32) * 16 } else { dim }
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        Self::row_bytes_for(self.dim, self.bits)
    }


    fn levels(&self) -> f32 {
        if self.bits == 4 { LEVELS4 } else { LEVELS }
    }

    /// The copy's value (0 to 255, or 0 to 15) for number `j` of `slot`.
    #[inline]
    fn value(&self, slot: usize, j: usize) -> u8 {
        let row = self.row(slot);
        if self.bits == 8 {
            return row[j];
        }
        let (block, w) = (j / 32, j % 32);
        let byte = row[block * 16 + w % 16];
        if w < 16 { byte & 0x0F } else { byte >> 4 }
    }

    /// Grow the slot capacity (new slots are zeros).
    pub fn reserve_slots(&mut self, capacity: usize) {
        let rb = self.row_bytes();
        if self.data.len() < capacity * rb {
            self.data.resize(capacity * rb, 0);
            if self.metric == Metric::L2 {
                self.norm2.resize(capacity, 0.0);
            }
        }
    }

    /// Encode `x` into slot `slot`. Rounds half to even, like numpy's `rint`.
    pub fn set(&mut self, slot: usize, x: &[f32]) {
        let (d, rb, levels, bits) = (self.dim, self.row_bytes(), self.levels(), self.bits);
        let row = &mut self.data[slot * rb..(slot + 1) * rb];
        row.fill(0);
        for j in 0..d {
            let v = ((x[j] - self.low[j]) / self.step[j]).round_ties_even().clamp(0.0, levels) as u8;
            if bits == 8 {
                row[j] = v;
            } else {
                let (block, w) = (j / 32, j % 32);
                row[block * 16 + w % 16] |= if w < 16 { v } else { v << 4 };
            }
        }
        if self.metric == Metric::L2 {
            let mut s = 0f32;
            for j in 0..d {
                let y = self.low[j] + self.value(slot, j) as f32 * self.step[j];
                s += y * y;
            }
            self.norm2[slot] = s;
        }
    }

    /// The numbers a copy stands for.
    pub fn decode(&self, slot: usize) -> Vec<f32> {
        let d = self.dim;
        (0..d).map(|j| self.low[j] + self.value(slot, j) as f32 * self.step[j]).collect()
    }

    #[inline]
    pub fn row(&self, slot: usize) -> &[u8] {
        let rb = self.row_bytes();
        &self.data[slot * rb..(slot + 1) * rb]
    }

    /// Prepare a question once; then `distance` is one pass over the copy's bytes.
    pub fn prepare(&self, q: &[f32], out: &mut Prepared) {
        out.a.clear();
        out.a.extend(q.iter().zip(&self.step).map(|(x, s)| x * s));
        out.c = dot(q, &self.low);
        out.qn2 = if self.metric == Metric::L2 { dot(q, q) } else { 0.0 };
        if self.qbits == 32 {
            return;
        }
        // whole numbers: the largest |a| maps to the largest allowed value; 16-bit sums of up to dim products of
        // 255 must stay inside a 32-bit sum
        let lim = if self.qbits == 16 {
            (i32::MAX as f64 / (self.levels() as f64 * self.dim as f64)).floor().min(32767.0)
        } else {
            127.0
        };
        let top = out.a.iter().fold(0f32, |m, v| m.max(v.abs()));
        out.scale = if top > 0.0 { top / lim as f32 } else { 1.0 };
        let inv = 1.0 / out.scale;
        if self.qbits == 16 {
            out.a16.clear();
            out.a16.extend(out.a.iter().map(|v| (v * inv).round() as i16));
            if self.bits == 4 {
                out.a16.resize(self.row_bytes() * 2, 0); // the padded numbers of the last block count nothing
            }
        } else {
            out.a8.clear();
            out.a8.extend(out.a.iter().map(|v| (v * inv).round() as i8));
            out.a8_sum = out.a8.iter().map(|&v| v as i32).sum();
        }
    }

    /// The question's sum against the copy's bytes, `sum(a * byte)`, in the question's own number type.
    #[inline]
    fn sums<const N: usize>(&self, p: &Prepared, rows: [&[u8]; N]) -> [f32; N] {
        match self.qbits {
            16 if self.bits == 4 => dot_u4_i16_n(&p.a16, rows).map(|v| p.scale * v as f32),
            16 => dot_u8_i16_n(&p.a16, rows).map(|v| p.scale * v as f32),
            8 => dot_u8_i8_n(&p.a8, rows).map(|v| p.scale * (v + 128 * p.a8_sum) as f32),
            _ => dot_u8_n(&p.a, rows),
        }
    }

    /// Distance from a prepared question to the copy in `slot`.
    #[inline]
    pub fn distance(&self, p: &Prepared, slot: usize) -> f32 {
        let s = p.c + self.sums(p, [self.row(slot)])[0];
        match self.metric {
            Metric::Cos => 1.0 - s,
            Metric::Ip => -s,
            Metric::L2 => p.qn2 + self.norm2[slot] - 2.0 * s,
        }
    }

    /// `distance(p, slot)` for several copies at once, each to the last bit as `distance` gives it.
    #[inline]
    pub fn distance_n<const N: usize>(&self, p: &Prepared, slots: [usize; N]) -> [f32; N] {
        let dots = self.sums(p, slots.map(|u| self.row(u)));
        let mut out = [0f32; N];
        for r in 0..N {
            let s = p.c + dots[r];
            out[r] = match self.metric {
                Metric::Cos => 1.0 - s,
                Metric::Ip => -s,
                Metric::L2 => p.qn2 + self.norm2[slots[r]] - 2.0 * s,
            };
        }
        out
    }

    /// Bytes kept in memory for the copies (the scale arrays are tiny and left out, as in the prototype).
    pub fn bytes_per_item(&self) -> usize {
        self.row_bytes()
    }
}


/// Dot product of a prepared question with one copy (see `dot_u8_n`).
#[inline]
pub fn dot_u8(a: &[f32], c: &[u8]) -> f32 {
    dot_u8_n(a, [c])[0]
}

/// `dot_u8(a, c)` for several copies side by side. Each copy keeps 16 running sums: byte i (as a number) times
/// a[i] goes into sum i mod 16 by a fused multiply-add; the sums are then added one after another from 0 (sum 0,
/// then 1, ..., 15); the bytes past the last full block of 16 are multiply-added onto the result one by one.
/// Every copy's number is the same to the last bit as scoring it alone.
#[inline]
pub fn dot_u8_n<const N: usize>(a: &[f32], cs: [&[u8]; N]) -> [f32; N] {
    let n = a.len();
    assert!(cs.iter().all(|c| c.len() == n), "copies must have the question's length");
    let blocks = n / LANES;
    let mut out = [0f32; N];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every copy has n bytes (checked above) and loads stay below blocks * 16 <= n.
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [[vdupq_n_f32(0.0); 4]; N];
        let ap = a.as_ptr();
        for b in 0..blocks {
            let o = b * LANES;
            let x = [vld1q_f32(ap.add(o)), vld1q_f32(ap.add(o + 4)), vld1q_f32(ap.add(o + 8)), vld1q_f32(ap.add(o + 12))];
            for r in 0..N {
                let y = vld1q_u8(cs[r].as_ptr().add(o));
                let (lo, hi) = (vmovl_u8(vget_low_u8(y)), vmovl_high_u8(y));
                let f = [
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
                    vcvtq_f32_u32(vmovl_high_u16(lo)),
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
                    vcvtq_f32_u32(vmovl_high_u16(hi)),
                ];
                for k in 0..4 {
                    acc[r][k] = vfmaq_f32(acc[r][k], f[k], x[k]);
                }
            }
        }
        for r in 0..N {
            let mut lanes = [0f32; LANES];
            for k in 0..4 {
                vst1q_f32(lanes.as_mut_ptr().add(4 * k), acc[r][k]);
            }
            let mut s = 0f32;
            for v in lanes {
                s += v;
            }
            out[r] = s;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for r in 0..N {
        out[r] = portable_u8_blocks(a, cs[r], blocks);
    }
    for r in 0..N {
        for i in blocks * LANES..n {
            out[r] = (cs[r][i] as f32).mul_add(a[i], out[r]);
        }
    }
    out
}

/// `sum(a[i] * c[i])` for a 16-bit question, for several copies side by side. Whole numbers, so the sum is exact
/// and the same in any order; `prepare` keeps it inside 32 bits.
#[inline]
pub fn dot_u8_i16_n<const N: usize>(a: &[i16], cs: [&[u8]; N]) -> [i32; N] {
    let n = a.len();
    assert!(cs.iter().all(|c| c.len() == n), "copies must have the question's length");
    let blocks = n / LANES;
    let mut out = [0i32; N];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every copy has n bytes (checked above) and loads stay below blocks * 16 <= n.
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [[vdupq_n_s32(0); 4]; N];
        let ap = a.as_ptr();
        for b in 0..blocks {
            let o = b * LANES;
            let (qa, qb) = (vld1q_s16(ap.add(o)), vld1q_s16(ap.add(o + 8)));
            for r in 0..N {
                let y = vld1q_u8(cs[r].as_ptr().add(o));
                let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y)));
                let hi = vreinterpretq_s16_u16(vmovl_high_u8(y));
                acc[r][0] = vmlal_s16(acc[r][0], vget_low_s16(lo), vget_low_s16(qa));
                acc[r][1] = vmlal_high_s16(acc[r][1], lo, qa);
                acc[r][2] = vmlal_s16(acc[r][2], vget_low_s16(hi), vget_low_s16(qb));
                acc[r][3] = vmlal_high_s16(acc[r][3], hi, qb);
            }
        }
        for r in 0..N {
            out[r] = vaddvq_s32(vaddq_s32(vaddq_s32(acc[r][0], acc[r][1]), vaddq_s32(acc[r][2], acc[r][3])));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for r in 0..N {
        out[r] = (0..blocks * LANES).map(|i| a[i] as i32 * cs[r][i] as i32).sum();
    }
    for r in 0..N {
        for i in blocks * LANES..n {
            out[r] += a[i] as i32 * cs[r][i] as i32;
        }
    }
    out
}

/// `sum(a[i] * value i)` for a 16-bit question against 4-bit copies (see `Codes::bits` for the packing; `a` is
/// padded to whole blocks of 32), for several copies side by side. Exact whole numbers.
#[inline]
pub fn dot_u4_i16_n<const N: usize>(a: &[i16], cs: [&[u8]; N]) -> [i32; N] {
    let nb = cs[0].len();
    assert!(nb.is_multiple_of(16) && a.len() == nb * 2 && cs.iter().all(|c| c.len() == nb), "4-bit rows come in blocks of 16 bytes");
    let mut out = [0i32; N];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: each row has nb bytes and the question 2 * nb numbers (checked above); loads stay inside both.
    unsafe {
        use std::arch::aarch64::*;
        let low4 = vdupq_n_u8(0x0F);
        let mut acc = [[vdupq_n_s32(0); 4]; N];
        let ap = a.as_ptr();
        for b in 0..nb / 16 {
            let (o, ob) = (b * 32, b * 16);
            let q = [vld1q_s16(ap.add(o)), vld1q_s16(ap.add(o + 8)), vld1q_s16(ap.add(o + 16)), vld1q_s16(ap.add(o + 24))];
            for r in 0..N {
                let y = vld1q_u8(cs[r].as_ptr().add(ob));
                for (half, v) in [vandq_u8(y, low4), vshrq_n_u8::<4>(y)].into_iter().enumerate() {
                    let lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
                    let hi = vreinterpretq_s16_u16(vmovl_high_u8(v));
                    let (qa, qb) = (q[2 * half], q[2 * half + 1]);
                    acc[r][0] = vmlal_s16(acc[r][0], vget_low_s16(lo), vget_low_s16(qa));
                    acc[r][1] = vmlal_high_s16(acc[r][1], lo, qa);
                    acc[r][2] = vmlal_s16(acc[r][2], vget_low_s16(hi), vget_low_s16(qb));
                    acc[r][3] = vmlal_high_s16(acc[r][3], hi, qb);
                }
            }
        }
        for r in 0..N {
            out[r] = vaddvq_s32(vaddq_s32(vaddq_s32(acc[r][0], acc[r][1]), vaddq_s32(acc[r][2], acc[r][3])));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for r in 0..N {
        for i in 0..nb * 2 {
            let (block, w) = (i / 32, i % 32);
            let byte = cs[r][block * 16 + w % 16];
            out[r] += a[i] as i32 * if w < 16 { byte & 0x0F } else { byte >> 4 } as i32;
        }
    }
    out
}

/// `sum(a[i] * (c[i] - 128))` for an 8-bit question, for several copies side by side: the chip's signed dot
/// instruction multiplies and adds 16 pairs at once, so each byte is shifted by 128 first (the caller adds
/// `128 * sum(a)` back). Exact whole numbers.
#[inline]
pub fn dot_u8_i8_n<const N: usize>(a: &[i8], cs: [&[u8]; N]) -> [i32; N] {
    let n = a.len();
    assert!(cs.iter().all(|c| c.len() == n), "copies must have the question's length");
    let blocks = n / LANES;
    let mut out = [0i32; N];
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    // SAFETY: every copy has n bytes (checked above) and loads stay below blocks * 16 <= n; SDOT only reads and
    // writes the registers named.
    unsafe {
        use std::arch::aarch64::*;
        let flip = vdupq_n_u8(0x80);
        let mut acc = [[vdupq_n_s32(0); 2]; N];
        let ap = a.as_ptr();
        for b in 0..blocks {
            let o = b * LANES;
            let q = vld1q_s8(ap.add(o));
            for r in 0..N {
                let y = vreinterpretq_s8_u8(veorq_u8(vld1q_u8(cs[r].as_ptr().add(o)), flip));
                let k = b & 1; // two sums per copy, so one dot does not wait for the last
                std::arch::asm!("sdot {acc:v}.4s, {y:v}.16b, {q:v}.16b", acc = inout(vreg) acc[r][k], y = in(vreg) y,
                                q = in(vreg) q, options(pure, nomem, nostack, preserves_flags));
            }
        }
        for r in 0..N {
            out[r] = vaddvq_s32(vaddq_s32(acc[r][0], acc[r][1]));
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    for r in 0..N {
        out[r] = (0..blocks * LANES).map(|i| a[i] as i32 * (cs[r][i] as i32 - 128)).sum();
    }
    for r in 0..N {
        for i in blocks * LANES..n {
            out[r] += a[i] as i32 * (cs[r][i] as i32 - 128);
        }
    }
    out
}

/// The same sums one number at a time: the fallback on other chips, and the tests' bit-for-bit reference.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn portable_u8_blocks(a: &[f32], c: &[u8], blocks: usize) -> f32 {
    let mut acc = [0f32; LANES];
    for b in 0..blocks {
        for j in 0..LANES {
            acc[j] = (c[b * LANES + j] as f32).mul_add(a[b * LANES + j], acc[j]);
        }
    }
    let mut s = 0f32;
    for v in acc {
        s += v;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_copies_pack_unpack_and_sum_exactly() {
        let mut rng = crate::rng::Rng::new(9);
        let mut f = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        for &d in &[1usize, 16, 31, 32, 33, 100, 200, 768] {
            let rows: Vec<Vec<f32>> = (0..9).map(|_| (0..d).map(|_| f()).collect()).collect();
            let q: Vec<f32> = (0..d).map(|_| f()).collect();
            for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
                let mut c = Codes::train_bits(rows.iter().map(|r| r.as_slice()), d, metric, rows.len(), 4);
                c.qbits = 16;
                rows.iter().enumerate().for_each(|(i, r)| c.set(i, r));
                assert_eq!(c.row_bytes(), d.div_ceil(32) * 16);
                let mut p = Prepared::default();
                c.prepare(&q, &mut p);
                for i in 0..rows.len() {
                    // each number comes back within half a step, and the vector sum is the plain one
                    let back = c.decode(i);
                    assert!((0..d).all(|j| (back[j] - rows[i][j]).abs() <= 0.5 * c.step[j] + 1e-6), "d {d}");
                    let plain: i64 = (0..d).map(|j| p.a16[j] as i64 * c.value(i, j) as i64).sum();
                    assert_eq!(dot_u4_i16_n(&p.a16, [c.row(i)])[0] as i64, plain, "d {d}");
                    // and the distance is the one to the decoded copy, within the question's rounding
                    let exact: f32 = match metric {
                        Metric::Cos => 1.0 - dot(&q, &back),
                        Metric::Ip => -dot(&q, &back),
                        Metric::L2 => crate::metric::l2sq(&q, &back),
                    };
                    let bound = d as f32 * 0.5 * p.scale * 15.0 * if metric == Metric::L2 { 2.0 } else { 1.0 } + 1e-4;
                    assert!((c.distance(&p, i) - exact).abs() <= bound, "d {d} {metric:?}");
                }
                let four = c.distance_n(&p, [0, 2, 4, 8]);
                assert_eq!(four, [0, 2, 4, 8].map(|i| c.distance(&p, i)));
            }
        }
    }

    #[test]
    fn whole_number_questions_give_exact_sums_close_to_the_float_ones() {
        let mut rng = crate::rng::Rng::new(5);
        let mut f = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        for &d in &[1usize, 15, 16, 17, 33, 100, 200, 768, 1025] {
            let rows: Vec<Vec<f32>> = (0..9).map(|_| (0..d).map(|_| f()).collect()).collect();
            let q: Vec<f32> = (0..d).map(|_| f()).collect();
            for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
                for qbits in [16u8, 8] {
                    let mut c = Codes::train(rows.iter().map(|r| r.as_slice()), d, metric, rows.len());
                    rows.iter().enumerate().for_each(|(i, r)| c.set(i, r));
                    let exact = c.clone();
                    c.qbits = qbits;
                    let (mut p, mut pf) = (Prepared::default(), Prepared::default());
                    c.prepare(&q, &mut p);
                    exact.prepare(&q, &mut pf);
                    for i in 0..rows.len() {
                        // the vector sum is the plain whole-number sum
                        let row = c.row(i);
                        let plain: i64 = if qbits == 16 {
                            (0..d).map(|j| p.a16[j] as i64 * row[j] as i64).sum()
                        } else {
                            (0..d).map(|j| p.a8[j] as i64 * (row[j] as i64 - 128)).sum()
                        };
                        let got = if qbits == 16 { dot_u8_i16_n(&p.a16, [row])[0] } else { dot_u8_i8_n(&p.a8, [row])[0] };
                        assert_eq!(got as i64, plain, "d {d} qbits {qbits}");
                        // close to the float distance: each of the d terms is off by at most half a step of 255
                        let bound = d as f32 * 0.5 * p.scale * 255.0 * if metric == Metric::L2 { 2.0 } else { 1.0 } + 1e-5;
                        let (di, df) = (c.distance(&p, i), exact.distance(&pf, i));
                        assert!((di - df).abs() <= bound, "d {d} qbits {qbits} {metric:?}: {di} vs {df}");
                    }
                    // four side by side give each one's own number
                    let four = c.distance_n(&p, [0, 3, 5, 8]);
                    assert_eq!(four, [0, 3, 5, 8].map(|i| c.distance(&p, i)));
                }
            }
        }
    }

    #[test]
    fn vector_byte_sums_match_the_one_at_a_time_sums_bit_for_bit() {
        let mut r = crate::rng::Rng::new(12);
        for n in [0usize, 1, 5, 15, 16, 17, 100, 128, 200, 768, 1000] {
            let a: Vec<f32> = (0..n).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
            let cs: Vec<Vec<u8>> = (0..4).map(|_| (0..n).map(|_| r.below(256) as u8).collect()).collect();
            let four = dot_u8_n(&a, [&cs[0], &cs[1], &cs[2], &cs[3]]);
            for (j, c) in cs.iter().enumerate() {
                let blocks = n / LANES;
                let mut want = portable_u8_blocks(&a, c, blocks);
                for i in blocks * LANES..n {
                    want = (c[i] as f32).mul_add(a[i], want);
                }
                assert_eq!(dot_u8(&a, c).to_bits(), want.to_bits(), "n = {n}");
                assert_eq!(four[j].to_bits(), want.to_bits(), "n = {n}");
            }
        }
    }
    use crate::metric::normalize;

    fn rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn copies_decode_to_within_half_a_step() {
        let r = rows(200, 37, 3);
        let mut c = Codes::train(r.iter().map(|v| v.as_slice()), 37, Metric::L2, r.len());
        for (i, v) in r.iter().enumerate() {
            c.set(i, v);
            let back = c.decode(i);
            for j in 0..37 {
                assert!((back[j] - v[j]).abs() <= 0.5 * c.step[j] + 1e-6);
            }
        }
    }

    #[test]
    fn copy_distances_match_distances_to_the_decoded_numbers() {
        for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
            let mut r = rows(100, 48, 7);
            if metric == Metric::Cos {
                r.iter_mut().for_each(|v| normalize(v));
            }
            let mut c = Codes::train(r.iter().map(|v| v.as_slice()), 48, metric, r.len());
            for (i, v) in r.iter().enumerate() {
                c.set(i, v);
            }
            let q = &r[5];
            let mut p = Prepared::default();
            c.prepare(q, &mut p);
            for i in 0..r.len() {
                let want = metric.distance(q, &c.decode(i));
                let got = c.distance(&p, i);
                assert!((want - got).abs() < 1e-4 * (1.0 + want.abs()), "{metric:?} {i}: {want} vs {got}");
            }
        }
    }

    #[test]
    fn values_outside_the_trained_range_are_clipped() {
        let r = [vec![0.0f32, 1.0], vec![1.0, 2.0]];
        let mut c = Codes::train(r.iter().map(|v| v.as_slice()), 2, Metric::Ip, 3);
        c.set(2, &[5.0, -5.0]);
        assert_eq!(c.row(2), &[255, 0]);
    }
}
