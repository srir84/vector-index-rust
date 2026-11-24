//! A small seeded random generator (SplitMix64), so builds are repeatable without extra libraries.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A whole number in 0..n (n > 0), without bias.
    pub fn below(&mut self, n: u64) -> u64 {
        let zone = u64::MAX - u64::MAX % n;
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % n;
            }
        }
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// `k` distinct numbers from 0..n.
    pub fn sample(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut all: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + self.below((n - i) as u64) as usize;
            all.swap(i, j);
        }
        all.truncate(k);
        all
    }
}
