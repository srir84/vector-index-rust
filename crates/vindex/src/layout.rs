//! Renumbering items so linked items sit near each other in memory (part 2, DESIGN.md section 18).
//!
//! After the build, items take the order in which a breadth-first walk of the links from the start item meets them.
//! A search then reads copies, link lists and full numbers from fewer, closer memory pages. Answers and checks do
//! not change: the walk depends on the links, their order and the distances, never on slot numbers.

use std::collections::VecDeque;

use crate::index::{Index, NO_ID};
use crate::store::Store;

/// Rows of `width` values moved to their new places: row `new` gets old row `order[new]`; rows past the order keep
/// their place (spare slots).
fn permute_rows<T: Copy>(v: &mut [T], width: usize, order: &[u32]) {
    let moved: Vec<T> = order.iter().flat_map(|&o| v[o as usize * width..(o as usize + 1) * width].iter().copied()).collect();
    v[..moved.len()].copy_from_slice(&moved);
}

impl Index {
    /// Put the used slots in breadth-first order from the start item (unreached slots last, in their old order),
    /// following each item's links, then its extra links (part 3), so popular items end up near each other.
    pub(crate) fn renumber(&mut self) {
        let n = self.n_used;
        if n < 2 {
            return;
        }
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut queue = VecDeque::from([self.start]);
        seen[self.start as usize] = true;
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in self.graph.links(u as usize).iter().chain(self.graph.extra_links(u as usize)) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    queue.push_back(v);
                }
            }
        }
        order.extend((0..n as u32).filter(|&u| !seen[u as usize]));
        let mut new_of = vec![0u32; n];
        for (new, &old) in order.iter().enumerate() {
            new_of[old as usize] = new as u32;
        }
        let d = self.dim;
        if let Store::Ram { data, .. } = &mut self.store {
            permute_rows(data, d, &order);
        }
        if let Some(b) = &mut self.flipped {
            permute_rows(b, d, &order);
        }
        if self.graph.has_extra() {
            let lists: Vec<(u32, Vec<u32>)> = (0..n)
                .filter(|&s| !self.graph.extra_links(order[s] as usize).is_empty())
                .map(|s| (s as u32, self.graph.extra_links(order[s] as usize).iter().map(|&v| new_of[v as usize]).collect()))
                .collect();
            self.graph.set_extra(&lists);
        }
        let s = self.graph.slots;
        permute_rows(&mut self.graph.nbrs, s, &order);
        permute_rows(&mut self.graph.deg, 1, &order);
        permute_rows(&mut self.graph.alive, 1, &order);
        for u in 0..n {
            let k = self.graph.deg[u] as usize;
            for v in &mut self.graph.nbrs[u * s..u * s + k] {
                *v = new_of[*v as usize];
            }
        }
        if let Some(c) = &mut self.codes {
            let rb = c.row_bytes();
            permute_rows(&mut c.data, rb, &order);
            if !c.norm2.is_empty() {
                permute_rows(&mut c.norm2, 1, &order);
            }
        }
        permute_rows(&mut self.ext, 1, &order);
        self.slot_of = (0..n).filter(|&u| self.graph.is_alive(u) && self.ext[u] != NO_ID).map(|u| (self.ext[u], u as u32)).collect();
        for f in self.free.iter_mut().chain(self.pending_free.iter_mut()) {
            *f = new_of[*f as usize];
        }
        self.start = new_of[self.start as usize];
    }
}

#[cfg(test)]
mod tests {
    use crate::index::{Effort, Index, Settings, Tune};
    use crate::metric::Metric;
    use crate::rng::Rng;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    #[test]
    fn renumbering_changes_no_answer() {
        let d = 12;
        let (x, tq, q) = (data(3000, d, 1), data(600, d, 2), data(80, d, 3));
        for metric in [Metric::Cos, Metric::Ip, Metric::L2] {
            let mut out = Vec::new();
            for renumber in [false, true] {
                let s = Settings { r: 12, l_build: 40, effort_questions: 300, renumber, tune: Tune::Off, ..Settings::default() };
                let mut a = Index::new(d, metric, s).unwrap();
                a.build(&x, &(0..3000u64).map(|i| i * 7 + 1).collect::<Vec<_>>(), Some(&tq)).unwrap();
                let r1 = a.search(&q, 10, Effort::Target(0.95)).unwrap();
                let r2 = a.search(&q, 10, Effort::List(40)).unwrap();
                out.push((r1.ids, r1.dists, r1.checks, r2.ids, r2.dists, r2.checks, a.stats().build_checks));
                if renumber {
                    // linked items now sit near each other: the start item first, its links right after
                    assert_eq!(a.start, 0);
                    let first: Vec<u32> = a.graph.links(0).to_vec();
                    assert!(first.iter().all(|&v| (v as usize) <= first.len()), "{first:?}");
                }
            }
            assert_eq!(out[0], out[1], "{metric:?}");
        }
    }
}
