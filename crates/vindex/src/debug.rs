//! Helpers for the step checks: run pieces of the index on inputs supplied from outside.

use rayon::prelude::*;

use crate::graph::Graph;
use crate::metric::Metric;
use crate::search::{finish_plain, seed, walk, Scratch};
use crate::store::Store;

/// Search every question on a given map with full numbers and a fixed list size `l`.
/// Returns (ids, dists, checks per question); ids are slots, -1 where fewer than k were found.
pub fn search_full_fixed(g: &Graph, x: &Store, metric: Metric, entry: u32, q: &[f32], l: usize, k: usize) -> (Vec<i64>, Vec<f32>, Vec<u64>) {
    let d = x.dim();
    let nq = q.len() / d;
    let mut ids = vec![-1i64; nq * k];
    let mut dists = vec![f32::INFINITY; nq * k];
    let mut checks = vec![0u64; nq];
    ids.par_chunks_mut(k)
        .zip(dists.par_chunks_mut(k))
        .zip(checks.par_iter_mut())
        .enumerate()
        .for_each_init(Scratch::default, |s, (i, ((oi, od), oc))| {
            let qi = &q[i * d..(i + 1) * d];
            s.visited.begin(g.capacity());
            s.list.reset(l + 1);
            let mut dist = |j: u32| metric.distance(qi, x.row(j as usize));
            let c0 = seed(g, &mut s.list, &mut s.visited, &[entry], &mut dist);
            let (c1, _) = walk(g, &mut s.list, &mut s.visited, &mut dist, l + 1, None);
            finish_plain(g, &s.list, k, oi, od);
            *oc = c0 + c1;
        });
    (ids, dists, checks)
}
