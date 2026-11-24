//! Linking new items into the map (DESIGN.md section 7): used by the build and by adds.

use rayon::prelude::*;

use crate::graph::Graph;
use crate::prune::{prune, BuildSpace, PruneScratch};
use crate::score::{BuildScore, Score};
use crate::search::{seed, walk, Scratch};

/// Working memory for linking one item.
#[derive(Default)]
pub struct LinkScratch {
    pub search: Scratch,
    pub prune: PruneScratch,
    pub cand: Vec<u32>,
    pub cand_d: Vec<f32>,
}

/// Out-links for one new item `u`: walk to its own numbers from `start` with list size `l` in build space,
/// then prune the opened items (other than itself, alive only) to at most `r`. Returns the checks made.
pub fn out_links_one(g: &Graph, sp: &BuildSpace, start: u32, u: u32, l: usize, r: usize, alpha: f32, s: &mut LinkScratch, out: &mut Vec<u32>) -> u64 {
    let v = sp.row(u);
    s.search.visited.begin(g.capacity());
    s.search.list.reset(l + 1);
    s.search.pool.clear();
    let mut sc = BuildScore { sp: *sp, v };
    let c0 = seed(g, &mut s.search.list, &mut s.search.visited, &[start], &mut sc);
    let (c1, _) = walk(g, &mut s.search.list, &mut s.search.visited, &mut sc, l + 1, Some((&mut s.search.pool, 4 * l + 64)));
    s.cand.clear();
    for &w in &s.search.pool {
        if w == u || !g.is_alive(w as usize) {
            continue;
        }
        s.cand.push(w);
    }
    s.cand_d.resize(s.cand.len(), 0.0);
    sc.many(&s.cand, &mut s.cand_d);
    prune(sp, &s.cand, &s.cand_d, r, alpha, &[], out, &mut s.prune);
    c0 + c1
}

/// Out-links for a batch of new items, in parallel, against the map as it is. Returns (links, checks).
pub fn out_links_batch(g: &Graph, sp: &BuildSpace, start: u32, items: &[u32], l: usize, r: usize, alpha: f32) -> (Vec<Vec<u32>>, u64) {
    let res: Vec<(Vec<u32>, u64)> = items
        .par_iter()
        .map_init(LinkScratch::default, |s, &u| {
            let mut out = Vec::with_capacity(r);
            let c = out_links_one(g, sp, start, u, l, r, alpha, s, &mut out);
            (out, c)
        })
        .collect();
    let checks = res.iter().map(|x| x.1).sum();
    (res.into_iter().map(|x| x.0).collect(), checks)
}

/// Give every item a new item links to a link back. A list that grows past the slots is pruned back to `r`,
/// with its current alive links plus the new ones as candidates. Lists are worked out in parallel, then written.
pub fn back_links(g: &mut Graph, sp: &BuildSpace, new_items: &[u32], r: usize, alpha: f32) {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for &u in new_items {
        for &t in g.links(u as usize) {
            pairs.push((t, u));
        }
    }
    pairs.sort_by_key(|p| p.0); // stable: each target's sources stay in batch order, as in the prototype
    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut a = 0;
    while a < pairs.len() {
        let t = pairs[a].0;
        let mut b = a;
        while b < pairs.len() && pairs[b].0 == t {
            b += 1;
        }
        groups.push((t, a, b));
        a = b;
    }
    let slots = g.slots;
    let gr: &Graph = g;
    let lists: Vec<(u32, Vec<u32>)> = groups
        .par_iter()
        .map_init(|| (PruneScratch::default(), Vec::<f32>::new()), |(ps, md), &(t, a, b)| {
            let mut merged: Vec<u32> = gr.links(t as usize).iter().copied().filter(|&x| gr.is_alive(x as usize)).collect();
            for &(_, s) in &pairs[a..b] {
                if s != t && !merged.contains(&s) {
                    merged.push(s);
                }
            }
            if merged.len() <= slots {
                return (t, merged);
            }
            md.clear();
            md.extend(merged.iter().map(|&x| sp.dist(t, x)));
            let mut out = Vec::with_capacity(r);
            prune(sp, &merged, md, r, alpha, &[], &mut out, ps);
            (t, out)
        })
        .collect();
    for (t, l) in lists {
        g.set_links(t as usize, &l);
    }
}
