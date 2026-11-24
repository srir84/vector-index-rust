//! Question links (Fix 3, automatic; Fix 8, twice) — DESIGN.md section 8.

use rayon::prelude::*;

use crate::graph::Graph;
use crate::metric::dot;
use crate::prune::{prune, BuildSpace, PruneScratch};
use crate::rng::Rng;
use crate::store::Store;

/// How many questions and items the "do questions look unlike items?" test samples.
pub const SAMPLE: usize = 256;

/// The test: median angle distance from a sampled training question to its nearest item, over the median
/// from a sampled item to its nearest other item (about 1.0 when questions look like items).
/// Returns (ratio, the items' median, checks made).
pub fn questions_unlike_items(store: &Store, alive: &[u32], train_q: &[f32], dim: usize, seed: u64) -> (f32, f32, u64) {
    let nq = train_q.len() / dim;
    let mut rng = Rng::new(seed);
    let qs = rng.sample(nq, SAMPLE.min(nq));
    let its = rng.sample(alive.len(), SAMPLE.min(alive.len()));
    let inv_norm = inv_norms(store, alive);
    let mut dq: Vec<f32> =
        qs.par_iter().map(|&i| nearest_angle(store, alive, &inv_norm, &train_q[i * dim..(i + 1) * dim], None)).collect();
    let mut di: Vec<f32> =
        its.par_iter().map(|&i| nearest_angle(store, alive, &inv_norm, store.row(alive[i] as usize), Some(i))).collect();
    let checks = ((qs.len() + its.len()) * alive.len()) as u64;
    let gap = median(&mut di);
    (median(&mut dq) / gap.max(1e-12), gap, checks)
}

/// The second half of the test alone: the median angle distance from a sampled item to its nearest other item,
/// for indexes built without the test (the usage stats compare questions with it). Returns (median, checks made).
pub fn items_gap(store: &Store, alive: &[u32], seed: u64) -> (f32, u64) {
    let its = Rng::new(seed).sample(alive.len(), SAMPLE.min(alive.len()));
    let inv_norm = inv_norms(store, alive);
    let mut di: Vec<f32> =
        its.par_iter().map(|&i| nearest_angle(store, alive, &inv_norm, store.row(alive[i] as usize), Some(i))).collect();
    (median(&mut di), (its.len() * alive.len()) as u64)
}

fn inv_norms(store: &Store, alive: &[u32]) -> Vec<f32> {
    alive
        .par_iter()
        .map(|&u| {
            let r = store.row(u as usize);
            let n = dot(r, r).sqrt();
            if n > 0.0 { 1.0 / n } else { 0.0 }
        })
        .collect()
}

/// Angle distance to the nearest item: 1 - the largest cosine over the items (skipping position `skip`).
fn nearest_angle(store: &Store, alive: &[u32], inv_norm: &[f32], v: &[f32], skip: Option<usize>) -> f32 {
    let vn = dot(v, v).sqrt().max(1e-30);
    let mut best = f32::NEG_INFINITY;
    for (i, &u) in alive.iter().enumerate() {
        if Some(i) == skip {
            continue;
        }
        let c = dot(v, store.row(u as usize)) * inv_norm[i] / vn;
        if c > best {
            best = c;
        }
    }
    1.0 - best
}

/// numpy's median: the middle value, or the mean of the two middle values.
pub fn median(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        ((v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0) as f32
    }
}

/// Link each training question's answers (DESIGN.md section 8, steps 2-3). `answers[q]` holds the slots of
/// question q's top answers, closest first (negative = none). The closest one (the hub) gets the others as
/// candidates and each of them gets the hub; every item with candidates keeps a spread-out subset of at most
/// `max_forced` first, then is pruned back to `r` together with its current links.
pub fn link_answers(g: &mut Graph, sp: &BuildSpace, answers: &[Vec<i64>], r: usize, alpha: f32, max_forced: usize) -> usize {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for row in answers {
        let row: Vec<u32> = row.iter().filter(|&&v| v >= 0).map(|&v| v as u32).collect();
        if row.len() < 2 {
            continue;
        }
        let h = row[0];
        for &o in &row[1..] {
            pairs.push((h, o));
            pairs.push((o, h));
        }
    }
    pairs.sort_by_key(|p| p.0); // stable: each item's candidates stay in question order, as in the prototype
    let mut groups: Vec<(u32, usize, usize)> = Vec::new();
    let mut a = 0;
    while a < pairs.len() {
        let w = pairs[a].0;
        let mut b = a;
        while b < pairs.len() && pairs[b].0 == w {
            b += 1;
        }
        groups.push((w, a, b));
        a = b;
    }
    let gr: &Graph = g;
    let lists: Vec<(u32, Vec<u32>)> = groups
        .par_iter()
        .map_init(PruneScratch::default, |ps, &(w, a, b)| {
            let mut fz: Vec<u32> = Vec::new();
            for &(_, s) in &pairs[a..b] {
                if s != w && gr.is_alive(s as usize) && !fz.contains(&s) {
                    fz.push(s);
                }
            }
            let fd: Vec<f32> = fz.iter().map(|&s| sp.dist(w, s)).collect();
            let mut fsel = Vec::with_capacity(max_forced);
            prune(sp, &fz, &fd, max_forced, alpha, &[], &mut fsel, ps);
            let mut allp: Vec<u32> = gr.links(w as usize).iter().copied().filter(|&x| gr.is_alive(x as usize)).collect();
            let m = allp.len();
            for &f in &fsel {
                if !allp[..m].contains(&f) {
                    allp.push(f);
                }
            }
            let ad: Vec<f32> = allp.iter().map(|&x| sp.dist(w, x)).collect();
            let mut out = Vec::with_capacity(r);
            prune(sp, &allp, &ad, r, alpha, &fsel, &mut out, ps);
            (w, out)
        })
        .collect();
    let touched = lists.len();
    for (w, l) in lists {
        g.set_links(w as usize, &l);
    }
    touched
}

/// Extra links for popular items (part 3, DESIGN.md section 22), from the training questions' answers
/// (`answers[q]`: slots, closest first). Every item in some question's first `top` answers is popular. It gets up
/// to `max` extra links: the popular items that share a question's first `top` answers with it most often (ties by
/// slot), leaving out its links already and deleted items. None when the answers do not bunch up: the distinct
/// items among them are more than `share` of the most there could be (`min(alive items, top x questions)`).
pub fn popular_links(g: &Graph, answers: &[Vec<i64>], alive: usize, top: usize, max: usize, share: f64) -> Option<Vec<(u32, Vec<u32>)>> {
    let mut seen = vec![false; g.capacity()];
    let (mut distinct, mut nq) = (0usize, 0usize);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for row in answers {
        let row: Vec<u32> = row.iter().filter(|&&v| v >= 0).take(top).map(|&v| v as u32).collect();
        if row.is_empty() {
            continue;
        }
        nq += 1;
        for &a in &row {
            if !seen[a as usize] {
                seen[a as usize] = true;
                distinct += 1;
            }
            pairs.extend(row.iter().filter(|&&b| b != a).map(|&b| (a, b)));
        }
    }
    if distinct as f64 > share * alive.min(top * nq).max(1) as f64 {
        return None;
    }
    pairs.sort_unstable();
    let mut lists = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        let a = pairs[i].0;
        let mut shared: Vec<(u32, u32)> = Vec::new(); // (how often, item)
        while i < pairs.len() && pairs[i].0 == a {
            let b = pairs[i].1;
            let mut c = 0;
            while i < pairs.len() && pairs[i] == (a, b) {
                c += 1;
                i += 1;
            }
            shared.push((c, b));
        }
        shared.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
        let have = g.links(a as usize);
        let l: Vec<u32> =
            shared.iter().map(|s| s.1).filter(|b| !have.contains(b) && g.is_alive(*b as usize)).take(max).collect();
        if !l.is_empty() {
            lists.push((a, l));
        }
    }
    Some(lists)
}
