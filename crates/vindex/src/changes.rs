//! Changes one at a time (Fix 7; DESIGN.md section 9): add, delete with repair right away, update in place.

use crate::build::{back_links, out_links_one, LinkScratch};
use crate::graph::Graph;
use crate::index::{space_of, Error, Index, Repair, Result, NO_ID};
use crate::metric::{normalize, Metric};
use crate::prune::{prune, BuildSpace, PruneScratch};
use crate::score::BuildScore;
use crate::search::{seed, walk};
use crate::store::Store;

/// Delete slot `u` and repair the map around it right away:
/// 1. walk to u's own numbers (build space, list size `l`), recording the opened items; then mark u dead;
/// 2. each opened item that links to u drops that link and takes up to `n_replace` of u's alive links (closest to
///    it first, not already linked): `Append` adds them, pruning back to `r` only if over the slots; `Prune` prunes
///    the merged list as a build does; `Occlude` adds each only if no link the item keeps (or has just taken)
///    already reaches that way (alpha * d(y, x) <= d(w, x)). Returns the checks made.
#[allow(clippy::too_many_arguments)]
pub fn delete_repair(
    g: &mut Graph,
    sp: &BuildSpace,
    start: u32,
    u: u32,
    l: usize,
    r: usize,
    alpha: f32,
    n_replace: usize,
    repair: Repair,
    ls: &mut LinkScratch,
) -> u64 {
    let v = sp.row(u);
    ls.search.visited.begin(g.capacity());
    ls.search.list.reset(l + 1);
    ls.search.pool.clear();
    let mut sc = BuildScore { sp: *sp, v };
    let mut checks = seed(g, &mut ls.search.list, &mut ls.search.visited, &[start], &mut sc);
    checks += walk(g, &mut ls.search.list, &mut ls.search.visited, &mut sc, l + 1, Some((&mut ls.search.pool, 4 * l + 64))).0;
    g.alive[u as usize] = 0;
    let ud: Vec<u32> = g.links(u as usize).to_vec();
    let pool = std::mem::take(&mut ls.search.pool);
    let mut ps = PruneScratch::default();
    for &w in &pool {
        if w == u {
            continue;
        }
        let Some(has) = g.links(w as usize).iter().position(|&x| x == u) else { continue };
        let mut lw: Vec<u32> = g.links(w as usize).to_vec();
        lw.remove(has);
        let mut cand: Vec<(f32, u32)> = Vec::new();
        for &x in &ud {
            if x == w || !g.is_alive(x as usize) || lw.contains(&x) {
                continue;
            }
            cand.push((sp.dist(w, x), x));
            checks += 1;
        }
        cand.sort_by(|a, b| a.0.total_cmp(&b.0));
        let take = n_replace.min(cand.len());
        if repair == Repair::Occlude {
            for &(dx, x) in &cand[..take] {
                if lw.len() >= g.slots {
                    break;
                }
                checks += lw.len() as u64;
                if lw.iter().all(|&y| alpha * sp.dist(y, x) > dx) {
                    lw.push(x);
                }
            }
        } else if repair == Repair::Append && lw.len() + take <= g.slots {
            lw.extend(cand[..take].iter().map(|c| c.1));
        } else {
            let mut merged = lw.clone();
            merged.extend(cand[..take].iter().map(|c| c.1));
            let md: Vec<f32> = merged.iter().map(|&x| sp.dist(w, x)).collect();
            let mut out = Vec::with_capacity(r);
            prune(sp, &merged, &md, r, alpha, &[], &mut out, &mut ps);
            lw = out;
        }
        g.set_links(w as usize, &lw);
    }
    ls.search.pool = pool;
    g.deg[u as usize] = 0;
    checks
}

/// Remove links that point at dead slots, for the given items.
pub fn drop_dead_links(g: &mut Graph, nodes: impl Iterator<Item = usize>) {
    for w in nodes {
        let keep: Vec<u32> = g.links(w).iter().copied().filter(|&x| g.is_alive(x as usize)).collect();
        g.set_links(w, &keep);
    }
    // extra links (part 3): a dead item keeps none, and no one keeps one to a dead item
    if g.has_extra() {
        let lists: Vec<(u32, Vec<u32>)> = (0..g.capacity())
            .filter(|&u| g.is_alive(u) && !g.extra_links(u).is_empty())
            .map(|u| (u as u32, g.extra_links(u).iter().copied().filter(|&x| g.is_alive(x as usize)).collect()))
            .collect();
        g.set_extra(&lists);
    }
}

impl Index {
    fn prepared_row(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.dim {
            return Err(Error(format!("an item needs {} numbers, got {}", self.dim, x.len())));
        }
        crate::index::check_finite(x, "the item's numbers")?;
        if self.n_used == 0 {
            return Err(Error("build the index before changing it".into()));
        }
        if matches!(self.store, Store::Mapped { .. }) {
            return Err(Error("the full numbers are on SSD (read-only); load with numbers=\"ram\" to change items".into()));
        }
        let mut v = x.to_vec();
        if self.metric == Metric::Cos {
            normalize(&mut v);
        }
        Ok(v)
    }

    /// Room for more items: every per-slot array grows together.
    fn grow(&mut self, capacity: usize) {
        self.graph.grow(capacity);
        self.store.grow(capacity);
        if let Some(f) = &mut self.flipped {
            f.resize(capacity * self.dim, 0.0);
        }
        if let Some(c) = &mut self.codes {
            c.reserve_slots(capacity);
        }
        self.ext.resize(capacity, NO_ID);
    }

    /// A slot for a new item: a free one; else, when 1,000 freed slots wait (or the index is full), clear every
    /// link to a dead slot in one pass and recycle them; else a new slot (growing the index if needed).
    fn new_slot(&mut self) -> u32 {
        if let Some(u) = self.free.pop() {
            return u;
        }
        if !self.pending_free.is_empty() && (self.pending_free.len() >= self.s.cleanup_batch || self.n_used >= self.graph.capacity()) {
            drop_dead_links(&mut self.graph, 0..self.n_used);
            self.free.append(&mut self.pending_free);
            return self.free.pop().expect("not empty");
        }
        if self.n_used >= self.graph.capacity() {
            let cap = self.graph.capacity();
            self.grow((cap + cap / 4).max(cap + 16));
        }
        self.n_used += 1;
        (self.n_used - 1) as u32
    }

    /// Put numbers `v` (already scaled) in slot `u` and link it in, as one new item of a build batch.
    fn insert_slot(&mut self, u: u32, v: &[f32]) {
        let (d, ui) = (self.dim, u as usize);
        self.store.set(ui, v);
        if let Some(f) = &mut self.flipped {
            Index::flip_into(v, &mut f[ui * d..(ui + 1) * d]);
        }
        let (r, l, alpha) = (self.s.r, self.s.l_build, self.s.alpha);
        let sp = space_of(&self.flipped, &self.store, d, self.metric);
        let mut ls = LinkScratch::default();
        let mut out = Vec::with_capacity(r);
        let checks = out_links_one(&self.graph, &sp, self.start, u, l, r, alpha, &mut ls, &mut out);
        self.graph.set_links(ui, &out);
        if self.graph.has_extra() {
            self.graph.extra_n[ui] = 0; // a reused slot does not keep the deleted item's extra links
        }
        self.graph.alive[ui] = 1;
        back_links(&mut self.graph, &sp, &[u], r, alpha);
        if let Some(c) = &mut self.codes {
            c.set(ui, v);
        }
        self.change_checks += checks;
    }

    /// Count one change; every `qlinks_refresh_every` changes, redo the question links (when they are on).
    fn note_change(&mut self) {
        self.changes += 1;
        let every = self.s.qlinks_refresh_every;
        if every > 0 && self.changes.is_multiple_of(every) && self.train_q.is_some() {
            let c = self.question_links_round();
            self.change_checks += c;
        }
    }

    /// The start moves to the deleted item's first alive neighbour (or, failing that, any alive item).
    fn move_start(&mut self, u: u32) {
        if let Some(&w) = self.graph.links(u as usize).iter().find(|&&w| w != u && self.graph.is_alive(w as usize)) {
            self.start = w;
            return;
        }
        if let Some(w) = (0..self.n_used as u32).find(|&w| w != u && self.graph.is_alive(w as usize)) {
            self.start = w;
        }
    }

    fn delete_inner(&mut self, id: u64, reuse_slot: bool) -> Result<u32> {
        let Some(u) = self.slot_of.remove(&id) else {
            return Err(Error(format!("no item with id {id}")));
        };
        if u == self.start {
            self.move_start(u);
        }
        let (r, l, alpha, nr) = (self.s.r, self.s.l_build, self.s.alpha, self.s.n_replace);
        let sp = space_of(&self.flipped, &self.store, self.dim, self.metric);
        let mut ls = LinkScratch::default();
        let checks = delete_repair(&mut self.graph, &sp, self.start, u, l, r, alpha, nr, self.s.repair, &mut ls);
        self.change_checks += checks;
        self.ext[u as usize] = NO_ID;
        if reuse_slot {
            self.pending_free.push(u);
            self.note_change();
        }
        Ok(u)
    }

    /// Add an item. Its id must be new.
    pub fn add(&mut self, id: u64, x: &[f32]) -> Result<()> {
        let v = self.prepared_row(x)?;
        if self.slot_of.contains_key(&id) {
            return Err(Error(format!("id {id} is already in the index; use update")));
        }
        let u = self.new_slot();
        self.ext[u as usize] = id;
        self.slot_of.insert(id, u);
        self.insert_slot(u, &v);
        self.note_change();
        Ok(())
    }

    /// Delete an item: its neighbours are repaired right away and its slot is recycled later.
    pub fn delete(&mut self, id: u64) -> Result<()> {
        if self.n_used == 0 {
            return Err(Error("the index is empty".into()));
        }
        self.delete_inner(id, true).map(|_| ())
    }

    /// Replace an item's numbers: delete without freeing the slot, then link the new numbers into the same slot.
    pub fn update(&mut self, id: u64, x: &[f32]) -> Result<()> {
        let v = self.prepared_row(x)?;
        let u = self.delete_inner(id, false)?;
        drop_dead_links(&mut self.graph, std::iter::once(u as usize));
        self.ext[u as usize] = id;
        self.slot_of.insert(id, u);
        self.insert_slot(u, &v);
        self.note_change();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::index::{Effort, Index, Settings, Switch};
    use crate::metric::Metric;
    use crate::rng::Rng;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    fn small_index(x: &[f32], d: usize) -> Index {
        let s = Settings { r: 16, l_build: 48, code_bits: 8, learned_effort: false, question_links: Switch::Off, ..Settings::default() };
        let mut idx = Index::new(d, Metric::Cos, s).unwrap();
        let ids: Vec<u64> = (0..(x.len() / d) as u64).collect();
        idx.build(x, &ids, None).unwrap();
        idx
    }

    #[test]
    fn deleted_items_are_never_returned_and_updates_use_new_numbers() {
        let d = 24;
        let x = data(3000, d, 3);
        let q = data(50, d, 4);
        let mut idx = small_index(&x, d);
        let gone: Vec<u64> = (0..400).step_by(2).collect();
        for &i in &gone {
            idx.delete(i).unwrap();
        }
        // move item 1 onto question 0: it must now come back first for question 0
        idx.update(1, &q[..d]).unwrap();
        let out = idx.search(&q, 10, Effort::List(200)).unwrap();
        assert!(out.ids.iter().all(|i| !gone.contains(&(*i as u64))));
        assert_eq!(out.ids[0], 1);
        assert!(out.dists[0].abs() < 1e-4);
        // the deleted ids can no longer be deleted or updated; a deleted id can be added again
        assert!(idx.delete(0).is_err());
        assert!(idx.update(0, &q[..d]).is_err());
        idx.add(0, &q[d..2 * d]).unwrap();
        let out = idx.search(&q[d..2 * d], 1, Effort::List(100)).unwrap();
        assert_eq!(out.ids[0], 0);
    }

    #[test]
    fn freed_slots_are_recycled_and_the_index_grows_when_full() {
        let d = 16;
        let x = data(1200, d, 5);
        let mut idx = small_index(&x, d);
        idx.s.cleanup_batch = 100;
        for i in 0..150u64 {
            idx.delete(i).unwrap();
        }
        let used = idx.n_used;
        let extra = data(150, d, 6);
        for j in 0..150usize {
            idx.add(10_000 + j as u64, &extra[j * d..(j + 1) * d]).unwrap();
        }
        assert!(idx.n_used < used + 150, "slots were recycled ({} used before, {} after)", used, idx.n_used);
        // past capacity: the index grows instead of failing
        let more = data(400, d, 7);
        for j in 0..400usize {
            idx.add(20_000 + j as u64, &more[j * d..(j + 1) * d]).unwrap();
        }
        assert_eq!(idx.stats().items, 1200 - 150 + 150 + 400);
        assert_eq!(idx.stats().unreachable, 0);
    }

    #[test]
    fn every_repair_keeps_deleted_items_out_and_finds_about_what_a_fresh_build_finds() {
        use crate::index::Repair;
        let d = 16;
        let x = data(4000, d, 8);
        let q = data(100, d, 9);
        let mut truth = Vec::new();
        for qi in q.chunks(d) {
            let mut all: Vec<(f32, u64)> = (1000..4000u64).map(|i| (Metric::Cos.distance(qi, &x[i as usize * d..(i as usize + 1) * d]), i)).collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            truth.push(all[..10].iter().map(|a| a.1).collect::<Vec<_>>());
        }
        let found = |ids: &[i64], off: i64| -> usize {
            ids.chunks(10).zip(&truth).map(|(f, t)| f.iter().filter(|&&i| i >= 0 && t.contains(&((i + off) as u64))).count()).sum()
        };
        let fresh = small_index(&x[1000 * d..], d);
        let fresh_found = found(&fresh.search(&q, 10, Effort::List(100)).unwrap().ids, 1000);
        for repair in [Repair::Append, Repair::Prune, Repair::Occlude] {
            let mut idx = small_index(&x, d);
            idx.s.repair = repair;
            for i in 0..1000u64 {
                idx.delete(i).unwrap();
            }
            let out = idx.search(&q, 10, Effort::List(100)).unwrap();
            assert!(out.ids.iter().all(|&i| i >= 1000), "{repair:?}: a deleted item came back");
            let got = found(&out.ids, 0);
            assert!(got + 60 >= fresh_found, "{repair:?}: {got} of 1000, a fresh build {fresh_found}");
        }
    }
}
