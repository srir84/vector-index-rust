//! The walk (DESIGN.md section 4): a list sorted by distance; open the closest entry not yet opened,
//! add its unseen alive links, repeat until every entry within the limit is opened.

use crate::graph::Graph;
use crate::score::Score;

/// One list entry: a distance, and a slot whose top bit marks "opened". Kept together so that making room for a
/// new entry moves one array, not three.
#[derive(Clone, Copy, Default)]
struct Entry {
    d: f32,
    tag: u32,
}

const OPENED: u32 = 1 << 31;

/// The walk's list. Equal distances keep arrival order reversed (a new entry goes before equal ones),
/// as in the prototype's `list_insert`. Slots must be below 2^31 (the top bit is the "opened" mark).
#[derive(Default)]
pub struct SortedList {
    e: Vec<Entry>,
    pub size: usize,
    cap: usize,
    /// Every entry before `cursor` is opened (lets us skip the prototype's rescan from the top).
    cursor: usize,
}

impl SortedList {
    pub fn reset(&mut self, cap: usize) {
        if self.e.len() < cap + 1 {
            self.e.resize(cap + 1, Entry::default());
        }
        self.size = 0;
        self.cap = cap;
        self.cursor = 0;
    }

    /// Slot of entry `j`.
    #[inline]
    pub fn id(&self, j: usize) -> u32 {
        self.e[j].tag & !OPENED
    }

    /// Distance of entry `j`.
    #[inline]
    pub fn dist(&self, j: usize) -> f32 {
        self.e[j].d
    }

    /// Keep at most `cap` entries from now on. Results are unchanged as long as `cap` is at least every later
    /// walk limit and every entry the answer reads: the first `cap` entries are the same either way, and an
    /// entry past them can only move further back.
    pub fn shrink_cap(&mut self, cap: usize) {
        if cap < self.cap {
            self.cap = cap;
            self.size = self.size.min(cap);
            self.cursor = self.cursor.min(self.size);
        }
    }

    #[inline]
    pub fn insert(&mut self, d: f32, id: u32) {
        debug_assert!(id < OPENED);
        let size = self.size;
        if size == self.cap && self.e[size - 1].d < d {
            return;
        }
        let lo = lower_bound(&self.e[..size], d);
        let last = if size < self.cap { size } else { self.cap - 1 };
        if last > lo {
            self.e.copy_within(lo..last, lo + 1);
        }
        self.e[lo] = Entry { d, tag: id };
        if size < self.cap {
            self.size += 1;
        }
        if lo < self.cursor {
            self.cursor = lo;
        }
    }

    /// Position of the closest unopened entry among the first `limit`, if any.
    #[inline]
    pub fn next_to_open(&mut self, limit: usize) -> Option<usize> {
        while self.cursor < self.size && self.e[self.cursor].tag & OPENED != 0 {
            self.cursor += 1;
        }
        if self.cursor < self.size.min(limit) {
            Some(self.cursor)
        } else {
            None
        }
    }

    /// Mark entry `j` opened and return its slot.
    #[inline]
    fn open(&mut self, j: usize) -> u32 {
        self.e[j].tag |= OPENED;
        self.e[j].tag & !OPENED
    }
}

/// First position whose distance is not below `d`, by halving without unpredictable branches.
#[inline]
fn lower_bound(e: &[Entry], d: f32) -> usize {
    if e.is_empty() {
        return 0;
    }
    let (mut lo, mut n) = (0usize, e.len());
    while n > 1 {
        let half = n / 2;
        lo = if e[lo + half].d < d { lo + half } else { lo };
        n -= half;
    }
    lo + (e[lo].d < d) as usize
}

/// "Seen in this search" marks, reset by bumping a stamp instead of clearing the array.
#[derive(Default)]
pub struct Visited {
    stamp: u32,
    marks: Vec<u32>,
}

impl Visited {
    pub fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.stamp = 1;
        }
    }

    #[inline]
    pub fn seen(&self, i: u32) -> bool {
        self.marks[i as usize] == self.stamp
    }

    #[inline]
    pub fn mark(&mut self, i: u32) {
        self.marks[i as usize] = self.stamp;
    }
}

/// Working memory for the re-check.
#[derive(Default)]
pub struct Rerank {
    ids: Vec<u32>,
    d: Vec<f32>,
    pairs: Vec<(f32, u32)>,
}

/// Per-thread working memory, reused across searches.
#[derive(Default)]
pub struct Scratch {
    pub list: SortedList,
    pub visited: Visited,
    /// Items opened by the walk (only recorded when building or deleting).
    pub pool: Vec<u32>,
    pub prep: crate::codes::Prepared,
    pub rerank: Rerank,
}

/// Put the start items in the list. Returns the checks made.
pub fn seed<S: Score>(g: &Graph, list: &mut SortedList, visited: &mut Visited, starts: &[u32], sc: &mut S) -> u64 {
    let mut checks = 0;
    for &s in starts {
        if visited.seen(s) {
            continue;
        }
        visited.mark(s);
        let d = sc.one(s);
        checks += 1;
        let _ = g;
        list.insert(d, s);
    }
    checks
}

/// Open entries until every entry among the first `limit` is opened. Returns (checks, hops).
/// When `pool` is given, opened items are recorded in it (at most `pool_cap`), for building links.
///
/// For each opened item, its new links are all asked to start loading (`fetch`) and then all scored (`many`)
/// before any goes into the list, so their memory reads overlap and the distance work of one does not wait on the
/// list work of another. The same items go into the list in the same order with the same distances as scoring
/// and inserting them one at a time, so results and checks are unchanged.
#[inline]
pub fn walk<S: Score>(
    g: &Graph,
    list: &mut SortedList,
    visited: &mut Visited,
    sc: &mut S,
    limit: usize,
    pool: Option<(&mut Vec<u32>, usize)>,
) -> (u64, u64) {
    walk_with::<S, true>(g, list, visited, sc, limit, pool)
}

/// `walk`, with `CHECK_ALIVE` false when no link can lead to a deleted item (no slot in use is deleted): each link
/// then costs one lookup (seen?) instead of two (seen? alive?), with the same results.
#[inline]
pub fn walk_with<S: Score, const CHECK_ALIVE: bool>(
    g: &Graph,
    list: &mut SortedList,
    visited: &mut Visited,
    sc: &mut S,
    limit: usize,
    mut pool: Option<(&mut Vec<u32>, usize)>,
) -> (u64, u64) {
    let (mut checks, mut hops) = (0u64, 0u64);
    let mut fresh = [0u32; 256];
    let mut fd = [0f32; 256];
    while let Some(pos) = list.next_to_open(limit) {
        let node = list.open(pos);
        hops += 1;
        if let Some((p, cap)) = pool.as_mut() {
            if p.len() < *cap {
                p.push(node);
            }
        }
        // its links, then its extra links (popular items only: DESIGN.md section 22)
        for links in [g.links(node as usize), g.extra_links(node as usize)] {
            let mut done = 0;
            while done < links.len() {
                // in chunks of up to 256 links (lists are at most the slot count, 41 by default)
                let chunk = &links[done..(done + fresh.len()).min(links.len())];
                done += chunk.len();
                // keep the links not seen yet and not deleted, without a branch per link so the reads overlap;
                // marking a deleted item as seen changes nothing (it is skipped either way)
                let mut m = 0;
                for &c in chunk {
                    let take = !visited.seen(c) & (!CHECK_ALIVE || g.is_alive(c as usize));
                    visited.mark(c);
                    fresh[m] = c;
                    m += take as usize;
                }
                for &c in &fresh[..m] {
                    sc.fetch(c);
                }
                sc.many(&fresh[..m], &mut fd[..m]);
                for j in 0..m {
                    list.insert(fd[j], fresh[j]);
                }
                checks += m as u64;
            }
        }
        // start loading the link lists of the next two items the walk is likely to open
        if let Some(next) = list.next_to_open(limit) {
            g.prefetch_links(list.id(next) as usize);
            let end = list.size.min(limit);
            let mut j = next + 1;
            while j < end && list.e[j].tag & OPENED != 0 {
                j += 1;
            }
            if j < end {
                g.prefetch_links(list.id(j) as usize);
            }
        }
    }
    (checks, hops)
}

/// The answer without re-checks: the first k alive entries. Returns how many were written.
pub fn finish_plain(g: &Graph, list: &SortedList, k: usize, ids: &mut [i64], dists: &mut [f32]) -> usize {
    let mut n = 0;
    for j in 0..list.size {
        if n == k {
            break;
        }
        let i = list.id(j);
        if !g.is_alive(i as usize) {
            continue;
        }
        ids[n] = i as i64;
        dists[n] = list.dist(j);
        n += 1;
    }
    n
}

/// The answer with re-checks: the first max(rerank, k) alive entries get their full-number distance,
/// and the best k of those are returned. Returns (how many written, full checks made).
#[allow(clippy::too_many_arguments)]
pub fn finish_rerank<S: Score>(
    g: &Graph,
    list: &SortedList,
    k: usize,
    rerank: usize,
    full: &mut S,
    buf: &mut Rerank,
    ids: &mut [i64],
    dists: &mut [f32],
) -> (usize, u64) {
    let rr = list.size.min(rerank.max(k));
    buf.ids.clear();
    for j in 0..list.size {
        if buf.ids.len() == rr {
            break;
        }
        let i = list.id(j);
        if g.is_alive(i as usize) {
            buf.ids.push(i);
        }
    }
    buf.d.resize(buf.ids.len(), 0.0);
    full.many(&buf.ids, &mut buf.d);
    buf.pairs.clear();
    buf.pairs.extend(buf.d.iter().copied().zip(buf.ids.iter().copied()));
    let checks = buf.pairs.len() as u64;
    buf.pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = k.min(buf.pairs.len());
    for j in 0..n {
        ids[j] = buf.pairs[j].1 as i64;
        dists[j] = buf.pairs[j].0;
    }
    (n, checks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_keeps_order_drops_the_farthest_and_puts_new_ties_first() {
        let mut l = SortedList::default();
        l.reset(3);
        for (d, i) in [(0.5, 1), (0.2, 2), (0.9, 3), (0.1, 4), (0.95, 5)] {
            l.insert(d, i);
        }
        let ids = |l: &SortedList| (0..l.size).map(|j| l.id(j)).collect::<Vec<_>>();
        assert_eq!(ids(&l), [4, 2, 1]);
        l.insert(0.2, 6);
        assert_eq!(ids(&l), [4, 6, 2]);
        // equal to the last entry of a full list: goes in before it, and the last one drops out
        l.insert(0.2, 7);
        assert_eq!(ids(&l), [4, 7, 6]);
        l.shrink_cap(2);
        assert_eq!(ids(&l), [4, 7]);
    }

    #[test]
    fn walk_on_a_line_finds_the_closest_items() {
        // items 0..20 on a line, each linked to its neighbours; question at 13.2
        let n = 20;
        let mut g = Graph::new(n, 2);
        for u in 0..n {
            let mut l = vec![];
            if u > 0 {
                l.push(u as u32 - 1);
            }
            if u + 1 < n {
                l.push(u as u32 + 1);
            }
            g.set_links(u, &l);
            g.alive[u] = 1;
        }
        let mut s = Scratch::default();
        s.visited.begin(n);
        s.list.reset(4);
        let mut dist = |i: u32| (i as f32 - 13.2).abs();
        seed(&g, &mut s.list, &mut s.visited, &[0], &mut dist);
        walk(&g, &mut s.list, &mut s.visited, &mut dist, 4, None);
        let (mut ids, mut d) = (vec![-1i64; 3], vec![0f32; 3]);
        let n_found = finish_plain(&g, &s.list, 3, &mut ids, &mut d);
        assert_eq!(n_found, 3);
        assert_eq!(ids, vec![13, 14, 12]);
        // a dead item is never returned
        g.alive[13] = 0;
        s.visited.begin(n);
        s.list.reset(4);
        seed(&g, &mut s.list, &mut s.visited, &[0], &mut dist);
        walk(&g, &mut s.list, &mut s.visited, &mut dist, 4, None);
        finish_plain(&g, &s.list, 3, &mut ids, &mut d);
        assert!(!ids.contains(&13));
    }
}
