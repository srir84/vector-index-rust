//! The link map: a fixed number of link slots per item, a degree, and an alive mark; plus extra links for
//! popular items (part 3).

/// Links stored flat: `slots` entries per item, of which the first `deg[u]` are used.
#[derive(Clone, Debug)]
pub struct Graph {
    pub slots: usize,
    pub nbrs: Vec<u32>,
    pub deg: Vec<u32>,
    pub alive: Vec<u8>,
    /// Extra links for popular items (DESIGN.md section 22): item u's are `extra[extra_at[u]..][..extra_n[u]]`, and
    /// most items have none. All three are empty when the map has no extra links.
    pub extra_at: Vec<u32>,
    pub extra_n: Vec<u8>,
    pub extra: Vec<u32>,
}

impl Graph {
    pub fn new(capacity: usize, slots: usize) -> Self {
        Graph {
            slots,
            nbrs: vec![0; capacity * slots],
            deg: vec![0; capacity],
            alive: vec![0; capacity],
            extra_at: Vec::new(),
            extra_n: Vec::new(),
            extra: Vec::new(),
        }
    }

    /// Build from arrays (e.g. a map exported from the prototype). Every item is alive.
    pub fn from_parts(nbrs: Vec<u32>, deg: Vec<u32>, slots: usize) -> Self {
        let n = deg.len();
        assert_eq!(nbrs.len(), n * slots, "nbrs must be n x slots");
        Graph { slots, nbrs, deg, alive: vec![1; n], extra_at: Vec::new(), extra_n: Vec::new(), extra: Vec::new() }
    }

    /// Item u's extra links (none for most items, and for every item when the map has none).
    #[inline]
    pub fn extra_links(&self, u: usize) -> &[u32] {
        if self.extra_n.is_empty() {
            return &[];
        }
        let a = self.extra_at[u] as usize;
        &self.extra[a..a + self.extra_n[u] as usize]
    }

    pub fn has_extra(&self) -> bool {
        !self.extra_n.is_empty()
    }

    /// Replace every extra link: `lists` holds (item, its extra links), at most 255 each.
    pub fn set_extra(&mut self, lists: &[(u32, Vec<u32>)]) {
        let cap = self.capacity();
        self.extra_at = vec![0; cap];
        self.extra_n = vec![0; cap];
        self.extra.clear();
        for (u, l) in lists {
            let l = &l[..l.len().min(255)];
            self.extra_at[*u as usize] = self.extra.len() as u32;
            self.extra_n[*u as usize] = l.len() as u8;
            self.extra.extend_from_slice(l);
        }
    }

    /// Drop every extra link.
    pub fn clear_extra(&mut self) {
        self.extra_at = Vec::new();
        self.extra_n = Vec::new();
        self.extra = Vec::new();
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.deg.len()
    }

    #[inline]
    pub fn links(&self, u: usize) -> &[u32] {
        &self.nbrs[u * self.slots..u * self.slots + self.deg[u] as usize]
    }

    /// Start loading an item's link list (and its extra links, if any).
    #[inline]
    pub fn prefetch_links(&self, u: usize) {
        let row = &self.nbrs[u * self.slots..u * self.slots + self.deg[u] as usize];
        crate::prefetch::prefetch_bytes(row.as_ptr() as *const u8, std::mem::size_of_val(row));
        let ex = self.extra_links(u);
        if !ex.is_empty() {
            crate::prefetch::prefetch_bytes(ex.as_ptr() as *const u8, std::mem::size_of_val(ex));
        }
    }

    #[inline]
    pub fn is_alive(&self, u: usize) -> bool {
        self.alive[u] != 0
    }

    pub fn set_links(&mut self, u: usize, list: &[u32]) {
        debug_assert!(list.len() <= self.slots);
        let base = u * self.slots;
        self.nbrs[base..base + list.len()].copy_from_slice(list);
        self.deg[u] = list.len() as u32;
    }

    pub fn grow(&mut self, capacity: usize) {
        if capacity > self.capacity() {
            self.nbrs.resize(capacity * self.slots, 0);
            self.deg.resize(capacity, 0);
            self.alive.resize(capacity, 0);
            if self.has_extra() {
                self.extra_at.resize(capacity, 0);
                self.extra_n.resize(capacity, 0);
            }
        }
    }

    /// How many alive items link to each item.
    pub fn in_degrees(&self, n_used: usize) -> Vec<u32> {
        let mut indeg = vec![0u32; self.capacity()];
        for u in 0..n_used {
            if self.is_alive(u) {
                for &v in self.links(u) {
                    indeg[v as usize] += 1;
                }
            }
        }
        indeg
    }
}
