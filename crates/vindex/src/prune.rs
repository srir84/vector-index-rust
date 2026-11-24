//! Picking a few spread-out links (DESIGN.md section 6): DiskANN's rule, ported from the prototype's `prune`.

use crate::metric::Metric;

/// The space links are built in: the full numbers, or the flipped copies for best-score data.
#[derive(Clone, Copy)]
pub struct BuildSpace<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub metric: Metric,
}

impl<'a> BuildSpace<'a> {
    #[inline]
    pub fn row(&self, i: u32) -> &'a [f32] {
        let i = i as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    pub fn dist(&self, a: u32, b: u32) -> f32 {
        self.metric.distance(self.row(a), self.row(b))
    }
}

/// Working memory for `prune`, reused across calls.
#[derive(Default)]
pub struct PruneScratch {
    order: Vec<u32>,
    occl: Vec<f32>,
    last: Vec<u32>,
    kept: Vec<u32>,
}

/// Pick at most `r` links for an item from `pool` (distances `pool_d` from it, in build space).
///
/// `forced` go in first (they do not occlude others). Then candidates are taken closest first, in rounds
/// with a = 1.0, a * 1.2, ... up to `alpha`. A candidate p is occluded by an already-kept link j when
/// d(v,p) / d(j,p) > a (distance formulas), or for best score when d(j,p) < a * d(v,p) (DiskANN's
/// "occluding" rule). Each candidate keeps its worst ratio so far and is only checked against links kept
/// since its last check.
pub fn prune(
    space: &BuildSpace,
    pool: &[u32],
    pool_d: &[f32],
    r: usize,
    alpha: f32,
    forced: &[u32],
    out: &mut Vec<u32>,
    sc: &mut PruneScratch,
) {
    out.clear();
    let m = pool.len();
    sc.order.clear();
    sc.order.extend(0..m as u32);
    sc.order.sort_by(|&a, &b| pool_d[a as usize].total_cmp(&pool_d[b as usize]));
    sc.occl.clear();
    sc.occl.resize(m, 0.0);
    sc.last.clear();
    sc.last.resize(m, 0);
    sc.kept.clear();
    let occluding = space.metric == Metric::Ip;
    for &f in forced {
        if out.len() == r {
            break;
        }
        out.push(f);
        for oi in 0..m {
            if pool[sc.order[oi] as usize] == f {
                sc.occl[oi] = f32::INFINITY;
            }
        }
    }
    let mut cur = 1.0f32;
    let inc = alpha.min(1.2);
    while out.len() < r {
        for oi in 0..m {
            if out.len() >= r {
                break;
            }
            if sc.occl[oi] > cur {
                continue;
            }
            let i = sc.order[oi] as usize;
            let dv = pool_d[i];
            let mut f = sc.occl[oi];
            let mut n = sc.last[oi] as usize;
            while n < sc.kept.len() {
                let jpos = sc.kept[n] as usize;
                n += 1;
                if jpos >= oi {
                    continue;
                }
                let j = sc.order[jpos] as usize;
                let djk = space.dist(pool[j], pool[i]);
                if occluding {
                    if djk < cur * dv {
                        f = cur + 0.01;
                    }
                } else if djk <= 0.0 {
                    f = f32::INFINITY;
                } else {
                    f = f.max(dv / djk);
                }
                if f > cur {
                    break;
                }
            }
            sc.last[oi] = n as u32;
            if f > cur {
                sc.occl[oi] = f;
                continue;
            }
            sc.occl[oi] = f32::INFINITY;
            sc.kept.push(oi as u32);
            out.push(pool[i]);
        }
        if cur >= alpha {
            break;
        }
        cur = (cur * inc).min(alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_closest_and_drops_ones_hidden_behind_it() {
        // v at 0; a at 1, b at 1.1 (right behind a), c at -1 (other side)
        let data = vec![0.0, 1.0, 1.1, -1.0];
        let sp = BuildSpace { data: &data, dim: 1, metric: Metric::L2 };
        let pool = [1u32, 2, 3];
        let d: Vec<f32> = pool.iter().map(|&p| sp.dist(0, p)).collect();
        let (mut out, mut sc) = (vec![], PruneScratch::default());
        prune(&sp, &pool, &d, 3, 1.2, &[], &mut out, &mut sc);
        assert_eq!(out, vec![1, 3]);
        // with room and no slack it still never keeps more than r
        prune(&sp, &pool, &d, 1, 1.2, &[], &mut out, &mut sc);
        assert_eq!(out, vec![1]);
        // forced links come first
        prune(&sp, &pool, &d, 2, 1.2, &[2], &mut out, &mut sc);
        assert_eq!(out[0], 2);
        assert_eq!(out.len(), 2);
    }
}
