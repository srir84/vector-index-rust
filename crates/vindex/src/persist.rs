//! Saving to a folder and loading back (ARCHITECTURE.md, "Files on disk").
//!
//! Every file is little-endian. `meta.bin` holds a magic word, a format version, and everything that is not a
//! big array; loading checks the header and every file's size before using it. The flipped build copies are not
//! saved: they are recomputed from the full numbers.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::codes::Codes;
use crate::effort::Learned;
use crate::graph::Graph;
use crate::index::{BuildSpaceKind, Effort, Error, Index, Result, Settings, Switch, NO_ID};
use crate::metric::Metric;
use crate::store::Store;
use crate::audit::SpotChecks;
use crate::usage::Usage;

const MAGIC: &[u8; 8] = b"VINDEX\0\0";
/// Version 2 adds part 2's settings and facts at the end of meta.bin, as text pairs, so later ones need no new
/// version. Version 1 folders still load.
const VERSION: u32 = 2;

/// Where the full numbers go when loading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numbers {
    /// Read into memory: searches and changes.
    Ram,
    /// Left in the file, read by the OS when needed: searches only.
    Ssd,
}

fn io(e: std::io::Error) -> Error {
    Error(format!("file error: {e}"))
}

// ---------------------------------------------------------------- byte writing and reading

struct W(Vec<u8>);

impl W {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        v.iter().for_each(|x| self.f32(*x));
    }
    fn f64s(&mut self, v: &[f64]) {
        self.u64(v.len() as u64);
        v.iter().for_each(|x| self.f64(*x));
    }
    fn usizes(&mut self, v: &[usize]) {
        self.u64(v.len() as u64);
        v.iter().for_each(|x| self.u64(*x as u64));
    }
    fn text(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at + n > self.b.len() {
            return Err(Error("meta.bin is cut short".into()));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self, limit: usize) -> Result<usize> {
        let n = self.u64()? as usize;
        if n > limit {
            return Err(Error("meta.bin holds an impossible length".into()));
        }
        Ok(n)
    }
    fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.len(1 << 30)?;
        (0..n).map(|_| self.f32()).collect()
    }
    fn f64s(&mut self) -> Result<Vec<f64>> {
        let n = self.len(1 << 30)?;
        (0..n).map(|_| self.f64()).collect()
    }
    fn usizes(&mut self) -> Result<Vec<usize>> {
        let n = self.len(1 << 30)?;
        (0..n).map(|_| self.u64().map(|v| v as usize)).collect()
    }
    fn text(&mut self) -> Result<String> {
        let n = self.len(1 << 20)?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error("meta.bin holds text that is not UTF-8".into()))
    }
}

/// Raw little-endian bytes of a slice of plain numbers.
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: T is a plain number type (u8/u32/u64/f32) with no padding; reading its bytes is always valid.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The files of one `save`. Each is written to `<name>.tmp` first; only when all are written do they replace the old
/// ones, so a save that fails part-way (disk full, a crash) leaves the folder's previous index whole. The old
/// meta.bin goes first and the new one comes last, so a folder caught between the two has no meta.bin and is never
/// taken for a complete index.
struct Staged<'a> {
    dir: &'a Path,
    names: Vec<&'static str>,
    gone: Vec<&'static str>,
    committed: bool,
}

impl<'a> Staged<'a> {
    fn new(dir: &'a Path) -> Self {
        Staged { dir, names: Vec::new(), gone: Vec::new(), committed: false }
    }

    fn tmp(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.tmp"))
    }

    fn write(&mut self, name: &'static str, bytes: &[u8]) -> Result<()> {
        self.names.push(name);
        let mut f = fs::File::create(self.tmp(name)).map_err(io)?;
        f.write_all(bytes).map_err(io)?;
        f.sync_all().map_err(io)
    }

    /// Remove `name` once the new files are in place.
    fn remove(&mut self, name: &'static str) {
        self.gone.push(name);
    }

    fn commit(mut self) -> Result<()> {
        self.committed = true;
        match fs::remove_file(self.dir.join("meta.bin")) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io(e)),
            _ => {}
        }
        for name in &self.names {
            fs::rename(self.tmp(name), self.dir.join(name)).map_err(io)?;
        }
        for name in &self.gone {
            let _ = fs::remove_file(self.dir.join(name));
        }
        fs::File::open(self.dir).and_then(|d| d.sync_all()).map_err(io)
    }
}

impl Drop for Staged<'_> {
    fn drop(&mut self) {
        if !self.committed {
            for name in &self.names {
                let _ = fs::remove_file(self.tmp(name));
            }
        }
    }
}

/// Read a whole file of plain numbers of type T straight into `out`, checking first that its size is exactly
/// `out`'s (so nothing is read twice or held twice while loading).
fn read_into<T: Copy>(dir: &Path, name: &str, out: &mut [T]) -> Result<()> {
    let path = dir.join(name);
    let size = fs::metadata(&path).map_err(|e| Error(format!("{name}: {e}")))?.len() as usize;
    let want = std::mem::size_of_val(out);
    if size != want {
        return Err(Error(format!("{name} has {size} bytes, expected {want}: the folder is damaged or from another index")));
    }
    // SAFETY: `out` holds exactly `want` bytes of plain numbers; any byte pattern is a valid value for them.
    let buf = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, want) };
    fs::File::open(&path).map_err(io)?.read_exact(buf).map_err(io)
}

/// Read a whole file of `count` plain numbers of type T, checking its size first.
fn read_numbers<T: Copy + Default>(dir: &Path, name: &str, count: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); count];
    read_into(dir, name, &mut v)?;
    Ok(v)
}

/// usage.json: the usage counters, the spot checks' counts, and the spot-checked questions (their numbers are in
/// checked_q.f32, in the same order).
#[derive(serde::Serialize)]
struct MonitorOut<'a> {
    usage: &'a Usage,
    spot_checks: &'a SpotChecks,
    checked: Vec<CheckedRec>,
}

#[derive(serde::Deserialize)]
struct MonitorIn {
    usage: Usage,
    spot_checks: SpotChecks,
    #[serde(default)]
    checked: Vec<CheckedRec>,
}

/// A spot-checked question without its numbers. Exactly one of `target` and `list` is set.
#[derive(serde::Serialize, serde::Deserialize)]
struct CheckedRec {
    target: Option<f64>,
    list: Option<usize>,
    k: usize,
    correct: u32,
    kth: f32,
    tenth: f32,
}

fn metric_code(m: Metric) -> u32 {
    match m {
        Metric::Cos => 0,
        Metric::Ip => 1,
        Metric::L2 => 2,
    }
}

fn write_settings(w: &mut W, s: &Settings) {
    w.u64(s.r as u64);
    w.u64(s.l_build as u64);
    w.f32(s.alpha);
    w.f32(s.degree_slack);
    w.u32(match s.build_space {
        BuildSpaceKind::Auto => 0,
        BuildSpaceKind::Same => 1,
        BuildSpaceKind::Flipped => 2,
    });
    w.u64(s.seed);
    w.f32(s.max_batch_frac);
    w.u32(s.code_bits as u32);
    w.u64(s.rerank as u64);
    w.u32(match s.question_links {
        Switch::Auto => 0,
        Switch::On => 1,
        Switch::Off => 2,
    });
    w.f32(s.ood_ratio);
    w.u64(s.q_topk as u64);
    w.u64(s.q_links_max as u64);
    w.u64(s.q_rounds as u64);
    w.u32(s.learned_effort as u32);
    w.u64(s.effort_questions as u64);
    w.u64(s.n_replace as u64);
    w.u64(s.qlinks_refresh_every);
    w.u64(s.cleanup_batch as u64);
    w.f32(s.spare);
}

fn read_settings(r: &mut R) -> Result<Settings> {
    let bad = |what: &str| Error(format!("meta.bin: bad {what}"));
    Ok(Settings {
        r: r.u64()? as usize,
        l_build: r.u64()? as usize,
        alpha: r.f32()?,
        degree_slack: r.f32()?,
        build_space: match r.u32()? {
            0 => BuildSpaceKind::Auto,
            1 => BuildSpaceKind::Same,
            2 => BuildSpaceKind::Flipped,
            _ => return Err(bad("build space")),
        },
        seed: r.u64()?,
        max_batch_frac: r.f32()?,
        code_bits: r.u32()? as u8,
        rerank: r.u64()? as usize,
        question_links: match r.u32()? {
            0 => Switch::Auto,
            1 => Switch::On,
            2 => Switch::Off,
            _ => return Err(bad("question-link switch")),
        },
        ood_ratio: r.f32()?,
        q_topk: r.u64()? as usize,
        q_links_max: r.u64()? as usize,
        q_rounds: r.u64()? as usize,
        learned_effort: r.u32()? != 0,
        effort_questions: r.u64()? as usize,
        n_replace: r.u64()? as usize,
        qlinks_refresh_every: r.u64()?,
        cleanup_batch: r.u64()? as usize,
        spare: r.f32()?,
        // part 2's settings follow at the end of meta.bin as text pairs (see `load`)
        ..Settings::default()
    })
}

impl Index {
    /// Save to `dir` (created if needed; existing index files in it are replaced).
    pub fn save(&self, dir: &Path) -> Result<()> {
        if cfg!(target_endian = "big") {
            return Err(Error("saving needs a little-endian machine".into()));
        }
        if self.n_used == 0 {
            return Err(Error("nothing to save: the index is empty".into()));
        }
        fs::create_dir_all(dir).map_err(io)?;
        let mut st = Staged::new(dir);
        let (n, d, slots) = (self.n_used, self.dim, self.graph.slots);
        let mut w = W(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u32(VERSION);
        w.u32(0x0102_0304); // byte-order check
        w.u64(d as u64);
        w.u32(metric_code(self.metric));
        write_settings(&mut w, &self.s);
        w.u64(n as u64);
        w.u64(slots as u64);
        w.u64(self.start as u64);
        w.u64(self.build_checks);
        w.u64(self.change_checks);
        w.u64(self.changes);
        w.f32(self.ood_ratio.unwrap_or(f32::NAN));
        w.u64(self.question_link_items as u64);
        w.u64(self.free.len() as u64);
        w.u64(self.pending_free.len() as u64);
        match &self.codes {
            Some(c) => {
                w.u32(1);
                w.f32s(&c.low);
                w.f32s(&c.step);
            }
            None => w.u32(0),
        }
        match self.learned().as_deref() {
            Some(e) => {
                w.u32(1);
                w.f32s(&e.cuts);
                w.usizes(&e.levels);
                w.f64s(&e.correct);
                w.f64s(&e.checks);
                w.u64(e.questions as u64);
            }
            None => w.u32(0),
        }
        w.u64(self.train_q.as_ref().map_or(0, |t| (t.len() / d) as u64));
        // version 2: part 2's settings and facts as text pairs
        let mut extras: Vec<(&str, String)> = vec![
            ("track_usage", self.s.track_usage.to_string()),
            ("audit_budget", format!("{:?}", self.s.audit_budget)),
            ("effort_apart", self.s.effort_apart.to_string()),
            ("relearn_every", self.s.relearn_every.to_string()),
            ("tune", if self.s.tune == crate::index::Tune::Auto { "auto" } else { "off" }.to_string()),
            ("tune_target", format!("{:?}", self.s.tune_target)),
            ("max_bytes_per_item", self.s.max_bytes_per_item.to_string()),
            ("renumber", self.s.renumber.to_string()),
            ("question_bits", self.s.question_bits.to_string()),
        ];
        // part 3; left out at its default, so that a folder without extra links reads in a build before it
        if self.s.popular_links != Switch::Auto {
            extras.push(("popular_links", if self.s.popular_links == Switch::On { "on" } else { "off" }.to_string()));
        }
        // part 5; likewise left out at its default
        if self.s.repair != crate::index::Repair::Occlude {
            extras.push(("repair", match self.s.repair {
                crate::index::Repair::Prune => "prune",
                _ => "append",
            }.to_string()));
        }
        if let Some(g) = self.item_gap.get() {
            extras.push(("item_gap", format!("{g:?}")));
        }
        w.u64(extras.len() as u64);
        for (k, v) in &extras {
            w.text(k);
            w.text(v);
        }
        st.write("links.u32", bytes_of(&self.graph.nbrs[..n * slots]))?;
        st.write("degree.u32", bytes_of(&self.graph.deg[..n]))?;
        st.write("alive.u8", &self.graph.alive[..n])?;
        st.write("ids.u64", bytes_of(&self.ext[..n]))?;
        let mut fl: Vec<u32> = self.free.clone();
        fl.extend_from_slice(&self.pending_free);
        st.write("free.u32", bytes_of(&fl))?;
        let numbers: Vec<f32> = (0..n).flat_map(|u| self.store.row(u).iter().copied()).collect();
        st.write("numbers.f32", bytes_of(&numbers))?;
        if let Some(c) = &self.codes {
            st.write("codes.u8", &c.data[..n * c.row_bytes()])?;
        }
        if let Some(t) = &self.train_q {
            st.write("train_q.f32", bytes_of(t))?;
        }
        // popular items' extra links (part 3): how many each slot has, then all of them in slot order
        if self.graph.has_extra() {
            let flat: Vec<u32> = (0..n).flat_map(|u| self.graph.extra_links(u).iter().copied()).collect();
            st.write("extra_n.u8", &self.graph.extra_n[..n])?;
            st.write("extra.u32", bytes_of(&flat))?;
        } else {
            st.remove("extra_n.u8");
            st.remove("extra.u32");
        }
        // the usage stats and the spot checks' counts, readable as they are
        let (usage, checked_q) = {
            let m = self.monitor();
            // a question checked on an empty index has no finite distances: not worth keeping
            let kept: Vec<&crate::audit::Checked> =
                m.spot.window.iter().filter(|c| c.kth.is_finite() && c.tenth.is_finite()).collect();
            let checked = kept
                .iter()
                .map(|c| {
                    let (target, list) = match c.effort {
                        Effort::Target(t) => (Some(t), None),
                        Effort::List(l) => (None, Some(l)),
                    };
                    CheckedRec { target, list, k: c.k, correct: c.correct, kth: c.kth, tenth: c.tenth }
                })
                .collect();
            let q: Vec<f32> = kept.iter().flat_map(|c| c.q.iter().copied()).collect();
            let json = serde_json::to_vec_pretty(&MonitorOut { usage: &m.usage, spot_checks: &m.spot, checked })
                .map_err(|e| Error(format!("usage.json: {e}")))?;
            (json, q)
        };
        st.write("usage.json", &usage)?;
        if let Some(t) = &self.tuning {
            let json = serde_json::to_vec_pretty(t).map_err(|e| Error(format!("tuning.json: {e}")))?;
            st.write("tuning.json", &json)?;
        }
        st.write("checked_q.f32", bytes_of(&checked_q))?;
        // meta.bin last: a folder without it is never taken for a complete index
        st.write("meta.bin", &w.0)?;
        st.commit()
    }

    /// Load an index saved with `save`. With `Numbers::Ssd` the full numbers stay in the file (searches only).
    pub fn load(dir: &Path, numbers: Numbers) -> Result<Index> {
        if cfg!(target_endian = "big") {
            return Err(Error("loading needs a little-endian machine".into()));
        }
        let meta = fs::read(dir.join("meta.bin")).map_err(|e| Error(format!("meta.bin: {e}")))?;
        let mut r = R { b: &meta, at: 0 };
        if r.take(8)? != MAGIC {
            return Err(Error("not a vindex folder (meta.bin has the wrong magic word)".into()));
        }
        let version = r.u32()?;
        if version != 1 && version != VERSION {
            return Err(Error(format!("saved with format version {version}; this build reads versions 1 to {VERSION}")));
        }
        if r.u32()? != 0x0102_0304 {
            return Err(Error("saved on a machine with another byte order".into()));
        }
        let d = r.u64()? as usize;
        let metric = match r.u32()? {
            0 => Metric::Cos,
            1 => Metric::Ip,
            2 => Metric::L2,
            _ => return Err(Error("meta.bin: bad metric".into())),
        };
        let mut s = read_settings(&mut r)?;
        let (n, slots) = (r.u64()? as usize, r.u64()? as usize);
        if d == 0 || n == 0 || slots != s.slots() {
            return Err(Error("meta.bin: sizes do not fit together".into()));
        }
        let start = r.u64()? as u32;
        let (build_checks, change_checks, changes) = (r.u64()?, r.u64()?, r.u64()?);
        let ood = r.f32()?;
        let question_link_items = r.u64()? as usize;
        let (n_free, n_pending) = (r.u64()? as usize, r.u64()? as usize);
        let code_params = if r.u32()? == 1 { Some((r.f32s()?, r.f32s()?)) } else { None };
        let effort = if r.u32()? == 1 {
            Some(Learned { cuts: r.f32s()?, levels: r.usizes()?, correct: r.f64s()?, checks: r.f64s()?, questions: r.u64()? as usize })
        } else {
            None
        };
        let n_train = r.u64()? as usize;
        let extras: Vec<(String, String)> = if version >= 2 {
            let m = r.len(1 << 10)?;
            (0..m).map(|_| Ok((r.text()?, r.text()?))).collect::<Result<_>>()?
        } else {
            Vec::new()
        };
        // part 2's settings, before the index is made: its checks need them (4-bit copies need 16-bit questions)
        for (k, v) in extras.iter().filter(|(k, _)| k != "item_gap") {
            s.set(k, v)?;
        }
        if start as usize >= n || n_free + n_pending > n {
            return Err(Error("meta.bin: start or free slots out of range".into()));
        }
        let cap = n.max((n as f64 * (1.0 + s.spare as f64)) as usize);
        let mut graph = Graph::new(cap, slots);
        read_into(dir, "links.u32", &mut graph.nbrs[..n * slots])?;
        read_into(dir, "degree.u32", &mut graph.deg[..n])?;
        read_into(dir, "alive.u8", &mut graph.alive[..n])?;
        if graph.deg[..n].iter().any(|&k| k as usize > slots) || graph.nbrs[..n * slots].iter().any(|&v| v as usize >= n) {
            return Err(Error("links.u32 or degree.u32 point outside the index: the folder is damaged".into()));
        }
        if dir.join("extra_n.u8").exists() {
            let counts: Vec<u8> = read_numbers(dir, "extra_n.u8", n)?;
            let total: usize = counts.iter().map(|&k| k as usize).sum();
            let flat: Vec<u32> = read_numbers(dir, "extra.u32", total)?;
            if flat.iter().any(|&v| v as usize >= n) {
                return Err(Error("extra.u32 points outside the index: the folder is damaged".into()));
            }
            let mut lists = Vec::new();
            let mut at = 0;
            for (u, &k) in counts.iter().enumerate() {
                if k > 0 {
                    lists.push((u as u32, flat[at..at + k as usize].to_vec()));
                    at += k as usize;
                }
            }
            graph.set_extra(&lists);
        }
        let mut ext = vec![NO_ID; cap];
        read_into(dir, "ids.u64", &mut ext[..n])?;
        let mut slot_of = HashMap::with_capacity(n);
        for u in 0..n {
            if graph.alive[u] != 0 && ext[u] != NO_ID && slot_of.insert(ext[u], u as u32).is_some() {
                return Err(Error(format!("ids.u64 lists id {} twice", ext[u])));
            }
        }
        let fl: Vec<u32> = read_numbers(dir, "free.u32", n_free + n_pending)?;
        let store = match numbers {
            Numbers::Ram => {
                let mut data = vec![0f32; cap * d];
                read_into(dir, "numbers.f32", &mut data[..n * d])?;
                Store::Ram { dim: d, data }
            }
            Numbers::Ssd => {
                let st = Store::mapped(&dir.join("numbers.f32"), d).map_err(io)?;
                if st.rows() != n {
                    return Err(Error("numbers.f32 does not hold one row per slot".into()));
                }
                st
            }
        };
        let codes = match code_params {
            Some((low, step)) => {
                if low.len() != d || step.len() != d {
                    return Err(Error("meta.bin: copy scales do not fit the dimension".into()));
                }
                let rb = Codes::row_bytes_for(d, s.code_bits);
                let mut c = Codes { dim: d, metric, low, step, data: vec![0; cap * rb], norm2: Vec::new(), qbits: 32, bits: s.code_bits };
                read_into(dir, "codes.u8", &mut c.data[..n * rb])?;
                if metric == Metric::L2 {
                    c.norm2 = vec![0.0; cap];
                    for u in 0..n {
                        let row = c.decode(u);
                        c.norm2[u] = row.iter().map(|x| x * x).sum();
                    }
                }
                Some(c)
            }
            None => None,
        };
        let train_q = if n_train > 0 { Some(read_numbers::<f32>(dir, "train_q.f32", n_train * d)?) } else { None };
        let mut idx = Index::new(d, metric, s)?;
        idx.store = store;
        idx.graph = graph;
        idx.ext = ext;
        idx.slot_of = slot_of;
        idx.n_used = n;
        idx.start = start;
        idx.build_checks = build_checks;
        idx.change_checks = change_checks;
        idx.changes = changes;
        idx.ood_ratio = if ood.is_nan() { None } else { Some(ood) };
        idx.question_link_items = question_link_items;
        idx.free = fl[..n_free].to_vec();
        idx.pending_free = fl[n_free..].to_vec();
        idx.codes = codes;
        idx.set_learned(effort);
        idx.train_q = train_q;
        if let Some((_, v)) = extras.iter().find(|(k, _)| k == "item_gap") {
            let g: f32 = v.parse().map_err(|_| Error(format!("meta.bin: item_gap {v:?} is not a number")))?;
            let _ = idx.item_gap.set(g);
        }
        // part 2's settings are known only now: the copies take their question form from them
        let qbits = idx.s.question_bits;
        if let Some(c) = idx.codes.as_mut() {
            c.qbits = qbits;
        }
        // what the build's tuning tried, when it ran
        let tp = dir.join("tuning.json");
        if tp.exists() {
            let text = fs::read(&tp).map_err(io)?;
            idx.tuning = Some(serde_json::from_slice(&text).map_err(|e| Error(format!("tuning.json: {e}")))?);
        }
        // usage stats: missing in version 1 folders
        let up = dir.join("usage.json");
        if up.exists() {
            let text = fs::read(&up).map_err(io)?;
            let mut m: MonitorIn = serde_json::from_slice(&text).map_err(|e| Error(format!("usage.json: {e}")))?;
            let q: Vec<f32> = read_numbers(dir, "checked_q.f32", m.checked.len() * d)?;
            for (rec, row) in m.checked.iter().zip(q.chunks_exact(d)) {
                let effort = match (rec.target, rec.list) {
                    (Some(t), None) => Effort::Target(t),
                    (None, Some(l)) => Effort::List(l),
                    _ => return Err(Error("usage.json: a checked question needs a target or a list size".into())),
                };
                m.spot_checks.window.push_back(crate::audit::Checked {
                    q: row.to_vec(),
                    effort,
                    k: rec.k,
                    correct: rec.correct,
                    kth: rec.kth,
                    tenth: rec.tenth,
                });
            }
            idx.set_monitor(m.usage, m.spot_checks);
        }
        idx.recompute_flipped();
        Ok(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Effort;
    use crate::rng::Rng;

    fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n * d).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("vindex-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn a_failed_save_keeps_the_old_index() {
        let d = 8;
        let s = Settings { learned_effort: false, tune: crate::index::Tune::Off, ..Settings::default() };
        let mut a = Index::new(d, Metric::L2, s).unwrap();
        a.build(&data(800, d, 1), &(0..800u64).collect::<Vec<_>>(), None).unwrap();
        let dir = tmp("failed-save");
        a.save(&dir).unwrap();
        let before = Index::load(&dir, Numbers::Ram).unwrap().search(&data(5, d, 2), 10, Effort::List(40)).unwrap().ids;
        // a later save that cannot finish: one of its files cannot be written
        a.delete(3).unwrap();
        fs::create_dir(dir.join("ids.u64.tmp")).unwrap();
        assert!(a.save(&dir).is_err());
        fs::remove_dir(dir.join("ids.u64.tmp")).unwrap();
        let left: Vec<String> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        assert!(left.iter().all(|f| !f.ends_with(".tmp")), "{left:?}");
        let after = Index::load(&dir, Numbers::Ram).unwrap().search(&data(5, d, 2), 10, Effort::List(40)).unwrap().ids;
        assert_eq!(before, after, "the folder still holds the first save");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_loaded_index_searches_and_changes_exactly_like_the_saved_one() {
        let d = 20;
        let x = data(2000, d, 1);
        let tq = data(400, d, 2);
        let q = data(60, d, 3);
        for (metric, bits) in [(Metric::Cos, 8), (Metric::Ip, 8), (Metric::L2, 8), (Metric::Cos, 4), (Metric::L2, 4)] {
            let s = Settings {
                r: 16,
                l_build: 40,
                q_rounds: 2,
                question_links: Switch::On,
                effort_questions: 300,
                code_bits: bits,
                question_bits: if bits == 4 { 16 } else { 32 },
                ..Settings::default()
            };
            let mut a = Index::new(d, metric, s).unwrap();
            let ids: Vec<u64> = (0..2000u64).map(|i| i * 3 + 7).collect();
            a.build(&x, &ids, Some(&tq)).unwrap();
            for i in 0..50u64 {
                a.delete(i * 3 + 7).unwrap();
            }
            let dir = tmp(&format!("{}-{bits}", metric.name()));
            a.save(&dir).unwrap();
            for mode in [Numbers::Ram, Numbers::Ssd] {
                let b = Index::load(&dir, mode).unwrap();
                for effort in [Effort::Target(0.95), Effort::List(60)] {
                    let (ra, rb) = (a.search(&q, 10, effort).unwrap(), b.search(&q, 10, effort).unwrap());
                    assert_eq!(ra.ids, rb.ids, "{metric:?} {mode:?}");
                    assert_eq!(ra.checks, rb.checks);
                }
            }
            // changes after loading behave exactly like changes before saving
            let mut b = Index::load(&dir, Numbers::Ram).unwrap();
            let extra = data(30, d, 4);
            for j in 0..30usize {
                a.add(100_000 + j as u64, &extra[j * d..(j + 1) * d]).unwrap();
                b.add(100_000 + j as u64, &extra[j * d..(j + 1) * d]).unwrap();
            }
            a.update(10 * 3 + 7 + 300, &extra[..d]).unwrap();
            b.update(10 * 3 + 7 + 300, &extra[..d]).unwrap();
            assert_eq!(a.search(&q, 10, Effort::List(60)).unwrap().ids, b.search(&q, 10, Effort::List(60)).unwrap().ids);
            // an index on SSD refuses changes
            let mut c = Index::load(&dir, Numbers::Ssd).unwrap();
            assert!(c.add(999_999, &extra[..d]).is_err());
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn damaged_or_foreign_folders_are_refused() {
        let d = 8;
        let x = data(300, d, 9);
        let s = Settings { r: 8, l_build: 20, learned_effort: false, question_links: Switch::Off, ..Settings::default() };
        let mut a = Index::new(d, Metric::L2, s).unwrap();
        a.build(&x, &(0..300u64).collect::<Vec<_>>(), None).unwrap();
        let dir = tmp("damaged");
        a.save(&dir).unwrap();
        // a cut-short links file
        let links = fs::read(dir.join("links.u32")).unwrap();
        fs::write(dir.join("links.u32"), &links[..links.len() - 4]).unwrap();
        assert!(Index::load(&dir, Numbers::Ram).err().expect("refused").0.contains("links.u32"));
        fs::write(dir.join("links.u32"), &links).unwrap();
        // a link pointing outside the index
        let mut bad = links.clone();
        bad[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(dir.join("links.u32"), &bad).unwrap();
        assert!(Index::load(&dir, Numbers::Ram).is_err());
        fs::write(dir.join("links.u32"), &links).unwrap();
        // another format version, and not an index at all
        let mut meta = fs::read(dir.join("meta.bin")).unwrap();
        meta[8] = 99;
        fs::write(dir.join("meta.bin"), &meta).unwrap();
        assert!(Index::load(&dir, Numbers::Ram).err().expect("refused").0.contains("version"));
        fs::write(dir.join("meta.bin"), b"hello").unwrap();
        assert!(Index::load(&dir, Numbers::Ram).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn usage_and_part_2_facts_survive_saving_and_version_1_folders_still_load() {
        let d = 12;
        let x = data(800, d, 5);
        let tq = data(200, d, 6);
        let s = Settings { r: 12, l_build: 30, effort_questions: 150, track_usage: false, ..Settings::default() };
        let mut a = Index::new(d, Metric::Cos, s).unwrap();
        a.build(&x, &(0..800u64).collect::<Vec<_>>(), Some(&tq)).unwrap();
        a.s.track_usage = true;
        a.s.audit_budget = 1e12; // spot-check every question
        a.search(&data(40, d, 7), 10, Effort::Target(0.9)).unwrap();
        a.search(&data(5, d, 8), 10, Effort::List(30)).unwrap();
        a.audit();
        let gap = a.item_gap().unwrap();
        let dir = tmp("usage");
        a.save(&dir).unwrap();
        let b = Index::load(&dir, Numbers::Ram).unwrap();
        assert_eq!(a.usage(), b.usage());
        assert_eq!((b.usage().calls, b.usage().questions, b.usage().fixed), (2, 45, 5));
        assert_eq!(b.item_gap.get().copied(), Some(gap));
        assert!(b.s.track_usage);
        assert_eq!(b.s.audit_budget, 1e12);
        // the spot checks, margins included (they use the checked questions' spread)
        assert_eq!(a.spot_report(), b.spot_report());
        assert_eq!(a.spot_report().estimates.len(), 2);
        let (qa, qb) = (a.checked_questions(), b.checked_questions());
        assert_eq!(qa.len(), 45);
        assert!(qa.iter().zip(&qb).all(|(x, y)| x.q == y.q && x.correct == y.correct && x.kth == y.kth));
        // a version 1 folder: no text pairs at the end of meta.bin, no usage.json
        let mut meta = fs::read(dir.join("meta.bin")).unwrap();
        let tail = 8 + 2 * 8 + "track_usage".len() + "true".len() + 2 * 8 + "audit_budget".len()
            + format!("{:?}", a.s.audit_budget).len() + 2 * 8 + "effort_apart".len() + "true".len()
            + 2 * 8 + "relearn_every".len() + a.s.relearn_every.to_string().len()
            + 2 * 8 + "tune".len() + "auto".len() + 2 * 8 + "tune_target".len() + format!("{:?}", a.s.tune_target).len()
            + 2 * 8 + "max_bytes_per_item".len() + 1
            + 2 * 8 + "renumber".len() + "true".len()
            + 2 * 8 + "question_bits".len() + 2
            + 2 * 8 + "item_gap".len() + format!("{gap:?}").len();
        meta.truncate(meta.len() - tail);
        meta[8..12].copy_from_slice(&1u32.to_le_bytes());
        fs::write(dir.join("meta.bin"), &meta).unwrap();
        fs::remove_file(dir.join("usage.json")).unwrap();
        let c = Index::load(&dir, Numbers::Ram).unwrap();
        assert_eq!(c.usage(), crate::usage::Usage::default());
        assert!(c.item_gap.get().is_none());
        assert_eq!(c.search(&tq[..d * 3], 10, Effort::List(30)).unwrap().ids, a.search(&tq[..d * 3], 10, Effort::List(30)).unwrap().ids);
        let _ = fs::remove_dir_all(&dir);
    }
}
