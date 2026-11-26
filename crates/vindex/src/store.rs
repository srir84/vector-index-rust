//! The full numbers, one row of `dim` floats per slot: in memory, or a file mapped from disk (SSD mode).

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

/// Where the full numbers live.
pub enum Store {
    /// In memory, slot-major.
    Ram { dim: usize, data: Vec<f32> },
    /// A read-only file of slot-major little-endian floats, read on demand by the OS (SSD mode).
    Mapped { dim: usize, rows: usize, map: Mmap },
}

impl Store {
    pub fn ram(dim: usize, capacity: usize) -> Self {
        Store::Ram { dim, data: vec![0.0; capacity * dim] }
    }

    /// Map `path` (written by `persist`) read-only. Rows must be complete.
    pub fn mapped(path: &Path, dim: usize) -> std::io::Result<Self> {
        let f = File::open(path)?;
        // SAFETY: the file is only read, and nothing in this process writes it while mapped.
        let map = unsafe { Mmap::map(&f)? };
        let bytes = map.len();
        if dim == 0 || bytes % (dim * 4) != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "numbers file is not whole rows"));
        }
        if !(map.as_ptr() as usize).is_multiple_of(4) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "numbers file is not 4-byte aligned"));
        }
        Ok(Store::Mapped { dim, rows: bytes / (dim * 4), map })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        match self {
            Store::Ram { dim, .. } | Store::Mapped { dim, .. } => *dim,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Store::Ram { dim, data } => data.len() / dim.max(&1),
            Store::Mapped { rows, .. } => *rows,
        }
    }

    #[inline]
    pub fn row(&self, slot: usize) -> &[f32] {
        match self {
            Store::Ram { dim, data } => &data[slot * dim..(slot + 1) * dim],
            Store::Mapped { dim, map, .. } => {
                let b = &map[slot * dim * 4..(slot + 1) * dim * 4];
                // SAFETY: the map is 4-byte aligned (checked in `mapped`), rows are whole, and any bit
                // pattern is a valid f32. Little-endian is assumed (checked when saving and loading).
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, *dim) }
            }
        }
    }

    /// Write a row (memory mode only; changes need the numbers in memory).
    pub fn set(&mut self, slot: usize, x: &[f32]) {
        match self {
            Store::Ram { dim, data } => data[slot * *dim..(slot + 1) * *dim].copy_from_slice(x),
            Store::Mapped { .. } => panic!("the full numbers are on SSD (read-only); load with numbers=\"ram\" to change items"),
        }
    }

    pub fn grow(&mut self, capacity: usize) {
        if let Store::Ram { dim, data } = self {
            if data.len() < capacity * *dim {
                data.resize(capacity * *dim, 0.0);
            }
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self, Store::Mapped { .. })
    }

    /// Ask the OS to start reading these rows now (SSD mode), so the re-checks overlap their reads.
    pub fn prefetch_rows(&self, slots: &[u32]) {
        if let Store::Mapped { dim, map, .. } = self {
            for &s in slots {
                let start = s as usize * dim * 4;
                let _ = map.advise_range(memmap2::Advice::WillNeed, start, dim * 4);
            }
        }
    }
}
