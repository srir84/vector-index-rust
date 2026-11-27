//! vindex: a fast, memory-lean vector search index.
//!
//! A flat link map (DiskANN's method) with the fixes that survived the experiments in
//! `../vector-index-experiments`: links built on flipped copies for best-score data, a learned
//! search effort per question, question links where questions look unlike the items, repair right
//! away after a delete, and 1-byte copies with a full-number re-check. See `DESIGN.md`.

// Index loops are kept where they follow the vector kernels' lane order or index several arrays at once, and the
// build's inner functions take their parts as separate arguments so the borrows stay disjoint.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod audit;
pub mod buckets;
pub mod build;
pub mod changes;
pub mod codes;
pub mod debug;
pub mod effort;
pub mod graph;
pub mod index;
pub mod layout;
pub mod metric;
pub mod persist;
pub mod prefetch;
pub mod progress;
pub mod prune;
pub mod qlinks;
pub mod rng;
pub mod score;
pub mod search;
pub mod store;
pub mod suggest;
pub mod tune;
pub mod usage;

pub use codes::{Codes, Prepared};
pub use graph::Graph;
pub use audit::{Checked, Estimate};
pub use index::{Effort, Error, Index, Memory, Settings, SpotReport, Stats, Tune};
pub use metric::Metric;
pub use persist::Numbers;
pub use progress::{Progress, ProgressNow};
pub use store::Store;
pub use suggest::Suggestion;
pub use tune::{Trial, Tuning};
pub use usage::{Usage, UsageReport};
