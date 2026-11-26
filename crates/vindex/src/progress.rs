//! What a build is doing now, for showing progress (the Python package's `build_progress()`).
//!
//! It sits outside the index's lock, so another thread can read it while the build runs. A build writes to it only
//! at a step change or after a batch of links (about 70 times for a million items), so a mutex costs nothing that
//! shows. It reports; nothing in the build reads it back.

use std::sync::Mutex;

/// The build's progress. `phase` names the part of the build (a settings trial, or the build itself), `step` what
/// that part is doing, and `done` of `total` how far linking has got (both 0 in steps that do not count).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProgressNow {
    pub phase: String,
    pub step: String,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, Default)]
pub struct Progress(Mutex<ProgressNow>);

impl Progress {
    /// What the build is doing now.
    pub fn now(&self) -> ProgressNow {
        self.0.lock().map(|p| p.clone()).unwrap_or_default()
    }

    /// A new part of the build; its step starts empty.
    pub fn phase(&self, phase: String) {
        if let Ok(mut p) = self.0.lock() {
            *p = ProgressNow { phase, ..ProgressNow::default() };
        }
    }

    /// A new step, with `total` things to do (0: not counted).
    pub fn step(&self, step: &str, total: u64) {
        if let Ok(mut p) = self.0.lock() {
            p.step = step.to_string();
            p.done = 0;
            p.total = total;
        }
    }

    /// `by` more things done in this step.
    pub fn advance(&self, by: u64) {
        if let Ok(mut p) = self.0.lock() {
            p.done += by;
        }
    }
}
