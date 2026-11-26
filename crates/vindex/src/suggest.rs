//! Suggestions (part 2, DESIGN.md section 17): changes that need a rebuild, worked out from how the index is used.
//! Each says why. A change of settings is measured on a trial built from the current items and the spot-checked
//! questions, timed against the index as it is now, never guessed.

use crate::audit::RELEARN_MIN;
use crate::index::{Index, Settings, Switch};
use crate::rng::Rng;
use crate::tune::{self, Tuning};

/// A rebuild is suggested only when its trial is at least this much faster than the index as it is.
pub const WORTH_IT: f64 = 1.15;
/// Slots holding deleted items get a note from this share of the used slots.
pub const EMPTY_SHARE: f64 = 0.2;
/// A suggestion's trial takes the whole index up to this many items.
pub const SUGGEST_TRIAL_MAX: usize = 100_000;

/// One suggestion.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    /// "rebuild" (measured on a trial) or "note".
    pub kind: &'static str,
    /// What to do, and the evidence, in plain words.
    pub what: String,
    pub why: String,
    /// Settings for the rebuild, as name and value.
    pub settings: Vec<(String, String)>,
    /// Whether the rebuild should take the spot-checked questions (`Index::checked_questions`) as its sample
    /// questions: true when they change whether question links turn on.
    pub use_checked_questions: bool,
    /// Searches per second at the tuning target in the trial: the index as it is, and after the rebuild.
    pub now: Option<f64>,
    pub after: Option<f64>,
}

fn note(what: String, why: String) -> Suggestion {
    Suggestion { kind: "note", what, why, settings: Vec::new(), use_checked_questions: false, now: None, after: None }
}

impl Index {
    /// Rebuilds worth doing, and notes, with the settings trial when it ran (DESIGN.md section 17). The trial needs
    /// at least 800 spot-checked questions and 10,000 items; it builds several trial indexes, so it takes a while.
    /// Searches continue meanwhile.
    pub fn suggest(&self) -> (Vec<Suggestion>, Option<Tuning>) {
        let mut out = Vec::new();
        let (used, alive) = (self.n_used, self.slot_of.len());
        let links_on = self.train_q.is_some();

        // slots holding deleted items
        let empty = used.saturating_sub(alive);
        if used > 0 && empty as f64 >= EMPTY_SHARE * used as f64 {
            let per = self.memory().total as f64 / used as f64;
            out.push(note(
                "Rebuild to free memory.".into(),
                format!("{empty} of {used} slots ({:.0}%) hold deleted items: about {:.0} MB that adds will only partly reuse.",
                        100.0 * empty as f64 / used as f64, empty as f64 * per / 1e6),
            ));
        }

        // real questions unlike the items while question links are off
        let use_report = self.usage_report();
        if let Some(u) = use_report.unlike_items {
            if !links_on && self.s.question_links != Switch::Off && u > self.s.ood_ratio as f64 {
                out.push(note(
                    "Real questions look unlike the items, but question links are off.".into(),
                    format!("In use, a question's best answer is {u:.1} times as far as items are from each other \
                             (question links turn on above {:.1}); the build's sample questions gave {}. Rebuilding \
                             with the spot-checked questions as sample questions would turn them on.",
                            self.s.ood_ratio, self.ood_ratio.map_or("no reading".into(), |r| format!("{r:.1}"))),
                ));
            }
        }

        // accuracy below a target before learning again can start
        let spot = self.spot_report();
        let checked = self.checked_questions();
        for e in &spot.estimates {
            let t = e.label.strip_prefix("target ").and_then(|t| t.parse::<f64>().ok());
            if let (Some(t), Some(m)) = (t, e.margin) {
                if t - e.accuracy > m && checked.len() < RELEARN_MIN {
                    out.push(note(
                        format!("Accuracy is below its {:.0}% target.", 100.0 * t),
                        format!("Spot checks found {:.1}% ± {:.1}%; the effort table is learned again from use once \
                                 {RELEARN_MIN} questions are spot-checked ({} so far).",
                                100.0 * e.accuracy, 100.0 * m, checked.len()),
                    ));
                }
            }
        }

        // a settings trial on the current items and the spot-checked questions
        let alive_slots: Vec<u32> = (0..used as u32).filter(|&u| self.graph.is_alive(u as usize)).collect();
        if checked.len() < tune::MIN_QUESTIONS || alive_slots.len() < tune::MIN_ITEMS {
            out.push(note(
                "No settings trial yet.".into(),
                format!("It needs {} spot-checked questions and {} items; there are {} and {}.",
                        tune::MIN_QUESTIONS, tune::MIN_ITEMS, checked.len(), alive_slots.len()),
            ));
            return (out, None);
        }
        let qs: Vec<f32> = checked.iter().flat_map(|c| c.q.iter().copied()).collect();
        // the whole index, up to 100,000 items: a fifth of it understates what better links gain at full size
        // (GloVe at 16 links: 1.8x in a fifth-size trial, 2.9x for real)
        let m = alive_slots.len().min(SUGGEST_TRIAL_MAX);
        let mut pick = Rng::new(self.s.seed ^ 0x7375_6767).sample(alive_slots.len(), m);
        pick.sort_unstable();
        let xs: Vec<f32> = pick.iter().flat_map(|&j| self.store.row(alive_slots[j] as usize).iter().copied()).collect();
        // the index as it is: its settings, with question links as they are now
        let now = Settings { question_links: if links_on { Switch::On } else { Switch::Off }, ..self.s.clone() };
        // the choices: question links decided by the spot-checked questions, unless switched off
        let base = Settings { question_links: if self.s.question_links == Switch::Off { Switch::Off } else { Switch::Auto }, ..self.s.clone() };
        // the baseline is the index itself when the trial holds all of its items: a trial of it built from the
        // spot-checked questions would have weaker question links than the real one (built from all its samples)
        let first = if m == alive_slots.len() { tune::First::Index(self) } else { tune::First::Settings(&now) };
        // timed on the last questions, as the build's tuning is: a quarter, at most 1,000. With about 2,000 real
        // questions a question-links rebuild's gain is predicted to about +-20% (step 19: Llama said 4.4x, gave 3.6x;
        // timing on half of them instead said 3.3x and gave 4.2x)
        let t = tune::search(&base, first, self.metric, self.dim, &xs, &qs, (checked.len() / 4).min(1000), &Default::default());
        if let (Some(c), Some(v0)) = (t.chosen, t.trials.first().and_then(|f| f.searches_per_s)) {
            let best = &t.trials[c];
            let v1 = best.searches_per_s.unwrap_or(0.0);
            if c != 0 && v1 >= WORTH_IT * v0 {
                let mut settings = Vec::new();
                let mut changes = Vec::new();
                if best.r != self.s.r {
                    settings.push(("r".to_string(), best.r.to_string()));
                    changes.push(format!("{} links per item (now {})", best.r, self.s.r));
                }
                if best.alpha != self.s.alpha {
                    settings.push(("alpha".to_string(), format!("{:?}", best.alpha)));
                    changes.push(format!("slack {:.1} (now {:.1})", best.alpha, self.s.alpha));
                }
                if best.code_bits != self.s.code_bits {
                    settings.push(("code_bits".to_string(), best.code_bits.to_string()));
                    changes.push(format!("{}-bit copies (now {}-bit)", best.code_bits, self.s.code_bits));
                }
                let links_change = best.question_links != links_on;
                if links_change {
                    changes.push(if best.question_links {
                        "the spot-checked questions as sample questions, which turn question links on".to_string()
                    } else {
                        "the spot-checked questions as sample questions, which leave question links off".to_string()
                    });
                }
                settings.push(("tune".to_string(), "off".to_string()));
                out.push(Suggestion {
                    kind: "rebuild",
                    what: format!("Rebuild with {}: about {:.1}x faster at {:.0}% accuracy.", changes.join(" and "), v1 / v0, 100.0 * t.target),
                    why: format!("A trial on {m} of the items with {} spot-checked questions: {v0:.0} searches/s as the \
                                  index is now, {v1:.0} with the change, one core. Memory {:.0} bytes per item (now {:.0}).",
                                 checked.len(), best.bytes_per_item, t.trials[0].bytes_per_item),
                    settings,
                    use_checked_questions: links_change,
                    now: Some(v0),
                    after: Some(v1),
                });
            }
        }
        (out, Some(t))
    }
}
