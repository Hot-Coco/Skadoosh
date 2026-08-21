//! Procedural hold music: generates pleasant-sounding chord progressions
//! using additive sine-wave synthesis with no external audio files.
//!
//! When the agent is waiting for a long tool execution, hold music fills
//! the silence. The music auto-ducks when TTS clips arrive and resumes
//! when the agent is thinking again.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::tts::TTS_SAMPLE_RATE;

/// Number of simultaneous voices in the chord.
const VOICES: usize = 4;

/// Fade-in/out duration in seconds for smooth transitions.
const FADE_S: f32 = 0.15;

/// Frequency multiplier for the chord root. Keys cycle through a relaxed
/// progression: C → Am → F → G.
const PROGRESSION: [[f32; VOICES]; 4] = [
    [261.63, 329.63, 392.00, 523.25], // C major
    [220.00, 261.63, 329.63, 440.00], // Am
    [174.61, 220.00, 261.63, 349.23], // F major
    [196.00, 246.94, 293.66, 392.00], // G major
];

/// Generates a fixed number of 24 kHz f32 mono samples of hold music.
///
/// `t` is the global sample counter; each call advances it by the
/// returned frame length so successive calls produce a seamless stream.
/// `active` gates generation: when false the output is silence (used for
/// ducking under TTS).
pub struct HoldMusic {
    t: u64,
    active: Arc<AtomicBool>,
}

impl HoldMusic {
    /// Creates a new generator with the shared active flag.
    pub fn new(active: Arc<AtomicBool>) -> Self {
        Self { t: 0, active }
    }

    /// Generates up to `max_samples` of 24 kHz f32 mono, advancing the
    /// internal counter. Returns silence when `active` is false.
    pub fn generate(&mut self, max_samples: usize) -> Vec<f32> {
        let active = self.active.load(Ordering::Relaxed);
        let mut out = Vec::with_capacity(max_samples);
        let rate = TTS_SAMPLE_RATE as f64;
        let fade_samples = (FADE_S * TTS_SAMPLE_RATE as f32) as usize;
        let chord_dur_samples = (2.8 * rate) as u64;
        let n = max_samples.min(4096);

        for i in 0..n {
            let sample_t = self.t + i as u64;

            // Cycle through the progression every ~2.8 s.
            let chord_idx = ((sample_t / chord_dur_samples) % 4) as usize;
            let freqs = PROGRESSION[chord_idx];

            // Crossfade between adjacent chords.
            let fade_pos = (sample_t % chord_dur_samples) as usize;
            let crossfade: f32 = if fade_pos < fade_samples {
                fade_pos as f32 / fade_samples as f32
            } else if fade_pos >= chord_dur_samples as usize - fade_samples {
                1.0 - (fade_pos - (chord_dur_samples as usize - fade_samples)) as f32
                    / fade_samples as f32
            } else {
                1.0
            };

            let prev_idx = if chord_idx == 0 { 3 } else { chord_idx - 1 };
            let prev_freqs = PROGRESSION[prev_idx];

            let mut sample: f32 = 0.0;

            if active {
                let phase = sample_t as f64 * 2.0 * std::f64::consts::PI / rate;

                for v in 0..VOICES {
                    // Current chord voice.
                    let amp_cur = 0.08 * crossfade;
                    sample += ((phase * freqs[v] as f64).sin()) as f32 * amp_cur;

                    // Previous chord voice (crossfade).
                    let amp_prev = 0.08 * (1.0 - crossfade);
                    sample += ((phase * prev_freqs[v] as f64).sin()) as f32 * amp_prev;
                }

                // Gentle LFO for a warmer feel.
                let lfo_phase = sample_t as f64 * 0.3 * 2.0 * std::f64::consts::PI / rate;
                let lfo = (lfo_phase.sin() as f32) * 0.3 + 0.7;
                sample *= lfo;
            }

            out.push(sample);
        }

        self.t += n as u64;
        out
    }

    /// Returns how many samples have been generated total.
    pub fn elapsed_samples(&self) -> u64 {
        self.t
    }

    /// Resets the generator to t=0.
    pub fn reset(&mut self) {
        self.t = 0;
    }
}
