//! Zero-dependency linear resampler with a persistent fractional phase, so
//! chunk boundaries are click-free. Steady-state [`LinearResampler::process`]
//! performs no heap allocation: the caller owns the scratch `Vec`.

/// Linear-interpolating resampler between two fixed sample rates.
///
/// A ratio of 1.0 short-circuits to a plain copy. The fractional phase and
/// the last input sample are carried across calls, so chunked processing is
/// continuous with offline processing.
#[allow(dead_code)] // fields consumed by the task-1.1 implementation
pub struct LinearResampler {
    src_rate: u32,
    dst_rate: u32,
    // Persistent fractional phase + previous sample for boundary continuity.
    phase: f64,
    last_input: f32,
}

impl LinearResampler {
    /// Creates a resampler from `src_rate` to `dst_rate` (Hz). A ratio of 1.0
    /// short-circuits to a copy.
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        Self {
            src_rate,
            dst_rate,
            phase: 0.0,
            last_input: 0.0,
        }
    }

    /// Source sample rate in Hz.
    pub fn src_rate(&self) -> u32 {
        self.src_rate
    }

    /// Destination sample rate in Hz.
    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    /// Resamples `input`, appending to the caller-owned scratch `out`
    /// (`out.clear()` then fill) — zero allocation in steady state.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let _ = (input, out);
        todo!("task 1.1: linear resampler with persistent fractional phase")
    }
}

/// One-shot resampling convenience for tests and `--selftest`.
pub fn resample_offline(input: &[f32], src: u32, dst: u32) -> Vec<f32> {
    let mut resampler = LinearResampler::new(src, dst);
    let mut out = Vec::new();
    resampler.process(input, &mut out);
    out
}
