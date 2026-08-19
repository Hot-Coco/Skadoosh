//! Zero-dependency linear resampler with a persistent fractional phase, so
//! chunk boundaries are click-free. Steady-state [`LinearResampler::process`]
//! performs no heap allocation: the caller owns the scratch `Vec`.

/// Linear-interpolating resampler between two fixed sample rates.
///
/// A ratio of 1.0 short-circuits to a plain copy. The fractional phase and
/// the last input sample are carried across calls, so chunked processing is
/// continuous with offline processing.
pub struct LinearResampler {
    src_rate: u32,
    dst_rate: u32,
    /// Source samples advanced per output sample (`src_rate / dst_rate`).
    step: f64,
    /// Position of the next output sample, in source samples, relative to the
    /// start of the *next* input slice: index `-1` is `last_input` (the
    /// previous chunk's final sample) and index `0` is the upcoming
    /// `input[0]`. Invariant after every call: `phase ∈ [-1, step)`.
    phase: f64,
    /// Final sample of the previous input chunk, used to interpolate the
    /// interval that straddles a chunk boundary.
    last_input: f32,
}

impl LinearResampler {
    /// Creates a resampler from `src_rate` to `dst_rate` (Hz). A ratio of 1.0
    /// short-circuits to a copy.
    pub fn new(src_rate: u32, dst_rate: u32) -> Self {
        assert!(
            src_rate > 0 && dst_rate > 0,
            "sample rates must be non-zero"
        );
        Self {
            src_rate,
            dst_rate,
            step: src_rate as f64 / dst_rate as f64,
            phase: 0.0,
            last_input: 0.0,
        }
    }

    /// Destination sample rate in Hz.
    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    /// Resamples `input`, appending to the caller-owned scratch `out`
    /// (`out.clear()` then fill) — zero allocation in steady state.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        out.clear();
        if self.src_rate == self.dst_rate {
            // Ratio 1.0: bit-exact passthrough.
            out.extend_from_slice(input);
            if let Some(&last) = input.last() {
                self.last_input = last;
            }
            return;
        }
        let len = input.len();
        // Upper-bound the emitted count and reserve once; in steady state the
        // capacity already suffices, so this is a no-op.
        out.reserve((len as f64 / self.step) as usize + 2);
        let step = self.step;
        let mut pos = self.phase;
        // The last interpolatable interval is [len - 2, len - 1]; outputs
        // landing in [len - 1, len) need the next chunk's first sample and
        // are deferred (that's what `last_input` + negative phase are for).
        let last_index = len as isize - 1;
        while (pos.floor() as isize) < last_index {
            let i = pos.floor() as isize;
            let frac = (pos - i as f64) as f32;
            let a = if i < 0 {
                self.last_input
            } else {
                input[i as usize]
            };
            let b = input[(i + 1) as usize];
            out.push(a + frac * (b - a));
            pos += step;
        }
        self.phase = pos - len as f64;
        if let Some(&last) = input.last() {
            self.last_input = last;
        }
    }
}

/// One-shot resampling convenience for tests and `--selftest`.
pub fn resample_offline(input: &[f32], src: u32, dst: u32) -> Vec<f32> {
    let mut resampler = LinearResampler::new(src, dst);
    let mut out = Vec::new();
    resampler.process(input, &mut out);
    out
}
