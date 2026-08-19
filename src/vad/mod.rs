//! VAD: [`SileroVad`] (stateful ort wrapper) plus the pure [`VadSegmenter`]
//! state machine (probabilities injected, so it is unit-testable headless).

pub mod silero;

pub use silero::{SileroVad, FRAME_LEN};

/// Duration of one [`FRAME_LEN`] frame in milliseconds (512 / 16 000 = 32).
const FRAME_MS: u32 = (FRAME_LEN as u32 * 1000) / 16_000;

/// Preroll ring depth: 8 frames = 256 ms, prepended so onsets aren't clipped.
const PREROLL_FRAMES: usize = 8;

/// Minimum active (non-preroll, non-trailing-silence) speech per segment.
/// Shorter bursts are rejected as clicks.
const MIN_SEGMENT_MS: u32 = 250;

/// Events emitted by [`VadSegmenter`].
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    /// Speech onset: fired on the first frame at or above the threshold while
    /// `Listening`.
    SpeechStart,
    /// A complete speech segment (8-frame preroll included), 16 kHz f32 mono.
    Segment(Vec<f32>),
}

/// Internal segmenter state.
#[derive(Debug, Clone, Copy)]
enum State {
    /// No speech in progress; incoming frames only update the preroll ring.
    Listening,
    /// Speech in progress; `silent_frames` counts the current run of trailing
    /// sub-threshold frames.
    Speaking { silent_frames: u32 },
}

/// Pure VAD segmentation state machine.
///
/// States `Listening` / `Speaking { silent_frames }`; a fixed preroll ring of
/// 8 frames (256 ms) is prepended so onsets are not clipped. `SpeechStart`
/// fires on the first frame ≥ `threshold` from `Listening`; `Segment` fires
/// when trailing silence exceeds `silence_ms` (a ~250 ms min-segment-length
/// guard rejects clicks). The segmenter returns to `Listening` immediately —
/// continuous listening, including during playback (required for barge-in).
pub struct VadSegmenter {
    threshold: f32,
    silence_ms: u32,
    state: State,
    /// Fixed preroll ring: the last [`PREROLL_FRAMES`] frames seen, in any
    /// state, so a fresh onset is always preceded by its audio context.
    preroll: [[f32; FRAME_LEN]; PREROLL_FRAMES],
    /// Next ring write slot.
    preroll_pos: usize,
    /// Frames currently valid in the ring (≤ [`PREROLL_FRAMES`]).
    preroll_len: usize,
    /// Samples of the segment currently being accumulated.
    segment: Vec<f32>,
    /// How many preroll frames were prepended at the current onset (the ring
    /// may be partially filled at stream start); excluded from the
    /// min-segment-length guard.
    preroll_at_onset: usize,
}

impl VadSegmenter {
    /// Creates a segmenter closing segments after `silence_ms` of trailing
    /// sub-`threshold` frames.
    pub fn new(threshold: f32, silence_ms: u32) -> Self {
        Self {
            threshold,
            silence_ms,
            state: State::Listening,
            preroll: [[0.0; FRAME_LEN]; PREROLL_FRAMES],
            preroll_pos: 0,
            preroll_len: 0,
            segment: Vec::new(),
            preroll_at_onset: 0,
        }
    }

    /// Feeds one 512-sample frame and its speech probability; returns an
    /// event when a boundary fires.
    pub fn push(&mut self, frame: &[f32; FRAME_LEN], prob: f32) -> Option<VadEvent> {
        let is_speech = prob >= self.threshold;
        match self.state {
            State::Listening => {
                if is_speech {
                    self.segment.clear();
                    // Drain the ring oldest → newest, then the onset frame.
                    let first =
                        (self.preroll_pos + PREROLL_FRAMES - self.preroll_len) % PREROLL_FRAMES;
                    for i in 0..self.preroll_len {
                        let idx = (first + i) % PREROLL_FRAMES;
                        let preroll_frame = self.preroll[idx];
                        self.segment.extend_from_slice(&preroll_frame);
                    }
                    self.preroll_at_onset = self.preroll_len;
                    self.segment.extend_from_slice(frame);
                    self.ring_push(frame);
                    self.state = State::Speaking { silent_frames: 0 };
                    Some(VadEvent::SpeechStart)
                } else {
                    self.ring_push(frame);
                    None
                }
            }
            State::Speaking { silent_frames } => {
                self.segment.extend_from_slice(frame);
                self.ring_push(frame);
                if is_speech {
                    self.state = State::Speaking { silent_frames: 0 };
                    return None;
                }
                let silent_frames = silent_frames + 1;
                // Exact ms math: close once trailing silence exceeds
                // silence_ms (300 ms → 10 frames at 32 ms/frame).
                if silent_frames * FRAME_MS > self.silence_ms {
                    self.state = State::Listening;
                    let total_frames = self.segment.len() / FRAME_LEN;
                    let active_frames = total_frames
                        .saturating_sub(self.preroll_at_onset)
                        .saturating_sub(silent_frames as usize);
                    if (active_frames as u32 * FRAME_MS) >= MIN_SEGMENT_MS {
                        Some(VadEvent::Segment(std::mem::take(&mut self.segment)))
                    } else {
                        // Sub-min-length blip (click): reject silently.
                        self.segment.clear();
                        None
                    }
                } else {
                    self.state = State::Speaking { silent_frames };
                    None
                }
            }
        }
    }

    /// Appends a frame to the fixed preroll ring.
    fn ring_push(&mut self, frame: &[f32; FRAME_LEN]) {
        self.preroll[self.preroll_pos] = *frame;
        self.preroll_pos = (self.preroll_pos + 1) % PREROLL_FRAMES;
        self.preroll_len = (self.preroll_len + 1).min(PREROLL_FRAMES);
    }
}
