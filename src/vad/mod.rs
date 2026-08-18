//! VAD: [`SileroVad`] (stateful ort wrapper) plus the pure [`VadSegmenter`]
//! state machine (probabilities injected, so it is unit-testable headless).

pub mod silero;

pub use silero::{SileroVad, FRAME_LEN};

/// Events emitted by [`VadSegmenter`].
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    /// Speech onset: fired on the first frame at or above the threshold while
    /// `Listening`.
    SpeechStart,
    /// A complete speech segment (8-frame preroll included), 16 kHz f32 mono.
    Segment(Vec<f32>),
}

/// Pure VAD segmentation state machine.
///
/// States `Listening` / `Speaking { silent_frames }`; a fixed preroll ring of
/// 8 frames (256 ms) is prepended so onsets are not clipped. `SpeechStart`
/// fires on the first frame ≥ `threshold` from `Listening`; `Segment` fires
/// when trailing silence exceeds `silence_ms` (a ~250 ms min-segment-length
/// guard rejects clicks). The segmenter returns to `Listening` immediately —
/// continuous listening, including during playback (required for barge-in).
#[allow(dead_code)] // state fields consumed by the task-2.2 implementation
pub struct VadSegmenter {
    threshold: f32,
    silence_ms: u32,
}

impl VadSegmenter {
    /// Creates a segmenter closing segments after `silence_ms` of trailing
    /// sub-`threshold` frames.
    pub fn new(threshold: f32, silence_ms: u32) -> Self {
        Self {
            threshold,
            silence_ms,
        }
    }

    /// Feeds one 512-sample frame and its speech probability; returns an
    /// event when a boundary fires.
    pub fn push(&mut self, frame: &[f32; FRAME_LEN], prob: f32) -> Option<VadEvent> {
        let _ = (frame, prob);
        todo!("task 2.2: Listening/Speaking state machine with preroll")
    }
}
