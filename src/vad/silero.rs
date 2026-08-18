//! Silero VAD v5 ONNX wrapper (stateful: carries the LSTM state tensor
//! across frames). Owned by the VAD task; not `Sync`.

use std::path::Path;

use crate::error::Result;

/// Samples per VAD frame: 32 ms at 16 kHz.
pub const FRAME_LEN: usize = 512;

/// Stateful Silero VAD inference wrapper.
///
/// Inputs per frame: `input [1, 512] f32`, `state [2, 1, 128] f32`, `sr i64`;
/// the returned state is carried into the next call. Input/output buffers are
/// preallocated — no per-frame allocation beyond `ort` internals.
#[allow(dead_code)] // session/state buffers wired up by the task-2.1 implementation
pub struct SileroVad {
    session: Option<ort::session::Session>,
}

impl SileroVad {
    /// Loads the ONNX model (`Session::builder()?.commit_from_file(...)`) and
    /// initializes the state tensor (`[2, 1, 128]` zeros) and `sr` input.
    pub fn new(model_path: &Path) -> Result<Self> {
        let _ = model_path;
        todo!("task 2.1: ort session + state init")
    }

    /// Runs one frame, returning the speech probability in `[0, 1]`.
    /// [`VadError::BadFrameLen`](crate::error::VadError::BadFrameLen) unless
    /// `frame.len() == FRAME_LEN`.
    pub fn process(&mut self, frame: &[f32]) -> Result<f32> {
        let _ = frame;
        todo!("task 2.1: single-frame inference with carried state")
    }

    /// Resets the LSTM state; called after each emitted segment.
    pub fn reset_state(&mut self) {
        todo!("task 2.1: zero the state tensor")
    }
}
