//! Silero VAD v5 ONNX wrapper (stateful: carries the LSTM state tensor
//! across frames). Owned by the VAD task; not `Sync`.
//!
//! Buffer strategy: the model reads the caller's frame slice and our carried
//! state `Vec` in place via [`TensorRef`] (zero-copy), and the scalar `sr`
//! tensor is constructed once at load and re-fed by reference — so no input
//! is ever copied or reallocated per frame. The output tensors (`output`,
//! `stateN`) are allocated by `ort` inside every [`Session::run`] call (the
//! `SessionOutputs` owner); we copy the returned state back into our reusable
//! `state` buffer (1 KiB memcpy per 32 ms frame). That per-run output
//! allocation is the one `ort` internals force on us.

use std::path::Path;

use ort::session::Session;
use ort::value::{Tensor, TensorRef};

use crate::error::{Result, VadError};

/// Samples per VAD frame: 32 ms at 16 kHz.
pub const FRAME_LEN: usize = 512;

/// Silero v5 LSTM state tensor shape (`[2, 1, 128]`) and element count.
const STATE_SHAPE: [usize; 3] = [2, 1, 128];
const STATE_LEN: usize = STATE_SHAPE[0] * STATE_SHAPE[1] * STATE_SHAPE[2];

/// Stateful Silero VAD inference wrapper.
///
/// Inputs per frame: `input [1, 512] f32`, `state [2, 1, 128] f32`, `sr i64`
/// (scalar); the returned `stateN` is carried into the next call. See the
/// module docs for the buffer-reuse strategy.
pub struct SileroVad {
    session: Session,
    /// Carried LSTM state (`STATE_SHAPE`, row-major), fed back each frame.
    state: Vec<f32>,
    /// Scalar `sr` input (16 000), built once and fed by reference.
    sr: Tensor<i64>,
}

impl SileroVad {
    /// Loads the ONNX model (`Session::builder()?.commit_from_file(...)`) and
    /// initializes the state tensor (`[2, 1, 128]` zeros) and `sr` input.
    pub fn new(model_path: &Path) -> Result<Self> {
        let mut builder = Session::builder().map_err(|e| VadError::ModelLoad(e.to_string()))?;
        crate::gpu::apply_gpu_ep(&mut builder)
            .map_err(|e| VadError::ModelLoad(format!("GPU EP: {e}")))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| VadError::ModelLoad(format!("{}: {e}", model_path.display())))?;
        let sr = Tensor::from_array((Vec::<i64>::new(), vec![16_000_i64]))
            .map_err(|e| VadError::ModelLoad(format!("failed to build sr tensor: {e}")))?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            sr,
        })
    }

    /// Runs one frame, returning the speech probability in `[0, 1]`.
    /// [`VadError::BadFrameLen`] unless
    /// `frame.len() == FRAME_LEN`.
    pub fn process(&mut self, frame: &[f32]) -> Result<f32> {
        if frame.len() != FRAME_LEN {
            return Err(VadError::BadFrameLen {
                expected: FRAME_LEN,
                actual: frame.len(),
            }
            .into());
        }
        // Zero-copy input tensors: the model reads `frame` and `self.state`
        // in place for the duration of the synchronous `run` call.
        let input = TensorRef::from_array_view(([1_usize, FRAME_LEN], frame))
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let state = TensorRef::from_array_view((STATE_SHAPE, &self.state[..]))
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => &self.sr])
            .map_err(|e| VadError::Inference(e.to_string()))?;
        let prob = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Inference(e.to_string()))?
            .1[0];
        let new_state = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VadError::Inference(e.to_string()))?
            .1;
        self.state.copy_from_slice(new_state);
        Ok(prob)
    }

    /// Resets the LSTM state; called after each emitted segment.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }
}
