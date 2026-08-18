//! Kokoro-82M ONNX TTS engine (24 kHz f32 output).

use std::path::Path;

use super::{TtsClip, TtsEngine};
use crate::error::Result;

/// Kokoro TTS via `ort`.
///
/// Per clause: phonemize → [`normalize_ipa`](super::phonemes::normalize_ipa)
/// → tokenize → inputs `input_ids int64 [1, n]`, `style f32 [1, 256]`,
/// `speed f32 [1]` → f32 PCM @ 24 kHz.
///
/// Style row selection matches the reference Kokoro ONNX implementations:
/// `style = voices[voice][min(n_tokens, 510)]` — the bank is indexed by
/// phoneme-token count, *not* a fixed row (always picking row 0 degrades
/// prosody). Hard cap: clauses over 510 tokens (512 context − 2 BOS/EOS pads)
/// are split at the last whitespace/punctuation before the cap.
#[allow(dead_code)] // session/voices bank wired up by the task-5.3 implementation
pub struct OnnxTts {
    session: Option<ort::session::Session>,
}

impl OnnxTts {
    /// Loads the model and the voices bank (`voices.bin`, a 2D
    /// `[n_rows, 256]` f32 style bank); `voice` names the row group, `speed`
    /// scales duration.
    pub fn load(model: &Path, voices: &Path, voice: &str, speed: f32) -> Result<Self> {
        let _ = (model, voices, voice, speed);
        todo!("task 5.3: ort session + voices bank load")
    }
}

impl TtsEngine for OnnxTts {
    fn synthesize(&mut self, text: &str) -> Result<TtsClip> {
        let _ = text;
        todo!("task 5.3: phonemize → tokenize → ort inference")
    }
}
