//! espeak-ng subprocess phonemizer + Kokoro tokenizer.

use std::collections::HashMap;

use crate::error::Result;

/// Phonemizes English text via `espeak-ng -q --ipa=3 -v en-us` with captured
/// stdout. A non-zero exit or a missing binary yields
/// [`TtsError::Phonemize`](crate::error::TtsError::Phonemize) — never a panic.
pub fn phonemize(text: &str) -> Result<String> {
    let _ = text;
    todo!("task 5.2: espeak-ng subprocess wrapper")
}

/// Normalizes raw espeak `--ipa=3` output toward Kokoro's misaki-trained G2P
/// (stress marks, allophone variants, tie bars). The replacement map is
/// derived from the kokoro-onnx/kokoro-rs reference at implementation time;
/// v1 pronunciation quality is below misaki-based pipelines.
pub fn normalize_ipa(raw: &str) -> String {
    let _ = raw;
    todo!("task 5.2: IPA normalization map")
}

/// Maps phoneme chars to token ids with BOS/EOS padding; unknown chars are
/// skipped and warn-counted. `vocab` is loaded from Kokoro's `config.json`.
pub fn tokenize(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let _ = (phonemes, vocab);
    todo!("task 5.2: char→id tokenizer with BOS/EOS")
}
