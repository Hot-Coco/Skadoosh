//! Pure-Rust G2P via [`misaki_rs`] — replaces the espeak-ng subprocess path.
//!
//! [`MisakiG2p`] wraps `misaki_rs::G2P` and produces Kokoro-compatible IPA
//! phonemes directly, no external binary needed. The espeak-ng path in
//! [`super::phonemes`] is kept as a fallback for environments where misaki
//! cannot be loaded.
//!
//! [`MisakiG2p::phonemize`] returns the same IPA string format that
//! [`super::phonemes::phonemize`] does, so the downstream
//! `normalize_ipa → tokenize` pipeline works unchanged.

use crate::error::{Result, TtsError};
use misaki_rs::{Language, G2P};
use std::sync::{LazyLock, Mutex};

/// Global misaki G2P instance, lazily initialised on first use.
pub(crate) static PHONEMIZER: LazyLock<MisakiG2p> = LazyLock::new(MisakiG2p::new);

/// A lazily-initialised, thread-safe misaki G2P engine.
pub struct MisakiG2p {
    inner: Mutex<Option<G2P>>,
}

impl MisakiG2p {
    /// Create a new (uninitialised) engine. Construction is cheap; the model
    /// data is loaded on first use.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    /// Ensure the engine is loaded, returning a reference.
    fn get_or_init(&self) -> Result<std::sync::MutexGuard<'_, Option<G2P>>> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| TtsError::Phonemize(format!("misaki mutex poisoned: {e}")))?;
        if guard.is_none() {
            *guard = Some(G2P::new(Language::EnglishUS));
        }
        Ok(guard)
    }

    /// Phonemize English text into Kokoro-compatible IPA.
    ///
    /// Returns the same format as `phonemes::phonemize()` — space-joined IPA
    /// with terminal punctuation preserved.
    pub fn phonemize(&self, text: &str) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }
        let mut guard = self.get_or_init()?;
        let g2p = guard.as_mut().expect("just initialised");
        let (ipa, _tokens) = g2p
            .g2p(text)
            .map_err(|e| TtsError::Phonemize(format!("misaki g2p: {e}")))?;
        Ok(ipa.trim().to_string())
    }
}

impl Default for MisakiG2p {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misaki_basic_phonemes() {
        let g2p = MisakiG2p::new();
        let ipa = g2p.phonemize("Hello, world!").expect("phonemize");
        assert!(!ipa.is_empty(), "should produce phonemes");
        assert!(
            !ipa.contains('❓'),
            "should not contain unknown marker: {ipa}"
        );
    }

    #[test]
    fn misaki_empty_input() {
        let g2p = MisakiG2p::new();
        assert_eq!(g2p.phonemize("").unwrap(), "");
        assert_eq!(g2p.phonemize("   ").unwrap(), "");
    }

    #[test]
    fn misaki_known_words() {
        let g2p = MisakiG2p::new();
        for word in ["hello", "world", "testing", "skadoosh", "voice", "agent"] {
            let ipa = g2p.phonemize(word).expect(word);
            assert!(!ipa.is_empty(), "{word}: empty output");
        }
    }
}
