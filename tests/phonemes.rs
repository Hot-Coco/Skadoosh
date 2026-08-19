//! Phonemizer + tokenizer tests (plan task 5.2 acceptance). The espeak-ng
//! binary is a documented runtime dependency (plan §5); when it is absent
//! the espeak-dependent test skips loudly instead of failing.

use std::collections::HashMap;
use std::path::Path;

use skadoosh::error::{SkadooshError, TtsError};
use skadoosh::tts::phonemes::{kokoro_vocab, normalize_ipa, phonemize, phonemize_with, tokenize};

#[test]
fn phonemize_hello_produces_ipa() {
    let ipa = match phonemize("hello") {
        Ok(ipa) => ipa,
        Err(e) => {
            eprintln!(
                "skipping phonemize_hello_produces_ipa: espeak-ng unavailable ({e}); \
                 install with `apt install espeak-ng`"
            );
            return;
        }
    };
    assert!(!ipa.trim().is_empty(), "expected non-empty IPA");
    // espeak-ng 1.51 renders "hello" as `həlˈo‍ʊ`: sanity-check a vowel/stress.
    assert!(
        ipa.contains('h') && ipa.contains('ə'),
        "unexpected IPA for \"hello\": {ipa:?}"
    );
}

#[test]
fn phonemize_re_appends_terminal_punctuation() {
    let ipa = match phonemize("What time is it?") {
        Ok(ipa) => ipa,
        Err(e) => {
            eprintln!("skipping terminal-punct test: espeak-ng unavailable ({e})");
            return;
        }
    };
    assert!(ipa.ends_with('?'), "expected trailing '?', got {ipa:?}");
}

#[test]
fn missing_binary_returns_phonemize_error_not_panic() {
    let err = phonemize_with(Path::new("/nonexistent/espeak-ng-bogus"), "hi")
        .expect_err("missing binary must error");
    assert!(
        matches!(err, SkadooshError::Tts(TtsError::Phonemize(_))),
        "expected TtsError::Phonemize, got {err:?}"
    );
}

#[test]
fn failing_binary_returns_phonemize_error() {
    // `/bin/false` exits non-zero with no output: same error class.
    let err = phonemize_with(Path::new("/bin/false"), "hi").expect_err("non-zero exit must error");
    assert!(
        matches!(err, SkadooshError::Tts(TtsError::Phonemize(_))),
        "expected TtsError::Phonemize, got {err:?}"
    );
}

#[test]
fn normalize_strips_joiners_and_maps_allophones() {
    // ZWJ (U+200D) and tie bar (U+0361) are stripped, leaving the digraphs
    // the Kokoro vocab expects ("oʊ", "dʒ", "tʃ") — matching the kokoros
    // reference behavior.
    assert_eq!(normalize_ipa("həlˈo\u{200D}ʊ"), "həlˈoʊ");
    assert_eq!(normalize_ipa("d\u{200D}ʒˈʌmps"), "dʒˈʌmps");
    assert_eq!(normalize_ipa("t\u{0361}ʃ"), "tʃ");
    // Allophone replacements: ʲ→j, r→ɹ (x→k, ɬ→l share the same code path).
    assert_eq!(normalize_ipa("ɹˈɛdʲi"), "ɹˈɛdji");
    assert_eq!(normalize_ipa("red"), "ɹed");
    // Stress and length marks pass through unchanged (they are in-vocab).
    assert_eq!(normalize_ipa("ˌhɛlˈoː"), "ˌhɛlˈoː");
}

#[test]
fn normalize_applies_reference_word_rules() {
    // misaki en.py / kokoros reference rules:
    assert_eq!(normalize_ipa("kəkˈoːɹoʊ"), "kˈoʊkəɹoʊ"); // the model's own name
    assert_eq!(normalize_ipa("nˈaɪnti"), "nˈaɪndi"); // en-US flap
    assert_eq!(normalize_ipa("nˈaɪntiː"), "nˈaɪntiː"); // not before ː
    assert_eq!(normalize_ipa("tˈuːhˈʌndɹɪd"), "tˈuː hˈʌndɹɪd"); // word boundary
    assert_eq!(normalize_ipa("wɜːd z."), "wɜːdz."); // word-final z merge
}

#[test]
fn tokenize_pads_bos_eos_and_skips_unknown() {
    let vocab: HashMap<char, i64> = [('h', 50), ('ə', 83), ('l', 54), ('o', 57), ('ʊ', 135)]
        .into_iter()
        .collect();
    assert_eq!(tokenize("həloʊ", &vocab), vec![0, 50, 83, 54, 57, 135, 0]);
    // Unknown chars are skipped (with a warn count), not mapped or panicked.
    assert_eq!(tokenize("hXə", &vocab), vec![0, 50, 83, 0]);
    // Empty input still gets both pads.
    assert_eq!(tokenize("", &vocab), vec![0, 0]);
}

#[test]
fn tokenizer_maps_every_char_of_the_real_vocab() {
    let vocab = kokoro_vocab();
    assert!(
        vocab.len() > 100,
        "Kokoro-82M vocab should have 114 entries"
    );
    for (c, id) in &vocab {
        assert_eq!(
            tokenize(&c.to_string(), &vocab),
            vec![0, *id, 0],
            "vocab char {c:?} (id {id}) must round-trip"
        );
    }
}
