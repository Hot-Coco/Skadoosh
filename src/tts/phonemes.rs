//! espeak-ng subprocess phonemizer + Kokoro tokenizer.
//!
//! [`phonemize`] shells out to `espeak-ng -q --ipa=3 -v en-us` (raw IPA),
//! [`normalize_ipa`] aligns that output with Kokoro's misaki-trained G2P
//! inventory, and [`tokenize`] maps chars to token ids with BOS/EOS padding.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use crate::error::{Result, TtsError};

/// espeak-ng binary name resolved via `PATH`.
const ESPEAK_BIN: &str = "espeak-ng";

/// Terminal punctuation that Kokoro was trained to hear at clause ends
/// (subset of the vocab's punctuation); appended back after phonemization
/// because `espeak-ng --ipa` drops all punctuation from its output.
const TERMINAL_PUNCT: &[char] = &['.', '!', '?', ',', ';', ':'];

/// Phonemizes English text. Uses the pure-Rust [`misaki_rs`] G2P engine
/// when available; falls back to `espeak-ng` if misaki fails to load.
/// Terminal punctuation (`. ! ? , ; :`) that phonemizers strip from the
/// output is re-appended afterwards so Kokoro still sees clause-final
/// markers.
/// See [`phonemize_with`] for direct espeak control.
pub fn phonemize(text: &str) -> Result<String> {
    // Try misaki first — pure Rust, no external binary.
    let (ipa, from_misaki) = match super::misaki_g2p::PHONEMIZER.phonemize(text) {
        Ok(ipa) if !ipa.is_empty() || text.trim().is_empty() => (ipa, true),
        Ok(ipa) => (ipa, true),
        Err(e) => {
            tracing::debug!(error = %e, "misaki g2p failed; falling back to espeak-ng");
            (phonemize_with(Path::new(ESPEAK_BIN), text)?, false)
        }
    };
    // Re-append terminal punctuation that phonemizers drop. misaki may
    // preserve some punctuation, espeak always strips it in IPA mode.
    let mut result = if from_misaki {
        ipa
    } else {
        // espeak already has re-append logic in phonemize_with, so skip.
        return Ok(ipa);
    };
    if let Some(last) = text.trim_end().chars().last() {
        if TERMINAL_PUNCT.contains(&last) && !result.ends_with(last) {
            result.push(' ');
            result.push(last);
        }
    }
    Ok(result)
}

/// Phonemizes English text with an explicit espeak-ng binary path (the
/// binary path is a parameter so tests can exercise the missing-binary
/// failure without mutating the process-global `PATH`).
///
/// Text is fed on stdin (argv would misparse leading `-`); espeak prints one
/// IPA line per sentence, which are joined with single spaces. Since espeak
/// drops punctuation in IPA mode, the input's terminal punctuation (when one
/// of `. ! ? , ; :`) is re-appended so Kokoro still sees a clause-final
/// marker. A non-zero exit or a missing binary yields
/// [`TtsError::Phonemize`] — never a panic.
pub fn phonemize_with(binary: &Path, text: &str) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }
    let mut child = Command::new(binary)
        .args(["-q", "--ipa=3", "-v", "en-us"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| TtsError::Phonemize(format!("failed to spawn {}: {e}", binary.display())))?;
    if let Some(mut stdin) = child.stdin.take() {
        // Input is one clause (a few hundred bytes at most), far below the
        // pipe buffer, so this cannot deadlock against the child's output.
        if let Err(e) = stdin.write_all(text.as_bytes()) {
            return Err(TtsError::Phonemize(format!(
                "failed to write to {} stdin: {e}",
                binary.display()
            ))
            .into());
        }
        // Dropping stdin closes the pipe so espeak sees EOF.
    }
    let out = child
        .wait_with_output()
        .map_err(|e| TtsError::Phonemize(format!("failed to wait on {}: {e}", binary.display())))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(TtsError::Phonemize(format!(
            "{} exited with {}: {}",
            binary.display(),
            out.status,
            stderr.trim()
        ))
        .into());
    }
    let ipa = String::from_utf8_lossy(&out.stdout);
    let mut joined = ipa
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(last) = text.trim_end().chars().last() {
        if TERMINAL_PUNCT.contains(&last) && !joined.ends_with(last) {
            joined.push(last);
        }
    }
    Ok(joined)
}

/// Normalizes raw espeak `--ipa=3` output toward Kokoro's misaki-trained G2P
/// inventory.
///
/// The replacement map mirrors the reference implementations — misaki's
/// `en.py` post-processing and its Rust port in kokoros
/// (`kokoros/src/tts/phonemizer.rs`, github.com/lucasjinreal/kokoros):
///
/// * Kokoro word-name fixes (`kəkˈoːɹoʊ` → `kˈoʊkəɹoʊ`, British variant too);
/// * allophone char replacements `ʲ`→`j`, `r`→`ɹ`, `x`→`k`, `ɬ`→`l`;
/// * espeak `--ipa=3` joins affricates/diphthongs with U+200D ZWJ (and
///   occasionally U+0361 tie bar): the joiner is stripped, leaving the
///   digraphs (`dʒ`, `oʊ`, `aɪ`) the reference vocab expects — exactly what
///   kokoros' vocab filter produces;
/// * misaki's `hundred` word-boundary fix (`tˈuːhˈʌndɹɪd` → `tˈuː hˈʌndɹɪd`),
///   the word-final ` z` merge before punctuation, and the en-US flap
///   (`nˈaɪnti` → `nˈaɪndi`, unless followed by `ː`).
///
/// espeak stress marks `ˈ`/`ˌ` and length `ː` are already in the Kokoro
/// vocab and pass through unchanged. v1 pronunciation quality is below
/// misaki-based pipelines (README caveat); unknown chars that survive this
/// map are skipped later by [`tokenize`] with a warning.
pub fn normalize_ipa(raw: &str) -> String {
    let mut ps = raw.to_string();
    // Kokoro word-name fixes (misaki en.py).
    ps = ps
        .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
        .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");
    // Allophone char replacements (misaki en.py / kokoros).
    ps = ps
        .replace('\u{02B2}', "j") // ʲ
        .replace('r', "ɹ")
        .replace('x', "k")
        .replace('ɬ', "l");
    // Strip joiners/tie bars, leaving digraphs (see fn docs).
    ps = ps.replace(['\u{200D}', '\u{0361}'], "");
    ps = hundred_rule(&ps);
    ps = z_merge_rule(&ps);
    ps = ninety_flap_rule(&ps);
    ps.trim().to_string()
}

/// misaki `(?<=[a-zɹː])(?=hˈʌndɹɪd)` → `" "`: espeak runs "two hundred"
/// together as `tˈuːhˈʌndɹɪd`; re-insert the missing word boundary.
fn hundred_rule(ps: &str) -> String {
    const NEEDLE: &str = "hˈʌndɹɪd";
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len() + 4);
    let needle: Vec<char> = NEEDLE.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i > 0 && chars[i..].starts_with(&needle) && matches!(chars[i - 1], 'a'..='z' | 'ɹ' | 'ː')
        {
            out.push(' ');
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// misaki ` z(?=[;:,.!?¡¿—…"«»“” ]|$)` → `z`: espeak sometimes detaches a
/// word-final `z`; merge it back when punctuation/space/end follows.
fn z_merge_rule(ps: &str) -> String {
    const PUNCT: &[char] = &[
        ';', ':', ',', '.', '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', ' ',
    ];
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' && i + 1 < chars.len() && chars[i + 1] == 'z' {
            let after = chars.get(i + 2);
            match after {
                None => i += 1, // " z" at end: drop the space
                Some(&c) if PUNCT.contains(&c) => i += 1,
                _ => {}
            }
        }
        if i < chars.len() {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// misaki en-US flap `(?<=nˈaɪn)ti(?!ː)` → `di` ("ninety" → `nˈaɪndi`).
fn ninety_flap_rule(ps: &str) -> String {
    const NEEDLE: &str = "nˈaɪnti";
    let needle: Vec<char> = NEEDLE.chars().collect();
    let chars: Vec<char> = ps.chars().collect();
    let mut out = String::with_capacity(ps.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i..].starts_with(&needle) && chars.get(i + needle.len()) != Some(&'ː') {
            // Emit "nˈaɪn" + "di" (the "ti" → "di" flap).
            out.extend(needle[..needle.len() - 2].iter());
            out.push('d');
            out.push('i');
            i += needle.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Maps phoneme chars to token ids with BOS/EOS padding (id `0` at both
/// ends); unknown chars are skipped and warn-counted. `vocab` is Kokoro's
/// vocab — see [`kokoro_vocab`].
pub fn tokenize(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let mut ids = Vec::with_capacity(phonemes.len() + 2);
    ids.push(0); // BOS
    let mut unknown = 0usize;
    for c in phonemes.chars() {
        match vocab.get(&c) {
            Some(&id) => ids.push(id),
            None => unknown += 1,
        }
    }
    if unknown > 0 {
        tracing::warn!(unknown, "tokenize: skipped unknown phoneme chars");
    }
    ids.push(0); // EOS
    ids
}

/// Kokoro-82M's character vocab (114 entries; ids are non-contiguous, `0` is
/// the BOS/EOS pad). Embedded from `hexgrad/Kokoro-82M` `config.json`
/// (`vocab` dict) — the model version this crate targets
/// (`kokoro-v0_19.onnx` from thewh1teagle/kokoro-onnx `model-files` release)
/// is pinned, so the vocab is a compile-time constant rather than a third
/// downloaded file.
pub fn kokoro_vocab() -> HashMap<char, i64> {
    KOKORO_VOCAB.iter().copied().collect()
}

const KOKORO_VOCAB: &[(char, i64)] = &[
    (';', 1),
    (':', 2),
    (',', 3),
    ('.', 4),
    ('!', 5),
    ('?', 6),
    ('\u{2014}', 9),
    ('\u{2026}', 10),
    ('"', 11),
    ('(', 12),
    (')', 13),
    ('\u{201C}', 14),
    ('\u{201D}', 15),
    (' ', 16),
    ('\u{303}', 17),
    ('\u{2A3}', 18),
    ('\u{2A5}', 19),
    ('\u{2A6}', 20),
    ('\u{2A8}', 21),
    ('\u{1D5D}', 22),
    ('\u{AB67}', 23),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('\u{1D4A}', 42),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('\u{251}', 69),
    ('\u{250}', 70),
    ('\u{252}', 71),
    ('\u{E6}', 72),
    ('\u{3B2}', 75),
    ('\u{254}', 76),
    ('\u{255}', 77),
    ('\u{E7}', 78),
    ('\u{256}', 80),
    ('\u{F0}', 81),
    ('\u{2A4}', 82),
    ('\u{259}', 83),
    ('\u{25A}', 85),
    ('\u{25B}', 86),
    ('\u{25C}', 87),
    ('\u{25F}', 90),
    ('\u{261}', 92),
    ('\u{265}', 99),
    ('\u{268}', 101),
    ('\u{26A}', 102),
    ('\u{29D}', 103),
    ('\u{26F}', 110),
    ('\u{270}', 111),
    ('\u{14B}', 112),
    ('\u{273}', 113),
    ('\u{272}', 114),
    ('\u{274}', 115),
    ('\u{F8}', 116),
    ('\u{278}', 118),
    ('\u{3B8}', 119),
    ('\u{153}', 120),
    ('\u{279}', 123),
    ('\u{27E}', 125),
    ('\u{27B}', 126),
    ('\u{281}', 128),
    ('\u{27D}', 129),
    ('\u{282}', 130),
    ('\u{283}', 131),
    ('\u{288}', 132),
    ('\u{2A7}', 133),
    ('\u{28A}', 135),
    ('\u{28B}', 136),
    ('\u{28C}', 138),
    ('\u{263}', 139),
    ('\u{264}', 140),
    ('\u{3C7}', 142),
    ('\u{28E}', 143),
    ('\u{292}', 147),
    ('\u{294}', 148),
    ('\u{2C8}', 156),
    ('\u{2CC}', 157),
    ('\u{2D0}', 158),
    ('\u{2B0}', 162),
    ('\u{2B2}', 164),
    ('\u{2193}', 169),
    ('\u{2192}', 171),
    ('\u{2197}', 172),
    ('\u{2198}', 173),
    ('\u{1D7B}', 177),
];
