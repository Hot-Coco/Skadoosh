//! Kokoro-82M ONNX TTS engine (24 kHz f32 output).
//!
//! Per clause: phonemize → [`normalize_ipa`]
//! → tokenize → inputs `input_ids int64 [1, n]`, `style f32 [1, 256]`,
//! `speed f32 [1]` → f32 PCM @ 24 kHz.
//!
//! Style row selection matches the reference Kokoro ONNX implementations:
//! `style = voices[voice][min(n_tokens, 510)]` — the bank is indexed by
//! phoneme-token count, *not* a fixed row (always picking row 0 degrades
//! prosody). Hard cap: clauses over 510 tokens (512 context − 2 BOS/EOS pads)
//! are split at the last whitespace/punctuation before the cap.

use std::collections::HashMap;
use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;

use super::phonemes::{kokoro_vocab, normalize_ipa, phonemize, tokenize};
use super::{TtsClip, TtsEngine, TTS_SAMPLE_RATE};
use crate::error::{Result, TtsError};

/// Style bank width (Kokoro `style_dim`).
const STYLE_DIM: usize = 256;
/// Maximum phoneme tokens per inference (512 context − 2 BOS/EOS pads).
const MAX_PHONEME_TOKENS: usize = 510;

/// Kokoro TTS via `ort`.
pub struct OnnxTts {
    session: Session,
    /// Style bank for the selected voice, flattened row-major
    /// `[rows][STYLE_DIM]` (`rows` is 511 for the stock `voices.bin`).
    styles: Vec<f32>,
    /// Number of style rows in `styles`.
    style_rows: usize,
    /// Speaking-rate multiplier fed to the `speed` input.
    speed: f32,
    /// Kokoro char vocab (BOS/EOS = id 0 handled by [`tokenize`]).
    vocab: HashMap<char, i64>,
}

impl OnnxTts {
    /// Loads the model and the voices bank; `voice` names the row group
    /// (`"af"`, `"am_adam"`, ...), `speed` scales duration.
    ///
    /// `voices` accepts the stock `voices.bin` (an **npz bundle**: an
    /// *uncompressed/stored* ZIP of per-voice `.npy` members, verified
    /// 2026-08-18 against thewh1teagle/kokoro-onnx release `model-files`,
    /// 5 758 648 bytes, 11 stored `<f4` `[511, 1, 256]` members) or a plain
    /// single-voice `.npy` file. Only the requested voice is decoded into
    /// memory. Deflated ZIP members are *not* supported (no zip crate dep):
    /// they yield [`TtsError::MissingVoices`].
    pub fn load(model: &Path, voices: &Path, voice: &str, speed: f32) -> Result<Self> {
        let mut builder = Session::builder().map_err(|e| TtsError::ModelLoad(e.to_string()))?;
        let session = builder
            .commit_from_file(model)
            .map_err(|e| TtsError::ModelLoad(format!("{}: {e}", model.display())))?;
        let bank = load_voice_bank(voices, voice)?;
        Ok(Self {
            session,
            styles: bank.data,
            style_rows: bank.rows,
            speed,
            vocab: kokoro_vocab(),
        })
    }

    /// Runs one inference on an already-normalized phoneme chunk that fits
    /// the model context, returning 24 kHz f32 PCM.
    fn infer_chunk(&mut self, phonemes: &str) -> Result<Vec<f32>> {
        let tokens = tokenize(phonemes, &self.vocab); // [BOS, .., EOS]
        let n_phoneme_tokens = tokens.len() - 2;
        let n = tokens.len();
        // Style row = min(phoneme-token count, 510), indexed before padding
        // (reference: kokoros `mix_styles(style_name, tokens.len())`).
        let row = n_phoneme_tokens
            .min(MAX_PHONEME_TOKENS)
            .min(self.style_rows.saturating_sub(1));
        let start = row * STYLE_DIM;
        let style = self.styles[start..start + STYLE_DIM].to_vec();

        let input_ids = Tensor::from_array((vec![1_i64, n as i64], tokens))
            .map_err(|e| TtsError::Inference(format!("input_ids tensor: {e}")))?;
        let style_t = Tensor::from_array((vec![1_i64, STYLE_DIM as i64], style))
            .map_err(|e| TtsError::Inference(format!("style tensor: {e}")))?;
        let speed_t = Tensor::from_array((vec![1_i64], vec![self.speed]))
            .map_err(|e| TtsError::Inference(format!("speed tensor: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "style" => style_t,
                "speed" => speed_t,
            ])
            .map_err(|e| TtsError::Inference(e.to_string()))?;
        let (_shape, pcm) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Inference(format!("extract audio tensor: {e}")))?;
        Ok(pcm.to_vec())
    }
}

impl TtsEngine for OnnxTts {
    fn synthesize(&mut self, text: &str) -> Result<TtsClip> {
        let raw = phonemize(text)?;
        let phonemes = normalize_ipa(&raw);
        let mut samples = Vec::new();
        let mut remaining = phonemes.trim();
        while !remaining.is_empty() {
            let (head, tail) = split_at_token_cap(remaining, &self.vocab);
            if !head.trim().is_empty() {
                samples.extend_from_slice(&self.infer_chunk(head)?);
            }
            remaining = tail.trim_start();
        }
        Ok(TtsClip {
            samples,
            sample_rate: TTS_SAMPLE_RATE,
        })
    }
}

/// Splits a phoneme string into `(head, tail)` where `head` tokenizes to at
/// most [`MAX_PHONEME_TOKENS`] ids. When the whole string fits, `tail` is
/// empty. Otherwise the cut lands just after the last whitespace or
/// punctuation char before the cap (falling back to a hard cut at the cap
/// char when there is none).
fn split_at_token_cap<'a>(phonemes: &'a str, vocab: &HashMap<char, i64>) -> (&'a str, &'a str) {
    let mut count = 0usize;
    let mut cap_byte = None;
    for (byte_idx, c) in phonemes.char_indices() {
        if vocab.contains_key(&c) {
            if count == MAX_PHONEME_TOKENS {
                // This char would become token 511; cut before it.
                cap_byte = Some(byte_idx);
                break;
            }
            count += 1;
        }
    }
    let Some(cap_byte) = cap_byte else {
        return (phonemes, "");
    };
    // Last whitespace/punctuation at or before the cap.
    let mut split = None;
    for (byte_idx, c) in phonemes[..cap_byte].char_indices() {
        if c.is_whitespace() || matches!(c, ',' | '.' | '!' | '?' | ';' | ':' | '—' | '…') {
            split = Some(byte_idx + c.len_utf8());
        }
    }
    match split {
        Some(b) if !phonemes[..b].trim().is_empty() => (&phonemes[..b], &phonemes[b..]),
        _ => (&phonemes[..cap_byte], &phonemes[cap_byte..]),
    }
}

/// A decoded per-voice style bank: `rows` rows of [`STYLE_DIM`] f32.
#[derive(Debug)]
struct VoiceBank {
    data: Vec<f32>,
    rows: usize,
}

/// Loads one voice's style bank from an npz bundle (stored/uncompressed zip,
/// the `voices.bin` format) or a plain npy file.
fn load_voice_bank(path: &Path, voice: &str) -> Result<VoiceBank> {
    let bytes = std::fs::read(path)
        .map_err(|e| TtsError::MissingVoices(format!("cannot read {}: {e}", path.display())))?;
    let entry: Vec<u8> = if bytes.starts_with(b"PK\x03\x04") {
        let entries = walk_stored_zip_entries(&bytes);
        let wanted = format!("{voice}.npy");
        let (_, method, range) = entries
            .iter()
            .find(|(name, _, _)| *name == wanted)
            .ok_or_else(|| {
                let available: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
                TtsError::MissingVoices(format!(
                    "voice '{voice}' not in {}; available: {}",
                    path.display(),
                    available.join(", ")
                ))
            })?;
        if *method != 0 {
            return Err(TtsError::MissingVoices(format!(
                "{wanted} in {} is deflate-compressed; only stored (uncompressed) npz \
                 members are supported",
                path.display()
            ))
            .into());
        }
        bytes[range.clone()].to_vec()
    } else if bytes.starts_with(b"\x93NUMPY") {
        bytes
    } else {
        return Err(TtsError::MissingVoices(format!(
            "{} is neither an npz bundle (zip) nor an npy file",
            path.display()
        ))
        .into());
    };
    Ok(parse_npy_f32(&entry).map_err(|e| match e {
        TtsError::MissingVoices(m) => TtsError::MissingVoices(format!("{}: {m}", path.display())),
        other => other,
    })?)
}

/// Walks a ZIP archive's local file headers, returning `(name, compression
/// method, data byte range)` per member. Only local headers are parsed —
/// sufficient for npz bundles, whose members are stored sequentially ahead
/// of the central directory. ZIP64 size extra fields (used by `voices.bin`)
/// are honored; members with data descriptors are not expected in npz files.
fn walk_stored_zip_entries(zip: &[u8]) -> Vec<(String, u16, std::ops::Range<usize>)> {
    const LOCAL_HEADER_SIG: u32 = 0x0403_4B50; // "PK\x03\x04"
    const ZIP64_EXTRA_TAG: u16 = 0x0001;
    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos + 30 <= zip.len() {
        if u32_le(zip, pos) != Some(LOCAL_HEADER_SIG) {
            break; // central directory reached
        }
        let (Some(method), Some(csize32), Some(usize32), Some(name_len), Some(extra_len)) = (
            u16_le(zip, pos + 8),
            u32_le(zip, pos + 18),
            u32_le(zip, pos + 22),
            u16_le(zip, pos + 26),
            u16_le(zip, pos + 28),
        ) else {
            break;
        };
        let name_start = pos + 30;
        let Some(name_bytes) = zip.get(name_start..name_start + name_len as usize) else {
            break;
        };
        let extra_start = name_start + name_len as usize;
        let Some(extra) = zip.get(extra_start..extra_start + extra_len as usize) else {
            break;
        };
        let mut csize = u64::from(csize32);
        if csize32 == u32::MAX || usize32 == u32::MAX {
            // ZIP64 extra field: u64 uncompressed size, then u64 compressed
            // size (each present only when its 32-bit field is 0xFFFFFFFF).
            // Only the compressed size is needed to skip/copy the data (for
            // stored members the two are equal).
            let mut e = 0usize;
            while e + 4 <= extra.len() {
                let (Some(tag), Some(sz)) = (u16_le(extra, e), u16_le(extra, e + 2)) else {
                    break;
                };
                let body_start = e + 4;
                let Some(body) = extra.get(body_start..body_start + sz as usize) else {
                    break;
                };
                if tag == ZIP64_EXTRA_TAG {
                    let mut off = 0usize;
                    if usize32 == u32::MAX && body.len() >= off + 8 {
                        off += 8; // skip uncompressed size
                    }
                    if csize32 == u32::MAX && body.len() >= off + 8 {
                        csize = u64_le(body, off).unwrap_or(0);
                    }
                }
                e = body_start + sz as usize;
            }
        }
        let Some(data_start) = extra_start.checked_add(extra_len as usize) else {
            break;
        };
        let Some(data_end) = data_start.checked_add(csize as usize) else {
            break;
        };
        if data_end > zip.len() {
            break;
        }
        if let Ok(name) = std::str::from_utf8(name_bytes) {
            entries.push((name.to_string(), method, data_start..data_end));
        }
        pos = data_end;
    }
    entries
}

/// Parses a `.npy` v1/v2/v3 file holding little-endian f32 data, returning
/// the flattened samples plus the row count of a `[rows, 256]`,
/// `[rows, 1, 256]`, or bare `[256]` bank. Malformed or truncated input
/// (short header reads, absurd shapes, overflowing sizes) is always a clean
/// [`TtsError::MissingVoices`] — never a panic.
fn parse_npy_f32(buf: &[u8]) -> std::result::Result<VoiceBank, TtsError> {
    if !buf.starts_with(b"\x93NUMPY") {
        return Err(TtsError::MissingVoices("bad npy magic".to_string()));
    }
    let Some(&major) = buf.get(6) else {
        return Err(TtsError::MissingVoices("truncated npy magic".to_string()));
    };
    let (header_len, header_start) = match major {
        1 => {
            let Some(hb) = buf.get(8..10) else {
                return Err(TtsError::MissingVoices(
                    "truncated npy v1 header".to_string(),
                ));
            };
            (usize::from(u16::from_le_bytes([hb[0], hb[1]])), 10_usize)
        }
        2 | 3 => {
            let Some(hb) = buf.get(8..12) else {
                return Err(TtsError::MissingVoices(
                    "truncated npy v2/v3 header".to_string(),
                ));
            };
            (
                u32::from_le_bytes([hb[0], hb[1], hb[2], hb[3]]) as usize,
                12_usize,
            )
        }
        v => {
            return Err(TtsError::MissingVoices(format!(
                "unsupported npy version {v}"
            )));
        }
    };
    let header_end = header_start + header_len;
    let header = buf
        .get(header_start..header_end)
        .and_then(|h| std::str::from_utf8(h).ok())
        .ok_or_else(|| TtsError::MissingVoices("truncated npy header".to_string()))?;
    if !header.contains("'descr': '<f4'") {
        return Err(TtsError::MissingVoices(format!(
            "expected npy descr '<f4', got: {}",
            header.trim()
        )));
    }
    if header.contains("'fortran_order': True") {
        return Err(TtsError::MissingVoices(
            "fortran-order npy arrays are unsupported".to_string(),
        ));
    }
    let shape_pos = header
        .find("'shape':")
        .ok_or_else(|| TtsError::MissingVoices("npy header has no shape".to_string()))?;
    let dims = parse_npy_shape(&header[shape_pos..])?;
    let rows = match dims.as_slice() {
        [d] if *d == STYLE_DIM => 1,
        [r, d] if *d == STYLE_DIM => *r,
        [r, one, d] if *one == 1 && *d == STYLE_DIM => *r,
        _ => {
            return Err(TtsError::MissingVoices(format!(
                "unexpected voice bank shape {dims:?}"
            )));
        }
    };
    if rows == 0 {
        return Err(TtsError::MissingVoices(
            "voice bank has 0 style rows".to_string(),
        ));
    }
    // Checked arithmetic: a crafted shape (e.g. 2^62 rows) must not overflow
    // into a spuriously-passing truncation check (which would leave
    // `VoiceBank.rows` huge and panic on style-row indexing later).
    let total = rows.checked_mul(STYLE_DIM).ok_or_else(|| {
        TtsError::MissingVoices(format!("voice bank shape {dims:?} overflows usize"))
    })?;
    let need = total.checked_mul(4).ok_or_else(|| {
        TtsError::MissingVoices(format!("voice bank shape {dims:?} overflows usize"))
    })?;
    // In-bounds by construction: the `header` slice above proved
    // `header_end <= buf.len()`.
    let data = &buf[header_end..];
    if data.len() < need {
        return Err(TtsError::MissingVoices(format!(
            "npy data truncated: need {need} bytes, have {}",
            data.len()
        )));
    }
    let mut floats = Vec::with_capacity(total);
    for b in data[..need].chunks_exact(4) {
        floats.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(VoiceBank { data: floats, rows })
}

/// Parses the ints of an npy shape tuple, starting at `'shape': ...`.
fn parse_npy_shape(header_from_shape: &str) -> std::result::Result<Vec<usize>, TtsError> {
    let open = header_from_shape
        .find('(')
        .ok_or_else(|| TtsError::MissingVoices("malformed npy shape".to_string()))?;
    let close = header_from_shape[open..]
        .find(')')
        .map(|i| open + i)
        .ok_or_else(|| TtsError::MissingVoices("malformed npy shape".to_string()))?;
    let inner = &header_from_shape[open + 1..close];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let d: usize = part
            .parse()
            .map_err(|_| TtsError::MissingVoices(format!("bad npy shape dim '{part}'")))?;
        dims.push(d);
    }
    Ok(dims)
}

fn u16_le(buf: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(buf.get(at..at + 2)?.try_into().ok()?))
}

fn u32_le(buf: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(buf.get(at..at + 4)?.try_into().ok()?))
}

fn u64_le(buf: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(buf.get(at..at + 8)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    //! Fuzz-style robustness tests for the npy/npz parsers: malformed,
    //! truncated, and adversarial inputs must always produce a clean
    //! `TtsError::MissingVoices`, never a panic (the parser reads
    //! user-supplied model files).

    use super::*;
    use crate::error::SkadooshError;

    /// Builds a minimal npy v1 file: magic, version, header dict with the
    /// given shape text, then `data_bytes` zero bytes of payload.
    fn npy_v1(shape_text: &str, data_bytes: usize) -> Vec<u8> {
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
        let mut v = b"\x93NUMPY".to_vec();
        v.extend_from_slice(&[1, 0]);
        v.extend_from_slice(&(header.len() as u16).to_le_bytes());
        v.extend_from_slice(header.as_bytes());
        let total = v.len() + data_bytes;
        v.resize(total, 0);
        v
    }

    /// Builds a minimal stored (uncompressed) npz holding one `af.npy`
    /// member with the given payload.
    fn npz_stored(member_name: &str, payload: &[u8]) -> Vec<u8> {
        let mut v = b"PK\x03\x04".to_vec();
        v.extend_from_slice(&20u16.to_le_bytes()); // version needed
        v.extend_from_slice(&[0, 0]); // flags
        v.extend_from_slice(&[0, 0]); // method 0 = stored
        v.extend_from_slice(&[0; 8]); // time/date/crc (unchecked by parser)
        v.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // csize
        v.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // usize
        v.extend_from_slice(&(member_name.len() as u16).to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes()); // extra len
        v.extend_from_slice(member_name.as_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn missing_voices<T>(r: std::result::Result<T, TtsError>) -> bool {
        matches!(r, Err(TtsError::MissingVoices(_)))
    }

    #[test]
    fn npy_valid_minimal_bank_decodes() {
        let buf = npy_v1("(2, 1, 256)", 2 * 256 * 4);
        let bank = parse_npy_f32(&buf).expect("valid npy should decode");
        assert_eq!(bank.rows, 2);
        assert_eq!(bank.data.len(), 512);
        // Bare [256] single-row bank also decodes.
        let bare = npy_v1("(256,)", 256 * 4);
        assert_eq!(parse_npy_f32(&bare).expect("bare bank").rows, 1);
        // v2 header (u32 header length) decodes too.
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1, 256), }";
        let mut v2 = b"\x93NUMPY".to_vec();
        v2.extend_from_slice(&[2, 0]);
        v2.extend_from_slice(&(header.len() as u32).to_le_bytes());
        v2.extend_from_slice(header.as_bytes());
        let total = v2.len() + 2 * 256 * 4;
        v2.resize(total, 0);
        assert_eq!(parse_npy_f32(&v2).expect("v2 npy").rows, 2);
    }

    #[test]
    fn npy_every_truncated_prefix_errors_without_panic() {
        let full = npy_v1("(2, 1, 256)", 2 * 256 * 4);
        for cut in 0..full.len() {
            let r = parse_npy_f32(&full[..cut]);
            assert!(
                missing_voices(r),
                "prefix of len {cut} must be MissingVoices, not panic/Ok"
            );
        }
        // Sanity: the untruncated buffer is Ok.
        assert!(parse_npy_f32(&full).is_ok());
    }

    #[test]
    fn npy_tiny_garbage_buffers_error_without_panic() {
        for len in 0..=16 {
            // Zeros.
            assert!(
                missing_voices(parse_npy_f32(&vec![0u8; len])),
                "zeros len {len}"
            );
            // Valid magic, zeroed rest (includes a 10-11 byte buffer claiming
            // v2 — the historical index-out-of-bounds panic).
            let mut b = b"\x93NUMPY".to_vec();
            b.resize(len, 0);
            assert!(missing_voices(parse_npy_f32(&b)), "magic+zeros len {len}");
            // Valid magic + v2/v3 version byte, truncated length field.
            for major in [2u8, 3] {
                let mut b = b"\x93NUMPY".to_vec();
                b.extend_from_slice(&[major, 0]);
                b.resize(len.max(8), 0);
                assert!(
                    missing_voices(parse_npy_f32(&b)),
                    "v{major} magic+zeros len {len}"
                );
            }
        }
    }

    #[test]
    fn npy_mutated_version_and_header_fields_error() {
        for major in [0u8, 4, 9, 42, 255] {
            let mut b = npy_v1("(2, 1, 256)", 2 * 256 * 4);
            b[6] = major;
            assert!(missing_voices(parse_npy_f32(&b)), "version {major}");
        }
        // v1 buffer relabeled v2: the u32 header length swallows header text.
        let mut relabeled = npy_v1("(2, 1, 256)", 2 * 256 * 4);
        relabeled[6] = 2;
        assert!(missing_voices(parse_npy_f32(&relabeled)));
        // Wrong dtype.
        let mut wrong_descr = npy_v1("(2, 1, 256)", 2 * 256 * 4);
        let pos = wrong_descr.windows(4).position(|w| w == b"<f4'").unwrap();
        wrong_descr[pos..pos + 3].copy_from_slice(b"<f8");
        assert!(missing_voices(parse_npy_f32(&wrong_descr)));
        // Fortran order.
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 1, 256), }";
        let mut f = b"\x93NUMPY".to_vec();
        f.extend_from_slice(&[1, 0]);
        f.extend_from_slice(&(header.len() as u16).to_le_bytes());
        f.extend_from_slice(header.as_bytes());
        let total = f.len() + 2 * 256 * 4;
        f.resize(total, 0);
        assert!(missing_voices(parse_npy_f32(&f)));
    }

    #[test]
    fn npy_absurd_shapes_error_without_overflow() {
        // rows = 2^62: rows * 256 overflows usize.
        assert!(missing_voices(parse_npy_f32(&npy_v1(
            "(4611686018427387904, 1, 256)",
            0
        ))));
        // rows = 2^54: rows * 256 fits but * 4 bytes overflows.
        assert!(missing_voices(parse_npy_f32(&npy_v1(
            "(18014398509481984, 1, 256)",
            0
        ))));
        // Zero rows: no overflow, but a row-less bank would panic the
        // style-row indexing downstream.
        assert!(missing_voices(parse_npy_f32(&npy_v1("(0, 1, 256)", 0))));
        // Wrong inner dims / ranks.
        assert!(missing_voices(parse_npy_f32(&npy_v1("(2, 2, 256)", 0))));
        assert!(missing_voices(parse_npy_f32(&npy_v1("(512,)", 0))));
        assert!(missing_voices(parse_npy_f32(&npy_v1(
            "(1, 1, 1, 1, 256)",
            0
        ))));
        assert!(missing_voices(parse_npy_f32(&npy_v1("()", 0))));
        // Shape dim too large to parse as usize.
        assert!(missing_voices(parse_npy_f32(&npy_v1(
            "(99999999999999999999999999, 1, 256)",
            0
        ))));
        // Plausible shape, missing payload.
        assert!(missing_voices(parse_npy_f32(&npy_v1("(511, 1, 256)", 64))));
    }

    #[test]
    fn npz_stored_entry_roundtrips_and_truncations_error() {
        let payload = npy_v1("(2, 1, 256)", 2 * 256 * 4);
        let zip = npz_stored("af.npy", &payload);
        let dir = std::env::temp_dir().join(format!("skadoosh-npz-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("voices.bin");
        std::fs::write(&path, &zip).unwrap();
        let bank = load_voice_bank(&path, "af").expect("stored npz should load");
        assert_eq!(bank.rows, 2);
        assert_eq!(bank.data.len(), 512);
        // Every strict prefix: clean error, never a panic.
        for cut in 0..zip.len() {
            std::fs::write(&path, &zip[..cut]).unwrap();
            let r = load_voice_bank(&path, "af");
            assert!(
                matches!(r, Err(SkadooshError::Tts(TtsError::MissingVoices(_)))),
                "npz prefix len {cut} must be MissingVoices"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn npz_deflated_entry_is_rejected_cleanly() {
        let mut zip = npz_stored("af.npy", &[0xDE, 0xAD, 0xBE, 0xEF]);
        // Method byte at offset 8: 8 = deflate.
        zip[8] = 8;
        let dir = std::env::temp_dir().join(format!("skadoosh-npz-defl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("voices.bin");
        std::fs::write(&path, &zip).unwrap();
        let r = load_voice_bank(&path, "af");
        std::fs::remove_dir_all(&dir).ok();
        match r {
            Err(SkadooshError::Tts(TtsError::MissingVoices(m))) => {
                assert!(m.contains("deflate"), "message should say why: {m}");
            }
            other => panic!("expected MissingVoices for deflated entry, got {other:?}"),
        }
        // A member name that is absent lists available voices, no panic.
        let zip = npz_stored("zz.npy", &[0u8; 4]);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, &zip).unwrap();
        let r = load_voice_bank(&path, "af");
        std::fs::remove_dir_all(&dir).ok();
        match r {
            Err(SkadooshError::Tts(TtsError::MissingVoices(m))) => {
                assert!(m.contains("zz.npy"), "available voices listed: {m}");
            }
            other => panic!("expected MissingVoices for absent voice, got {other:?}"),
        }
    }
}
