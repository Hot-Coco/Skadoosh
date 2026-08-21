//! Minimal 16-bit PCM WAV writer, shared by `--selftest` / `--say --out-wav`
//! (in the `pipeline` module, behind the `audio` feature) and
//! [`Agent::say_to_wav`](crate::agent::Agent::say_to_wav) (always available,
//! even in a no-`audio` build). Kept in its own module so the WAV writer does
//! not inherit the `audio` feature gate of the pipeline orchestrator.
//!
//! `hound` is a dev-dependency only, so library code writes the RIFF container
//! itself.

use std::path::Path;

use crate::error::Result;

/// Writes a canonical 44-byte-header 16-bit PCM mono wav.
pub(crate) fn write_wav16(path: &Path, samples: &[f32], rate: u32) -> Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, &out)
        .map_err(|err| anyhow::anyhow!("failed to write {}: {err}", path.display()).into())
}
