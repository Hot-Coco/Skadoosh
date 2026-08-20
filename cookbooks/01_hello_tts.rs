//! Cookbook 01 — Hello TTS.
//!
//! Synthesizes the text "Hello from Skadoosh!" with the zero-model
//! [`MockTts`] engine and writes the result to a 24 kHz WAV file using
//! `hound` (a dev-dependency, available to examples). No Kokoro model, no
//! audio device, no LLM — just the TTS stage.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 01_hello_tts
//! ```

use std::path::PathBuf;

use skadoosh::tts::{MockTts, TtsEngine};

/// Wraps any `Display` error (here `hound`/io errors) into the crate's
/// umbrella error so `?` propagation works inside `fn main -> skadoosh::Result`.
fn wrap<E: std::fmt::Display>(e: E) -> skadoosh::SkadooshError {
    anyhow::anyhow!("{e}").into()
}

fn main() -> skadoosh::Result<()> {
    let text = "Hello from Skadoosh!";

    // MockTts: a 220 Hz sine whose duration scales with the input length
    // (clamped to 250 ms..2.5 s). No model files are loaded.
    let mut tts = MockTts::new();
    let clip = tts.synthesize(text)?;

    println!("synthesized {text:?}");
    println!("  sample rate : {} Hz", clip.sample_rate);
    println!("  samples     : {}", clip.samples.len());
    println!(
        "  duration    : {:.0} ms",
        clip.samples.len() as f32 / clip.sample_rate as f32 * 1000.0
    );
    let peak = clip.samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
    println!("  peak amplitude : {peak:.3} (mock ceiling is 0.3)");

    // Write a 32-bit float, mono WAV — a lossless round trip for the f32
    // samples the engine produces (the crate's own writer uses 16-bit PCM;
    // here we keep full float precision for clarity).
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cookbook_01_hello.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: clip.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    {
        let mut writer = hound::WavWriter::create(&path, spec).map_err(wrap)?;
        for &s in &clip.samples {
            writer.write_sample(s).map_err(wrap)?;
        }
        writer.finalize().map_err(wrap)?;
    }

    println!("wrote {}", path.display());

    // Sanity checks (the cookbook "actually runs and produces output").
    assert_eq!(clip.sample_rate, 24_000, "MockTts emits 24 kHz");
    assert!(!clip.samples.is_empty(), "clip must contain samples");
    assert!(clip.samples.iter().all(|s| s.is_finite()), "no NaN/inf");
    assert!(path.exists(), "wav file was written");

    println!("01_hello_tts: OK");
    Ok(())
}
