//! MockTts → wav tests (plan task 5.1 acceptance): duration math, peak
//! levels, no NaNs, and the engine factory's mock fallback. The real Kokoro
//! path is covered by the `SKADOOSH_KOKORO_TESTS=1` gated test at the bottom.

use std::path::Path;

use clap::Parser;
use skadoosh::config::Config;
use skadoosh::tts::{self, MockTts, OnnxTts, TtsClip, TtsEngine};

fn synth(text: &str) -> TtsClip {
    MockTts::new().synthesize(text).expect("mock synthesize")
}

#[test]
fn duration_peak_and_no_nans() {
    let text = "Hello world."; // 12 chars → 660 ms
    let clip = synth(text);
    assert_eq!(clip.sample_rate, 24_000);

    let expected_ms = text.chars().count() as f32 * 55.0;
    let actual_ms = clip.samples.len() as f32 / 24.0;
    let tol = 0.2 * expected_ms;
    assert!(
        (actual_ms - expected_ms).abs() <= tol,
        "duration {actual_ms} ms not within ±20% of {expected_ms} ms"
    );

    let peak = clip.samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
    assert!(peak <= 0.3 + 1e-6, "peak {peak} exceeds 0.3");
    assert!(peak > 0.29, "peak {peak} implausibly low for a steady sine");
    assert!(
        clip.samples.iter().all(|s| s.is_finite()),
        "clip contains NaN/inf"
    );

    // Raised-cosine edges: the clip starts and ends near zero (no click).
    assert!(clip.samples[0].abs() < 1e-3, "fade-in: {}", clip.samples[0]);
    let last = *clip.samples.last().unwrap();
    assert!(last.abs() < 1e-2, "fade-out: {last}");
}

#[test]
fn duration_clamps_at_bounds() {
    // Empty/short text clamps to 250 ms = 6000 samples.
    let short = synth("");
    assert_eq!(short.samples.len(), 6_000);
    let tiny = synth("hi");
    assert_eq!(tiny.samples.len(), 6_000);

    // 1000 chars × 55 ms = 55 s clamps to 2.5 s = 60 000 samples.
    let long = synth(&"a".repeat(1000));
    assert_eq!(long.samples.len(), 60_000);
}

#[test]
fn clause_to_wav_file_roundtrip() {
    let text = "Round trip clause."; // 18 chars → 990 ms
    let clip = synth(text);
    let path = std::env::temp_dir().join(format!("skadoosh-mock-tts-{}.wav", std::process::id()));

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: clip.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    {
        let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
        for &s in &clip.samples {
            writer.write_sample(s).expect("write sample");
        }
        writer.finalize().expect("finalize wav");
    }

    let mut reader = hound::WavReader::open(&path).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 24_000);
    assert_eq!(spec.channels, 1);
    let back: Vec<f32> = reader
        .samples::<f32>()
        .collect::<Result<_, _>>()
        .expect("read samples");
    assert_eq!(back, clip.samples, "wav roundtrip must be lossless (f32)");
    let expected_ms = text.chars().count() as f32 * 55.0;
    let wav_ms = back.len() as f32 / 24.0;
    assert!(
        (wav_ms - expected_ms).abs() <= 0.2 * expected_ms,
        "wav duration {wav_ms} ms vs expected {expected_ms} ms"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn factory_returns_mock_when_forced() {
    let cfg = Config::try_parse_from(["skadoosh", "--mock-tts"]).expect("parse config");
    let mut engine = tts::build_engine(&cfg).expect("build engine");
    // Behavioral proof it's the mock: 4 chars → clamped 250 ms sine.
    let clip = engine.synthesize("test").expect("synthesize");
    assert_eq!(clip.sample_rate, 24_000);
    assert_eq!(clip.samples.len(), 6_000);
}

#[test]
fn factory_falls_back_to_mock_when_kokoro_files_missing() {
    let cfg = Config::try_parse_from([
        "skadoosh",
        "--tts-model",
        "/nonexistent/kokoro-v0_19.onnx",
        "--tts-voices",
        "/nonexistent/voices.bin",
    ])
    .expect("parse config");
    let mut engine = tts::build_engine(&cfg).expect("build engine");
    let clip = engine.synthesize("test").expect("synthesize");
    assert_eq!(clip.samples.len(), 6_000, "expected MockTts fallback");
}

/// Real Kokoro-82M inference (plan task 5.3 acceptance). Gated: needs the
/// ~325 MB model + voices bank from `scripts/download_models.sh
/// --with-kokoro` and the espeak-ng binary.
#[test]
fn kokoro_real_model() {
    if std::env::var("SKADOOSH_KOKORO_TESTS").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping kokoro_real_model: set SKADOOSH_KOKORO_TESTS=1 and fetch the model \
             via scripts/download_models.sh --with-kokoro"
        );
        return;
    }
    let model = Path::new("models/kokoro-v0_19.onnx");
    let voices = Path::new("models/voices.bin");
    if !model.exists() || !voices.exists() {
        eprintln!(
            "skipping kokoro_real_model: {} or {} missing",
            model.display(),
            voices.display()
        );
        return;
    }

    let mut tts = OnnxTts::load(model, voices, "af", 1.0).expect("load kokoro");
    let clip = tts.synthesize("Hello world.").expect("synthesize");
    assert_eq!(clip.sample_rate, 24_000);
    assert!(
        clip.samples.len() > 2_400 && clip.samples.len() < 24_000 * 30,
        "implausible duration: {} samples",
        clip.samples.len()
    );
    let peak = clip.samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
    assert!(peak > 1e-3, "output is silent (peak {peak})");
    assert!(clip.samples.iter().all(|s| s.is_finite()));

    // Voice selection changes the output.
    let mut other = OnnxTts::load(model, voices, "am_adam", 1.0).expect("load kokoro");
    let clip2 = other.synthesize("Hello world.").expect("synthesize");
    assert_ne!(clip.samples, clip2.samples, "voice must change the output");

    // Speed scales duration monotonically.
    let mut fast = OnnxTts::load(model, voices, "af", 2.0).expect("load kokoro");
    let clip_fast = fast.synthesize("Hello world.").expect("synthesize");
    assert!(
        clip_fast.samples.len() < clip.samples.len(),
        "speed 2.0 ({} samples) should be shorter than 1.0 ({} samples)",
        clip_fast.samples.len(),
        clip.samples.len()
    );
}
