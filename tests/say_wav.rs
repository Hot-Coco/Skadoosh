//! `--say --out-wav` / `Agent::say_to_wav` tests: one-shot text→speech to a
//! 24 kHz wav with the MockTts engine — fully headless, no audio device, no
//! models, no LLM server.

use std::path::PathBuf;

use skadoosh::agent::Agent;
use skadoosh::config::{Config, OutputMode};

/// MockTts sample count for a clause: `clamp(chars * 55 ms, 250..2500 ms)`
/// at 24 kHz (mirrors `src/tts/mock.rs`).
fn mock_clip_samples(clause: &str) -> usize {
    let ms = (clause.chars().count() as f32 * 55.0).clamp(250.0, 2_500.0);
    (24_000.0 * ms / 1000.0).round() as usize
}

fn say_config() -> Config {
    // --say needs no whisper/vad models and no LLM server; the config's
    // default model paths intentionally do not exist to prove they are not
    // touched.
    Config {
        mock_tts: true,
        say: Some("unused here — the SDK call carries the text".to_string()),
        ..Config::default()
    }
}

fn out_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/{name}"))
}

#[test]
fn say_to_wav_produces_non_silent_24khz_wav() {
    let out_wav = out_path("say_test_out.wav");
    let mut agent = Agent::builder()
        .config(say_config())
        .build()
        .expect("build");
    agent
        .say_to_wav("Hello world.", &out_wav)
        .expect("say_to_wav");

    let reader = hound::WavReader::open(&out_wav).expect("open wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 24_000, "24 kHz output");
    assert_eq!(spec.channels, 1, "mono");
    assert_eq!(spec.bits_per_sample, 16, "16-bit PCM");

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("sample read"))
        .collect();
    let expected = mock_clip_samples("Hello world.");
    let tol = expected / 20;
    assert!(
        samples.len().abs_diff(expected) <= tol,
        "wav has {} samples, expected ≈ {expected}",
        samples.len()
    );
    let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(peak > 3_000, "non-silent output (peak {peak})");

    let _ = std::fs::remove_file(&out_wav);
}

/// Multi-clause text is clause-split and the clips concatenated.
#[test]
fn say_to_wav_concatenates_clauses() {
    let out_wav = out_path("say_clauses_out.wav");
    let mut agent = Agent::builder()
        .config(say_config())
        .build()
        .expect("build");
    agent
        .say_to_wav("One. Two. Three.", &out_wav)
        .expect("say_to_wav");

    let reader = hound::WavReader::open(&out_wav).expect("open wav");
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("sample read"))
        .collect();
    let expected =
        mock_clip_samples("One.") + mock_clip_samples(" Two.") + mock_clip_samples(" Three.");
    let tol = expected / 20;
    assert!(
        samples.len().abs_diff(expected) <= tol,
        "wav has {} samples, expected ≈ {expected} (sum of clause durations)",
        samples.len()
    );

    let _ = std::fs::remove_file(&out_wav);
}

/// An injected TTS engine is honored on the `say_to_wav` path (here: the
/// MockTts an SDK user would substitute with their own engine).
#[test]
fn say_to_wav_uses_injected_tts_engine() {
    use skadoosh::tts::MockTts;
    let out_wav = out_path("say_injected_out.wav");
    let mut config = say_config();
    config.mock_tts = false; // the injected engine must win over the fallback
    let mut agent = Agent::builder()
        .config(config)
        .tts(Box::new(MockTts::new()))
        .build()
        .expect("build");
    agent
        .say_to_wav("Injected engine.", &out_wav)
        .expect("say_to_wav");

    let reader = hound::WavReader::open(&out_wav).expect("open wav");
    assert_eq!(reader.spec().sample_rate, 24_000);
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("sample read"))
        .collect();
    let expected = mock_clip_samples("Injected engine.");
    let tol = expected / 20;
    assert!(
        samples.len().abs_diff(expected) <= tol,
        "wav has {} samples, expected ≈ {expected}",
        samples.len()
    );

    let _ = std::fs::remove_file(&out_wav);
}

/// Empty/whitespace text has no speakable clauses: a clean error, no file.
#[test]
fn say_to_wav_rejects_unspeakable_text() {
    let out_wav = out_path("say_empty_out.wav");
    let mut agent = Agent::builder()
        .config(say_config())
        .build()
        .expect("build");
    let err = agent
        .say_to_wav("   ", &out_wav)
        .expect_err("whitespace-only text must fail");
    assert!(
        err.to_string().contains("no speakable clauses"),
        "got {err:?}"
    );
    assert!(!out_wav.exists(), "no file written on failure");
}

/// `--output text` agents can still write wavs (an explicit `say_to_wav`
/// call is speech by request, not the voice loop's modality).
#[test]
fn say_to_wav_works_in_text_output_mode() {
    let out_wav = out_path("say_textmode_out.wav");
    let mut config = say_config();
    config.output = OutputMode::Text;
    let mut agent = Agent::builder().config(config).build().expect("build");
    agent
        .say_to_wav("Still speaks.", &out_wav)
        .expect("say_to_wav");
    assert!(out_wav.exists());
    let _ = std::fs::remove_file(&out_wav);
}
