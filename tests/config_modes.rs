//! CLI modality validation: the new `--repl`/`--say`/`--output`/`--out-wav`
//! flag combinations and their effect on which model files are required.

use std::path::PathBuf;

use skadoosh::config::{Config, OutputMode};

const WHISPER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-tiny.en.bin");
const VAD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jfk.wav");

/// A config whose whisper/vad model paths exist (when present on this
/// machine); tests that must NOT need them use `missing_models_config`.
fn valid_config() -> Config {
    Config {
        whisper_model: PathBuf::from(WHISPER),
        vad_model: PathBuf::from(VAD),
        ..Config::default()
    }
}

/// Model paths that definitely do not exist.
fn missing_models_config() -> Config {
    Config {
        whisper_model: PathBuf::from("models/definitely-missing-whisper.bin"),
        vad_model: PathBuf::from("models/definitely-missing-vad.onnx"),
        ..Config::default()
    }
}

#[test]
fn default_config_is_valid_when_models_exist() {
    if !PathBuf::from(WHISPER).exists() || !PathBuf::from(VAD).exists() {
        eprintln!("skipping: models absent (run scripts/download_models.sh)");
        return;
    }
    valid_config().validate().expect("default config validates");
}

#[test]
fn repl_skips_model_checks() {
    let mut config = missing_models_config();
    config.repl = true;
    config.validate().expect("--repl needs no models");
}

#[test]
fn say_skips_stt_vad_model_checks() {
    let mut config = missing_models_config();
    config.say = Some("hello".to_string());
    config.out_wav = Some(PathBuf::from("target/say_validate_out.wav"));
    config
        .validate()
        .expect("--say needs no whisper/vad models");
}

#[test]
fn repl_and_say_conflict() {
    let mut config = missing_models_config();
    config.repl = true;
    config.say = Some("hello".to_string());
    let err = config.validate().expect_err("must conflict");
    assert!(err.to_string().contains("--repl and --say"), "got {err:?}");
}

#[test]
fn repl_and_selftest_conflict() {
    let mut config = missing_models_config();
    config.repl = true;
    config.selftest = Some(PathBuf::from(JFK));
    let err = config.validate().expect_err("must conflict");
    assert!(
        err.to_string().contains("--repl and --selftest"),
        "got {err:?}"
    );
}

#[test]
fn say_and_selftest_conflict() {
    let mut config = missing_models_config();
    config.say = Some("hello".to_string());
    config.selftest = Some(PathBuf::from(JFK));
    let err = config.validate().expect_err("must conflict");
    assert!(
        err.to_string().contains("--say and --selftest"),
        "got {err:?}"
    );
}

#[test]
fn say_conflicts_with_output_text() {
    let mut config = missing_models_config();
    config.say = Some("hello".to_string());
    config.output = OutputMode::Text;
    let err = config.validate().expect_err("must conflict");
    assert!(err.to_string().contains("--output text"), "got {err:?}");
}

#[test]
fn out_wav_requires_say() {
    let mut config = valid_config();
    config.out_wav = Some(PathBuf::from("target/nowhere.wav"));
    let err = config
        .validate()
        .expect_err("--out-wav without --say fails");
    assert!(err.to_string().contains("--out-wav"), "got {err:?}");
}

#[test]
fn output_text_still_needs_stt_vad_models() {
    let mut config = missing_models_config();
    config.output = OutputMode::Text;
    let err = config.validate().expect_err("voice-in needs the models");
    assert!(err.to_string().contains("--whisper-model"), "got {err:?}");
}

/// The clap surface: flags parse, env names and defaults hold.
#[test]
fn clap_parses_new_flags() {
    use clap::Parser;
    let config = Config::try_parse_from([
        "skadoosh",
        "--repl",
        "--output",
        "text",
        "--api-key",
        "sk-test",
    ])
    .expect("parses");
    assert!(config.repl);
    assert_eq!(config.output, OutputMode::Text);
    assert_eq!(config.api_key.as_deref(), Some("sk-test"));

    let config =
        Config::try_parse_from(["skadoosh", "--say", "hello world", "--out-wav", "out.wav"])
            .expect("parses");
    assert_eq!(config.say.as_deref(), Some("hello world"));
    assert_eq!(config.out_wav, Some(PathBuf::from("out.wav")));
    assert_eq!(config.output, OutputMode::Audio, "default output is audio");
}

/// The hand-rolled `Debug` must never leak the API key (the flag documents
/// "never logged" — a derived Debug would be one `info!(?config)` away
/// from breaking that).
#[test]
fn debug_redacts_api_key() {
    let config = Config {
        api_key: Some("sk-super-secret-value".to_string()),
        ..Config::default()
    };
    let debug = format!("{config:?}");
    assert!(
        !debug.contains("sk-super-secret-value"),
        "api key leaked into Debug output: {debug}"
    );
    assert!(
        debug.contains("<redacted>"),
        "redaction marker present: {debug}"
    );
    // Everything else still prints.
    assert!(debug.contains("llm_url"), "{debug}");
    assert!(debug.contains(&config.llm_url), "{debug}");
}

/// `Config::default()` matches the clap flag defaults (SDK ergonomics: a
/// default config equals a bare `skadoosh` invocation).
#[test]
fn default_matches_clap_defaults() {
    use clap::Parser;
    let from_clap = Config::try_parse_from(["skadoosh"]).expect("parses");
    let default = Config::default();
    assert_eq!(from_clap.llm_url, default.llm_url);
    assert_eq!(from_clap.llm_model, default.llm_model);
    assert_eq!(from_clap.system_prompt, default.system_prompt);
    assert_eq!(from_clap.max_history_turns, default.max_history_turns);
    assert_eq!(from_clap.whisper_model, default.whisper_model);
    assert_eq!(from_clap.vad_model, default.vad_model);
    assert_eq!(from_clap.vad_threshold, default.vad_threshold);
    assert_eq!(from_clap.silence_ms, default.silence_ms);
    assert_eq!(from_clap.output, default.output);
    assert!(!default.repl);
    assert!(default.say.is_none());
    assert!(default.api_key.is_none());
    assert!(default.out_wav.is_none());
}
