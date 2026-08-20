//! Cookbook 10 — Config modes.
//!
//! Builds several [`Config`]s for the different run modes the CLI/SDK
//! supports and prints each one (the hand-rolled `Debug` redacts `api_key`):
//!
//! * **say** — one-shot text→speech to a wav (`--say --out-wav`): no STT/VAD
//!   models, no LLM.
//! * **repl** — interactive text loop (`--repl`): no models, no TTS, no audio.
//! * **text-mode voice** — voice-in → text-out (`--output text`): needs the
//!   Whisper + VAD models (STT still runs), but no TTS.
//! * **default audio** — the full voice loop.
//!
//! It then calls [`Config::validate`] to show which models each mode requires
//! — the headless modes (`--repl`, `--say`) pass with the (nonexistent)
//! default model paths, while the voice-in modes report the missing models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 10_config_modes
//! ```

use clap::Parser;
use skadoosh::config::{Config, OutputMode};

/// Builds a config from CLI-style args (clap `try_parse_from`), exactly as
/// the binary would parse them.
fn from_args(args: &[&str]) -> Config {
    Config::try_parse_from(args).expect("clap should parse these flags")
}

/// Prints a labeled config and the outcome of `validate`.
fn show(label: &str, config: &Config) {
    println!("=== {label} ===");
    println!("{config:#?}");
    match config.validate() {
        Ok(()) => println!("validate: OK\n"),
        Err(e) => println!("validate: {e}\n"),
    }
}

fn main() -> skadoosh::Result<()> {
    // 1. --say --out-wav: one-shot TTS to a file. Needs no STT/VAD models and
    //    no LLM; with no Kokoro files configured it falls back to MockTts.
    let say = from_args(&[
        "skadoosh",
        "--say",
        "Hello from a cookbook.",
        "--out-wav",
        "target/cookbook_10_say.wav",
        "--mock-tts",
    ]);
    show("say mode (--say --out-wav)", &say);

    // 2. --repl: text-only loop. No models, no TTS, no audio device.
    let repl = from_args(&["skadoosh", "--repl", "--output", "text"]);
    show("repl mode (--repl)", &repl);

    // 3. --output text: voice-in → text-out. STT/VAD models are still
    //    required (audio goes in), but no TTS is built.
    let mut text_voice = from_args(&["skadoosh", "--output", "text"]);
    show("text-mode voice (--output text)", &text_voice);

    // 4. Default: the full audio voice loop. Needs STT, VAD, and TTS.
    let default = Config::default();
    show("default audio voice loop", &default);

    // Assertions on the parsed modes.
    assert!(say.say.is_some(), "say mode carries --say text");
    assert!(say.out_wav.is_some(), "say mode carries --out-wav");
    assert!(say.mock_tts, "say mode forces mock TTS");
    assert!(repl.repl, "repl mode sets repl=true");
    assert_eq!(repl.output, OutputMode::Text, "repl uses text output");
    assert_eq!(text_voice.output, OutputMode::Text, "text-mode voice");
    assert_eq!(default.output, OutputMode::Audio, "default is audio");

    // validate() outcomes: headless modes pass; voice-in modes require the
    // (absent in the sandbox) model files.
    assert!(say.validate().is_ok(), "say needs no STT/VAD models");
    assert!(repl.validate().is_ok(), "repl needs no models");
    assert!(
        text_voice.validate().is_err(),
        "text-mode voice still needs the STT/VAD models"
    );
    // Quiet the "unused variable" lint while keeping the name readable.
    let _ = &mut text_voice;

    println!("10_config_modes: OK");
    Ok(())
}
