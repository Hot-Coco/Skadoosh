//! Cookbook 14 — Wake word gating.
//!
//! Demonstrates the wake-word gating the pipeline applies when
//! [`Config::wake_word`] is set: a transcript is only processed if it
//! contains the wake word (case-insensitive substring match, mirroring the
//! pipeline), and the wake word is then stripped to leave the command.
//!
//! No server, no models — just the gating logic over a [`Config`].
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 14_wake_word
//! ```

use skadoosh::Config;

/// Wake-word gate, matching the pipeline's rule: a case-insensitive
/// substring match on the transcript. Returns the stripped command when the
/// wake word is present, or `None` when the transcript should be skipped.
///
/// This mirrors `src/pipeline.rs`'s
/// `text.to_lowercase().contains(&ww.to_lowercase())` check, plus the
/// natural "remove the wake word to get the command" step an app performs.
fn gate(transcript: &str, wake_word: &str) -> Option<String> {
    let pos = transcript.to_lowercase().find(&wake_word.to_lowercase())?;
    let command = &transcript[pos + wake_word.len()..];
    Some(command.trim().to_string())
}

fn main() -> skadoosh::Result<()> {
    let config = Config {
        wake_word: Some("hey skadoosh".to_string()),
        ..Config::default()
    };

    println!("config.wake_word = {:?}", config.wake_word);
    println!("(the agent only processes speech after the wake word)\n");

    let cases = [
        ("hey skadoosh what time is it", true, "what time is it"),
        (
            "HEY SKADOOSH turn on the lights",
            true,
            "turn on the lights",
        ),
        ("what time is it", false, ""), // missing wake word -> skipped
        ("hey skadoosh", true, ""),     // wake word only -> empty command
        ("please hey skadoosh stop", true, "stop"), // mid-sentence: command is text after the wake word
    ];

    for (transcript, should_match, expected) in cases {
        match gate(transcript, config.wake_word.as_deref().unwrap()) {
            Some(command) => {
                println!("MATCH   {transcript:?} -> command {command:?}");
                assert!(should_match, "should have been skipped: {transcript:?}");
                assert_eq!(
                    command,
                    expected.trim(),
                    "stripped command for {transcript:?}"
                );
            }
            None => {
                println!("SKIP    {transcript:?} (missing wake word)");
                assert!(!should_match, "should have matched: {transcript:?}");
            }
        }
    }

    // With no wake word configured, every transcript passes through ungated.
    let no_ww = Config::default();
    assert!(no_ww.wake_word.is_none(), "wake word is opt-in");
    println!("\nwith wake_word unset, gating is disabled (everything passes)");

    println!("\n14_wake_word: OK");
    Ok(())
}
