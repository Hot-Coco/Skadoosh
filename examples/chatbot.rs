//! A minimal text-only chatbot built on the [`Agent`] SDK — no audio, no
//! models, no TTS. Point it at any OpenAI-compatible LLM server (local Ollama
//! by default) and chat on stdin/stdout.
//!
//! # Setup
//!
//! 1. Install and start Ollama (an OpenAI-compatible local server):
//!
//!    ```sh
//!    curl -fsSL https://ollama.com/install.sh | sh
//!    ollama serve            # listens on http://localhost:11434
//!    ollama pull llama3.2    # download the default model
//!    ```
//!
//! 2. Build and run the chatbot:
//!
//!    ```sh
//!    cargo run --example chatbot
//!    ```
//!
//! 3. Chat away. Type `/quit` (or press ctrl-d) to exit, and `/clear` to wipe
//!    the conversation history and start fresh.
//!
//! # Configuration (environment variables)
//!
//! | Variable            | Default                     | Meaning                          |
//! |---------------------|-----------------------------|----------------------------------|
//! | `SKADOOSH_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible LLM base URL   |
//! | `SKADOOSH_MODEL`    | `llama3.2`                  | Model name for chat completions  |
//! | `SKADOOSH_API_KEY`  | `ollama`                    | Bearer token (Ollama needs none) |
//!
//! Any hosted OpenAI-compatible provider works too — set the base URL, model,
//! and API key to point at it.
//!
//! # How it works
//!
//! The crate ships no `HttpLlm` type, so (like the `text_chat` example) this
//! uses the config-based [`LlmClient`]: [`Agent::builder`] assembles a
//! text-output agent ([`OutputMode::Text`] → no TTS, no playback), and the LLM
//! client is built lazily from the [`Config`] on the first
//! [`Agent::text_turn`]. Each input line is one `text_turn`; the full reply
//! text is returned and printed as `Bot> {reply}`. Conversation history
//! accumulates across turns (bounded by `Config::max_history_turns`); `/clear`
//! rebuilds the agent from the same config, resetting history to just the
//! system prompt.
//!
//! The REPL core ([`run_repl`]) is split out of [`main`] so
//! `tests/examples.rs` can drive it in-memory against a scripted LLM backend —
//! no server, no stdin/stdout — using exactly the same `Agent::text_turn`
//! path.

use std::io::{BufRead, Write};

use skadoosh::{Agent, Config, OutputMode, Result, SkadooshError};

/// OpenAI-compatible base URL (env `SKADOOSH_BASE_URL`).
const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";
/// Model name (env `SKADOOSH_MODEL`).
const DEFAULT_MODEL: &str = "llama3.2";
/// Bearer token / API key (env `SKADOOSH_API_KEY`).
const DEFAULT_API_KEY: &str = "ollama";

/// Builds a text-output [`Config`] from the `SKADOOSH_*` environment
/// variables, falling back to the Ollama defaults above.
fn config_from_env() -> Config {
    Config {
        llm_url: std::env::var("SKADOOSH_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
        llm_model: std::env::var("SKADOOSH_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
        api_key: Some(
            std::env::var("SKADOOSH_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.to_string()),
        ),
        output: OutputMode::Text, // text-only: no TTS, no audio device
        ..Config::default()
    }
}

/// Builds a fresh [`Agent`] from the env-derived config. A brand-new agent
/// starts with just the seeded system prompt, so calling this again on
/// `/clear` resets the conversation history.
fn build_agent() -> Result<Agent> {
    Agent::builder().config(config_from_env()).build()
}

/// Wraps a std I/O error as a [`SkadooshError`] (via `anyhow`), matching the
/// crate's own repl I/O-error handling — `SkadooshError` has no direct
/// `From<std::io::Error>`, so the `?` operator needs this bridge.
fn io_err(err: std::io::Error) -> SkadooshError {
    anyhow::anyhow!("chatbot I/O failed: {err}").into()
}

/// The chatbot REPL core, split from [`main`] so tests can drive it in-memory
/// (against a scripted LLM, no server) without stdin/stdout.
///
/// Prints `You> ` before reading each line from `input` and `Bot> {reply}`
/// after each [`Agent::text_turn`]. `make_agent` builds the initial agent and
/// is called again on `/clear` to reset history. `/quit` or EOF exits; LLM
/// errors are printed and the loop continues (I/O errors propagate, since a
/// broken stdin/stdout is terminal).
pub fn run_repl<R, W, F>(mut input: R, mut output: W, mut make_agent: F) -> Result<()>
where
    R: BufRead,
    W: Write,
    F: FnMut() -> Result<Agent>,
{
    let mut agent = make_agent()?;
    writeln!(
        output,
        "skadoosh chatbot — type a message; /quit to exit, /clear to reset history"
    )
    .map_err(io_err)?;

    loop {
        write!(output, "You> ").map_err(io_err)?;
        output.flush().map_err(io_err)?;
        let mut line = String::new();
        if input.read_line(&mut line).map_err(io_err)? == 0 {
            break; // EOF (ctrl-d)
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/quit" {
            break;
        }
        if text == "/clear" {
            match make_agent() {
                Ok(fresh) => {
                    agent = fresh;
                    writeln!(output, "(history cleared)").map_err(io_err)?;
                }
                Err(err) => writeln!(output, "(failed to reset history: {err})").map_err(io_err)?,
            }
            continue;
        }
        match agent.text_turn(text) {
            Ok(reply) => writeln!(output, "Bot> {}", reply.trim()).map_err(io_err)?,
            Err(err) => writeln!(output, "Bot> (error: {err})").map_err(io_err)?,
        }
    }

    writeln!(output, "bye").map_err(io_err)?;
    Ok(())
}

fn main() -> Result<()> {
    let stdin = std::io::stdin();
    run_repl(stdin.lock(), std::io::stdout(), build_agent)
}
