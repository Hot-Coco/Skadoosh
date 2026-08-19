//! Text chat over the SDK: an [`Agent`] text-turn loop driven by stdin,
//! using the real [`LlmClient`] — no Whisper/VAD/Kokoro models, no audio
//! devices, no TTS.
//!
//! # Running
//!
//! Point the agent at any OpenAI-compatible server — local Ollama:
//!
//! ```sh
//! ollama pull qwen2.5:0.5b
//! cargo run --example text_chat
//! ```
//!
//! or a hosted provider (any `$SKADOOSH_LLM_URL`), or the in-process mock
//! SSE server used by the test-suite — the test
//! `tests/sdk_agent.rs::text_chat_example_runs_against_mock_server` drives
//! exactly this file's [`chat`] function against the mock.
//!
//! Environment: `SKADOOSH_LLM_URL` (default `http://localhost:11434/v1`),
//! `SKADOOSH_LLM_MODEL` (default `qwen2.5:0.5b`), `SKADOOSH_API_KEY`
//! (optional bearer token for hosted providers).

use skadoosh::{Agent, AgentEvent, Config, OutputMode, Result};

/// The example's core, split from `main` so tests can drive it against a
/// mock server: three turns on one agent (history accumulates), events
/// observed on a side channel. Returns the replies in order.
fn chat(llm_url: &str) -> Result<Vec<String>> {
    let mut config = Config::default();
    config.llm_url = llm_url.to_string();
    config.llm_model = std::env::var("SKADOOSH_LLM_MODEL").unwrap_or(config.llm_model);
    config.api_key = std::env::var("SKADOOSH_API_KEY").ok().or(config.api_key);
    config.output = OutputMode::Text; // no TTS, no audio device
    let mut agent = Agent::builder().config(config).build()?;

    let mut events = agent.events();
    let seen = std::thread::spawn(move || {
        let mut clauses = Vec::new();
        while let Ok(event) = events.blocking_recv() {
            match event {
                AgentEvent::Clause(c) => clauses.push(c),
                AgentEvent::Error(e) => eprintln!("agent error: {e}"),
                _ => {}
            }
        }
        clauses
    });

    let mut replies = Vec::new();
    for prompt in ["Hello!", "What can you do?", "Thanks, bye."] {
        let reply = agent.text_turn(prompt)?;
        println!("you> {prompt}");
        println!("bot> {}", reply.trim());
        replies.push(reply);
    }

    agent.shutdown();
    drop(agent); // closes the event channel, ending the observer thread
    let clauses = seen.join().expect("observer thread");
    assert_eq!(
        clauses.concat(),
        replies.concat(),
        "every reply text arrived as Clause events"
    );
    Ok(replies)
}

fn main() -> Result<()> {
    chat(&std::env::var("SKADOOSH_LLM_URL").unwrap_or_else(|_| Config::default().llm_url))?;
    Ok(())
}
