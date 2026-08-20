//! Cookbook 02 — Single text turn with a mock LLM backend.
//!
//! Builds an [`Agent`] in text-output mode with a *scripted* [`LlmBackend`]
//! (no real server, no models, no audio device) and runs one
//! [`Agent::text_turn`]. The agent streams the reply as clauses and returns
//! the full reply text.
//!
//! This is the recommended headless pattern (see the crate docs): inject a
//! custom `LlmBackend` and drive `text_turn` / `repl` without any LLM server.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 02_text_turn
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::{Agent, Config, OutputMode, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A scripted LLM backend: pops one canned reply (a vec of clauses) per
/// user turn, in order. This is the trait an SDK user implements to plug in
/// their own model or serving stack — here it stands in for a real server.
struct ScriptedLlm {
    script: Mutex<VecDeque<Vec<String>>>,
}

impl ScriptedLlm {
    fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
        }
    }

    /// Queues one turn's reply (a list of clauses streamed in order).
    fn turn(self, clauses: &[&str]) -> Self {
        self.script
            .lock()
            .expect("script lock")
            .push_back(clauses.iter().map(|s| s.to_string()).collect());
        self
    }
}

impl LlmBackend for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        _user: &'a str,
        clauses: mpsc::Sender<String>,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        // Drain the script first so the MutexGuard does not cross an await.
        let reply: Vec<String> = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .unwrap_or_else(|| vec!["(no scripted reply)".to_string()]);
        Box::pin(async move {
            for clause in reply {
                let _ = clauses.send(clause).await;
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {
        self.script.lock().expect("script lock").clear();
    }
}

fn main() -> Result<()> {
    let config = Config {
        output: OutputMode::Text,
        ..Config::default()
    };

    let mut agent = Agent::builder()
        .config(config)
        .llm(Box::new(
            ScriptedLlm::new().turn(&["Hello! ", "I am a scripted assistant."]),
        ))
        .build()?;

    let reply = agent.text_turn("Hi there!")?;
    println!("you> Hi there!");
    println!("bot> {}", reply.trim());

    assert!(
        reply.contains("Hello!") && reply.contains("scripted assistant"),
        "reply should reassemble both clauses: {reply:?}"
    );

    println!("02_text_turn: OK");
    Ok(())
}
