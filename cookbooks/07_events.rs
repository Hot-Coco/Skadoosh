//! Cookbook 07 — Agent events.
//!
//! Subscribes to the [`AgentEvent`](skadoosh::AgentEvent) stream and logs
//! every event emitted during a single text turn, then asserts the expected
//! ordering: each [`AgentEvent::Clause`] precedes the final
//! [`AgentEvent::ReplyDone`]. Uses a scripted [`LlmBackend`] — no server,
//! no models, no audio device.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 07_events
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::{Agent, AgentEvent, Config, OutputMode, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A scripted LLM backend: one queued reply (a vec of clauses) per turn.
struct ScriptedLlm {
    script: Mutex<VecDeque<Vec<String>>>,
}

impl ScriptedLlm {
    fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
        }
    }

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
        .llm(Box::new(ScriptedLlm::new().turn(&[
            "Skadoosh ",
            "is listening. ",
            "Ask me anything!",
        ])))
        .build()?;

    // Subscribe BEFORE the turn: receivers only see events emitted after
    // they subscribe. A side thread logs each event as it arrives.
    let mut events = agent.events();
    let observer = std::thread::spawn(move || {
        let mut log = Vec::new();
        while let Ok(event) = events.blocking_recv() {
            let label = match &event {
                AgentEvent::Listening => "Listening".to_string(),
                AgentEvent::SpeechStart => "SpeechStart".to_string(),
                AgentEvent::Transcript(t) => format!("Transcript({t:?})"),
                AgentEvent::Clause(c) => format!("Clause({c:?})"),
                AgentEvent::ReplyDone => "ReplyDone".to_string(),
                AgentEvent::TurnCancelled => "TurnCancelled".to_string(),
                AgentEvent::ToolCall { name, .. } => format!("ToolCall({name})"),
                AgentEvent::StageLatency { total_ms, .. } => {
                    format!("StageLatency(total={total_ms}ms)")
                }
                AgentEvent::Error(e) => format!("Error({e})"),
            };
            println!("event: {label}");
            log.push(label);
        }
        log
    });

    let reply = agent.text_turn("Hello!")?;
    println!("reply: {}", reply.trim());

    // Dropping the agent closes the event channel, ending the observer.
    drop(agent);
    let log = observer.join().expect("observer thread");

    // On the text-turn path, every Clause precedes ReplyDone — and nothing
    // else fires (no Listening/SpeechStart/Transcript on a text turn).
    assert!(
        log.iter().filter(|e| e.starts_with("Clause")).count() == 3,
        "three clauses streamed: {log:?}"
    );
    assert_eq!(
        log.last().map(String::as_str),
        Some("ReplyDone"),
        "ends with ReplyDone: {log:?}"
    );
    assert!(
        log.iter()
            .rposition(|e| e.starts_with("Clause"))
            .zip(log.iter().position(|e| e == "ReplyDone"))
            .is_some_and(|(last_clause, done)| last_clause < done),
        "last Clause precedes ReplyDone: {log:?}"
    );

    println!("07_events: OK");
    Ok(())
}
