//! Cookbook 03 — REPL mode with a mock LLM replying inline.
//!
//! Drives [`Agent::repl`] fully in-memory: a scripted [`LlmBackend`] supplies
//! the replies and a `&[u8]` slice plays the role of stdin. The repl streams
//! each reply's clauses inline (`bot> …`) as they arrive, and exits on
//! `/quit` or EOF. No server, no models, no audio.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 03_repl_mode
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::{Agent, Config, Result};
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
    // The repl is always text-only (no VAD/STT/TTS), so OutputMode is moot,
    // but text mode keeps the config internally consistent.
    let config = Config {
        repl: true,
        ..Config::default()
    };

    let mut agent = Agent::builder()
        .config(config)
        .llm(Box::new(
            ScriptedLlm::new()
                .turn(&["Hello, ", "world."])
                .turn(&["Fine, ", "thanks!"]),
        ))
        .build()?;

    // In-memory stdin: two turns then `/quit`. A blank line in between is
    // skipped by the repl.
    let input: &[u8] = b"hello there\n\nhow are you?\n/quit\n";
    let mut output: Vec<u8> = Vec::new();
    agent.repl(input, &mut output)?;

    let out = String::from_utf8(output).expect("repl output is utf-8");
    print!("{out}");

    // The repl prints a banner, one `bot>` line per turn, and a final "bye".
    assert!(out.contains("skadoosh repl"), "banner present: {out:?}");
    assert!(
        out.contains("bot> Hello, world."),
        "turn 1 streamed: {out:?}"
    );
    assert!(
        out.contains("bot> Fine, thanks!"),
        "turn 2 streamed: {out:?}"
    );
    assert!(out.contains("bye"), "clean exit: {out:?}");

    println!("03_repl_mode: OK");
    Ok(())
}
