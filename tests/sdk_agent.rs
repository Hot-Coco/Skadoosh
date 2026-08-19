//! SDK facade tests (plan v0.2 "SDK & modalities"): the `Agent` builder
//! (defaults + engine injection), `text_turn` against the mock SSE server,
//! and the in-memory repl — all headless, no model files needed.

// Not every mock knob (serve_error/wait_peer_gone) is used by this suite.
#[allow(dead_code)]
#[path = "common/mock_openai.rs"]
mod mock_openai;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use mock_openai::{done_line, token_line, Chunk, MockOpenAi};
use skadoosh::agent::{Agent, AgentEvent};
use skadoosh::config::{Config, OutputMode};
use skadoosh::llm::LlmBackend;
use skadoosh::stt::MockStt;
use skadoosh::tts::MockTts;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A text-mode config pointed at the mock LLM (the only stage these tests
/// use); model paths intentionally do not exist — they must not be touched.
fn text_config(llm_url: &str) -> Config {
    Config {
        llm_url: llm_url.to_string(),
        llm_model: "mock-model".to_string(),
        system_prompt: "You are a test bot.".to_string(),
        output: OutputMode::Text,
        ..Default::default()
    }
}

/// A scripted `LlmBackend` written against the public trait (the same shape
/// an SDK user implements).
struct ScriptedLlm {
    replies: Arc<Mutex<Vec<String>>>,
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
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = skadoosh::Result<()>> + Send + 'a>>
    {
        Box::pin(async move {
            // Drain the script first so the MutexGuard does not cross an await.
            let script: Vec<String> = self.replies.lock().expect("replies").drain(..).collect();
            for clause in script {
                let _ = clauses.send(clause).await;
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {}
}

/// Builder defaults: no injected engines, no devices opened, no models
/// loaded — `build()` is pure assembly.
#[test]
fn builder_defaults_build_without_models_or_devices() {
    let agent = Agent::builder()
        .build()
        .expect("build must not open devices or load models");
    let _events = agent.events(); // subscribable before any run
    agent.shutdown(); // cancel with nothing running: no-op, no panic
}

/// Every engine slot accepts an injected trait object.
#[test]
fn builder_accepts_injected_engines() {
    let agent = Agent::builder()
        .config(text_config("http://127.0.0.1:1"))
        .stt(Box::new(MockStt::from_replies(["canned"])))
        .llm(Box::new(ScriptedLlm {
            replies: Arc::new(Mutex::new(Vec::new())),
        }))
        .tts(Box::new(MockTts::new()))
        .build()
        .expect("build");
    let _ = agent.events();
}

/// The MockStt scripted queue is consumed in order, and an exhausted script
/// yields empty transcripts (which the pipeline skips).
#[tokio::test]
async fn mock_stt_pops_scripted_replies_in_order() {
    use skadoosh::stt::SttEngine;
    let stt = MockStt::from_replies(["first", "second"]);
    assert_eq!(stt.name(), "mock-stt");

    let first = stt.transcribe(vec![0.0; 1600]).await.expect("reply 1");
    assert_eq!(first.expect("ok 1"), "first");
    let second = stt.transcribe(vec![0.0; 1600]).await.expect("reply 2");
    assert_eq!(second.expect("ok 2"), "second");
    let exhausted = stt.transcribe(vec![0.0; 1600]).await.expect("reply 3");
    assert_eq!(exhausted.expect("ok 3"), "", "exhausted script → empty");
}

/// MockStt delayed replies arrive after their delay (used by stale-turn
/// tests upstream).
#[tokio::test]
async fn mock_stt_delayed_reply_arrives() {
    use skadoosh::stt::SttEngine;
    let stt = MockStt::new();
    stt.push_delayed(Duration::from_millis(50), "slow");
    let start = std::time::Instant::now();
    let reply = stt.transcribe(vec![0.0; 1600]).await.expect("reply");
    assert_eq!(reply.expect("ok"), "slow");
    assert!(
        start.elapsed() >= Duration::from_millis(50),
        "delay honored: {:?}",
        start.elapsed()
    );
}

/// `text_turn` with the real `LlmClient` against the mock SSE server:
/// clauses stream out as events, the full reply is returned, and the shared
/// history machinery accumulates the turn.
///
/// (Multi-threaded runtime: the mock server's tasks keep progressing on a
/// worker thread while the main test thread blocks in the sync `text_turn`
/// facade.)
#[tokio::test(flavor = "multi_thread")]
async fn text_turn_against_mock_server_streams_and_returns_reply() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Hello, ")),
        Chunk::now(token_line("world. ")),
        Chunk::now(token_line("How are you?")),
        Chunk::now(done_line()),
    ])
    .await;

    let mut agent = Agent::builder()
        .config(text_config(&server.url()))
        .build()
        .expect("build");
    let mut events = agent.events();
    let seen = std::thread::spawn(move || {
        let mut order = Vec::new();
        while let Ok(event) = events.blocking_recv() {
            order.push(match event {
                AgentEvent::Clause(c) => format!("clause:{c:?}"),
                AgentEvent::ReplyDone => "reply-done".to_string(),
                other => panic!("unexpected event on the text_turn path: {other:?}"),
            });
        }
        order
    });

    let reply = agent.text_turn("Say hi").expect("text_turn");
    assert_eq!(reply, "Hello, world. How are you?");

    drop(agent); // closes the event channel, ending the observer
    let order = seen.join().expect("observer");
    // ORDER, not just presence: every clause of the turn precedes
    // ReplyDone, and nothing else fires on this path.
    assert_eq!(
        order,
        vec![
            "clause:\"Hello,\"".to_string(),
            "clause:\" world.\"".to_string(),
            "clause:\" How are you?\"".to_string(),
            "reply-done".to_string(),
        ]
    );

    // The request carried the system prompt + user message (same machinery
    // as the voice pipeline).
    let req = server.captured_request().expect("request captured");
    assert!(req.contains("You are a test bot."), "request: {req}");
    assert!(req.contains("Say hi"), "request: {req}");
}

/// `text_turn` history accumulates across turns on the same agent (the
/// second request carries the first turn's user+assistant messages).
#[tokio::test(flavor = "multi_thread")]
async fn text_turn_history_accumulates_across_turns() {
    let script = |reply: &str| vec![Chunk::now(token_line(reply)), Chunk::now(done_line())];
    let server = MockOpenAi::serve_many(vec![script("One."), script("Two.")]).await;

    let mut agent = Agent::builder()
        .config(text_config(&server.url()))
        .build()
        .expect("build");
    assert_eq!(agent.text_turn("first").expect("turn 1"), "One.");
    assert_eq!(agent.text_turn("second").expect("turn 2"), "Two.");

    let req = server.captured_request().expect("request 2 captured");
    assert!(req.contains("first"), "turn 1 user text in history: {req}");
    assert!(req.contains("One."), "turn 1 reply in history: {req}");
    assert!(req.contains("second"), "turn 2 user text in history: {req}");
}

/// `text_turn` surfaces LLM errors (connection refused) instead of
/// panicking or hanging.
#[test]
fn text_turn_reports_llm_errors() {
    let mut agent = Agent::builder()
        .config(text_config("http://127.0.0.1:1")) // nothing listens
        .build()
        .expect("build");
    let err = agent.text_turn("hi").expect_err("must fail");
    assert!(
        matches!(err, skadoosh::SkadooshError::Llm(_)),
        "expected an LLM error, got {err:?}"
    );
}

/// The repl, driven fully in-memory: scripted input lines produce streamed
/// `bot> ` clauses in order, `/quit` exits, blank lines are skipped.
#[tokio::test(flavor = "multi_thread")]
async fn repl_in_memory_streams_clauses_in_order() {
    let script = |reply: &[&str]| {
        let mut chunks: Vec<Chunk> = reply.iter().map(|c| Chunk::now(token_line(c))).collect();
        chunks.push(Chunk::now(done_line()));
        chunks
    };
    let server = MockOpenAi::serve_many(vec![
        script(&["Hello, ", "world."]),
        script(&["Fine ", "thanks."]),
    ])
    .await;

    let mut agent = Agent::builder()
        .config(text_config(&server.url()))
        .build()
        .expect("build");

    let input = b"hello there\n\nhow are you?\n/quit\n".as_slice();
    let mut out: Vec<u8> = Vec::new();
    agent.repl(input, &mut out).expect("repl");
    let out = String::from_utf8(out).expect("utf8 output");

    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "skadoosh repl — type a line, /quit to exit");
    assert_eq!(lines[1], "bot> Hello, world.", "clauses in order: {out:?}");
    assert_eq!(lines[2], "bot> Fine thanks.", "second turn: {out:?}");
    assert_eq!(lines[3], "bye");
    assert_eq!(lines.len(), 4, "no extra output: {out:?}");

    // History accumulated both turns (second request carries the first).
    let req = server.captured_request().expect("request 2 captured");
    assert!(req.contains("hello there"), "request: {req}");
    assert!(req.contains("Hello, world."), "request: {req}");
}

/// EOF (no `/quit`) also ends the repl cleanly.
#[test]
fn repl_exits_on_eof() {
    let mut agent = Agent::builder()
        .config(text_config("http://127.0.0.1:1")) // never contacted
        .build()
        .expect("build");
    let mut out: Vec<u8> = Vec::new();
    agent.repl("".as_bytes(), &mut out).expect("repl");
    let out = String::from_utf8(out).expect("utf8");
    assert!(out.contains("bye"), "clean EOF exit: {out:?}");
}

/// The compiled `mock_agent` example runs green (zero models/servers): the
/// whole plugin story (MockStt + scripted LlmBackend + MockTts) drives a
/// real orchestrator turn.
#[test]
fn mock_agent_example_runs_green() {
    let mut path = std::env::current_exe().expect("current exe");
    path.pop(); // target/debug/deps → target/debug
    path.pop();
    path.push("examples/mock_agent");
    let output = std::process::Command::new(&path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", path.display()));
    assert!(
        output.status.success(),
        "mock_agent example failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("plugin turn OK"),
        "unexpected stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}
