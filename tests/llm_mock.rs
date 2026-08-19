//! LLM streaming client tests against the in-process mock OpenAI SSE server
//! (plan task 4.2 acceptance): ordering, reassembly, history bounds,
//! `[DONE]`, malformed-line tolerance, cancellation.

#[path = "common/mock_openai.rs"]
mod mock_openai;

use std::time::Duration;

use mock_openai::{done_line, token_line, Chunk, MockOpenAi};
use skadoosh::error::{LlmError, SkadooshError};
use skadoosh::llm::{parse_sse_line, LlmClient};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const SYSTEM: &str = "You are a test bot.";

fn client(url: &str, max_history_turns: usize) -> LlmClient {
    LlmClient::new(url, "mock-model", SYSTEM, max_history_turns, None)
}

/// Drains the clause channel after `stream_reply` has returned (its `Sender`
/// is dropped, so `recv` terminates).
async fn drain(rx: &mut mpsc::Receiver<String>) -> Vec<String> {
    let mut got = Vec::new();
    while let Some(c) = rx.recv().await {
        got.push(c);
    }
    got
}

#[tokio::test]
async fn clauses_arrive_in_order_and_reassemble() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(": stream-open comment\n\n"),
        Chunk::now(token_line("Hello, ")),
        Chunk::now(token_line("world. ")),
        Chunk::now(token_line("How are ")),
        Chunk::now(token_line("you?")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut client = client(&server.url(), 8);
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("Say hi", tx, CancellationToken::new())
        .await
        .expect("stream_reply should succeed");

    let got = drain(&mut rx).await;
    assert_eq!(
        got,
        vec![
            "Hello,".to_string(),
            " world.".to_string(),
            " How are you?".to_string()
        ]
    );
    assert_eq!(got.concat(), "Hello, world. How are you?");

    // History: system (message 0), user, completed assistant reply.
    let h = client.history();
    assert_eq!(h.len(), 3);
    assert_eq!(h[0].role, "system");
    assert_eq!(h[0].content, SYSTEM);
    assert_eq!(h[1].role, "user");
    assert_eq!(h[1].content, "Say hi");
    assert_eq!(h[2].role, "assistant");
    assert_eq!(h[2].content, "Hello, world. How are you?");

    // The outgoing request carries model, full messages, and stream: true.
    let req = server.captured_request().expect("request captured");
    assert!(req.contains("\"model\":\"mock-model\""), "request: {req}");
    assert!(req.contains("\"stream\":true"), "request: {req}");
    assert!(req.contains("\"role\":\"system\""), "request: {req}");
    assert!(req.contains("Say hi"), "request: {req}");

    assert!(
        server.wait_peer_gone(Duration::from_secs(2)).await,
        "server should observe the client closing the connection"
    );
}

#[tokio::test]
async fn history_truncates_to_max_history_turns() {
    let script = |reply: &str| vec![Chunk::now(token_line(reply)), Chunk::now(done_line())];
    let server =
        MockOpenAi::serve_many(vec![script("A one."), script("A two."), script("A three.")]).await;
    let mut client = client(&server.url(), 2);

    for user in ["u1", "u2", "u3"] {
        let (tx, _rx) = mpsc::channel(16);
        client
            .stream_reply(user, tx, CancellationToken::new())
            .await
            .expect("stream_reply should succeed");
    }

    // system + last 2 turns (2 messages each): u1/a1 fell off.
    let h = client.history();
    assert_eq!(h.len(), 5, "history: {h:?}");
    assert_eq!(h[0].role, "system");
    assert_eq!(h[0].content, SYSTEM);
    assert_eq!(h[1].content, "u2");
    assert_eq!(h[2].content, "A two.");
    assert_eq!(h[3].content, "u3");
    assert_eq!(h[4].content, "A three.");
}

#[tokio::test]
async fn malformed_and_non_data_lines_are_skipped() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(": comment / keep-alive\n\n"),
        Chunk::now("\n"),
        Chunk::now("event: message\n\n"),
        Chunk::now("id: 42\nretry: 1000\n\n"),
        Chunk::now("data: {this is not json}\n\n"),
        // A role-only delta chunk has no content and must be ignored.
        Chunk::now("data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"),
        Chunk::now(token_line("Good. ")),
        Chunk::now(token_line("Yes!")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut client = client(&server.url(), 8);
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("hi", tx, CancellationToken::new())
        .await
        .expect("malformed lines must not fail the stream");

    let got = drain(&mut rx).await;
    assert_eq!(got, vec!["Good.".to_string(), " Yes!".to_string()]);
    assert_eq!(client.history()[2].content, "Good. Yes!");
}

#[tokio::test]
async fn clean_close_without_done_is_tolerated() {
    // No `data: [DONE]`: the server just closes after the last chunk.
    let server = MockOpenAi::serve(vec![Chunk::now(token_line("Bye."))]).await;
    let mut client = client(&server.url(), 8);
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("hi", tx, CancellationToken::new())
        .await
        .expect("EOF must end the stream cleanly");

    let got = drain(&mut rx).await;
    assert_eq!(got, vec!["Bye.".to_string()]);
    assert_eq!(client.history()[2].content, "Bye.");
}

#[tokio::test]
async fn eof_flushes_unterminated_final_line() {
    // Server closes without a trailing newline after the last `data:` line —
    // the content must still be delivered (no `[DONE]`, no `\n`).
    let last_line = token_line("Tail."); // normally ends with "\n\n"
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("First. ")),
        Chunk::now(last_line.trim_end().to_string()),
    ])
    .await;
    let mut client = client(&server.url(), 8);
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("hi", tx, CancellationToken::new())
        .await
        .expect("EOF with unterminated line must still parse");

    let got = drain(&mut rx).await;
    assert_eq!(got, vec!["First.".to_string(), " Tail.".to_string()]);
    assert_eq!(client.history()[2].content, "First. Tail.");
}

#[tokio::test]
async fn cancelled_turns_still_truncate_history() {
    // max_history_turns = 1 → budget is 2 non-system messages. A storm of
    // cancelled turns must not grow history past that, even though no
    // assistant reply is ever appended on the cancel path.
    let mut storm = client("http://127.0.0.1:1", 1); // never connected
    for i in 0..5 {
        let token = CancellationToken::new();
        token.cancel(); // pre-cancelled: no request is ever sent
        let (tx, _rx) = mpsc::channel(16);
        let res = tokio::time::timeout(
            Duration::from_secs(2),
            storm.stream_reply(&format!("u{i}"), tx, token),
        )
        .await
        .expect("pre-cancelled stream_reply must return promptly");
        assert!(matches!(res, Err(SkadooshError::Llm(LlmError::Cancelled))));
    }
    let h = storm.history();
    assert_eq!(h.len(), 3, "system + 2-message budget: {h:?}");
    assert_eq!(h[0].role, "system");
    assert!(
        h[1..].iter().all(|m| m.role == "user"),
        "cancelled turns leave only (bounded) user messages: {h:?}"
    );
    assert_eq!(h[2].content, "u4", "newest unanswered user message kept");

    // A later successful turn still records its reply and stays in budget.
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Back.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut client2 = client(&server.url(), 1);
    let (tx, _rx) = mpsc::channel(16);
    client2
        .stream_reply("u", tx, CancellationToken::new())
        .await
        .expect("stream succeeds");
    let token = CancellationToken::new();
    token.cancel();
    let (tx, _rx) = mpsc::channel(16);
    let _ = client2.stream_reply("u2", tx, token).await;
    let h = client2.history();
    assert_eq!(h.len(), 3, "mixed success+cancel stays bounded: {h:?}");
}

#[tokio::test]
async fn api_error_status_is_reported() {
    let server = MockOpenAi::serve_error(500, "model exploded").await;
    let mut client = client(&server.url(), 8);
    let (tx, _rx) = mpsc::channel(16);
    let err = client
        .stream_reply("hi", tx, CancellationToken::new())
        .await
        .expect_err("500 must be an error");
    match err {
        SkadooshError::Llm(LlmError::Api { status, body }) => {
            assert_eq!(status, 500);
            assert!(body.contains("model exploded"), "body: {body}");
        }
        other => panic!("expected LlmError::Api, got {other:?}"),
    }
    // No assistant reply was appended on failure.
    assert_eq!(client.history().len(), 2);
}

#[tokio::test]
async fn cancel_mid_stream_returns_cancelled_and_drops_partial_reply() {
    let mut script = vec![Chunk::now(token_line("First. "))];
    // Effectively unbounded slow stream: cancellation must cut it short.
    for _ in 0..2_000 {
        script.push(Chunk::after(
            Duration::from_millis(5),
            token_line("more text "),
        ));
    }
    let server = MockOpenAi::serve(script).await;
    let mut client = client(&server.url(), 8);
    let token = CancellationToken::new();
    let canceller = {
        let token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            token.cancel();
        })
    };

    let (tx, mut rx) = mpsc::channel(16);
    let res = tokio::time::timeout(
        // Generous bound: the only requirement is "eventually" (no tight
        // wall-clock bounds on loaded machines).
        Duration::from_secs(2),
        client.stream_reply("hi", tx, token),
    )
    .await
    .expect("stream_reply must return within 2 s of cancellation");
    assert!(
        matches!(res, Err(SkadooshError::Llm(LlmError::Cancelled))),
        "expected Cancelled, got {res:?}"
    );

    // No clauses are delivered after cancel returns (the sender was consumed
    // and dropped by `stream_reply`); anything received predates the return.
    let got = drain(&mut rx).await;
    assert!(
        got.len() < 200,
        "stream must have been cut short, got {} clauses",
        got.len()
    );
    assert_eq!(got.first().map(String::as_str), Some("First."));

    // The partial assistant reply is NOT appended to history.
    let h = client.history();
    assert_eq!(h.len(), 2, "history after cancel: {h:?}");
    assert_eq!(h[1].role, "user");

    // The server observes the dropped connection.
    assert!(
        server.wait_peer_gone(Duration::from_secs(2)).await,
        "server should observe the cancelled connection dropping"
    );
    canceller.abort();
}

#[tokio::test]
async fn cancel_before_request_returns_cancelled() {
    let server = MockOpenAi::serve(vec![Chunk::now(token_line("Never."))]).await;
    let mut client = client(&server.url(), 8);
    let token = CancellationToken::new();
    token.cancel();
    let (tx, _rx) = mpsc::channel(16);
    let res = tokio::time::timeout(Duration::from_secs(2), client.stream_reply("hi", tx, token))
        .await
        .expect("pre-cancelled stream_reply must return promptly");
    assert!(matches!(res, Err(SkadooshError::Llm(LlmError::Cancelled))));
    assert_eq!(client.history().len(), 2, "no assistant reply on cancel");
}

#[tokio::test]
async fn api_key_sets_bearer_auth_header() {
    // With a key, every request carries `Authorization: Bearer <key>`
    // (unlocks hosted OpenAI-compatible providers).
    let server =
        MockOpenAi::serve(vec![Chunk::now(token_line("Hi.")), Chunk::now(done_line())]).await;
    let mut client = LlmClient::new(
        &server.url(),
        "mock-model",
        SYSTEM,
        8,
        Some("sk-test-secret".to_string()),
    );
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("hello", tx, CancellationToken::new())
        .await
        .expect("stream_reply should succeed");
    let got = drain(&mut rx).await;
    assert_eq!(got.concat(), "Hi.");

    let req = server.captured_request().expect("request captured");
    assert!(
        req.to_lowercase()
            .contains("authorization: bearer sk-test-secret"),
        "request must carry the bearer token: {req}"
    );
}

#[tokio::test]
async fn no_api_key_sends_no_authorization_header() {
    // Ollama-style local servers need no key: the header must be absent.
    let server =
        MockOpenAi::serve(vec![Chunk::now(token_line("Hi.")), Chunk::now(done_line())]).await;
    let mut client = client(&server.url(), 8);
    let (tx, _rx) = mpsc::channel(16);
    client
        .stream_reply("hello", tx, CancellationToken::new())
        .await
        .expect("stream_reply should succeed");

    let req = server.captured_request().expect("request captured");
    assert!(
        !req.to_lowercase().contains("authorization:"),
        "no key → no Authorization header: {req}"
    );
}

#[tokio::test]
async fn clear_history_resets_to_system_prompt() {
    let server =
        MockOpenAi::serve(vec![Chunk::now(token_line("Hi.")), Chunk::now(done_line())]).await;
    let mut client = client(&server.url(), 8);
    let (tx, _rx) = mpsc::channel(16);
    client
        .stream_reply("hello", tx, CancellationToken::new())
        .await
        .expect("stream_reply should succeed");
    assert_eq!(client.history().len(), 3);

    use skadoosh::llm::LlmBackend;
    client.clear_history();
    let h = client.history();
    assert_eq!(h.len(), 1, "history resets to the system prompt: {h:?}");
    assert_eq!(h[0].role, "system");
    assert_eq!(h[0].content, SYSTEM);
    // The LlmBackend trait method does the same thing.
    LlmBackend::clear_history(&mut client);
    assert_eq!(client.history().len(), 1);
    // LlmBackend::name is the model name.
    assert_eq!(LlmBackend::name(&client), "mock-model");
}

#[test]
fn parse_sse_line_tolerates_dialect_variants() {
    assert!(parse_sse_line("").is_none());
    assert!(parse_sse_line("   ").is_none());
    assert!(parse_sse_line(": keep-alive").is_none());
    assert!(parse_sse_line("event: message").is_none());
    assert!(parse_sse_line("id: 7").is_none());

    // [DONE] sentinel, with or without the conventional space.
    assert!(matches!(parse_sse_line("data: [DONE]"), Some(Ok(None))));
    assert!(matches!(parse_sse_line("data:[DONE]"), Some(Ok(None))));

    // A content token parses.
    let line = token_line("hi");
    match parse_sse_line(line.trim_end()) {
        Some(Ok(Some(tok))) => assert_eq!(tok, "hi"),
        other => panic!("expected token, got {other:?}"),
    }

    // Malformed JSON surfaces as Err for the caller to warn-and-skip.
    assert!(matches!(parse_sse_line("data: {oops"), Some(Err(_))));

    // Role-only / finish-reason chunks carry no content: ignored.
    assert!(parse_sse_line("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}").is_none());
    assert!(
        parse_sse_line("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}").is_none()
    );
}
