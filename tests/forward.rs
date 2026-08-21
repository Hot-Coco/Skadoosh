//! Call-forwarding integration tests: the `forward_call` tool POSTs the full
//! conversation context to an external service and relays the response back,
//! and `--forward-url` is parsed from the CLI.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use skadoosh::config::Config;
use skadoosh::forward::{
    forward_conversation, forward_tool_definition, ForwardConfig, ForwardTool, FORWARD_TOOL_NAME,
};
use skadoosh::llm::{Message, MessageContent};
use skadoosh::tools::ToolExecutor;

/// A one-shot mock HTTP server: captures the raw request and replies with a
/// fixed `status` + `body` (plain text, Content-Length framed). Modeled on the
/// shared `mock_openai` helper but tailored to non-SSE forwarding responses.
struct MockForward {
    addr: SocketAddr,
    captured: Arc<Mutex<Option<String>>>,
}

impl MockForward {
    /// Serves exactly one connection, replying with `reply_body`.
    async fn serve(reply_body: &str) -> Self {
        Self::serve_with(200, reply_body).await
    }

    /// Serves exactly one connection with an explicit `status` and body.
    async fn serve_with(status: u16, reply_body: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock forward server");
        let addr = listener.local_addr().expect("local addr");
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let captured_clone = Arc::clone(&captured);
        let reply = reply_body.to_string();
        tokio::spawn(async move {
            let Ok((mut sock, _)) = listener.accept().await else {
                return;
            };
            // Read headers, then exactly Content-Length body bytes.
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let header_end = loop {
                let n = match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => n,
                };
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = find(&buf, b"\r\n\r\n") {
                    break pos + 4;
                }
                if buf.len() > 64 * 1024 {
                    return;
                }
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]);
            let mut content_len = 0usize;
            for line in headers.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    if k.trim().eq_ignore_ascii_case("content-length") {
                        content_len = v.trim().parse().unwrap_or(0);
                    }
                }
            }
            while buf.len() - header_end < content_len {
                match sock.read(&mut tmp).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(&tmp[..n]),
                }
            }
            *captured_clone.lock().unwrap() = Some(String::from_utf8_lossy(&buf).into_owned());

            let head = format!(
                "HTTP/1.1 {status} OK\r\ncontent-type: text/plain\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n",
                reply.len()
            );
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(reply.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
        Self { addr, captured }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// The captured request body (JSON), once received.
    fn body(&self) -> String {
        let raw = self.captured.lock().unwrap().clone().unwrap_or_default();
        // Split headers from body at the first blank line.
        match find(raw.as_bytes(), b"\r\n\r\n") {
            Some(pos) => raw[pos + 4..].to_string(),
            None => raw,
        }
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn user_msg(text: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: MessageContent::Text(text.to_string()),
        tool_call_id: None,
        tool_calls: None,
    }
}

fn system_msg(text: &str) -> Message {
    Message {
        role: "system".to_string(),
        content: MessageContent::Text(text.to_string()),
        tool_call_id: None,
        tool_calls: None,
    }
}

/// `forward_conversation` POSTs the conversation history, current query, and
/// forwarding reason/summary, and returns the forwarded service's text reply.
#[tokio::test]
async fn forward_conversation_posts_context_and_relays_response() {
    let server = MockForward::serve("The billing team says: your invoice is paid.").await;
    let config = ForwardConfig {
        endpoint: server.url(),
        timeout_secs: 5,
    };
    let history = vec![
        system_msg("You are a voice assistant."),
        user_msg("What is the status of my invoice?"),
    ];

    let reply = forward_conversation(
        &config,
        &history,
        "What is the status of my invoice?",
        "I cannot access billing records.",
        "Ask for the invoice payment status.",
    )
    .await
    .expect("forward_conversation should succeed");

    assert_eq!(reply, "The billing team says: your invoice is paid.");

    // The POST body must carry the full context.
    let body: serde_json::Value = serde_json::from_str(&server.body()).expect("body is JSON");
    assert_eq!(body["reason"], "I cannot access billing records.");
    assert_eq!(body["summary"], "Ask for the invoice payment status.");
    assert_eq!(body["current_query"], "What is the status of my invoice?");
    let hist = body["history"].as_array().expect("history is array");
    assert_eq!(hist.len(), 2, "full history is forwarded");
    assert_eq!(hist[0]["role"], "system");
    assert_eq!(hist[1]["role"], "user");
    assert_eq!(hist[1]["content"], "What is the status of my invoice?");
}

/// `ForwardTool` implements `ToolExecutor`: `execute` POSTs the arguments and
/// returns the relayed response. (Multi-thread runtime so `block_in_place`
/// inside the sync trait method is legal.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn forward_tool_executor_posts_and_returns_response() {
    let server = MockForward::serve("transferred-agent-reply").await;
    let tool = ForwardTool::new(ForwardConfig {
        endpoint: server.url(),
        timeout_secs: 5,
    });

    let args = r#"{"reason":"out of scope","summary":"handle the refund request"}"#;
    let out = tool
        .execute(FORWARD_TOOL_NAME, args)
        .expect("execute should succeed");
    assert_eq!(out, "transferred-agent-reply");

    // The arguments are forwarded as reason/summary.
    let body: serde_json::Value = serde_json::from_str(&server.body()).expect("body is JSON");
    assert_eq!(body["reason"], "out of scope");
    assert_eq!(body["summary"], "handle the refund request");
}

/// A non-success status from the forwarding endpoint surfaces as an error.
#[tokio::test]
async fn forward_conversation_surfaces_non_success_as_error() {
    let server = MockForward::serve_with(503, "upstream down").await;
    let config = ForwardConfig {
        endpoint: server.url(),
        timeout_secs: 5,
    };
    let err = forward_conversation(&config, &[], "q", "r", "s")
        .await
        .expect_err("non-2xx should error");
    assert!(
        err.to_string().contains("503"),
        "error should mention status: {err}"
    );
}

/// The `forward_call` tool definition is registered with the expected name and
/// description.
#[test]
fn forward_tool_definition_has_expected_shape() {
    let tool = forward_tool_definition();
    assert_eq!(tool.function.name, FORWARD_TOOL_NAME);
    assert!(
        tool.function
            .description
            .as_deref()
            .unwrap_or("")
            .contains("Forward this conversation"),
        "description should describe forwarding: {:?}",
        tool.function.description
    );
    // Parameters declare reason + summary as required strings.
    let params = &tool.function.parameters;
    assert_eq!(params["type"], "object");
    assert_eq!(params["properties"]["reason"]["type"], "string");
    assert_eq!(params["properties"]["summary"]["type"], "string");
    assert_eq!(params["required"][0], "reason");
    assert_eq!(params["required"][1], "summary");
}

/// `--forward-url` is parsed from the CLI.
#[test]
fn forward_url_flag_is_parsed() {
    use clap::Parser;
    let config = Config::try_parse_from(["skadoosh", "--forward-url", "http://example/forward"])
        .expect("parses");
    assert_eq!(
        config.forward_url.as_deref(),
        Some("http://example/forward")
    );

    // Default (no flag) is unset.
    let default = Config::try_parse_from(["skadoosh"]).expect("parses");
    assert!(default.forward_url.is_none(), "default forward_url is None");
}

/// `Config::default()` matches the clap default for `forward_url` (SDK
/// ergonomics: a default config equals a bare `skadoosh` invocation).
#[test]
fn default_config_has_no_forward_url() {
    let default = Config::default();
    assert!(default.forward_url.is_none());
}
