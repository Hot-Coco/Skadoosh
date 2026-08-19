//! In-process mock of an OpenAI-compatible SSE chat-completions server,
//! shared by the LLM integration tests (and later the pipeline tests).
//!
//! Bound on `127.0.0.1` with an ephemeral port via a tokio `TcpListener`.
//! The response body is a script of [`Chunk`]s — raw byte strings written
//! verbatim, each with an optional pre-write delay (for cancel tests).
//! Malformed lines are injected by simply scripting them. The server records
//! the full raw request (`captured_request`) and observes the client closing
//! the connection (`peer_gone` / `wait_peer_gone`).
//!
//! `serve` handles exactly one connection; `serve_many` handles one
//! connection per script, sequentially (multi-turn history tests).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

/// One scripted chunk of the SSE response body.
pub struct Chunk {
    /// Raw bytes written verbatim (typically `data: {...}\n\n` lines).
    pub body: String,
    /// Delay applied *before* this chunk is written.
    pub delay: Duration,
}

impl Chunk {
    /// A chunk written immediately.
    pub fn now(body: impl Into<String>) -> Self {
        Self {
            body: body.into(),
            delay: Duration::ZERO,
        }
    }

    /// A chunk written after `delay`.
    pub fn after(delay: Duration, body: impl Into<String>) -> Self {
        Self {
            body: body.into(),
            delay,
        }
    }
}

/// Builds one `data:` line of an OpenAI `chat.completion.chunk` carrying
/// `content` as the delta text (JSON-escaped via `serde_json`).
pub fn token_line(content: &str) -> String {
    let payload = serde_json::json!({
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock-model",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}],
    });
    format!("data: {payload}\n\n")
}

/// The `data: [DONE]` end sentinel line.
pub fn done_line() -> String {
    "data: [DONE]\n\n".to_string()
}

/// A running mock server. Dropping it aborts the accept task.
pub struct MockOpenAi {
    addr: SocketAddr,
    captured: Arc<Mutex<Option<String>>>,
    peer_gone: Arc<AtomicBool>,
    task: Option<JoinHandle<()>>,
}

impl MockOpenAi {
    /// Serves exactly one connection: 200 OK + the scripted SSE chunks, then
    /// the socket stays open until the client closes.
    pub async fn serve(chunks: Vec<Chunk>) -> Self {
        Self::serve_many(vec![chunks]).await
    }

    /// Serves one connection per script, in order (each `stream_reply` call
    /// opens a fresh HTTP connection).
    pub async fn serve_many(scripts: Vec<Vec<Chunk>>) -> Self {
        Self::start(200, None, scripts).await
    }

    /// Serves one connection that answers `status` with a plain body.
    pub async fn serve_error(status: u16, body: &str) -> Self {
        Self::start(status, Some(body.to_string()), vec![Vec::new()]).await
    }

    async fn start(status: u16, error_body: Option<String>, scripts: Vec<Vec<Chunk>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let addr = listener.local_addr().expect("local addr");
        let captured = Arc::new(Mutex::new(None));
        let peer_gone = Arc::new(AtomicBool::new(false));
        let (c2, p2) = (Arc::clone(&captured), Arc::clone(&peer_gone));
        let task = tokio::spawn(async move {
            for script in scripts {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                p2.store(false, Ordering::SeqCst);
                if !serve_conn(&mut sock, status, error_body.as_deref(), &script, &c2).await {
                    p2.store(true, Ordering::SeqCst);
                    return;
                }
                // Wait (bounded) for the client to close, then accept the
                // next scripted connection.
                let mut sink = [0u8; 1024];
                let _ = tokio::time::timeout(Duration::from_secs(10), async {
                    loop {
                        match sock.read(&mut sink).await {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                    }
                })
                .await;
                p2.store(true, Ordering::SeqCst);
            }
        });
        Self {
            addr,
            captured,
            peer_gone,
            task: Some(task),
        }
    }

    /// Base URL for [`LlmClient::new`](skadoosh::llm::LlmClient::new).
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// The full raw HTTP request (headers + body) of the current/last
    /// connection, once received.
    pub fn captured_request(&self) -> Option<String> {
        self.captured.lock().expect("captured lock").clone()
    }

    /// Whether the client has closed/dropped the connection.
    pub fn peer_gone(&self) -> bool {
        self.peer_gone.load(Ordering::SeqCst)
    }

    /// Polls [`peer_gone`](Self::peer_gone) until `timeout` elapses.
    pub async fn wait_peer_gone(&self, timeout: Duration) -> bool {
        let start = tokio::time::Instant::now();
        while start.elapsed() < timeout {
            if self.peer_gone() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.peer_gone()
    }
}

impl Drop for MockOpenAi {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Reads the HTTP request (capturing it), writes the response head, then the
/// scripted chunks. Returns `false` when the peer vanished mid-serve.
async fn serve_conn(
    sock: &mut tokio::net::TcpStream,
    status: u16,
    error_body: Option<&str>,
    script: &[Chunk],
    captured: &Mutex<Option<String>>,
) -> bool {
    // Read headers, then exactly Content-Length body bytes.
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => return false,
            Ok(n) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return false; // header never terminated; bail
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
            Ok(0) | Err(_) => return false,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    *captured.lock().expect("captured lock") = Some(String::from_utf8_lossy(&buf).into_owned());

    let reason = if (200..300).contains(&status) {
        "OK"
    } else {
        "Error"
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: text/event-stream\r\n\
         cache-control: no-cache\r\nconnection: close\r\n\r\n"
    );
    if sock.write_all(head.as_bytes()).await.is_err() {
        return false;
    }
    if let Some(body) = error_body {
        let _ = sock.write_all(body.as_bytes()).await;
        // No content-length was sent: the body is delimited by the close.
        let _ = sock.shutdown().await;
        return true;
    }
    for chunk in script {
        if !chunk.delay.is_zero() {
            tokio::time::sleep(chunk.delay).await;
        }
        if sock.write_all(chunk.body.as_bytes()).await.is_err() {
            return false; // peer dropped mid-stream (e.g. barge-in cancel)
        }
        if sock.flush().await.is_err() {
            return false;
        }
    }
    // Half-close: the client sees EOF now (a script end without `[DONE]`
    // must not hang the client), while the caller's read loop can still
    // observe the client dropping the connection.
    let _ = sock.shutdown().await;
    true
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
