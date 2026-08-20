//! OpenAI-compatible streaming chat-completions client with bounded history.
//!
//! The reqwest client uses rustls and sets no request timeout — SSE streams
//! are long-lived. History is seeded with the system prompt
//! ([`Config::system_prompt`](crate::Config::system_prompt)) and truncated to
//! the last `--max-history-turns` user/assistant turns so long sessions
//! cannot overflow small local models' context.

use std::path::Path;

use base64::Engine;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::{LlmError, Result, SkadooshError};
use crate::llm::splitter::ClauseSplitter;

/// One content block in a multimodal message (OpenAI-compatible format).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// The text.
        text: String,
    },
    /// An image referenced by URL or base64 data URI.
    #[serde(rename = "image_url")]
    Image {
        /// Image URL wrapper.
        image_url: ImageUrl,
    },
}

/// An image reference (URL or base64 data URI) for multimodal models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// Image URL: `https://...` or `data:image/<type>;base64,...`.
    pub url: String,
    /// Optional detail level: `"auto"`, `"low"`, or `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Content of a chat message: plain text or multimodal blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content (backward-compatible with v0.2.0).
    Text(String),
    /// Multimodal content blocks for vision/document models.
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Returns the text content if this is a plain-text message.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Blocks(_) => None,
        }
    }
}

impl PartialEq<&str> for MessageContent {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, MessageContent::Text(s) if s == other)
    }
}

impl PartialEq<MessageContent> for &str {
    fn eq(&self, other: &MessageContent) -> bool {
        matches!(other, MessageContent::Text(s) if s == *self)
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// Loads an image file and returns a base64 data URI suitable for
/// multimodal LLM requests. Auto-detects the MIME type from the extension.
pub fn image_to_data_uri(path: &Path) -> std::result::Result<String, std::io::Error> {
    let bytes = std::fs::read(path)?;
    let mime = mime_from_ext(path);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

/// Best-effort MIME type from a file extension.
fn mime_from_ext(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("tiff") | Some("tif") => "image/tiff",
        Some("pdf") => "application/pdf",
        _ => "image/png", // default for unknown
    }
}

/// Clause length bounds for the splitter fed by [`LlmClient::stream_reply`]
/// (plan §6: `min_len ≈ 4`, `max_len ≈ 160`). Shared with the pipeline's
/// selftest, which splits the token stream directly.
pub(crate) const CLAUSE_MIN_LEN: usize = 4;
pub(crate) const CLAUSE_MAX_LEN: usize = 160;

/// One chat message in the conversation history.
///
/// The `content` field is [`MessageContent`] — plain text (`String`) for
/// text-only models, or a vec of [`ContentBlock`]s for multimodal models.
/// Serde's `#[serde(untagged)]` on `MessageContent` serialises a `Text`
/// variant as a JSON string and `Blocks` as a JSON array, exactly matching
/// the OpenAI chat-completions schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// `"system"`, `"user"`, or `"assistant"` (or `"tool"` for tool results).
    pub role: String,
    /// Message content: plain text or multimodal blocks.
    pub content: MessageContent,
}

/// Streaming LLM client for an OpenAI-compatible `/chat/completions` API.
///
/// Implements [`LlmBackend`](crate::llm::LlmBackend). Supports multimodal
/// (vision) inputs via [`ContentBlock::Image`] and tool/function calling.
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    /// Optional bearer token for hosted providers (Ollama needs none).
    /// Never logged.
    api_key: Option<String>,
    max_history_turns: usize,
    system_prompt: String,
    history: Vec<Message>,
    /// Image paths for the next user turn (cleared after the turn).
    image_paths: Vec<std::path::PathBuf>,
}

impl LlmClient {
    /// Creates a client; history is seeded with `system_prompt`. The HTTP
    /// client deliberately has no request timeout (streams are long-lived).
    ///
    /// `api_key` (from `--api-key` / `SKADOOSH_API_KEY`) unlocks hosted
    /// OpenAI-compatible providers: when `Some`, every request carries an
    /// `Authorization: Bearer <key>` header. Local Ollama needs no key —
    /// pass `None`. The key is never logged.
    pub fn new(
        base_url: &str,
        model: &str,
        system_prompt: &str,
        max_history_turns: usize,
        api_key: Option<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            max_history_turns,
            system_prompt: system_prompt.to_string(),
            history: vec![Message {
                role: "system".to_string(),
                content: MessageContent::Text(system_prompt.to_string()),
            }],
            image_paths: Vec::new(),
        }
    }

    /// The model name this client requests (also its
    /// [`LlmBackend::name`](crate::llm::LlmBackend::name)).
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Sets image paths for the next user turn. Images are loaded and
    /// base64-encoded when the turn starts; the paths are cleared after
    /// the turn completes. Supported formats: png, jpg, gif, webp, bmp.
    pub fn with_images(mut self, paths: Vec<std::path::PathBuf>) -> Self {
        self.image_paths = paths;
        self
    }

    /// Clears any queued image paths.
    pub fn clear_images(&mut self) {
        self.image_paths.clear();
    }

    /// Builds the config-default client (the one shared construction used
    /// by the binary's pipeline and the SDK facade).
    pub(crate) fn from_config(config: &crate::config::Config) -> Self {
        Self::new(
            &config.llm_url,
            &config.llm_model,
            &config.system_prompt,
            config.max_history_turns,
            config.api_key.clone(),
        )
    }

    /// Streams a reply: appends the user message, POSTs
    /// `{base}/chat/completions` with `stream: true`, buffers partial SSE
    /// lines, feeds the [`ClauseSplitter`], and
    /// sends completed clauses on `clauses`.
    ///
    /// `tokio::select!` on `cancel`: on cancellation the in-flight clause is
    /// discarded, the *partial assistant reply is dropped from history* (the
    /// user never heard it), and
    /// [`LlmError::Cancelled`] is returned. History is still truncated to the
    /// last `max_history_turns` turns on this path so repeated barge-ins
    /// cannot pile up unanswered user messages.
    /// At `data: [DONE]` (or a clean connection close) the splitter is
    /// flushed and the completed assistant reply is appended to history
    /// (truncated to the last `max_history_turns` user/assistant turns).
    pub async fn stream_reply(
        &mut self,
        user: &str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let images = std::mem::take(&mut self.image_paths);
        let user_content = if images.is_empty() {
            MessageContent::Text(user.to_string())
        } else {
            let mut blocks = vec![ContentBlock::Text {
                text: user.to_string(),
            }];
            for path in &images {
                match image_to_data_uri(path) {
                    Ok(data_uri) => blocks.push(ContentBlock::Image {
                        image_url: ImageUrl {
                            url: data_uri,
                            detail: Some("auto".to_string()),
                        },
                    }),
                    Err(e) => {
                        tracing::warn!(path=%path.display(), error=%e, "failed to load image; skipping");
                    }
                }
            }
            if blocks.len() == 1 {
                // All images failed to load; fall back to text-only.
                MessageContent::Text(user.to_string())
            } else {
                MessageContent::Blocks(blocks)
            }
        };
        self.history.push(Message {
            role: "user".to_string(),
            content: user_content,
        });
        let result = self.stream_reply_inner(&clauses, &cancel).await;
        if matches!(result, Err(SkadooshError::Llm(LlmError::Cancelled))) {
            self.truncate_history();
        }
        result
    }

    /// The streaming body of [`stream_reply`](Self::stream_reply), split out
    /// so the public method can run book-keeping (history truncation) on the
    /// cancellation path no matter which `select!` observed the token.
    async fn stream_reply_inner(
        &mut self,
        clauses: &mpsc::Sender<String>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": self.history,
            "stream": true,
        });
        let url = format!("{}/chat/completions", self.base_url);

        let mut request = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let resp = tokio::select! {
            _ = cancel.cancelled() => return Err(LlmError::Cancelled.into()),
            r = request.send() => r.map_err(LlmError::Http)?,
        };
        let resp = ensure_success(resp).await?;

        let mut stream = resp.bytes_stream();
        let mut splitter = ClauseSplitter::new(CLAUSE_MIN_LEN, CLAUSE_MAX_LEN);
        let mut reply = String::new();
        let mut lines = SseLineBuffer::default();
        let mut done = false;
        let mut eof = false;

        while !done && !eof {
            let chunk = tokio::select! {
                _ = cancel.cancelled() => return Err(LlmError::Cancelled.into()),
                c = stream.next() => c,
            };
            match chunk {
                Some(Ok(bytes)) => lines.feed(&bytes),
                Some(Err(e)) => return Err(LlmError::Http(e).into()),
                // Clean connection close (with or without `data: [DONE]`):
                // closing the buffer makes `next_line` yield any unterminated
                // final line once, so a server that omits the trailing `\n`
                // loses no content.
                None => {
                    lines.close();
                    eof = true;
                }
            }
            while let Some(line) = lines.next_line() {
                match parse_sse_line(&line) {
                    None => {}
                    Some(Ok(None)) => {
                        done = true;
                        break;
                    }
                    Some(Ok(Some(token))) => {
                        reply.push_str(&token);
                        for clause in splitter.push(&token) {
                            if !send_clause(clauses, cancel, clause).await? {
                                // Consumer is gone (pipeline shutdown): stop
                                // streaming, drop the partial reply.
                                tracing::debug!("clauses receiver dropped; aborting LLM stream");
                                return Ok(());
                            }
                        }
                    }
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "skipping malformed SSE data line");
                    }
                }
            }
        }

        // `[DONE]` (or EOF): flush the splitter remainder, then record the
        // completed assistant reply.
        if let Some(rest) = splitter.flush() {
            if !send_clause(clauses, cancel, rest).await? {
                tracing::debug!("clauses receiver dropped at stream end");
                return Ok(());
            }
        }
        self.history.push(Message {
            role: "assistant".to_string(),
            content: MessageContent::Text(reply),
        });
        self.truncate_history();
        Ok(())
    }

    /// Current conversation history (system prompt first).
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Resets the conversation to just the system prompt (the
    /// [`LlmBackend`](crate::llm::LlmBackend) history-discipline hook).
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history.push(Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
        });
    }

    /// Keeps the system message plus the last `max_history_turns`
    /// user/assistant turns (2 messages per turn). Runs after a successful
    /// assistant append and after a cancellation, so barge-in-heavy sessions
    /// stay bounded (the cancel path may cut right after an unanswered user
    /// message, which is fine — the budget, not pairing, is the invariant).
    fn truncate_history(&mut self) {
        let keep = 2 * self.max_history_turns;
        if self.history.len() > 1 + keep {
            let drop = self.history.len() - 1 - keep;
            self.history.drain(1..=drop);
        }
    }
}

/// Passes a success response through; a non-success status becomes
/// [`LlmError::Api`] with the response body (truncated to 1024 chars).
pub(crate) async fn ensure_success(resp: reqwest::Response) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let text = resp.text().await.unwrap_or_default();
    Err(LlmError::Api {
        status: status.as_u16(),
        body: text.chars().take(1024).collect(),
    }
    .into())
}

/// Sends one completed clause, watching `cancel` while waiting for channel
/// capacity. Returns `Ok(false)` when the receiver is gone (pipeline
/// shutdown: stop streaming quietly), `Err(Cancelled)` on cancellation.
async fn send_clause(
    clauses: &mpsc::Sender<String>,
    cancel: &CancellationToken,
    clause: String,
) -> std::result::Result<bool, LlmError> {
    tokio::select! {
        _ = cancel.cancelled() => Err(LlmError::Cancelled),
        sent = clauses.send(clause) => Ok(sent.is_ok()),
    }
}

/// Buffers raw SSE bytes and yields complete lines. A multi-byte UTF-8 char
/// split across stream chunks stays in the byte buffer until the
/// terminating `\n` arrives, so lines are always valid boundaries. Shared
/// by [`LlmClient::stream_reply`] and the pipeline's selftest, which drives
/// the SSE stream directly.
///
/// Call [`close`](Self::close) at end of stream: afterwards [`next_line`]
/// yields the unterminated trailing bytes once as a final line (a server
/// that closes without a trailing `\n` loses no content), then `None`.
#[derive(Default)]
pub(crate) struct SseLineBuffer {
    buf: Vec<u8>,
    eof: bool,
}

impl SseLineBuffer {
    /// Appends one stream chunk.
    pub(crate) fn feed(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Marks end of stream; idempotent.
    pub(crate) fn close(&mut self) {
        self.eof = true;
    }

    /// Pops the next complete `\n`-terminated line (lossy UTF-8 decode).
    /// After [`close`](Self::close), also yields the unterminated remainder
    /// once, then returns `None`.
    pub(crate) fn next_line(&mut self) -> Option<String> {
        if let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=nl).collect();
            return Some(String::from_utf8_lossy(&line_bytes).into_owned());
        }
        if self.eof && !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            return Some(String::from_utf8_lossy(&rest).into_owned());
        }
        None
    }
}

/// Parses one SSE line, tolerantly (OpenAI-compatible servers vary):
///
/// * blank lines, comments (`: ...`), `event:`/`id:`/`retry:` fields, and
///   data chunks without text content (role deltas, finish markers) → `None`;
/// * `data: [DONE]` → `Some(Ok(None))` (end sentinel);
/// * `data: {...}` with content → `Some(Ok(Some(token)))`;
/// * malformed JSON → `Some(Err(..))`; the caller warns and skips the line.
pub fn parse_sse_line(line: &str) -> Option<Result<Option<String>>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    let data = line.strip_prefix("data:")?;
    let data = data.trim();
    if data == "[DONE]" {
        return Some(Ok(None));
    }
    let parsed: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(LlmError::Sse(format!("malformed SSE data: {e}")).into()));
        }
    };
    // OpenAI `chat.completion.chunk`: `choices[0].delta.content`. Chunks
    // without content (role-only deltas, `finish_reason` markers, usage
    // summaries) are ignored.
    let token = parsed
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()?;
    if token.is_empty() {
        None
    } else {
        Some(Ok(Some(token.to_string())))
    }
}
