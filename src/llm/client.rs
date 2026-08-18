//! OpenAI-compatible streaming chat-completions client with bounded history.
//!
//! The reqwest client uses rustls and sets no request timeout — SSE streams
//! are long-lived. History is seeded with the system prompt
//! ([`Config::system_prompt`](crate::Config::system_prompt)) and truncated to
//! the last `--max-history-turns` user/assistant turns so long sessions
//! cannot overflow small local models' context.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::Result;

/// One chat message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// Streaming LLM client for an OpenAI-compatible `/chat/completions` API.
#[allow(dead_code)] // fields consumed by the task-4.2 implementation
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    max_history_turns: usize,
    history: Vec<Message>,
}

impl LlmClient {
    /// Creates a client; history is seeded with `system_prompt`. The HTTP
    /// client deliberately has no request timeout (streams are long-lived).
    pub fn new(base_url: &str, model: &str, system_prompt: &str, max_history_turns: usize) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            max_history_turns,
            history: vec![Message {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            }],
        }
    }

    /// Streams a reply: appends the user message, POSTs
    /// `{base}/chat/completions` with `stream: true`, buffers partial SSE
    /// lines, feeds the [`ClauseSplitter`](crate::llm::ClauseSplitter), and
    /// sends completed clauses on `clauses`.
    ///
    /// `tokio::select!` on `cancel`: on cancellation the in-flight clause is
    /// discarded, the *partial assistant reply is dropped from history* (the
    /// user never heard it), and
    /// [`LlmError::Cancelled`](crate::error::LlmError::Cancelled) is returned.
    /// At `data: [DONE]` the splitter is flushed and the completed assistant
    /// reply is appended to history (truncated to the last
    /// `max_history_turns` user/assistant turns).
    pub async fn stream_reply(
        &mut self,
        user: &str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let _ = (user, clauses, cancel);
        todo!("task 4.2: SSE stream with cancel + bounded history")
    }

    /// Current conversation history (system prompt first).
    pub fn history(&self) -> &[Message] {
        &self.history
    }
}

/// Parses one SSE line, tolerantly (OpenAI-compatible servers vary):
///
/// * blank lines, comments, and keep-alives → `None` (ignored);
/// * `data: [DONE]` → `Some(Ok(None))` (end sentinel);
/// * `data: {...}` with content → `Some(Ok(Some(token)))`;
/// * malformed JSON → skipped with a warning.
pub fn parse_sse_line(line: &str) -> Option<Result<Option<String>>> {
    let _ = line;
    todo!("task 4.2: tolerant SSE line parser")
}
