//! Call forwarding: when the LLM cannot handle a request, it invokes the
//! `forward_call` tool, which POSTs the conversation context to an external
//! service and relays that service's text response back as the tool result.
//!
//! The [`ForwardTool`] implements [`crate::tools::ToolExecutor`] so it slots
//! into the existing function-calling loop. Because the [`ToolExecutor`]
//! trait receives only the tool arguments (not the live history),
//! [`LlmClient`](crate::llm::LlmClient) routes `forward_call` through
//! [`ForwardTool::forward`], passing the real conversation history so the
//! forwarded service gets full context.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SkadooshError};
use crate::llm::Message;
use crate::tools::ToolExecutor;

/// The tool name the LLM invokes to forward a conversation.
pub const FORWARD_TOOL_NAME: &str = "forward_call";

/// Configuration for the call-forwarding endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardConfig {
    /// URL of the service to forward the conversation to.
    pub endpoint: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl ForwardConfig {
    /// Creates a new config targeting `endpoint` with a 30 s default timeout.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout_secs: 30,
        }
    }
}

/// The tool definition the LLM sees when call forwarding is enabled.
///
/// Arguments the model provides: `{"reason": "...", "summary": "..."}`.
pub fn forward_tool_definition() -> crate::llm::Tool {
    crate::llm::Tool::function(
        FORWARD_TOOL_NAME,
        "Forward this conversation to another service when you cannot answer \
         the user's question",
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this conversation is being forwarded."
                },
                "summary": {
                    "type": "string",
                    "description": "What to ask the forwarded service."
                }
            },
            "required": ["reason", "summary"]
        }),
    )
}

/// The POST body sent to the forwarding endpoint.
#[derive(Debug, Serialize)]
struct ForwardRequest<'a> {
    /// Why the conversation is being forwarded.
    reason: &'a str,
    /// What to ask the forwarded service.
    summary: &'a str,
    /// The current user query (last user message).
    current_query: &'a str,
    /// The full conversation history, newest last.
    history: &'a [Message],
}

/// Sends the full conversation context to the forwarding endpoint via HTTP
/// POST and returns the service's text response.
///
/// The body includes the conversation `history`, the `current_query`, the
/// forwarding `reason`, and the `summary` of what to ask the forwarded service.
pub async fn forward_conversation(
    config: &ForwardConfig,
    history: &[Message],
    current_query: &str,
    reason: &str,
    summary: &str,
) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.timeout_secs))
        .build()
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("forward HTTP client: {e}")))?;
    forward_with(&client, config, history, current_query, reason, summary).await
}

/// Shared inner POST routine used by both the free function and [`ForwardTool`].
async fn forward_with(
    client: &reqwest::Client,
    config: &ForwardConfig,
    history: &[Message],
    current_query: &str,
    reason: &str,
    summary: &str,
) -> Result<String> {
    let body = ForwardRequest {
        reason,
        summary,
        current_query,
        history,
    };
    let resp = client
        .post(&config.endpoint)
        .json(&body)
        .send()
        .await
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("forward request: {e}")))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("forward response: {e}")))?;
    if !status.is_success() {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "forward endpoint returned {status}: {}",
            text.chars().take(1024).collect::<String>()
        )));
    }
    Ok(text)
}

/// Parses the `{"reason": "...", "summary": "..."}` arguments the model
/// emits for a `forward_call`. Missing fields default to the empty string.
pub(crate) fn parse_forward_args(arguments: &str) -> (String, String) {
    let v: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let reason = v
        .get("reason")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let summary = v
        .get("summary")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    (reason, summary)
}

/// Tool executor that forwards the conversation to an external service.
///
/// Implements [`ToolExecutor`] so it can be registered alongside the
/// subprocess-based [`crate::tools::ShellExecutor`]. The async
/// [`ForwardTool::forward`] method is the entry point used by
/// [`LlmClient`](crate::llm::LlmClient), which has access to the live
/// conversation history; the sync [`ToolExecutor::execute`] impl forwards
/// just the reason/summary (no prior context) since the trait carries no
/// history.
#[derive(Debug, Clone)]
pub struct ForwardTool {
    config: ForwardConfig,
    client: reqwest::Client,
}

impl ForwardTool {
    /// Creates a new `ForwardTool` for `config`, reusing one HTTP client.
    pub fn new(config: ForwardConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { config, client }
    }

    /// Forwards the full conversation context (the async entry point used by
    /// the LLM client, which passes the live history and current query).
    pub async fn forward(
        &self,
        history: &[Message],
        current_query: &str,
        reason: &str,
        summary: &str,
    ) -> Result<String> {
        forward_with(
            &self.client,
            &self.config,
            history,
            current_query,
            reason,
            summary,
        )
        .await
    }

    /// The config this tool was built with.
    pub fn config(&self) -> &ForwardConfig {
        &self.config
    }
}

impl ToolExecutor for ForwardTool {
    fn execute(&self, _name: &str, arguments: &str) -> Result<String> {
        let (reason, summary) = parse_forward_args(arguments);
        // The trait method has no access to conversation history, so we
        // forward just the reason/summary with no prior context. The live
        // history is forwarded by `LlmClient` through `ForwardTool::forward`.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.forward(&[], "", &reason, &summary))
        })
    }
}
