//! Streaming LLM client (OpenAI-compatible SSE), the pluggable
//! [`LlmBackend`] trait, model registry, and clause splitting.

pub mod client;
pub mod model;
pub mod splitter;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub use crate::tools::ToolExecutor;
pub use client::{
    image_to_data_uri, load_tools_file, parse_sse_delta, parse_sse_line, ContentBlock, FunctionDef,
    ImageUrl, LlmClient, Message, MessageContent, SseDelta, Tool, ToolCall,
};
pub use splitter::ClauseSplitter;

use crate::error::Result;

/// Pluggable streaming LLM backend.
///
/// Object-safe (boxed futures; no extra crate dependencies). The contract
/// matches [`LlmClient::stream_reply`]: the backend appends `user` to its
/// conversation history, streams its reply, and sends completed clauses on
/// `clauses`. On cancellation ([`crate::error::LlmError::Cancelled`]) the
/// in-flight clause and the partial assistant reply are discarded — the
/// user never heard them.
///
/// # History discipline
///
/// A backend owns ONE conversation history (there is no per-conversation
/// id): [`clear_history`](LlmBackend::clear_history) resets it to a fresh
/// conversation (for [`LlmClient`]: the seeded system prompt). Call it
/// between logically separate conversations when reusing a backend.
pub trait LlmBackend: Send {
    /// Backend name, for logs and [`crate::AgentEvent`]s.
    fn name(&self) -> &str;

    /// Streams the reply to `user`, sending completed clauses on `clauses`
    /// as they are split out of the token stream; returns `Ok(())` at end
    /// of stream.
    fn stream_reply<'a>(
        &'a mut self,
        user: &'a str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Resets the conversation history (see the trait-level note).
    fn clear_history(&mut self);

    /// Returns the hold-music active flag, if this backend supports it.
    /// Default implementation returns `None`.
    fn hold_music_flag(&self) -> Option<&Arc<AtomicBool>> {
        None
    }
}

impl LlmBackend for LlmClient {
    fn name(&self) -> &str {
        // The model name identifies the serving endpoint's backend.
        self.model_name()
    }

    fn stream_reply<'a>(
        &'a mut self,
        user: &'a str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(LlmClient::stream_reply(self, user, clauses, cancel))
    }

    fn clear_history(&mut self) {
        LlmClient::clear_history(self);
    }

    fn hold_music_flag(&self) -> Option<&Arc<AtomicBool>> {
        self.hold_music_active.as_ref()
    }
}
