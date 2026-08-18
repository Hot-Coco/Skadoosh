//! Streaming LLM client (OpenAI-compatible SSE) and clause splitting.

pub mod client;
pub mod splitter;

pub use client::{parse_sse_line, LlmClient, Message};
pub use splitter::ClauseSplitter;
