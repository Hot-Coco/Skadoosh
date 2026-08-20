//! Tool execution: the [`ToolExecutor`] trait plus a couple of reference
//! implementations used by [`crate::llm::LlmClient`] to run the function calls
//! a model emits during tool calling.
//!
//! A tool executor turns a `(name, arguments)` pair — exactly what the model
//! produces in a [`crate::llm::ToolCall`] — into a string result that is fed
//! back into the conversation as a `role: "tool"` message on the next
//! round-trip.
//!
//! The trait is [`Send`] + [`Sync`] so an executor can be stored inside the
//! async [`crate::llm::LlmClient`], which is shared across tasks.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::error::{Result, SkadooshError};

/// Executes a tool/function call produced by the model.
///
/// Implementations receive the function `name` (as the model named it) and the
/// raw JSON `arguments` string the model emitted, and return either the tool
/// result (any string — conventionally JSON) or an error. The result is fed
/// back to the model as a `role: "tool"` message in the next round.
///
/// The trait is [`Send`] + [`Sync`] so an executor may be stored inside the
/// async [`crate::llm::LlmClient`] (see [`LlmClient::with_tool_executor`]).
///
/// [`LlmClient::with_tool_executor`]: crate::llm::LlmClient::with_tool_executor
pub trait ToolExecutor: Send + Sync {
    /// Runs the tool named `name` with the JSON `arguments` string and returns
    /// its result.
    fn execute(&self, name: &str, arguments: &str) -> Result<String>;
}

/// Runs each tool `name` as a subprocess, piping the JSON `arguments` to the
/// child's stdin and returning its captured stdout.
///
/// The command is invoked directly (no shell), so `name` must be an executable
/// on `PATH` or an absolute/relative path. The arguments JSON is sent on stdin
/// rather than as CLI args, which keeps the encoding lossless and sidesteps
/// shell-escaping pitfalls. A non-zero exit status becomes an error carrying
/// the child's (trimmed) stderr.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShellExecutor;

impl ShellExecutor {
    /// Creates a new [`ShellExecutor`].
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for ShellExecutor {
    fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        let mut child = Command::new(name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("spawn tool '{name}': {e}")))?;

        // Pipe the arguments JSON to the child's stdin. A write failure (the
        // child exited early, closed stdin, ...) is not fatal here — the
        // non-zero exit status / stderr below will surface the real cause.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(arguments.as_bytes());
        }

        let output = child
            .wait_with_output()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("wait for tool '{name}': {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "tool '{name}' exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// A no-op executor that never runs anything: every call returns a fixed
/// placeholder message. Handy for tests and for disabling tool execution
/// without removing the tool definitions.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopExecutor;

/// Placeholder result returned by [`NoopExecutor`]. Matches the shape the
/// `LlmClient` previously produced inline before an executor existed.
const NOOP_MESSAGE: &str = "{\"error\":\"tool execution not configured; respond with text\"}";

impl NoopExecutor {
    /// Creates a new [`NoopExecutor`].
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for NoopExecutor {
    fn execute(&self, _name: &str, _arguments: &str) -> Result<String> {
        Ok(NOOP_MESSAGE.to_string())
    }
}
