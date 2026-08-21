//! Tool execution for LLM function calling.
//!
//! The [`ToolExecutor`] trait powers the model's function-calling loop in
//! `LlmClient`. Subprocess-based execution
//! ([`ShellExecutor`]) spawns tool `name` as a subprocess with JSON
//! `arguments` on stdin and returns stdout. Streaming execution
//! ([`ShellExecutor::execute_streaming`]) calls back with each stdout line
//! as it arrives, letting the LLM react mid-execution.
//!
//! Parallel execution: when the model issues multiple tool calls in one
//! response, they run concurrently via `tokio::spawn`.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use crate::error::{Result, SkadooshError};

/// Runs a tool call produced by the model.
pub trait ToolExecutor: Send + Sync {
    /// Execute the tool named `name` with JSON `arguments`, returning the result.
    fn execute(&self, name: &str, arguments: &str) -> Result<String>;
}

/// Runs tool `name` as a subprocess; JSON `arguments` on stdin, returns stdout.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShellExecutor;

impl ShellExecutor {
    /// Creates a new `ShellExecutor`.
    pub fn new() -> Self {
        Self
    }

    /// Executes a tool call, streaming stdout lines through `on_line` as they
    /// arrive. Returns the full accumulated output and the wall-clock duration.
    pub async fn execute_streaming(
        name: &str,
        arguments: &str,
        mut on_line: impl FnMut(String),
    ) -> Result<(String, std::time::Duration)> {
        let started = Instant::now();

        let mut child = Command::new(name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("spawn '{name}': {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(arguments.as_bytes());
        }

        // Stream stdout lines as they arrive.
        let mut full_output = String::new();
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(l) => {
                        full_output.push_str(&l);
                        full_output.push('\n');
                        on_line(l);
                    }
                    Err(e) => {
                        tracing::warn!(tool=%name, error=%e, "failed reading tool stdout line");
                        break;
                    }
                }
            }
        }

        let output = child
            .wait_with_output()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("wait '{name}': {e}")))?;

        let elapsed = started.elapsed();

        if !output.status.success() {
            // Include any remaining stderr the streaming loop didn't capture.
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "'{name}' exited {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        // If stdout was fully captured, full_output already has it.
        // If we missed the streaming path (shouldn't happen), fall back.
        if full_output.is_empty() {
            full_output = String::from_utf8_lossy(&output.stdout).into_owned();
        }

        Ok((full_output, elapsed))
    }
}

impl ToolExecutor for ShellExecutor {
    fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        let mut child = Command::new(name)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("spawn '{name}': {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(arguments.as_bytes());
        }

        let output = child
            .wait_with_output()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("wait '{name}': {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "'{name}' exited {}: {}",
                output.status,
                stderr.trim()
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

/// Returns a placeholder message — tool execution not configured.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopExecutor;

impl NoopExecutor {
    /// Creates a new `NoopExecutor`.
    pub fn new() -> Self {
        Self
    }
}

impl ToolExecutor for NoopExecutor {
    fn execute(&self, _name: &str, _arguments: &str) -> Result<String> {
        Ok("{\"error\":\"tool execution not configured; respond with text\"}".to_string())
    }
}

/// Executes multiple tool calls in parallel via `tokio::spawn`, returning
/// results keyed by tool call id. Each call is a `(name, arguments, call_id)`
/// triple. Results arrive in completion order; the caller reassembles them
/// by `call_id`.
///
/// Each call spawns its own [`ShellExecutor`] in a tokio task so they run
/// concurrently without blocking each other.
pub async fn execute_parallel(
    calls: Vec<(String, String, String)>, // (name, args, call_id)
) -> BTreeMap<String, std::result::Result<String, SkadooshError>> {
    let mut handles = Vec::new();

    // Dispatch each call onto a separate tokio task.
    for (name, args, call_id) in calls {
        // Clone what we need for the spawned task.
        // We use a separate scope approach: pass owned data.
        let handle = tokio::spawn(async move {
            // Create a fresh executor per task since ShellExecutor is Copy.
            let exec = ShellExecutor::new();
            let result = exec.execute(&name, &args);
            (call_id, result)
        });
        handles.push(handle);
    }

    let mut results = BTreeMap::new();
    for handle in handles {
        match handle.await {
            Ok((call_id, result)) => {
                results.insert(call_id, result);
            }
            Err(join_err) => {
                tracing::error!(error=%join_err, "tool execution task panicked");
            }
        }
    }
    results
}
