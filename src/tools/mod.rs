//! Tool execution for LLM function calling.

use std::io::Write;
use std::process::{Command, Stdio};

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
