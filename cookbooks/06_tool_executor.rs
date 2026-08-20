//! Cookbook 06 — Tool executor.
//!
//! Demonstrates the [`ToolExecutor`] trait with the built-in executors:
//!
//! * [`ShellExecutor`] runs the tool `name` as a subprocess, pipes the JSON
//!   `arguments` to its **stdin**, and returns its **stdout**. So `cat`
//!   round-trips the arguments back — the simplest way to see the contract
//!   (a real tool would be a script that reads its JSON args from stdin and
//!   prints a JSON result).
//! * [`NoopExecutor`] returns a placeholder ("tool execution not configured")
//!   — the fallback used when no executor is wired into the [`LlmClient`].
//!
//! No server, no models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 06_tool_executor
//! ```

use skadoosh::tools::{NoopExecutor, ShellExecutor, ToolExecutor};

fn main() -> skadoosh::Result<()> {
    let arguments = r#"{"hello":"world"}"#;

    // ShellExecutor pipes `arguments` to the command's stdin and returns
    // stdout. `echo` ignores stdin, so we use `cat` to round-trip the JSON:
    // the model's function arguments reach the tool via stdin, and the
    // tool's stdout is fed back into the conversation.
    let shell = ShellExecutor::new();
    let stdout = shell.execute("cat", arguments)?;
    println!("ShellExecutor.execute(\"cat\", {arguments:?})");
    println!("  -> stdout: {:?}", stdout.trim_end());

    assert_eq!(
        stdout.trim(),
        arguments,
        "cat should echo back the JSON arguments it received on stdin"
    );

    // NoopExecutor: the "not configured" placeholder result.
    let noop = NoopExecutor::new();
    let result = noop.execute("get_weather", r#"{"city":"Berlin"}"#)?;
    println!("\nNoopExecutor.execute(\"get_weather\", {{\"city\":\"Berlin\"}})");
    println!("  -> {result}");
    assert!(
        result.contains("not configured"),
        "noop returns the placeholder result: {result}"
    );

    println!("\n06_tool_executor: OK");
    Ok(())
}
