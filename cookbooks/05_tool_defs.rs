//! Cookbook 05 — Tool/function definitions.
//!
//! Writes a JSON file of OpenAI-compatible tool/function definitions, loads
//! it back with [`load_tools_file`], and prints the parsed [`Tool`]s. Also
//! shows the [`Tool::function`] builder for constructing a tool in code. No
//! server, no models — just the tool-definition machinery the LLM client
//! sends with each request when `--tools-file` is set.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 05_tool_defs
//! ```

use std::path::PathBuf;

use skadoosh::llm::{load_tools_file, Tool};

/// Wraps any `Display` error (io / serde) into the crate's umbrella error.
fn wrap<E: std::fmt::Display>(e: E) -> skadoosh::SkadooshError {
    anyhow::anyhow!("{e}").into()
}

fn main() -> skadoosh::Result<()> {
    // 1. Write a tools JSON file in the OpenAI function-calling format:
    //    [{"type":"function","function":{"name","description","parameters"}}]
    let tools_json = serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Start a countdown timer in seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer", "description": "Duration in seconds"}
                    },
                    "required": ["seconds"]
                }
            }
        }
    ])
    .to_string();

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cookbook_05_tools.json");
    std::fs::write(&path, &tools_json).map_err(wrap)?;
    println!("wrote {} ({} bytes)", path.display(), tools_json.len());

    // 2. Load and parse it back into typed Tool values.
    let tools = load_tools_file(&path).map_err(wrap)?;
    println!("\n{} tool(s) available:", tools.len());
    for tool in &tools {
        println!(
            "  • {} — {}",
            tool.function.name,
            tool.function
                .description
                .as_deref()
                .unwrap_or("(no description)")
        );
        println!(
            "    parameters: {}",
            serde_json::to_string(&tool.function.parameters).unwrap_or_else(|_| "<invalid>".into())
        );
    }

    assert_eq!(tools.len(), 2, "two tools parsed from the file");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "set_timer");

    // 3. The in-code builder produces the same shape (useful for registering
    //    tools without a file).
    let built = Tool::function(
        "get_weather",
        "Get the current weather for a city.",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    );
    assert_eq!(built.tool_type, "function");
    assert_eq!(built.function.name, "get_weather");
    println!("\nTool::function builder -> {built:?}");

    // Cleanup.
    let _ = std::fs::remove_file(&path);

    println!("\n05_tool_defs: OK");
    Ok(())
}
