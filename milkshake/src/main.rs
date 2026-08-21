//! Milkshake — a clean web chat UI for StealthyLM with MCP tool support.
//!
//! Serves a single-page chat app at `http://localhost:PORT` that talks to
//! a local Ollama instance (StealthyLM by default) and can connect to MCP
//! servers for tool calling.

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::channel::mpsc;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, services::ServeDir};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Clone)]
#[command(name = "milkshake", about = "Clean web chat UI for StealthyLM")]
struct Args {
    /// Port to listen on.
    #[arg(long, env = "MILKSHAKE_PORT", default_value = "3000")]
    port: u16,

    /// Ollama base URL.
    #[arg(long, env = "OLLAMA_HOST", default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Default model name.
    #[arg(long, env = "MILKSHAKE_MODEL", default_value = "stealthylm")]
    model: String,
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct McpConnection {
    name: String,
    url: String,
    tools: Vec<ToolDef>,
}

struct AppState {
    args: Args,
    mcp_servers: Mutex<Vec<McpConnection>>,
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolDef {
    #[serde(rename = "type", default = "function_type")]
    typ: String,
    function: FunctionDef,
}

fn function_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Name of the tool this result is for. Ollama's native `/api/chat` uses a
    /// `tool_name` field on `tool`-role messages (rather than `tool_call_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct McpConnectRequest {
    name: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct McpDisconnectRequest {
    name: String,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "milkshake=info".into()),
        )
        .init();

    let args = Args::parse();
    let state = Arc::new(AppState {
        args: args.clone(),
        mcp_servers: Mutex::new(Vec::new()),
    });

    let app = Router::new()
        .route("/api/chat", post(chat_handler))
        .route("/api/models", get(models_handler))
        .route("/api/mcp/connect", post(mcp_connect_handler))
        .route("/api/mcp/servers", get(mcp_servers_handler))
        .route("/api/mcp/disconnect", post(mcp_disconnect_handler))
        .route("/api/health", get(health_handler))
        .layer(CorsLayer::permissive())
        .fallback_service(ServeDir::new("static"))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    tracing::info!(addr = %addr, "Milkshake starting");
    tracing::info!(ollama = %args.ollama_url, model = %args.model, "Ollama config");

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

async fn health_handler() -> &'static str {
    "ok"
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let url = format!("{}/api/tags", state.args.ollama_url);
    match reqwest::get(&url).await {
        Ok(resp) => {
            let body = resp.text().await.unwrap_or_default();
            (StatusCode::OK, body).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": format!("ollama unreachable: {e}")})),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Chat (streaming SSE proxy to Ollama)
// ---------------------------------------------------------------------------

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> impl IntoResponse {
    let model = req.model.unwrap_or_else(|| state.args.model.clone());

    // Inject the user's system prompt as a system message.
    let mut messages = req.messages;
    if let Some(ref sp) = req.system_prompt {
        if !sp.is_empty() {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: sp.clone(),
                    tool_calls: None,
                    tool_call_id: None,
                    tool_name: None,
                },
            );
        }
    }

    // Collect MCP tool definitions.
    let tools: Vec<ToolDef> = {
        let guard = state.mcp_servers.lock().unwrap();
        guard.iter().flat_map(|s| s.tools.clone()).collect()
    };

    // Build and send the first Ollama request synchronously so that a
    // connection failure is reported as a normal HTTP error (matching the
    // non-streaming error path) rather than an empty SSE stream.
    let mut first_payload = serde_json::json!({
        "model": &model,
        "stream": true,
    });
    first_payload["messages"] = serde_json::to_value(&messages).unwrap();
    if !tools.is_empty() {
        first_payload["tools"] = serde_json::to_value(&tools).unwrap();
    }

    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", state.args.ollama_url);
    let first_resp = match client.post(&url).json(&first_payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("{e}")})),
            )
                .into_response();
        }
    };

    // Drive the (possibly multi-round, tool-calling) exchange on a background
    // task, starting from the response we just obtained, and surface it to the
    // browser as a single SSE stream.
    let (tx, rx) = mpsc::unbounded::<Result<Event, std::convert::Infallible>>();
    tokio::spawn(run_chat_stream(
        tx, state, model, messages, tools, first_resp,
    ));

    Sse::new(rx)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ---------------------------------------------------------------------------
// MCP connect
// ---------------------------------------------------------------------------

async fn mcp_connect_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<McpConnectRequest>,
) -> impl IntoResponse {
    // List tools from the MCP server.
    let tools = match fetch_mcp_tools(&body.url).await {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("MCP handshake failed: {e}")})),
            )
                .into_response();
        }
    };

    let conn = McpConnection {
        name: body.name.clone(),
        url: body.url.clone(),
        tools,
    };

    let mut guard = state.mcp_servers.lock().unwrap();
    guard.retain(|c| c.name != body.name);
    guard.push(conn.clone());

    tracing::info!(name = %body.name, tools = conn.tools.len(), "MCP server connected");

    Json(serde_json::json!({"ok": true, "name": body.name, "tool_count": conn.tools.len()}))
        .into_response()
}

// ---------------------------------------------------------------------------
// MCP servers list
// ---------------------------------------------------------------------------

async fn mcp_servers_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let guard = state.mcp_servers.lock().unwrap();
    let servers: Vec<serde_json::Value> = guard
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "url": c.url,
                "tool_count": c.tools.len(),
                "tools": c.tools.iter().map(|t| t.function.name.clone()).collect::<Vec<_>>(),
            })
        })
        .collect();
    Json(serde_json::json!({"servers": servers})).into_response()
}

// ---------------------------------------------------------------------------
// MCP disconnect
// ---------------------------------------------------------------------------

async fn mcp_disconnect_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<McpDisconnectRequest>,
) -> impl IntoResponse {
    let mut guard = state.mcp_servers.lock().unwrap();
    guard.retain(|c| c.name != body.name);
    Json(serde_json::json!({"ok": true})).into_response()
}

// ---------------------------------------------------------------------------
// MCP tools fetch (JSON-RPC over HTTP)
// ---------------------------------------------------------------------------

async fn fetch_mcp_tools(server_url: &str) -> Result<Vec<ToolDef>, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    });
    let resp = client
        .post(server_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let tools_json = json["result"]["tools"].as_array().ok_or("no tools array")?;
    let mut tools = Vec::new();
    for t in tools_json {
        let name = t["name"].as_str().unwrap_or("?").to_string();
        let desc = t["description"].as_str().unwrap_or("").to_string();
        let params = t.get("inputSchema").cloned().unwrap_or(serde_json::json!({
            "type": "object",
            "properties": {}
        }));
        tools.push(ToolDef {
            typ: "function".into(),
            function: FunctionDef {
                name,
                description: desc,
                parameters: params,
            },
        });
    }
    Ok(tools)
}

// ---------------------------------------------------------------------------
// Chat streaming + MCP tool-calling loop
// ---------------------------------------------------------------------------

/// Maximum number of tool-calling round-trips before the loop gives up, to
/// prevent the model from spinning forever on tool requests.
const MAX_TOOL_ROUNDS: usize = 5;

/// Drives the Ollama `/api/chat` exchange, transparently executing MCP tool
/// calls between streaming rounds.
///
/// Every response chunk is forwarded to `tx` as an SSE event exactly as the
/// non-tool path would, so the browser sees one continuous stream. When a
/// round finishes with `tool_calls`, each requested tool is executed against
/// the MCP server that owns it, the assistant call + tool result messages are
/// appended to the conversation, and the expanded history is re-sent to Ollama.
/// Up to [`MAX_TOOL_ROUNDS`] round-trips are attempted before giving up.
async fn run_chat_stream(
    tx: mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
    state: Arc<AppState>,
    model: String,
    mut messages: Vec<ChatMessage>,
    tools: Vec<ToolDef>,
    first_resp: reqwest::Response,
) {
    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", state.args.ollama_url);

    let mut resp = first_resp;

    for round in 0..MAX_TOOL_ROUNDS {
        if tx.is_closed() {
            return; // browser disconnected
        }

        let (content, tool_calls) = match stream_and_collect(&tx, resp).await {
            Ok(v) => v,
            Err(e) => {
                send_json(
                    &tx,
                    serde_json::json!({"error": format!("stream read failed: {e}")}),
                );
                return;
            }
        };

        // No tool calls means the model produced its final answer; we're done.
        if tool_calls.is_empty() {
            return;
        }

        // Record the assistant's tool-call request in the conversation history.
        messages.push(ChatMessage {
            role: "assistant".into(),
            content,
            tool_calls: Some(serde_json::Value::Array(tool_calls.clone())),
            tool_call_id: None,
            tool_name: None,
        });

        // Execute each requested tool and append a `tool` result message.
        for tc in tool_calls {
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let raw_args = tc["function"]["arguments"].clone();
            // Ollama sends arguments as an object; tolerate a JSON string too.
            let arguments = match raw_args {
                serde_json::Value::Null => serde_json::json!({}),
                serde_json::Value::String(s) => {
                    serde_json::from_str(&s).unwrap_or(serde_json::json!({}))
                }
                other => other,
            };

            let result = match resolve_tool_server(&state, &name) {
                Some(server_url) => match fetch_mcp_call(&server_url, &name, arguments).await {
                    Ok(r) => r,
                    Err(e) => format!("Error: tool '{name}' failed: {e}"),
                },
                None => format!("Error: no MCP server provides tool '{name}'"),
            };

            tracing::info!(tool = %name, bytes = result.len(), "executed MCP tool call");

            messages.push(ChatMessage {
                role: "tool".into(),
                content: result,
                tool_calls: None,
                tool_call_id: None,
                tool_name: Some(name),
            });
        }

        // Stop before sending a follow-up request we won't be allowed to fully
        // process (the loop body above would just discard its output).
        if round + 1 >= MAX_TOOL_ROUNDS {
            break;
        }

        // Re-send the expanded conversation to Ollama for the next round.
        let mut payload = serde_json::json!({
            "model": &model,
            "stream": true,
        });
        payload["messages"] = serde_json::to_value(&messages).unwrap();
        if !tools.is_empty() {
            payload["tools"] = serde_json::to_value(&tools).unwrap();
        }

        resp = match client.post(&url).json(&payload).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "ollama unreachable mid-stream");
                send_json(
                    &tx,
                    serde_json::json!({"error": format!("ollama unreachable: {e}")}),
                );
                return;
            }
        };
    }

    // The model kept requesting tools past the round-trip cap.
    tracing::warn!("reached maximum tool-calling round-trips");
    send_json(
        &tx,
        serde_json::json!({"error": "reached maximum tool-calling round-trips"}),
    );
}

/// Streams every chunk from an Ollama `/api/chat` streaming response to `tx`
/// (preserving the existing wire format) while accumulating the assistant
/// `message.content` deltas and any `message.tool_calls`.
///
/// Returns the full concatenated content and the list of tool-call objects
/// (empty when the model requested no tools).
async fn stream_and_collect(
    tx: &mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
    resp: reqwest::Response,
) -> Result<(String, Vec<serde_json::Value>), String> {
    let mut content = String::new();
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut buf = String::new();

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&bytes).to_string();

        // Forward the raw chunk to the browser exactly as the non-tool path does.
        let _ = tx.unbounded_send(Ok(Event::default().data(text.clone())));

        // Ollama streams newline-delimited JSON; buffer and parse whole lines so
        // we reconstruct the full message even if chunks split a line in two.
        buf.push_str(&text);
        while let Some(idx) = buf.find('\n') {
            let line = buf[..idx].to_string();
            buf = buf[idx + 1..].to_string();
            parse_chunk_line(&line, &mut content, &mut tool_calls);
        }
    }
    // Trailing data without a final newline.
    if !buf.trim().is_empty() {
        parse_chunk_line(&buf, &mut content, &mut tool_calls);
    }

    Ok((content, tool_calls))
}

/// Accumulates `message.content` and `message.tool_calls` from a single Ollama
/// NDJSON line into the running totals.
fn parse_chunk_line(line: &str, content: &mut String, tool_calls: &mut Vec<serde_json::Value>) {
    if line.trim().is_empty() {
        return;
    }
    let Ok(j) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    if let Some(c) = j["message"]["content"].as_str() {
        if !c.is_empty() {
            content.push_str(c);
        }
    }
    if let Some(arr) = j["message"]["tool_calls"].as_array() {
        for tc in arr {
            tool_calls.push(tc.clone());
        }
    }
}

/// Forwards a single JSON object to the browser as an SSE data event.
fn send_json(
    tx: &mpsc::UnboundedSender<Result<Event, std::convert::Infallible>>,
    value: serde_json::Value,
) {
    let _ = tx.unbounded_send(Ok(Event::default().data(value.to_string())));
}

/// Finds the URL of the MCP server that exposes `tool_name`, if any.
fn resolve_tool_server(state: &AppState, tool_name: &str) -> Option<String> {
    let guard = state.mcp_servers.lock().unwrap();
    guard
        .iter()
        .find(|s| s.tools.iter().any(|t| t.function.name == tool_name))
        .map(|s| s.url.clone())
}

// ---------------------------------------------------------------------------
// MCP tool call (JSON-RPC over HTTP)
// ---------------------------------------------------------------------------

/// Sends a MCP JSON-RPC `tools/call` request to `server_url` and returns the
/// tool result as a string.
///
/// Text parts of the result `content` array are joined with newlines. If there
/// are no text parts, the raw `result` JSON is returned instead. JSON-RPC level
/// errors and transport failures are reported via [`Err`]; a tool that ran but
/// returned its own error (`isError`) is surfaced as [`Ok`] so the model can
/// react to it.
async fn fetch_mcp_call(
    server_url: &str,
    tool_name: &str,
    arguments: serde_json::Value,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        }
    });
    let resp = client
        .post(server_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    // Surface JSON-RPC level errors.
    if let Some(err) = json.get("error") {
        return Err(err.to_string());
    }

    let result = &json["result"];

    // MCP tool results are a `content` array of typed parts. Join any text
    // parts into a single string for the model; fall back to the raw JSON.
    if let Some(parts) = result["content"].as_array() {
        let text: String = parts
            .iter()
            .filter_map(|p| {
                if p["type"].as_str() == Some("text") {
                    p["text"].as_str().map(str::to_owned)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            return Ok(text);
        }
    }

    Ok(serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            args: Args {
                port: 0,
                ollama_url: "http://127.0.0.1:1".to_string(),
                model: "test-model".to_string(),
            },
            mcp_servers: Mutex::new(Vec::new()),
        })
    }

    fn test_app(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/api/chat", post(chat_handler))
            .route("/api/models", get(models_handler))
            .route("/api/mcp/connect", post(mcp_connect_handler))
            .route("/api/mcp/servers", get(mcp_servers_handler))
            .route("/api/mcp/disconnect", post(mcp_disconnect_handler))
            .route("/api/health", get(health_handler))
            .layer(CorsLayer::permissive())
            .fallback_service(ServeDir::new("static"))
            .with_state(state)
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let state = test_state();
        let app = test_app(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"ok");
    }

    #[tokio::test]
    async fn mcp_servers_empty_initially() {
        let state = test_state();
        let app = test_app(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/mcp/servers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["servers"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn mcp_connect_and_list() {
        let mock_url = start_mock_mcp_server().await;

        let state = test_state();
        let app = test_app(state.clone());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/mcp/connect")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"name": "test-mcp", "url": mock_url}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ok"], true);
        assert_eq!(json["tool_count"], 2);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/mcp/servers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let servers = json["servers"].as_array().unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0]["name"], "test-mcp");
        assert_eq!(servers[0]["tool_count"], 2);
        let tools: Vec<String> = servers[0]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tools, vec!["echo", "add"]);
    }

    #[tokio::test]
    async fn mcp_disconnect() {
        let mock_url = start_mock_mcp_server().await;

        let state = test_state();
        let app = test_app(state.clone());

        let _connect = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/mcp/connect")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"name": "disconnect-me", "url": mock_url}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/mcp/disconnect")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"name": "disconnect-me"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/mcp/servers")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["servers"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn chat_returns_error_when_ollama_down() {
        let state = test_state();
        let app = test_app(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .header("accept", "text/event-stream")
                    .body(Body::from(
                        serde_json::json!({
                            "messages": [{"role": "user", "content": "hi"}],
                            "model": "test-model"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let error = json["error"].as_str().unwrap();
        assert!(
            !error.is_empty(),
            "error field should be populated: {error}"
        );
    }

    #[tokio::test]
    async fn models_returns_error_when_ollama_down() {
        let state = test_state();
        let app = test_app(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let error = json["error"].as_str().unwrap();
        assert!(
            error.contains("ollama unreachable"),
            "unexpected error: {error}"
        );
    }

    async fn start_mock_mcp_server() -> String {
        async fn tools_handler() -> Json<serde_json::Value> {
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": { "type": "string" }
                                }
                            }
                        },
                        {
                            "name": "add",
                            "description": "Add two numbers",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": { "type": "number" },
                                    "b": { "type": "number" }
                                }
                            }
                        }
                    ]
                }
            }))
        }

        let app = Router::new().route("/", post(tools_handler));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        format!("http://{}", addr)
    }
}
