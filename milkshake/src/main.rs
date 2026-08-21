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
                },
            );
        }
    }

    // Collect MCP tool definitions.
    let tools: Vec<ToolDef> = {
        let guard = state.mcp_servers.lock().unwrap();
        guard.iter().flat_map(|s| s.tools.clone()).collect()
    };

    // Build Ollama payload.
    let mut payload = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
    });
    if !tools.is_empty() {
        payload["tools"] = serde_json::to_value(&tools).unwrap();
    }

    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", state.args.ollama_url);
    let resp = match client.post(&url).json(&payload).send().await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("{e}")})),
            )
                .into_response();
        }
    };

    // Stream SSE lines back to the browser.
    let stream = resp.bytes_stream().map(|chunk| {
        let data = chunk.unwrap_or_default();
        let text = String::from_utf8_lossy(&data).to_string();
        Ok::<_, std::convert::Infallible>(Event::default().data(text))
    });

    Sse::new(stream)
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
