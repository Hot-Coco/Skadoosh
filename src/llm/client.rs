//! OpenAI-compatible streaming chat-completions client with bounded history.
//!
//! The reqwest client uses rustls and sets no request timeout — SSE streams
//! are long-lived. History is seeded with the system prompt
//! ([`Config::system_prompt`](crate::Config::system_prompt)) and truncated to
//! the last `--max-history-turns` user/assistant turns so long sessions
//! cannot overflow small local models' context.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::config::DEFAULT_SYSTEM_PROMPT;
use crate::error::{LlmError, Result, SkadooshError};
use crate::llm::model;
use crate::llm::splitter::ClauseSplitter;
use crate::memory::MemoryStore;
use crate::rag::{OnnxEmbedder, RagStore};
use crate::tools::{execute_parallel, ShellExecutor, ToolExecutor};

/// One content block in a multimodal message (OpenAI-compatible format).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// The text.
        text: String,
    },
    /// An image referenced by URL or base64 data URI.
    #[serde(rename = "image_url")]
    Image {
        /// Image URL wrapper.
        image_url: ImageUrl,
    },
}

/// An image reference (URL or base64 data URI) for multimodal models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// Image URL: `https://...` or `data:image/<type>;base64,...`.
    pub url: String,
    /// Optional detail level: `"auto"`, `"low"`, or `"high"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Content of a chat message: plain text or multimodal blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content (backward-compatible with v0.2.0).
    Text(String),
    /// Multimodal content blocks for vision/document models.
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    /// Returns the text content if this is a plain-text message.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Blocks(_) => None,
        }
    }
}

impl PartialEq<&str> for MessageContent {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, MessageContent::Text(s) if s == other)
    }
}

impl PartialEq<MessageContent> for &str {
    fn eq(&self, other: &MessageContent) -> bool {
        matches!(other, MessageContent::Text(s) if s == *self)
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

impl From<serde_json::Value> for MessageContent {
    fn from(v: serde_json::Value) -> Self {
        match v {
            serde_json::Value::String(s) => MessageContent::Text(s),
            serde_json::Value::Array(ref arr) => {
                // Try to deserialize as ContentBlock array
                let val = serde_json::Value::Array(arr.clone());
                if let Ok(blocks) = serde_json::from_value::<Vec<ContentBlock>>(val) {
                    MessageContent::Blocks(blocks)
                } else {
                    MessageContent::Text(v.to_string())
                }
            }
            other => MessageContent::Text(other.to_string()),
        }
    }
}

/// Loads an image file and returns a base64 data URI suitable for
/// multimodal LLM requests. Auto-detects the MIME type from the extension.
pub fn image_to_data_uri(path: &Path) -> std::result::Result<String, std::io::Error> {
    let bytes = std::fs::read(path)?;
    let mime = mime_from_ext(path);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

/// Loads tool/function definitions from a JSON file.
/// Format: `[{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]`
pub fn load_tools_file(path: &Path) -> std::result::Result<Vec<Tool>, std::io::Error> {
    let bytes = std::fs::read(path)?;
    let tools: Vec<Tool> = serde_json::from_slice(&bytes).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid tools JSON: {e}"),
        )
    })?;
    Ok(tools)
}

/// Best-effort MIME type from a file extension.
fn mime_from_ext(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("tiff") | Some("tif") => "image/tiff",
        Some("pdf") => "application/pdf",
        _ => "image/png", // default for unknown
    }
}

/// Clause length bounds for the splitter fed by [`LlmClient::stream_reply`]
/// (plan §6: `min_len ≈ 4`, `max_len ≈ 160`). Shared with the pipeline's
/// selftest, which splits the token stream directly.
pub(crate) const CLAUSE_MIN_LEN: usize = 4;
pub(crate) const CLAUSE_MAX_LEN: usize = 160;

/// One chat message in the conversation history.
///
/// The `content` field is [`MessageContent`] — plain text (`String`) for
/// text-only models, or a vec of [`ContentBlock`]s for multimodal models.
/// Serde's `#[serde(untagged)]` on `MessageContent` serialises a `Text`
/// variant as a JSON string and `Blocks` as a JSON array, exactly matching
/// the OpenAI chat-completions schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message content: plain text or multimodal blocks.
    pub content: MessageContent,
    /// Tool call id (required for `role: "tool"` messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls made by the assistant (set on `role: "assistant"` when
    /// the model requests function calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A function definition for tool/function calling (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function's parameters.
    pub parameters: serde_json::Value,
}

/// A tool available to the model for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Must be `"function"` for OpenAI-compatible APIs.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDef,
}

impl Tool {
    /// Creates a new function-type tool.
    pub fn function(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: name.to_string(),
                description: Some(description.to_string()),
                parameters,
            },
        }
    }
}

/// A function call extracted from a tool-call delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Function name (set in the first delta chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Partial or complete JSON arguments string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// One tool call from a model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool call index in the response (distinguishes parallel calls).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u64>,
    /// Unique call id (set in the first delta chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Must be `"function"`.
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// The function name + arguments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunction>,
}

/// A delta from one SSE chunk: either a text token or a tool call fragment.
#[derive(Debug, Clone)]
pub enum SseDelta {
    /// A plain text content token.
    Text(String),
    /// A tool call fragment (accumulated across chunks by index).
    ToolCall(ToolCall),
    /// End of stream sentinel (`data: [DONE]`).
    Done,
}

/// Streaming LLM client for an OpenAI-compatible `/chat/completions` API.
///
/// Implements [`LlmBackend`](crate::llm::LlmBackend). Supports multimodal
/// (vision) inputs via [`ContentBlock::Image`] and tool/function calling.
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    /// Optional bearer token for hosted providers (Ollama needs none).
    /// Never logged.
    api_key: Option<String>,
    max_history_turns: usize,
    system_prompt: String,
    history: Vec<Message>,
    /// Image paths for the next user turn (cleared after the turn).
    image_paths: Vec<std::path::PathBuf>,
    /// Tool/function definitions sent with each request.
    tools: Vec<Tool>,
    /// Maximum tool-calling round-trips before forcing a text response.
    max_tool_rounds: usize,
    /// Optional tool executor for running the function calls the model
    /// requests. When `None`, tool calls get a placeholder "not configured"
    /// result message.
    tool_executor: Option<Box<dyn ToolExecutor>>,
    /// Optional call-forwarding tool. When `Some`, a `forward_call` tool is
    /// registered and routed here (with the live conversation history) instead
    /// of through the subprocess executor.
    forward_tool: Option<crate::forward::ForwardTool>,
    /// Optional WASM plugin manager (`--plugins-dir`). When `Some`, each
    /// loaded plugin is auto-registered as a tool and its calls are routed
    /// here — run in a wasmtime sandbox — instead of through the subprocess
    /// executor.
    plugin_manager: Option<crate::plugins::PluginManager>,
    /// Optional sandboxed code-execution tool. When `Some`, a `code_exec`
    /// tool is registered and routed here instead of through the subprocess
    /// executor (`--code-exec-timeout`).
    sandbox_tool: Option<Arc<crate::sandbox::SandboxExecutor>>,
    /// Optional mesh node. When `Some` (mesh networking enabled), a
    /// `forward_call` with a `target` peer name is routed to that peer's HTTP
    /// endpoint via the mesh instead of the `forward_tool` endpoint.
    mesh: Option<crate::mesh::MeshNode>,
    /// Shared flag toggled on during tool execution for hold-music ducking.
    /// When `Some`, the pipeline can play hold music while tools run.
    pub(crate) hold_music_active: Option<Arc<AtomicBool>>,
    /// Optional conversation memory (`--memory-file`). When `Some`,
    /// remembered user preferences were injected into the seeded system
    /// prompt (see [`build_system_prompt`]), and each completed turn is
    /// summarized back here (see [`Self::maybe_summarize_turn`]).
    memory: Option<Arc<Mutex<MemoryStore>>>,
    /// Optional RAG index (`--rag-dir`). When `Some`, the top-k chunks for
    /// the current query are injected into the system prompt before each turn
    /// (see [`Self::refresh_system_message`]).
    rag: Option<RagStore>,
}

impl LlmClient {
    /// Creates a client; history is seeded with `system_prompt`. The HTTP
    /// client deliberately has no request timeout (streams are long-lived).
    ///
    /// `api_key` (from `--api-key` / `SKADOOSH_API_KEY`) unlocks hosted
    /// OpenAI-compatible providers: when `Some`, every request carries an
    /// `Authorization: Bearer <key>` header. Local Ollama needs no key —
    /// pass `None`. The key is never logged.
    pub fn new(
        base_url: &str,
        model: &str,
        system_prompt: &str,
        max_history_turns: usize,
        api_key: Option<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            max_history_turns,
            system_prompt: system_prompt.to_string(),
            history: vec![Message {
                role: "system".to_string(),
                content: MessageContent::Text(system_prompt.to_string()),
                tool_call_id: None,
                tool_calls: None,
            }],
            image_paths: Vec::new(),
            tools: Vec::new(),
            max_tool_rounds: 5,
            tool_executor: None,
            forward_tool: None,
            plugin_manager: None,
            sandbox_tool: None,
            mesh: None,
            hold_music_active: None,
            memory: None,
            rag: None,
        }
    }

    /// The model name this client requests (also its
    /// [`LlmBackend::name`](crate::llm::LlmBackend::name)).
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Sets image paths for the next user turn. Images are loaded and
    /// base64-encoded when the turn starts; the paths are cleared after
    /// the turn completes. Supported formats: png, jpg, gif, webp, bmp.
    pub fn with_images(mut self, paths: Vec<std::path::PathBuf>) -> Self {
        self.image_paths = paths;
        self
    }

    /// Clears any queued image paths.
    pub fn clear_images(&mut self) {
        self.image_paths.clear();
    }

    /// Sets the tool/function definitions sent with every request.
    /// Tools are registered once and included in every turn; use an
    /// empty vec to disable tool calling.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// Sets the maximum number of tool-calling round-trips (default: 5).
    /// The tool loop terminates early when the model returns a text
    /// response instead of tool calls, or when this limit is reached.
    pub fn with_max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = max;
        self
    }

    /// Sets the tool executor used to run the function calls the model
    /// requests during tool calling (see [`ToolExecutor`]). When none is
    /// configured, tool calls fall back to a placeholder "not configured"
    /// result message. `LlmClient::from_config` wires in a
    /// [`ShellExecutor`] automatically when `tools_file` is set.
    pub fn with_tool_executor(mut self, executor: Box<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    /// Sets a shared flag that is toggled `true` during tool execution
    /// and `false` when the tool loop completes. The pipeline can use this
    /// to play hold music while tools are running.
    pub fn with_hold_music(mut self, flag: Arc<AtomicBool>) -> Self {
        self.hold_music_active = Some(flag);
        self
    }

    /// Returns the hold-music active flag, if one was configured.
    pub fn hold_music_flag(&self) -> Option<&Arc<AtomicBool>> {
        self.hold_music_active.as_ref()
    }

    /// Sets a mesh node, enabling `forward_call` routing to discovered peers.
    /// When set, a `forward_call` carrying a `target` peer name is routed to
    /// that peer's HTTP endpoint via the mesh; a call without `target` still
    /// falls back to the `forward_tool` / `--forward-url` endpoint.
    pub fn with_mesh(mut self, node: crate::mesh::MeshNode) -> Self {
        self.mesh = Some(node);
        self
    }

    /// Sets a RAG index, enabling retrieval-augmented generation: before each
    /// turn the top-k chunks for the user query are injected into the system
    /// prompt. `LlmClient::from_config` builds one from `--rag-dir`
    /// automatically; this builder lets the SDK inject a prebuilt store.
    pub fn with_rag(mut self, store: RagStore) -> Self {
        self.rag = Some(store);
        self
    }

    /// Builds the config-default client (the one shared construction used
    /// by the binary's pipeline and the SDK facade).
    pub(crate) fn from_config(config: &crate::config::Config) -> Self {
        // Built-in tools are always available — no --tools-file needed.
        let mut tools: Vec<Tool> = crate::tools::builtin::definitions();
        if let Some(ref path) = config.tools_file {
            tools.extend(load_tools_file(path).unwrap_or_default());
        }
        // When tool definitions are configured, wire in a ShellExecutor so the
        // model's function calls actually run instead of returning a
        // placeholder result.
        let tool_executor: Option<Box<dyn ToolExecutor>> = if config.tools_file.is_some() {
            Some(Box::new(ShellExecutor::new()))
        } else {
            None
        };
        // When mesh networking is enabled, start the node and auto-register
        // the mesh-extended `forward_call` tool (which adds an optional
        // `target` peer-name argument). The mesh routes targeted forwards to
        // a peer's HTTP endpoint; calls without a target fall back to the
        // `forward_tool` / `--forward-url` endpoint below.
        let mesh = if config.mesh {
            let agent_name = config
                .agent_name
                .clone()
                .unwrap_or_else(|| format!("skadoosh-{}", std::process::id()));
            tracing::info!(
                agent = %agent_name,
                port = config.mesh_port,
                "mesh networking enabled"
            );
            tools.push(crate::forward::mesh_forward_tool_definition());
            Some(crate::mesh::MeshNode::start(&agent_name, config.mesh_port))
        } else {
            None
        };
        // When a forwarding endpoint is configured, build the executor that
        // relays the conversation (with live history) to that endpoint. When
        // mesh is already enabled it registered the (richer) tool definition,
        // so the plain one is only pushed here in the non-mesh case.
        let forward_tool = if let Some(ref url) = config.forward_url {
            tracing::info!(forward_url = %url, "call forwarding enabled");
            if mesh.is_none() {
                tools.push(crate::forward::forward_tool_definition());
            }
            Some(crate::forward::ForwardTool::new(
                crate::forward::ForwardConfig::new(url.clone()),
            ))
        } else {
            None
        };
        // Sandboxed code execution (`--code-exec-timeout`): auto-register the
        // `code_exec` tool and build the sandbox executor that runs snippets
        // in a restricted subprocess (or Docker container) and returns stdout,
        // stderr, and the exit code — like `--forward-url` does for
        // `forward_call`.
        let sandbox_tool = if let Some(secs) = config.code_exec_timeout {
            tracing::info!(
                timeout_secs = secs,
                sandbox = ?config.code_exec_sandbox,
                "code_exec tool enabled"
            );
            tools.push(crate::sandbox::code_exec_tool_definition());
            Some(Arc::new(crate::sandbox::SandboxExecutor::new(
                secs,
                config.code_exec_sandbox,
            )))
        } else {
            None
        };
        let hold_music_active = if config.hold_music {
            tracing::info!("hold music enabled: will play during tool execution");
            Some(Arc::new(AtomicBool::new(false)))
        } else {
            None
        };
        // Conversation memory (`--memory-file`): open the store (auto-loads
        // any existing file), inject remembered preferences into the system
        // prompt, and keep the handle to summarize each completed turn.
        let memory = if let Some(ref path) = config.memory_file {
            tracing::info!(path = %path.display(), "conversation memory enabled");
            Some(Arc::new(Mutex::new(MemoryStore::open(path))))
        } else {
            None
        };
        // Retrieval-augmented generation (`--rag-dir`): load and embed the docs
        // once at startup. Best-effort — a missing model/vocab or an empty dir
        // warns and leaves RAG off: retrieval is an enhancement, never a
        // startup blocker. The companion vocab path is derived from the model
        // path (`<stem>-vocab.txt`).
        let rag = if let Some(ref dir) = config.rag_dir {
            let model = &config.rag_model;
            let vocab = OnnxEmbedder::companion_vocab(model);
            if !model.is_file() || !vocab.is_file() {
                tracing::warn!(
                    rag_model = %model.display(),
                    rag_vocab = %vocab.display(),
                    "RAG enabled (--rag-dir) but the embedding model or vocab is missing; \
                     retrieval disabled (run scripts/download_models.sh --with-rag)"
                );
                None
            } else {
                match OnnxEmbedder::load(model, &vocab, crate::rag::DEFAULT_MAX_SEQ_LEN) {
                    Ok(embedder) => {
                        match RagStore::build(dir, Box::new(embedder), config.rag_top_k) {
                            Ok(store) => {
                                tracing::info!(chunks = store.len(), "RAG index ready");
                                Some(store)
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "RAG index build failed; retrieval disabled");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "RAG embedding model load failed; retrieval disabled"
                        );
                        None
                    }
                }
            }
        } else {
            None
        };
        // WASM plugins (`--plugins-dir`): load each `.wasm` plugin, register
        // its tool definition, and keep the manager so the tool-call loop can
        // route plugin calls into the wasmtime sandbox. When `--plugins-dir`
        // is unset, fall back to the default `~/.skadoosh/plugins/` directory
        // only if it already exists (otherwise no plugins load, silently).
        let plugin_manager = match config.plugins_dir.as_deref() {
            Some(dir) => match crate::plugins::PluginManager::load_dir(dir) {
                Ok(pm) => {
                    let count = pm.len();
                    tools.extend(pm.tool_definitions());
                    tracing::info!(dir = %dir.display(), count, "WASM plugins loaded");
                    Some(pm)
                }
                Err(e) => {
                    tracing::warn!(
                        dir = %dir.display(),
                        error = %e,
                        "failed to load plugins dir; continuing without plugins"
                    );
                    None
                }
            },
            None => crate::plugins::default_plugins_dir()
                .filter(|dir| dir.exists())
                .and_then(|dir| match crate::plugins::PluginManager::load_dir(&dir) {
                    Ok(pm) => {
                        let count = pm.len();
                        tools.extend(pm.tool_definitions());
                        tracing::info!(
                            dir = %dir.display(),
                            count,
                            "WASM plugins loaded from default dir"
                        );
                        Some(pm)
                    }
                    Err(e) => {
                        tracing::warn!(
                            dir = %dir.display(),
                            error = %e,
                            "failed to load default plugins dir"
                        );
                        None
                    }
                }),
        };
        // Model registry: when the configured model is known and the user
        // hasn't explicitly overridden the system prompt, use the model's
        // own prompt. An explicit `--system-prompt` always wins.
        let known = model::find_model(&config.llm_model);
        let base_prompt = if config.system_prompt == DEFAULT_SYSTEM_PROMPT {
            known
                .and_then(|m| m.system_prompt)
                .unwrap_or(&config.system_prompt)
        } else {
            &config.system_prompt
        };
        let system_prompt =
            build_system_prompt(base_prompt, config.agent_name.as_deref(), memory.as_deref());
        // When the model is emotion-aware but the user hasn't enabled
        // --tts-emotion, suggest it (StealthyLM-Emotive pairs with it).
        if model::is_emotion_aware(&config.llm_model) && !config.tts_emotion {
            tracing::info!(
                "model '{}' is emotion-aware — try --tts-emotion for expressive speech",
                config.llm_model
            );
        }
        Self {
            image_paths: config.images.clone(),
            tools,
            max_tool_rounds: config.max_tool_rounds,
            tool_executor,
            forward_tool,
            plugin_manager,
            sandbox_tool,
            mesh,
            hold_music_active,
            memory,
            rag,
            ..Self::new(
                &config.llm_url,
                &config.llm_model,
                &system_prompt,
                config.max_history_turns,
                config.api_key.clone(),
            )
        }
    }

    /// Streams a reply: appends the user message, POSTs
    /// `{base}/chat/completions` with `stream: true`, buffers partial SSE
    /// lines, feeds the [`ClauseSplitter`], and
    /// sends completed clauses on `clauses`.
    ///
    /// `tokio::select!` on `cancel`: on cancellation the in-flight clause is
    /// discarded, the *partial assistant reply is dropped from history* (the
    /// user never heard it), and
    /// [`LlmError::Cancelled`] is returned. History is still truncated to the
    /// last `max_history_turns` turns on this path so repeated barge-ins
    /// cannot pile up unanswered user messages.
    /// At `data: [DONE]` (or a clean connection close) the splitter is
    /// flushed and the completed assistant reply is appended to history
    /// (truncated to the last `max_history_turns` user/assistant turns).
    pub async fn stream_reply(
        &mut self,
        user: &str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Result<()> {
        let images = std::mem::take(&mut self.image_paths);
        let user_content = if images.is_empty() {
            MessageContent::Text(user.to_string())
        } else {
            let mut blocks = vec![ContentBlock::Text {
                text: user.to_string(),
            }];
            for path in &images {
                match image_to_data_uri(path) {
                    Ok(data_uri) => blocks.push(ContentBlock::Image {
                        image_url: ImageUrl {
                            url: data_uri,
                            detail: Some("auto".to_string()),
                        },
                    }),
                    Err(e) => {
                        tracing::warn!(path=%path.display(), error=%e, "failed to load image; skipping");
                    }
                }
            }
            if blocks.len() == 1 {
                // All images failed to load; fall back to text-only.
                MessageContent::Text(user.to_string())
            } else {
                MessageContent::Blocks(blocks)
            }
        };
        self.history.push(Message {
            role: "user".to_string(),
            content: user_content,
            tool_call_id: None,
            tool_calls: None,
        });
        // RAG: inject the top-k chunks for this query into the system message
        // before the request is sent (no-op when `--rag-dir` is unset).
        self.refresh_system_message(user);
        let result = self.stream_reply_inner(user, &clauses, &cancel).await;
        if matches!(result, Err(SkadooshError::Llm(LlmError::Cancelled))) {
            self.truncate_history();
        }
        result
    }

    /// The streaming body of `stream_reply`, split out so the public method
    /// can run book-keeping (history truncation) on the cancellation path
    /// no matter which `select!` observed the token.
    async fn stream_reply_inner(
        &mut self,
        user: &str,
        clauses: &mpsc::Sender<String>,
        cancel: &CancellationToken,
    ) -> Result<()> {
        let tool_count = self.tools.len();
        let mut total_reply = String::new();

        for tool_round in 0..=self.max_tool_rounds {
            // On the final round, force text by omitting tools.
            let send_tools = tool_round < self.max_tool_rounds && tool_count > 0;

            let mut body = serde_json::json!({
                "model": self.model,
                "messages": self.history,
                "stream": true,
            });
            if send_tools {
                body["tools"] = serde_json::to_value(&self.tools).unwrap_or_default();
            }

            let url = format!("{}/chat/completions", self.base_url);
            let mut request = self.http.post(&url).json(&body);
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let resp = tokio::select! {
                _ = cancel.cancelled() => return Err(LlmError::Cancelled.into()),
                r = request.send() => r.map_err(LlmError::Http)?,
            };
            let resp = ensure_success(resp).await?;

            let mut stream = resp.bytes_stream();
            let mut splitter = ClauseSplitter::new(CLAUSE_MIN_LEN, CLAUSE_MAX_LEN);
            let mut round_reply = String::new();
            let mut tool_calls: BTreeMap<usize, ToolCall> = BTreeMap::new();
            let mut lines = SseLineBuffer::default();
            let mut done = false;
            let mut eof = false;

            while !done && !eof {
                let chunk = tokio::select! {
                    _ = cancel.cancelled() => return Err(LlmError::Cancelled.into()),
                    c = stream.next() => c,
                };
                match chunk {
                    Some(Ok(bytes)) => {
                        if !lines.feed(&bytes) {
                            return Err(
                                LlmError::Sse("SSE line exceeded maximum size".into()).into()
                            );
                        }
                    }
                    Some(Err(e)) => return Err(LlmError::Http(e).into()),
                    None => {
                        lines.close();
                        eof = true;
                    }
                }
                while let Some(line) = lines.next_line() {
                    match parse_sse_delta(&line) {
                        None => {}
                        Some(Ok(SseDelta::Done)) => {
                            done = true;
                            break;
                        }
                        Some(Ok(SseDelta::Text(token))) => {
                            total_reply.push_str(&token);
                            round_reply.push_str(&token);
                            for clause in splitter.push(&token) {
                                if !send_clause(clauses, cancel, clause).await? {
                                    tracing::debug!("clauses receiver dropped");
                                    return Ok(());
                                }
                            }
                        }
                        Some(Ok(SseDelta::ToolCall(tc))) => {
                            let idx = tc.index.unwrap_or(0) as usize;
                            let entry = tool_calls.entry(idx).or_insert_with(|| ToolCall {
                                index: None,
                                id: None,
                                call_type: None,
                                function: None,
                            });
                            if tc.index.is_some() {
                                entry.index = tc.index;
                            }
                            if tc.id.is_some() {
                                entry.id = tc.id;
                            }
                            if tc.call_type.is_some() {
                                entry.call_type = tc.call_type;
                            }
                            if let Some(ref f) = tc.function {
                                let ef = entry.function.get_or_insert(ToolCallFunction {
                                    name: None,
                                    arguments: None,
                                });
                                if f.name.is_some() {
                                    ef.name = f.name.clone();
                                }
                                if let Some(ref args) = f.arguments {
                                    ef.arguments =
                                        Some(ef.arguments.take().unwrap_or_default() + args);
                                }
                            }
                        }
                        Some(Err(e)) => {
                            tracing::warn!(error = %e, "skipping malformed SSE data line");
                        }
                    }
                }
            }

            // Flush splitter remainder.
            if let Some(rest) = splitter.flush() {
                if !send_clause(clauses, cancel, rest).await? {
                    return Ok(());
                }
            }

            // No tool calls → turn complete.
            if tool_calls.is_empty() {
                self.maybe_summarize_turn(user, &round_reply);
                self.history.push(Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text(round_reply),
                    tool_call_id: None,
                    tool_calls: None,
                });
                self.truncate_history();
                return Ok(());
            }

            // Tool calls received: record them + add tool results.
            let calls: Vec<ToolCall> = tool_calls.into_values().collect();
            tracing::info!(
                round = tool_round,
                count = calls.len(),
                "tool calls received"
            );

            // Signal hold music: tools are about to run.
            if let Some(ref flag) = self.hold_music_active {
                flag.store(true, Ordering::Relaxed);
            }

            // Assistant message with tool_calls.
            self.history.push(Message {
                role: "assistant".to_string(),
                content: MessageContent::Text(String::new()),
                tool_call_id: None,
                tool_calls: Some(calls.clone()),
            });

            // Tool call markers for observability.
            for tc in &calls {
                let name = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.name.as_deref())
                    .unwrap_or("?");
                let args = tc
                    .function
                    .as_ref()
                    .and_then(|f| f.arguments.as_deref())
                    .unwrap_or("{}");
                let _ = clauses.send(format!("\x00TOOL:{name}:{args}")).await;
            }

            // Tool result messages: run calls via the configured executor.
            // When multiple calls are present, execute them in parallel.
            let has_executor = self.tool_executor.is_some()
                || self.forward_tool.is_some()
                || self.mesh.is_some()
                || self.sandbox_tool.is_some()
                || self.plugin_manager.is_some();
            if has_executor {
                // Collect configured tool names for validation — rejects
                // tool calls the model hallucinates that weren't defined.
                let known: std::collections::HashSet<&str> = self
                    .tools
                    .iter()
                    .map(|t| t.function.name.as_str())
                    .collect();

                let batch: Vec<(String, String, String)> = calls
                    .iter()
                    .map(|tc| {
                        let call_id = tc.id.clone().unwrap_or_else(|| "call_unknown".to_string());
                        let name = tc
                            .function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .unwrap_or("")
                            .to_string();
                        let args = tc
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.as_deref())
                            .unwrap_or("{}")
                            .to_string();
                        (name, args, call_id)
                    })
                    .collect();

                if batch.len() > 1 {
                    tracing::info!(count = batch.len(), "tool calls received");
                }

                let mut results: BTreeMap<String, std::result::Result<String, SkadooshError>> =
                    BTreeMap::new();

                // Route `forward_call`: when a peer `target` is present and
                // the mesh is enabled, route to that peer's HTTP endpoint;
                // otherwise fall back to the ForwardTool (--forward-url
                // endpoint). The live conversation history is forwarded in
                // both cases so the recipient gets full context. Unknown tool
                // names are warned and skipped below.
                let mesh = self.mesh.as_ref();
                if self.forward_tool.is_some() || mesh.is_some() {
                    let current_query = last_user_text(&self.history);
                    let mut forwarded = 0usize;
                    for (name, args, call_id) in &batch {
                        if name == crate::forward::FORWARD_TOOL_NAME
                            && known.contains(name.as_str())
                        {
                            let (target, reason, summary) =
                                crate::forward::parse_forward_args_full(args);
                            let res = if let (Some(target), Some(mesh)) = (&target, mesh) {
                                mesh.forward_to_peer(
                                    target,
                                    &self.history,
                                    &current_query,
                                    &reason,
                                    &summary,
                                )
                                .await
                            } else if let Some(ref forward) = self.forward_tool {
                                forward
                                    .forward(&self.history, &current_query, &reason, &summary)
                                    .await
                            } else {
                                Err(SkadooshError::Other(anyhow::anyhow!(
                                    "forward_call set target {target:?} but mesh is not enabled \
                                     and no --forward-url endpoint is configured"
                                )))
                            };
                            forwarded += 1;
                            if let Err(ref e) = res {
                                tracing::warn!(tool = %name, error = %e, "forward call failed");
                            }
                            results.insert(call_id.clone(), res);
                        }
                    }
                    if forwarded > 0 {
                        tracing::info!(count = forwarded, "forwarded calls");
                    }
                }

                // Route `code_exec` to the SandboxExecutor (parallel via tokio
                // tasks, mirroring the shell executor below). The sandbox
                // returns stdout/stderr/exit_code as JSON.
                if let Some(ref sandbox) = self.sandbox_tool {
                    let mut sandboxed = 0usize;
                    let mut handles = Vec::new();
                    for (name, args, call_id) in &batch {
                        if name == crate::sandbox::CODE_EXEC_TOOL_NAME
                            && known.contains(name.as_str())
                        {
                            sandboxed += 1;
                            let sb = Arc::clone(sandbox);
                            let (n, a, cid) = (name.clone(), args.clone(), call_id.clone());
                            handles.push(tokio::spawn(async move { (cid, sb.execute(&n, &a)) }));
                        }
                    }
                    for h in handles {
                        match h.await {
                            Ok((cid, r)) => {
                                results.insert(cid, r);
                            }
                            Err(e) => tracing::error!(error = %e, "code_exec task panicked"),
                        }
                    }
                    if sandboxed > 0 {
                        tracing::info!(count = sandboxed, "executed code_exec calls in sandbox");
                    }
                }

                // Route plugin tool calls into the wasmtime sandbox via the
                // PluginManager. Each plugin's tool definition was registered in
                // `from_config`; the model's JSON arguments are passed verbatim
                // as the plugin's `input_json`. Runs inline (plugins are
                // CPU-bounded by the fuel budget), mirroring the forward
                // routing above.
                if let Some(ref plugins) = self.plugin_manager {
                    let mut ran = 0usize;
                    for (name, args, call_id) in &batch {
                        if plugins.has(name) && known.contains(name.as_str()) {
                            ran += 1;
                            let res = plugins.execute(name, args);
                            if let Err(ref e) = res {
                                tracing::warn!(tool = %name, error = %e, "plugin execution failed");
                            }
                            results.insert(call_id.clone(), res);
                        }
                    }
                    if ran > 0 {
                        tracing::info!(count = ran, "plugin tool calls executed");
                    }
                }

                // Route built-in tools (calculator, datetime, remember, recall).
                // These run in-process — no subprocess, no API.
                {
                    let mut builtin = 0usize;
                    for (name, args, call_id) in &batch {
                        if !crate::tools::builtin::is_builtin(name) {
                            continue;
                        }
                        // Try memory-less first (calculator, datetime).
                        let res = crate::tools::builtin::execute(name, args);
                        let res = if let Some(result) = res {
                            result
                        } else if let Some(ref mem) = self.memory {
                            // Memory-backed (remember, recall).
                            crate::tools::builtin::execute_with_memory(name, args, mem)
                                .unwrap_or_else(|| {
                                    Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                                        "built-in tool '{name}': memory not available"
                                    )))
                                })
                        } else {
                            Err(crate::error::SkadooshError::Other(anyhow::anyhow!(
                                "built-in tool '{name}' needs --memory-file"
                            )))
                        };
                        results.insert(call_id.clone(), res);
                        builtin += 1;
                    }
                    if builtin > 0 {
                        tracing::info!(count = builtin, "built-in tool calls executed");
                    }
                }

                // Remaining configured tool calls run as subprocesses through
                // the existing parallel executor.
                let shell_batch: Vec<(String, String, String)> = batch
                    .iter()
                    .filter(|(name, _, _)| {
                        if name == crate::forward::FORWARD_TOOL_NAME
                            || name == crate::sandbox::CODE_EXEC_TOOL_NAME
                            || self.plugin_manager.as_ref().is_some_and(|p| p.has(name))
                            || crate::tools::builtin::is_builtin(name)
                        {
                            false
                        } else if known.contains(name.as_str()) {
                            true
                        } else {
                            tracing::warn!(tool=%name, "rejected unknown tool call");
                            false
                        }
                    })
                    .cloned()
                    .collect();

                if self.tool_executor.is_some() && !shell_batch.is_empty() {
                    if shell_batch.len() > 1 {
                        tracing::info!(
                            count = shell_batch.len(),
                            "executing tool calls in parallel"
                        );
                    }
                    let shell_results = execute_parallel(shell_batch).await;
                    for (call_id, result) in shell_results {
                        results.insert(call_id, result);
                    }
                }

                // Re-assemble results in the original call order for history.
                for tc in &calls {
                    let call_id = tc.id.clone().unwrap_or_else(|| "call_unknown".to_string());
                    let content = match results.get(&call_id) {
                        Some(Ok(out)) => out.clone(),
                        Some(Err(e)) => {
                            let name = tc
                                .function
                                .as_ref()
                                .and_then(|f| f.name.as_deref())
                                .unwrap_or("?");
                            tracing::warn!(tool = %name, error = %e, "tool execution failed");
                            let body = serde_json::to_string(&e.to_string())
                                .unwrap_or_else(|_| "\"<unprintable error>\"".to_string());
                            format!("{{\"error\":{body}}}")
                        }
                        None => {
                            tracing::warn!(call_id = %call_id, "tool call result missing");
                            "{\"error\":\"tool execution result missing\"}".to_string()
                        }
                    };
                    self.history.push(Message {
                        role: "tool".to_string(),
                        content: MessageContent::Text(content),
                        tool_call_id: Some(call_id),
                        tool_calls: None,
                    });
                }
            } else {
                // No executor configured: placeholder results.
                for tc in &calls {
                    let call_id = tc.id.clone().unwrap_or_else(|| "call_unknown".to_string());
                    self.history.push(Message {
                        role: "tool".to_string(),
                        content: MessageContent::Text(
                            "{\"error\":\"tool execution not configured; respond with text\"}"
                                .to_string(),
                        ),
                        tool_call_id: Some(call_id),
                        tool_calls: None,
                    });
                }
            }

            // Tool execution done: clear hold music flag.
            if let Some(ref flag) = self.hold_music_active {
                flag.store(false, Ordering::Relaxed);
            }
        }

        // Max rounds exhausted: record whatever text we got.
        self.maybe_summarize_turn(user, &total_reply);
        self.history.push(Message {
            role: "assistant".to_string(),
            content: MessageContent::Text(total_reply),
            tool_call_id: None,
            tool_calls: None,
        });
        self.truncate_history();
        Ok(())
    }

    /// Current conversation history (system prompt first).
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Resets the conversation to just the system prompt (the
    /// [`LlmBackend`](crate::llm::LlmBackend) history-discipline hook).
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.history.push(Message {
            role: "system".to_string(),
            content: MessageContent::Text(self.system_prompt.clone()),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    /// Keeps the system message plus the last `max_history_turns`
    /// user/assistant turns (2 messages per turn). Runs after a successful
    /// assistant append and after a cancellation, so barge-in-heavy sessions
    /// stay bounded (the cancel path may cut right after an unanswered user
    /// message, which is fine — the budget, not pairing, is the invariant).
    fn truncate_history(&mut self) {
        let keep = 2 * self.max_history_turns;
        if self.history.len() > 1 + keep {
            let drop = self.history.len() - 1 - keep;
            self.history.drain(1..=drop);
        }
    }

    /// When RAG is enabled, rewrites the seeded system message
    /// (`history[0]`) to carry the top-k retrieved chunks for `query`,
    /// formatted as
    /// `"{system_prompt}\n\nRelevant context:\n{chunks}\n\nAnswer using this
    /// context if helpful."`. With no chunks (empty index or query-embedding
    /// failure) the base system prompt is restored, so a prior turn's context
    /// never leaks into the next. No-op when RAG is disabled.
    fn refresh_system_message(&mut self, query: &str) {
        // Borrow only `rag` here; the borrow ends before we touch `history`.
        let rag_context = if let Some(rag) = self.rag.as_mut() {
            let top_k = rag.top_k;
            let chunks = rag.search(query, top_k);
            if chunks.is_empty() {
                None
            } else {
                Some(format!(
                    "Relevant context:\n{}\n\nAnswer using this context if helpful.",
                    chunks.join("\n")
                ))
            }
        } else {
            None
        };

        // Only rewrite the system message when RAG is configured, so non-RAG
        // clients pay nothing and `history[0]` is left untouched.
        if self.rag.is_some() {
            if let Some(msg) = self.history.first_mut() {
                let content = match rag_context {
                    Some(ctx) => format!("{}\n\n{}", self.system_prompt, ctx),
                    None => self.system_prompt.clone(),
                };
                msg.content = MessageContent::Text(content);
            }
        }
    }

    /// If conversation memory is configured, appends a summary of this turn
    /// (`user` → `reply`) to the store (which auto-saves). Best-effort: a
    /// poisoned mutex is warned, not fatal, so memory can never break a turn.
    fn maybe_summarize_turn(&self, user: &str, reply: &str) {
        if let Some(memory) = &self.memory {
            match memory.lock() {
                Ok(mut store) => store.summarize_turn(user, reply),
                Err(err) => tracing::warn!(error = %err, "memory lock poisoned; turn not saved"),
            }
        }
    }
}

/// Returns the text of the most recent `user` message in `history`, or the
/// empty string if there is none. Used as the `current_query` forwarded to an
/// external service.
fn last_user_text(history: &[Message]) -> String {
    history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_text())
        .unwrap_or("")
        .to_string()
}

/// Builds the effective system prompt from the configured base:
///
/// * when `agent_name` is set (`--agent-name`), it is prefixed with
///   `"You are {name}, a helpful voice assistant. "`; and
/// * when a memory store is configured (`--memory-file`), any remembered
///   preferences are appended as
///   `"The user previously mentioned: {key: value; …}"`.
///
/// Both are best-effort: a missing name or memory leaves the base prompt
/// untouched.
fn build_system_prompt(
    base: &str,
    agent_name: Option<&str>,
    memory: Option<&Mutex<MemoryStore>>,
) -> String {
    let mut prompt = match agent_name.map(str::trim).filter(|name| !name.is_empty()) {
        Some(name) => format!("You are {name}, a helpful voice assistant. {base}"),
        None => base.to_string(),
    };
    if let Some(memory) = memory {
        if let Ok(store) = memory.lock() {
            if let Some(prefs) = store.preferences_summary() {
                prompt.push_str("\nThe user previously mentioned: ");
                prompt.push_str(&prefs);
            }
        }
    }
    prompt
}

/// Passes a success response through; a non-success status becomes
/// [`LlmError::Api`] with the response body (truncated to 1024 chars).
pub(crate) async fn ensure_success(resp: reqwest::Response) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let text = resp.text().await.unwrap_or_default();
    Err(LlmError::Api {
        status: status.as_u16(),
        body: text.chars().take(1024).collect(),
    }
    .into())
}

/// Sends one completed clause, watching `cancel` while waiting for channel
/// capacity. Returns `Ok(false)` when the receiver is gone (pipeline
/// shutdown: stop streaming quietly), `Err(Cancelled)` on cancellation.
async fn send_clause(
    clauses: &mpsc::Sender<String>,
    cancel: &CancellationToken,
    clause: String,
) -> std::result::Result<bool, LlmError> {
    tokio::select! {
        _ = cancel.cancelled() => Err(LlmError::Cancelled),
        sent = clauses.send(clause) => Ok(sent.is_ok()),
    }
}

/// Safety cap: a single SSE line must not exceed 1 MiB. Larger chunks from a
/// compromised or broken server are rejected to prevent unbounded memory growth.
const SSE_MAX_LINE_BYTES: usize = 1_048_576;

/// Buffers raw SSE bytes and yields complete lines. A multi-byte UTF-8 char
/// split across stream chunks stays in the byte buffer until the
/// terminating `\n` arrives, so lines are always valid boundaries. Shared
/// by [`LlmClient::stream_reply`] and the pipeline's selftest, which drives
/// the SSE stream directly.
///
/// Call [`close`](Self::close) at end of stream: afterwards [`next_line`](Self::next_line)
/// yields the unterminated trailing bytes once as a final line (a server
/// that closes without a trailing `\n` loses no content), then `None`.
#[derive(Default)]
pub(crate) struct SseLineBuffer {
    buf: Vec<u8>,
    eof: bool,
}

impl SseLineBuffer {
    /// Appends one stream chunk. Returns `false` when the buffered data would
    /// exceed [`SSE_MAX_LINE_BYTES`] — the caller should abort the stream.
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> bool {
        if self.buf.len() + chunk.len() > SSE_MAX_LINE_BYTES {
            return false;
        }
        self.buf.extend_from_slice(chunk);
        true
    }

    /// Marks end of stream; idempotent.
    pub(crate) fn close(&mut self) {
        self.eof = true;
    }

    /// Pops the next complete `\n`-terminated line (lossy UTF-8 decode).
    /// After [`close`](Self::close), also yields the unterminated remainder
    /// once, then returns `None`.
    pub(crate) fn next_line(&mut self) -> Option<String> {
        if let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=nl).collect();
            return Some(String::from_utf8_lossy(&line_bytes).into_owned());
        }
        if self.eof && !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            return Some(String::from_utf8_lossy(&rest).into_owned());
        }
        None
    }
}

/// Parses one SSE line, tolerantly (OpenAI-compatible servers vary):
///
/// * blank lines, comments (`: ...`), `event:`/`id:`/`retry:` fields, and
///   data chunks without text content (role deltas, finish markers) → `None`;
/// * `data: [DONE]` → `Some(Ok(None))` (end sentinel);
/// * `data: {...}` with content → `Some(Ok(Some(token)))`;
/// * malformed JSON → `Some(Err(..))`; the caller warns and skips the line.
///
/// For tool-calling support, use [`parse_sse_delta`] instead.
pub fn parse_sse_line(line: &str) -> Option<Result<Option<String>>> {
    match parse_sse_delta(line)? {
        Ok(SseDelta::Text(t)) => Some(Ok(Some(t))),
        Ok(SseDelta::Done) => Some(Ok(None)),
        Ok(SseDelta::ToolCall(_)) => None, // no text, no tool handling in legacy path
        Err(e) => Some(Err(e)),
    }
}

/// Parses one SSE line into an [`SseDelta`] — text content token, tool
/// call fragment, or the `[DONE]` sentinel. Handles both text deltas
/// (`delta.content`) and tool-call deltas (`delta.tool_calls[]`).
pub fn parse_sse_delta(line: &str) -> Option<Result<SseDelta>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with(':') {
        return None;
    }
    let data = line.strip_prefix("data:")?;
    let data = data.trim();
    if data == "[DONE]" {
        return Some(Ok(SseDelta::Done));
    }
    let parsed: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(LlmError::Sse(format!("malformed SSE data: {e}")).into()));
        }
    };
    let choices = parsed.get("choices")?.as_array()?;
    let choice = choices.first()?;
    let delta = choice.get("delta")?;

    // Tool calls take priority: if `delta.tool_calls` is present, parse them.
    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        if let Some(tc) = tool_calls.first() {
            if let Ok(tool_call) = serde_json::from_value::<ToolCall>(tc.clone()) {
                return Some(Ok(SseDelta::ToolCall(tool_call)));
            }
        }
        return None;
    }

    // Plain text content.
    let token = delta.get("content")?.as_str()?;
    if token.is_empty() {
        None
    } else {
        Some(Ok(SseDelta::Text(token.to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// `from_config` auto-registers the `forward_call` tool definition and a
    /// [`crate::forward::ForwardTool`] when `forward_url` is set, and
    /// registers neither when it is unset.
    #[test]
    fn from_config_registers_forward_tool_when_url_set() {
        let config = Config {
            forward_url: Some("http://example/forward".to_string()),
            ..Default::default()
        };

        let client = LlmClient::from_config(&config);
        let names: Vec<&str> = client
            .tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert!(
            names.contains(&crate::forward::FORWARD_TOOL_NAME),
            "forward_call tool must be registered: {names:?}"
        );
        assert!(
            client.forward_tool.is_some(),
            "ForwardTool executor must be wired when --forward-url is set"
        );

        // Without forward_url, neither the tool nor the executor is present.
        let bare = LlmClient::from_config(&Config::default());
        let bare_names: Vec<&str> = bare
            .tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert!(
            !bare_names.contains(&crate::forward::FORWARD_TOOL_NAME),
            "forward_call must not be registered without --forward-url: {bare_names:?}"
        );
        assert!(bare.forward_tool.is_none());
    }

    /// `from_config` auto-registers the `code_exec` tool definition and a
    /// [`crate::sandbox::SandboxExecutor`] when `code_exec_timeout` is set,
    /// and registers neither when it is unset.
    #[test]
    fn from_config_registers_code_exec_when_timeout_set() {
        let config = Config {
            code_exec_timeout: Some(30),
            ..Default::default()
        };

        let client = LlmClient::from_config(&config);
        let names: Vec<&str> = client
            .tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert!(
            names.contains(&crate::sandbox::CODE_EXEC_TOOL_NAME),
            "code_exec tool must be registered: {names:?}"
        );
        assert!(
            client.sandbox_tool.is_some(),
            "SandboxExecutor must be wired when --code-exec-timeout is set"
        );

        // Without code_exec_timeout, neither the tool nor the executor is present.
        let bare = LlmClient::from_config(&Config::default());
        let bare_names: Vec<&str> = bare
            .tools
            .iter()
            .map(|t| t.function.name.as_str())
            .collect();
        assert!(
            !bare_names.contains(&crate::sandbox::CODE_EXEC_TOOL_NAME),
            "code_exec must not be registered without --code-exec-timeout: {bare_names:?}"
        );
        assert!(bare.sandbox_tool.is_none());
    }
}
