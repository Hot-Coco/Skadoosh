# 🥤 Milkshake

Clean web chat UI for [StealthyLM](https://huggingface.co/StealthyML/StealthyLM-Emotive) with
MCP tool support. Part of the [Skadoosh](https://github.com/Hot-Coco/Skadoosh) family.

## Quick Start

```bash
# Start Ollama with StealthyLM
ollama serve

# Start Milkshake
cd milkshake
cargo run

# Open http://localhost:3000
```

Or with custom settings:

```bash
cargo run -- --port 8080 --ollama-url http://other-host:11434 --model llama3.2
```

## Features

- **Streaming chat** — responses appear word-by-word via SSE
- **StealthyLM by default** — purpose-built for voice and chat
- **Model switcher** — pick any Ollama model from the sidebar
- **Custom system prompt** — override the assistant's personality
- **MCP tools** — connect Model Context Protocol servers to add tool calling
- **Zero dependencies on the frontend** — single HTML file, no npm, no build step
- **Dark theme** — easy on the eyes, stays out of the way

## MCP (Model Context Protocol)

Click **+ Add MCP Server** in the sidebar, enter a name and the MCP server's URL.
Milkshake fetches the server's tool list and includes them in every chat request.
When the model calls a tool, the MCP server executes it.

```
# Example: connect a local MCP server
# MCP Name: filesystem
# MCP URL:  http://localhost:3001
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILKSHAKE_PORT` | `3000` | Server listen port |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `MILKSHAKE_MODEL` | `stealthylm` | Default model name |

## Architecture

```
Browser                     Milkshake (Rust/Axum)         Ollama
  │                              │                          │
  ├─ POST /api/chat ────────────►│                          │
  │                              ├─ POST /api/chat ────────►│
  │                              │◄─ SSE stream ────────────┤
  │◄─ SSE stream ────────────────┤                          │
  │                              │                          │
  ├─ POST /api/mcp/connect ─────►│                          │
  │                              ├─ tools/list ──► MCP Svr  │
  │                              │◄─ tool defs ───┤         │
```

## License

MIT OR Apache-2.0 (same as Skadoosh).
