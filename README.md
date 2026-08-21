# skadoosh

[![crates.io](https://img.shields.io/crates/v/skadoosh.svg)](https://crates.io/crates/skadoosh)
[![docs.rs](https://docs.rs/skadoosh/badge.svg)](https://docs.rs/skadoosh)
[![CI](https://github.com/Hot-Coco/Skadoosh/actions/workflows/ci.yml/badge.svg)](https://github.com/Hot-Coco/Skadoosh/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**A fast, fully local voice agent in Rust.** Mic in, Silero VAD listens, Whisper transcribes, your LLM replies, and Kokoro speaks each clause as it lands. Interrupt mid-sentence — it shuts up instantly. No cloud, no fuss. OpenAI-compatible backends also work.

Use it as a binary or as a library with pluggable engines.

## Install

```bash
cargo add skadoosh          # library
cargo install skadoosh      # binary (needs cmake, clang, ALSA dev headers)
```

## 5-line SDK

```rust
use skadoosh::{Agent, Config, Result};

fn main() -> Result<()> {
    Agent::builder().config(Config::default()).build()?.run()
}
```

That's a full voice agent. See `examples/` for text mode, mock engines, and custom backends.

## Quickstart (binary)

```bash
ollama pull qwen2.5:0.5b && ollama serve
./scripts/download_models.sh           # VAD + Whisper (~76 MB)
./scripts/download_models.sh --with-kokoro  # + Kokoro TTS (~320 MB, needs espeak-ng)
skadoosh
```

Talk. It answers. Talk over it — it stops.

```bash
skadoosh --repl                         # text chat mode
skadoosh --say "Hello." --out-wav hi.wav
skadoosh --selftest tests/data/jfk.wav  # headless end-to-end test
```

## Key features

- **Clause-level streaming** — TTS starts on the first clause, not the full reply
- **Real barge-in** — speak anytime, old audio stops in ~5–10 ms, lock-free
- **No cloud** — everything runs locally; OpenAI-compatible APIs also supported
- **Plug any engine** — implement `SttEngine`, `LlmBackend`, or `TtsEngine` and drop them in
- **GPU acceleration** — CUDA, CoreML, DirectML, ROCm via Cargo features (`gpu-cuda`, etc.)
- **Tool calling** — LLMs run shell commands, results feed back into conversation
- **Emotion-aware TTS** — keyword-driven speed modulation (excited/calm/neutral)
- **Hold music** — chord progression plays during long tool executions
- **`#![forbid(unsafe_code)]`** — zero unsafe, rustls everywhere

## Configuration

Every flag has a `SKADOOSH_` env var. `--api-key` is never logged.

| Flag | Default | What it does |
|---|---|---|
| `--llm-url` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `--llm-model` | `qwen2.5:0.5b` | Model name |
| `--api-key` | — | Bearer token for hosted providers |
| `--whisper-model` | `models/ggml-tiny.en.bin` | STT model |
| `--tts-model` | — | Kokoro ONNX (absent → sine mock) |
| `--mock-tts` | off | Force sine-wave TTS instead of Kokoro |
| `--silence-ms` | `300` | How much silence ends a segment |
| `--vad-threshold` | `0.5` | Speech probability threshold |
| `--push-to-talk` | off | Keyboard toggle for recording |
| `--wake-word` | — | Only process speech containing this |
| `--tts-voice` | `af` | Kokoro voice key |
| `--tts-speed` | `1.0` | Playback speed (0.5–2.0) |
| `--whisper-model-size` | `tiny` | Whisper model size for API backends |
| `--hold-music` | off | Chord progression during tool execution |
| `--tts-emotion` | off | Emotion-aware speech speed |
| `--selftest <wav>` | — | Headless end-to-end test |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
