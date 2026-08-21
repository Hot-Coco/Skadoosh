# Skadoosh — Setup Guide

Skadoosh is a fully local voice agent in Rust: mic → Silero VAD → Whisper STT →
streaming LLM → ONNX TTS → playback, with barge-in.

## 1. Prerequisites

- **Rust 1.88 or newer** — install via [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Ollama** — the default LLM backend (an OpenAI-compatible server on
  `http://localhost:11434/v1`). Install from <https://ollama.com>.
- **System audio libs** — only needed for voice mode (the default `audio`
  feature). On Debian/Ubuntu:
  ```bash
  sudo apt-get install -y cmake clang libclang-dev libasound2-dev pkg-config espeak-ng
  ```
  `espeak-ng` is only for Kokoro TTS; macOS uses CoreAudio (bundled), so skip
  `libasound2-dev` there. Text-only builds need none of these (§2).

## 2. Install

Install the binary from crates.io:

```bash
cargo install skadoosh
```

Or build from a clone:

```bash
git clone https://github.com/Hot-Coco/Skadoosh && cd Skadoosh
cargo build --release    # binary at target/release/skadoosh
```

> **No audio libs?** Build with `--no-default-features` to drop the `audio`
> feature (mic/VAD/STT/playback); the SDK, `--repl`, and `say_to_wav` remain:
> ```bash
> cargo install skadoosh --no-default-features
> ```

## 3. Basic chat (text-only)

This needs no models and no audio device — just an LLM server. Start Ollama,
pull the default model, and run the text REPL:

```bash
ollama pull qwen2.5:0.5b
ollama serve
skadoosh --repl
```

Type a message, press Enter, and the reply streams back as text. Exit with
`/quit` or `ctrl-d`.

## 4. Voice mode (full pipeline)

The download script lives in the repo (it is excluded from the published crate),
so `cargo install` users should clone first. It is idempotent — it skips files
already present:

```bash
git clone https://github.com/Hot-Coco/Skadoosh && cd Skadoosh
./scripts/download_models.sh                # Silero VAD + Whisper tiny.en (~76 MB)
./scripts/download_models.sh --with-kokoro  # + Kokoro TTS (~320 MB, needs espeak-ng)
./scripts/download_models.sh --with-rag     # + sentence embedder for --rag-dir (~90 MB)
```

It writes to the repo's `models/` directory, so launch the binary from the repo
root so its relative `models/...` defaults resolve:

```bash
./target/release/skadoosh
```

Talk — it transcribes, asks the LLM, and speaks each clause as it lands. Speak
over it at any time to barge in (it stops within milliseconds). Press `ctrl-c`
to quit. Without a Kokoro model it falls back to a sine-wave MockTts, so you can
verify the pipeline before downloading TTS.

## 5. Running the examples

Examples live in `examples/` (run from a repo clone):

```bash
cargo run --example text_chat              # text chat over the SDK (needs Ollama; no audio/models)
cargo run --example mock_agent             # full STT→LLM→TTS pipeline on mock engines — zero setup
cargo run --release --example voice_agent  # the real voice agent (needs models + mic + Ollama)
```

`text_chat` is the fastest smoke test: three scripted turns against your LLM
with no audio hardware. `mock_agent` runs the entire orchestrator on scripted
mock engines — green with no models, no server, and no microphone. `voice_agent`
is the full "hello world" voice loop; use `--release` for usable latency.

Standalone cookbook snippets also ship in `cookbooks/` (e.g. `03_repl_mode`,
`11_barge_in`, `12_mock_pipeline`) — run any with `cargo run --example <name>`.

## 6. Configuration

Every CLI flag has a `SKADOOSH_*` environment variable; flags win over env vars,
env vars win over defaults. The core ones:

| Env var | Flag | Default | Purpose |
|---|---|---|---|
| `SKADOOSH_LLM_URL` | `--llm-url` | `http://localhost:11434/v1` | OpenAI-compatible base URL |
| `SKADOOSH_LLM_MODEL` | `--llm-model` | `qwen2.5:0.5b` | Model name |
| `SKADOOSH_API_KEY` | `--api-key` | — | Bearer token for hosted providers (never logged; Ollama needs none) |

Point at a hosted provider instead of Ollama:

```bash
export SKADOOSH_LLM_URL=https://api.openai.com/v1
export SKADOOSH_LLM_MODEL=gpt-4o-mini
export SKADOOSH_API_KEY=sk-...        # sent as Authorization: Bearer <key>
skadoosh --repl
```

More flags: `--whisper-model`, `--tts-model`/`--tts-voices`, `--vad-threshold`,
`--silence-ms`, `--push-to-talk`, `--wake-word`, `--rag-dir`, `--hold-music`. Run
`skadoosh --help` for the full list; set `RUST_LOG=debug` for verbose logs.

## 7. Troubleshooting

**No audio device / no microphone.** List devices and pick one explicitly:
```bash
skadoosh --list-devices
skadoosh --input-device "Your Mic" --output-device "Your Speakers"
```
On a headless box, skip audio: `skadoosh --repl` (text in/out), `skadoosh
--output text` (voice in → text out), or `--say "Hi." --out-wav out.wav` (to a
file, no playback).

**Model not found.** The voice loop checks for `models/silero_vad.onnx` and
`models/ggml-tiny.en.bin` at startup. If you see `--whisper-model not found (run
scripts/download_models.sh)`, run the script from the repo root. Missing Kokoro
files only warn — the agent falls back to MockTts; add them with
`./scripts/download_models.sh --with-kokoro`.

**Ollama not running.** A connection refused to `localhost:11434` means the
server isn't up. Start it with `ollama serve`, confirm the model is pulled
(`ollama list`, else `ollama pull qwen2.5:0.5b`), or point elsewhere with
`SKADOOSH_LLM_URL=...`.

**Build fails on missing audio libs.** The default `audio` feature links
ALSA/clang/cmake. Install the packages from §1, or build text-only with
`cargo build --no-default-features`. For GPU acceleration, enable exactly one
execution provider: `--features gpu-cuda` (NVIDIA), `gpu-coreml` (macOS),
`gpu-directml` (Windows), or `gpu-rocm` (AMD).
