//! CLI configuration: clap derive with `SKADOOSH_*` env fallbacks.

use std::fmt;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

use crate::error::Result;

/// Default system prompt seeded into the LLM conversation history.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a voice assistant. Reply in short, \
     spoken-style sentences. Keep answers under three sentences unless asked for detail.";

/// Output modality of the agent (`--output` / `SKADOOSH_OUTPUT`).
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Speak replies through the output device (TTS + playback).
    #[default]
    Audio,
    /// Print transcripts and streamed reply clauses on stdout instead of
    /// playing audio (voice-in → text-out). Needs no TTS model and no
    /// output device.
    Text,
}

/// Command-line configuration for `skadoosh`.
///
/// Every flag falls back to a `SKADOOSH_*` environment variable; flags win
/// over env vars, env vars win over defaults.
///
/// `Debug` is hand-rolled (below) so `api_key` can never leak into logs —
/// "never logged" is a documented promise on the flag.
#[derive(Parser, Clone)]
#[command(
    name = "skadoosh",
    version,
    about = "Modular, low-latency local voice agent framework",
    long_about = "Skadoosh runs a continuous-listening local voice agent: \
                  mic → Silero VAD → Whisper STT → streaming LLM → ONNX TTS → playback, \
                  with barge-in."
)]
pub struct Config {
    /// Image file paths to include with the next user turn (multimodal
    /// vision models). Accepts PNG, JPEG, GIF, WebP, BMP, TIFF, and PDF.
    /// Repeat for multiple images: `--image a.png --image b.jpg`.
    #[arg(long = "image", env = "SKADOOSH_IMAGE", value_name = "PATH")]
    pub images: Vec<PathBuf>,

    /// Base URL of the OpenAI-compatible LLM API.
    #[arg(
        long,
        env = "SKADOOSH_LLM_URL",
        default_value = "http://localhost:11434/v1"
    )]
    pub llm_url: String,

    /// Model name passed to the chat-completions API.
    #[arg(long, env = "SKADOOSH_LLM_MODEL", default_value = "qwen2.5:0.5b")]
    pub llm_model: String,

    /// API key for hosted OpenAI-compatible providers; sent as
    /// `Authorization: Bearer <key>`. Local Ollama needs none. Never logged.
    #[arg(long, env = "SKADOOSH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// System prompt seeded into the LLM conversation history.
    #[arg(long, env = "SKADOOSH_SYSTEM_PROMPT", default_value = DEFAULT_SYSTEM_PROMPT)]
    pub system_prompt: String,

    /// Maximum number of trailing user/assistant turns kept in LLM history.
    #[arg(long, env = "SKADOOSH_MAX_HISTORY_TURNS", default_value_t = 8)]
    pub max_history_turns: usize,

    /// Path to the whisper.cpp ggml model.
    #[arg(
        long,
        env = "SKADOOSH_WHISPER_MODEL",
        default_value = "models/ggml-tiny.en.bin"
    )]
    pub whisper_model: PathBuf,

    /// Path to the Silero VAD ONNX model.
    #[arg(
        long,
        env = "SKADOOSH_VAD_MODEL",
        default_value = "models/silero_vad.onnx"
    )]
    pub vad_model: PathBuf,

    /// Path to the Kokoro TTS ONNX model. When absent (or the file is
    /// missing) the pipeline falls back to the sine-wave MockTts.
    #[arg(long, env = "SKADOOSH_TTS_MODEL")]
    pub tts_model: Option<PathBuf>,

    /// Path to the Kokoro voices bank (`voices.bin`). Required together with
    /// `--tts-model` for the real TTS engine.
    #[arg(long, env = "SKADOOSH_TTS_VOICES")]
    pub tts_voices: Option<PathBuf>,

    /// Speech probability threshold for the VAD, in `[0, 1)`.
    #[arg(long, env = "SKADOOSH_VAD_THRESHOLD", default_value_t = 0.5)]
    pub vad_threshold: f32,

    /// Trailing silence (milliseconds) that closes a speech segment.
    #[arg(long, env = "SKADOOSH_SILENCE_MS", default_value_t = 300)]
    pub silence_ms: u32,

    /// Input (capture) device name; the default input device is used when unset.
    #[arg(long, env = "SKADOOSH_INPUT_DEVICE")]
    pub input_device: Option<String>,

    /// Output (playback) device name; the default output device is used when unset.
    #[arg(long, env = "SKADOOSH_OUTPUT_DEVICE")]
    pub output_device: Option<String>,

    /// List available audio input/output devices and exit.
    #[arg(long)]
    pub list_devices: bool,

    /// Force the sine-wave mock TTS engine even if Kokoro files are present.
    #[arg(long, env = "SKADOOSH_MOCK_TTS")]
    pub mock_tts: bool,

    /// Run the headless self-test from a wav file (16-bit PCM, any rate),
    /// write `selftest_out.wav`, print the latency table, and exit.
    #[arg(long, env = "SKADOOSH_SELFTEST", value_name = "WAV_PATH")]
    pub selftest: Option<PathBuf>,

    /// Interactive text-in/text-out loop on stdin/stdout: no audio, no
    /// VAD/STT/TTS — the same LLM history/prompt machinery, printing reply
    /// clauses as they stream. Exit with `/quit` or EOF (ctrl-d).
    #[arg(long, env = "SKADOOSH_REPL")]
    pub repl: bool,

    /// One-shot text→speech: synthesize `TEXT` with the TTS engine and play
    /// it (or write it to `--out-wav` instead — no audio device needed).
    #[arg(long, env = "SKADOOSH_SAY", value_name = "TEXT")]
    pub say: Option<String>,

    /// Output modality of the voice loop: play replies on the output device
    /// (`audio`) or print transcripts and streamed replies on stdout
    /// (`text`).
    #[arg(long, env = "SKADOOSH_OUTPUT", value_enum, default_value_t = OutputMode::Audio)]
    pub output: OutputMode,

    /// With `--say`: write the synthesized speech to this 24 kHz wav file
    /// instead of playing it (headless-friendly).
    #[arg(long, env = "SKADOOSH_OUT_WAV", value_name = "PATH")]
    pub out_wav: Option<PathBuf>,

    /// Path to a JSON file of tool/function definitions for tool calling.
    /// Format: [{"type":"function","function":{"name":"...","description":"...","parameters":{...}}}]
    #[arg(long, env = "SKADOOSH_TOOLS_FILE", value_name = "PATH")]
    pub tools_file: Option<PathBuf>,

    /// Maximum tool-calling round-trips before forcing a text response (default: 5).
    #[arg(long, env = "SKADOOSH_MAX_TOOL_ROUNDS", default_value_t = 5)]
    pub max_tool_rounds: usize,

    /// URL of an external service to forward conversations to when the LLM
    /// cannot answer. When set, a `forward_call` tool is auto-registered so
    /// the model can hand off the conversation; the forwarded service's text
    /// response is relayed back to the user as the tool result.
    #[arg(long, env = "SKADOOSH_FORWARD_URL", value_name = "URL")]
    pub forward_url: Option<String>,

    /// Directory of `.wasm` skill plugins to load and auto-register with the
    /// LLM as function-calling tools (see `PLUGINS.md`). Each plugin runs in a
    /// sandboxed wasmtime instance with a fuel budget and no filesystem or
    /// network access. When unset, the default `~/.skadoosh/plugins/` is used
    /// if it exists (otherwise no plugins load silently); an explicitly
    /// configured directory that is missing logs a warning and continues
    /// without plugins.
    #[arg(long, env = "SKADOOSH_PLUGINS_DIR", value_name = "PATH")]
    pub plugins_dir: Option<PathBuf>,

    /// Kokoro TTS voice key (e.g. "af", "am_adam"). Requires Kokoro model.
    #[arg(long, env = "SKADOOSH_TTS_VOICE", default_value = "af")]
    pub tts_voice: String,

    /// TTS playback speed multiplier (0.5 – 2.0, default: 1.0).
    #[arg(long, env = "SKADOOSH_TTS_SPEED", default_value_t = 1.0)]
    pub tts_speed: f32,

    /// Wake word to trigger listening (e.g. "hey skadoosh"). When set, the agent
    /// only processes speech after the wake word is detected in the transcript.
    #[arg(long, env = "SKADOOSH_WAKE_WORD")]
    pub wake_word: Option<String>,

    /// Push-to-talk mode: press and hold Enter to record, release to send.
    /// Overrides VAD-based segmentation when enabled.
    #[arg(long, env = "SKADOOSH_PUSH_TO_TALK", default_value_t = false)]
    pub push_to_talk: bool,

    /// Play procedural hold music during long tool execution or thinking gaps.
    /// The agent auto-ducks the music when TTS begins speaking. No external
    /// audio files needed — music is generated from sine-wave chord progressions.
    #[arg(long, env = "SKADOOSH_HOLD_MUSIC", default_value_t = false)]
    pub hold_music: bool,

    /// Whisper model size to download/use. One of: tiny, base, small, medium.
    /// Only affects model path construction; the actual model file must exist.
    /// Default model is `ggml-tiny.en.bin`; `--whisper-model-size base` switches
    /// to `ggml-base.en.bin`.
    #[arg(long, env = "SKADOOSH_WHISPER_MODEL_SIZE", default_value = "tiny")]
    pub whisper_model_size: String,

    /// Enable emotion-aware TTS: the agent detects sentiment from the LLM
    /// output and adjusts speaking speed (faster for excitement, slower for
    /// calm/sad). Requires Kokoro TTS model.
    #[arg(long, env = "SKADOOSH_TTS_EMOTION", default_value_t = false)]
    pub tts_emotion: bool,

    /// Enable multi-agent mesh networking: LAN peer discovery via UDP
    /// broadcast and a tiny HTTP server that accepts `forward_call` requests
    /// from peer agents. When set, the `forward_call` tool is extended with a
    /// `target` peer-name argument so the model can route a call to a
    /// discovered agent instead of the `--forward-url` endpoint.
    #[arg(long, env = "SKADOOSH_MESH", default_value_t = false)]
    pub mesh: bool,

    /// UDP discovery + HTTP port for the mesh (default 9876). Both the UDP
    /// discovery socket and the HTTP server bind this port number — UDP and
    /// TCP are distinct protocols, so they never collide.
    #[arg(long, env = "SKADOOSH_MESH_PORT", default_value_t = 9876)]
    pub mesh_port: u16,

    /// Name for this node in the mesh. When `--mesh` is enabled but no name
    /// is given, the node defaults to `skadoosh-<pid>`.
    #[arg(long, env = "SKADOOSH_AGENT_NAME", value_name = "NAME")]
    pub agent_name: Option<String>,

    /// Directory of documents (`.txt`/`.md`) to index for retrieval-augmented
    /// generation. When set, the LLM client loads and embeds the docs once on
    /// startup, then injects the `--rag-top-k` most relevant chunks into the
    /// system prompt before each turn (run
    /// `scripts/download_models.sh --with-rag` for the embedding model).
    #[arg(long, env = "SKADOOSH_RAG_DIR", value_name = "PATH")]
    pub rag_dir: Option<PathBuf>,

    /// Number of retrieved chunks injected into the system prompt per turn
    /// (only used with `--rag-dir`).
    #[arg(long, env = "SKADOOSH_RAG_TOP_K", default_value_t = 3)]
    pub rag_top_k: usize,

    /// Path to the sentence-embedding ONNX model used by `--rag-dir`
    /// (all-MiniLM-L6-v2 by default). Its companion BERT vocab is expected at
    /// `<stem>-vocab.txt` next to the model, e.g.
    /// `models/all-MiniLM-L6-v2-vocab.txt`.
    #[arg(
        long,
        env = "SKADOOSH_RAG_MODEL",
        default_value = crate::rag::DEFAULT_RAG_MODEL
    )]
    pub rag_model: PathBuf,

    /// Path to the conversation-memory JSON file. When set, the agent
    /// remembers user preferences across runs (injecting them into the
    /// system prompt) and appends a summary of each completed turn. Unset
    /// by default — no memory unless configured.
    #[arg(long, env = "SKADOOSH_MEMORY_FILE", value_name = "PATH")]
    pub memory_file: Option<PathBuf>,

    /// File to watch for changes (`--watch-file` / `SKADOOSH_WATCH_FILE`).
    /// When the file changes, a `NOTIFICATION: The file <path> has changed.`
    /// user turn is injected into the conversation. Repeatable:
    /// `--watch-file a.txt --watch-file b.txt`.
    #[arg(long = "watch-file", env = "SKADOOSH_WATCH_FILE", value_name = "PATH")]
    pub watch_files: Vec<PathBuf>,

    /// Process ID to watch for exit (`--watch-process` /
    /// `SKADOOSH_WATCH_PROCESS`). When the process exits, a
    /// `NOTIFICATION: Process <pid> has exited.` user turn is injected.
    /// Repeatable. Linux only (polls `/proc/<pid>`).
    #[arg(
        long = "watch-process",
        env = "SKADOOSH_WATCH_PROCESS",
        value_name = "PID"
    )]
    pub watch_processes: Vec<u32>,

    /// Timer in seconds (`--watch-timer` / `SKADOOSH_WATCH_TIMER`). When it
    /// elapses, a `NOTIFICATION: Your <n>-second timer is up.` user turn is
    /// injected. Repeatable.
    #[arg(
        long = "watch-timer",
        env = "SKADOOSH_WATCH_TIMER",
        value_name = "SECONDS"
    )]
    pub watch_timers: Vec<u64>,

    /// Sandboxed code-execution tool. When set, a `code_exec` tool is
    /// auto-registered so the model can run Python, shell, or binary snippets
    /// in a restricted subprocess (or Docker container) and receive stdout,
    /// stderr, and the exit code back. The value is the per-run wall-clock
    /// timeout in seconds (the sandbox's own default is 30). Unset by default
    /// — no code execution unless configured.
    #[arg(long, env = "SKADOOSH_CODE_EXEC_TIMEOUT", value_name = "SECONDS")]
    pub code_exec_timeout: Option<u64>,

    /// Sandbox backend for `--code-exec-timeout`: `subprocess` (host ulimits
    /// + scrubbed env + best-effort network namespace) or `docker` (ephemeral
    ///   `--network none` container). Only meaningful when
    ///   `--code-exec-timeout` is set; defaults to `subprocess`.
    #[arg(
        long,
        env = "SKADOOSH_CODE_EXEC_SANDBOX",
        value_enum,
        default_value_t = crate::sandbox::SandboxMode::Subprocess
    )]
    pub code_exec_sandbox: crate::sandbox::SandboxMode,
}

impl Default for Config {
    /// Defaults matching the clap flag defaults (SDK entry point —
    /// `Config::default()` equals bare `skadoosh` with no flags).
    fn default() -> Self {
        Self {
            images: Vec::new(),
            llm_url: "http://localhost:11434/v1".to_string(),
            llm_model: "qwen2.5:0.5b".to_string(),
            api_key: None,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            max_history_turns: 8,
            whisper_model: PathBuf::from("models/ggml-tiny.en.bin"),
            vad_model: PathBuf::from("models/silero_vad.onnx"),
            tts_model: None,
            tts_voices: None,
            vad_threshold: 0.5,
            silence_ms: 300,
            input_device: None,
            output_device: None,
            list_devices: false,
            mock_tts: false,
            selftest: None,
            repl: false,
            say: None,
            output: OutputMode::Audio,
            out_wav: None,
            tools_file: None,
            max_tool_rounds: 5,
            forward_url: None,
            plugins_dir: None,
            tts_voice: "af".to_string(),
            tts_speed: 1.0,
            wake_word: None,
            push_to_talk: false,
            hold_music: false,
            whisper_model_size: "tiny".to_string(),
            tts_emotion: false,
            mesh: false,
            mesh_port: 9876,
            agent_name: None,
            rag_dir: None,
            rag_top_k: 3,
            rag_model: PathBuf::from(crate::rag::DEFAULT_RAG_MODEL),
            memory_file: None,
            watch_files: Vec::new(),
            watch_processes: Vec::new(),
            watch_timers: Vec::new(),
            code_exec_timeout: None,
            code_exec_sandbox: crate::sandbox::SandboxMode::Subprocess,
        }
    }
}

impl fmt::Debug for Config {
    /// Every field prints normally EXCEPT `api_key`, which is redacted —
    /// the flag documents "never logged", and a derived `Debug` would be
    /// one `info!(?config)` away from breaking that. The exhaustive
    /// destructuring means a future field fails to compile until it is
    /// considered here (and redacted too if ever secret).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self {
            images,
            llm_url,
            llm_model,
            api_key,
            system_prompt,
            max_history_turns,
            whisper_model,
            vad_model,
            tts_model,
            tts_voices,
            vad_threshold,
            silence_ms,
            input_device,
            output_device,
            list_devices,
            mock_tts,
            selftest,
            repl,
            say,
            output,
            out_wav,
            tools_file,
            max_tool_rounds,
            forward_url,
            plugins_dir,
            tts_voice,
            tts_speed,
            wake_word,
            push_to_talk,
            hold_music,
            whisper_model_size,
            tts_emotion,
            mesh,
            mesh_port,
            agent_name,
            rag_dir,
            rag_top_k,
            rag_model,
            memory_file,
            watch_files,
            watch_processes,
            watch_timers,
            code_exec_timeout,
            code_exec_sandbox,
        } = self;
        f.debug_struct("Config")
            .field("images", images)
            .field("llm_url", llm_url)
            .field("llm_model", llm_model)
            .field("api_key", &api_key.as_ref().map(|_| "<redacted>"))
            .field("system_prompt", system_prompt)
            .field("max_history_turns", max_history_turns)
            .field("whisper_model", whisper_model)
            .field("vad_model", vad_model)
            .field("tts_model", tts_model)
            .field("tts_voices", tts_voices)
            .field("vad_threshold", vad_threshold)
            .field("silence_ms", silence_ms)
            .field("input_device", input_device)
            .field("output_device", output_device)
            .field("list_devices", list_devices)
            .field("mock_tts", mock_tts)
            .field("selftest", selftest)
            .field("repl", repl)
            .field("say", say)
            .field("output", output)
            .field("out_wav", out_wav)
            .field("tools_file", tools_file)
            .field("max_tool_rounds", max_tool_rounds)
            .field("forward_url", forward_url)
            .field("plugins_dir", plugins_dir)
            .field("tts_voice", tts_voice)
            .field("tts_speed", tts_speed)
            .field("wake_word", wake_word)
            .field("push_to_talk", push_to_talk)
            .field("hold_music", hold_music)
            .field("whisper_model_size", whisper_model_size)
            .field("tts_emotion", tts_emotion)
            .field("mesh", mesh)
            .field("mesh_port", mesh_port)
            .field("agent_name", agent_name)
            .field("rag_dir", rag_dir)
            .field("rag_top_k", rag_top_k)
            .field("rag_model", rag_model)
            .field("memory_file", memory_file)
            .field("watch_files", watch_files)
            .field("watch_processes", watch_processes)
            .field("watch_timers", watch_timers)
            .field("code_exec_timeout", code_exec_timeout)
            .field("code_exec_sandbox", code_exec_sandbox)
            .finish()
    }
}

impl Config {
    /// Parses CLI arguments (with `SKADOOSH_*` env fallbacks). Exits the
    /// process with a usage message on parse failure, like clap's `parse`.
    pub fn parse() -> Config {
        <Config as Parser>::parse()
    }

    /// Validates the configuration:
    ///
    /// * `--vad-threshold` must lie in `[0, 1)`.
    /// * `--repl` conflicts with `--say` and `--selftest`; `--say` conflicts
    ///   with `--selftest` and `--output text`; `--out-wav` requires
    ///   `--say`.
    /// * Whisper/VAD model files must exist when the run uses them — i.e.
    ///   for the voice loop and `--selftest`, but not for `--repl`/`--say`
    ///   (except for `--list-devices`, which skips all checks).
    ///   If `--whisper-model` is not explicitly set and
    ///   `--whisper-model-size` is non-default, the whisper model path is
    ///   adjusted to `models/ggml-{size}.en.bin`.
    /// * The `--selftest` wav must exist when given.
    /// * TTS (Kokoro or the MockTts fallback) is needed by `--say` and by
    ///   the voice loop in `--output audio` mode; `--repl` and
    ///   `--output text` skip TTS entirely. Missing Kokoro files only
    ///   produce a warning — the pipeline falls back to MockTts (see
    ///   [`crate::tts::build_engine`]). `--say` without `--out-wav` plays to
    ///   an output device, which is checked when playback starts.
    pub fn validate(&mut self) -> Result<()> {
        // Apply whisper model size override when --whisper-model was not
        // explicitly set (still at the default).
        let default_model = PathBuf::from("models/ggml-tiny.en.bin");
        if self.whisper_model == default_model && self.whisper_model_size != "tiny" {
            let valid_sizes = ["tiny", "base", "small", "medium"];
            if valid_sizes.contains(&self.whisper_model_size.as_str()) {
                self.whisper_model =
                    PathBuf::from(format!("models/ggml-{}.en.bin", self.whisper_model_size));
            } else {
                tracing::warn!(
                    size = %self.whisper_model_size,
                    "unknown --whisper-model-size; using default ggml-tiny.en.bin"
                );
            }
        }
        if self.list_devices {
            return Ok(());
        }

        if !(0.0..1.0).contains(&self.vad_threshold) {
            return Err(anyhow::anyhow!(
                "--vad-threshold must be in [0, 1), got {}",
                self.vad_threshold
            )
            .into());
        }

        // Mode conflicts: the three run modes are mutually exclusive (the
        // first two, in flag order, are named in the error).
        let modes = [
            ("--repl", self.repl),
            ("--say", self.say.is_some()),
            ("--selftest", self.selftest.is_some()),
        ];
        let active: Vec<&str> = modes
            .iter()
            .filter(|(_, on)| *on)
            .map(|(name, _)| *name)
            .collect();
        if let [first, second, ..] = active.as_slice() {
            return Err(anyhow::anyhow!("{first} and {second} cannot be combined").into());
        }
        if self.say.is_some() && self.output == OutputMode::Text {
            return Err(anyhow::anyhow!(
                "--output text conflicts with --say (a spoken reply needs audio output)"
            )
            .into());
        }
        if self.out_wav.is_some() && self.say.is_none() {
            return Err(anyhow::anyhow!("--out-wav is only meaningful together with --say").into());
        }

        // Silently-inert combos are accepted but warned about, so a stray
        // flag isn't mistaken for having an effect. (Clap can't tell an
        // explicit `--output audio` from the default, so `--repl` always
        // warns; the say-side VAD/input warnings fire on non-default
        // values only.)
        if self.repl && self.output == OutputMode::Audio {
            tracing::warn!("--output audio has no effect with --repl (the repl is text-only)");
        }
        if self.selftest.is_some() && self.output == OutputMode::Text {
            tracing::warn!(
                "--output text has no effect with --selftest (the selftest always synthesizes a wav)"
            );
        }
        if self.say.is_some() {
            let defaults = Config::default();
            if self.input_device.is_some() {
                tracing::warn!("--input-device has no effect with --say (no audio is captured)");
            }
            if self.vad_threshold != defaults.vad_threshold {
                tracing::warn!("--vad-threshold has no effect with --say (no VAD runs)");
            }
            if self.silence_ms != defaults.silence_ms {
                tracing::warn!("--silence-ms has no effect with --say (no VAD runs)");
            }
        }

        // Whisper/VAD models are only needed when audio goes in (the voice
        // loop and --selftest); --repl and --say never load them.
        let needs_stt_vad = !self.repl && self.say.is_none();
        if needs_stt_vad {
            for (flag, path) in [
                ("--whisper-model", &self.whisper_model),
                ("--vad-model", &self.vad_model),
            ] {
                if !path.exists() {
                    return Err(anyhow::anyhow!(
                        "{flag} not found: {} (run scripts/download_models.sh)",
                        path.display()
                    )
                    .into());
                }
            }
        }

        if let Some(wav) = &self.selftest {
            if !wav.exists() {
                return Err(anyhow::anyhow!("--selftest wav not found: {}", wav.display()).into());
            }
        }

        // TTS is needed by --say and by the voice loop in audio mode; the
        // repl and --output text never build an engine.
        let needs_tts = self.say.is_some() || (!self.repl && self.output == OutputMode::Audio);
        if needs_tts && !self.mock_tts {
            match (&self.tts_model, &self.tts_voices) {
                (Some(model), Some(voices)) if model.exists() && voices.exists() => {}
                (None, None) => {
                    tracing::warn!(
                        "no TTS model configured (--tts-model/--tts-voices); \
                         falling back to sine-wave MockTts"
                    );
                }
                (model, voices) => {
                    tracing::warn!(
                        tts_model = ?model,
                        tts_voices = ?voices,
                        "Kokoro TTS files absent or incomplete; falling back to \
                         sine-wave MockTts (run scripts/download_models.sh --with-kokoro)"
                    );
                }
            }
        }

        Ok(())
    }
}
