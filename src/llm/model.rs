//! Model registry — known models with their canonical names, system prompts,
//! and default parameters.
//!
//! # Adding a model
//!
//! 1. Add a `const` for the model name below.
//! 2. Add the model to the [`known_models`] slice.
//! 3. The model's system prompt is auto-selected by
//!    `LlmClient::from_config` when the
//!    configured `--llm-model` matches a known model name (and the user hasn't
//!    explicitly overridden `--system-prompt`).
//!
//! Unknown models fall back to the generic [`crate::config::DEFAULT_SYSTEM_PROMPT`].

// ---------------------------------------------------------------------------
// Canonical model names
// ---------------------------------------------------------------------------

/// StealthyLM-Emotive — the default local voice-optimised model.
///
/// Qwen2.5-1.5B-Instruct fine-tuned for Skadoosh: zero markdown, short clauses,
/// emotion-aware tone, and grounded tool-calling. Ships as a Q4_K_M GGUF.
///
/// Download and setup:
/// ```bash
/// wget https://huggingface.co/StealthyML/StealthyLM-Emotive/resolve/main/StealthyLM_Q4KM.gguf
/// echo 'FROM ./StealthyLM_Q4KM.gguf' > Modelfile
/// ollama create stealthylm -f Modelfile
/// ```
///
/// Context: 32 KiB tokens. Pairs with `--tts-emotion` for expressive speech.
pub const STEALTHYLM: &str = "stealthylm";

/// Raw GGUF filename recognised as an alias for the StealthyLM model
/// (Ollama loads GGUF files by name when the file is in the models directory).
pub const STEALTHYLM_GGUF: &str = "StealthyLM_Q4KM.gguf";

// ---------------------------------------------------------------------------
// Model configurations
// ---------------------------------------------------------------------------

/// Metadata for one known model.
pub struct KnownModel {
    /// Canonical name passed to the chat-completions API (`--llm-model`).
    pub name: &'static str,
    /// System prompt seeded into the LLM conversation history for this model.
    /// When `None`, the generic default is used.
    pub system_prompt: Option<&'static str>,
    /// Maximum conversation context length in tokens (best-effort; the runtime
    /// may have its own tighter limit). Used for history-truncation tuning.
    /// When `None`, the caller's default applies.
    pub max_context_tokens: Option<usize>,
    /// Whether this model is optimised for emotion-aware TTS. When `true` the
    /// `--tts-emotion` flag is a natural pairing.
    pub emotion_aware: bool,
}

/// System prompt tailored for StealthyLM-Emotive.
///
/// StealthyLM is fine-tuned for voice: it avoids markdown, produces short
/// clauses that match Skadoosh's streaming TTS, and varies its emotional tone
/// to fit the context. The prompt reinforces these strengths without wasting
/// tokens on instructions the model already internalises from training.
const STEALTHYLM_SYSTEM_PROMPT: &str = "You are Skadoosh, an expressive voice \
     assistant. Speak in short, natural clauses — one to two sentences per \
     thought. Vary your tone to match the context: warm and empathetic for \
     personal topics, crisp and factual for information. Never use markdown, \
     lists, or formatting — plain speech only. If you don't know something, \
     say so honestly.";

/// Slice of all known models (add new models here).
pub fn known_models() -> &'static [KnownModel] {
    &[
        KnownModel {
            name: STEALTHYLM,
            system_prompt: Some(STEALTHYLM_SYSTEM_PROMPT),
            max_context_tokens: Some(32768),
            emotion_aware: true,
        },
        KnownModel {
            name: STEALTHYLM_GGUF,
            system_prompt: Some(STEALTHYLM_SYSTEM_PROMPT),
            max_context_tokens: Some(32768),
            emotion_aware: true,
        },
    ]
}

/// Looks up a known model by its canonical name (case-sensitive).
pub fn find_model(name: &str) -> Option<&'static KnownModel> {
    known_models().iter().find(|m| m.name == name)
}

/// Returns `true` when the model is optimised for emotion-aware TTS
/// (pairs well with `--tts-emotion`).
pub fn is_emotion_aware(name: &str) -> bool {
    find_model(name).is_some_and(|m| m.emotion_aware)
}
