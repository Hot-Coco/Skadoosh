//! The full voice agent via the SDK — the "hello world" of the crate.
//!
//! # Requirements (this example uses the real models + audio devices)
//!
//! * `models/silero_vad.onnx` and `models/ggml-tiny.en.bin` (run
//!   `scripts/download_models.sh`),
//! * a microphone + speaker (or the ALSA null devices for a smoke run),
//! * an OpenAI-compatible LLM server — local Ollama by default:
//!   `ollama create stealthylm -f Modelfile && ollama serve`.
//!
//! With no Kokoro TTS model configured, the agent speaks in the MockTts
//! sine wave (the pitch of pure triumph).
//!
//! ```sh
//! cargo run --release --example voice_agent
//! ```

use skadoosh::{Agent, AgentEvent, Config, Result};

fn main() -> Result<()> {
    let agent = Agent::builder().config(Config::default()).build()?;

    // Watch the event stream (transcripts, reply clauses, per-turn latency).
    let mut events = agent.events();
    std::thread::spawn(move || {
        while let Ok(event) = events.blocking_recv() {
            match event {
                AgentEvent::Transcript(text) => println!("you: {text}"),
                AgentEvent::StageLatency { total_ms, .. } => {
                    println!("  (turn took {total_ms} ms speech-end → first audio)")
                }
                _ => {}
            }
        }
    });

    println!("skadoosh voice agent — speak; ctrl-c to quit");
    agent.run()
}
