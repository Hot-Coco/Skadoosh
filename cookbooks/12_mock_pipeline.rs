//! Cookbook 12 — Full mock pipeline (end-to-end).
//!
//! Chains every voice-pipeline stage with mock/scripted engines — no model
//! files, no LLM server, no audio device:
//!
//! 1. **STT** — [`MockStt`] pops a canned transcript for the input segment.
//! 2. **LLM** — a scripted [`LlmBackend`] streams reply clauses for that
//!    transcript.
//! 3. **TTS** — [`MockTts`] synthesizes each streamed clause into a clip.
//!
//! This is the same composition the [`Agent`](skadoosh::Agent) facade drives
//! internally; here the stages are wired by hand to show how they plug
//! together via the public trait objects.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 12_mock_pipeline
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::stt::{MockStt, SttEngine};
use skadoosh::tts::{MockTts, TtsEngine};
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A scripted LLM backend: one queued reply (a vec of clauses) per turn.
struct ScriptedLlm {
    script: Mutex<VecDeque<Vec<String>>>,
}

impl ScriptedLlm {
    fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
        }
    }

    fn turn(self, clauses: &[&str]) -> Self {
        self.script
            .lock()
            .expect("script lock")
            .push_back(clauses.iter().map(|s| s.to_string()).collect());
        self
    }
}

impl LlmBackend for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        _user: &'a str,
        clauses: mpsc::Sender<String>,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let reply: Vec<String> = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .unwrap_or_else(|| vec!["(no scripted reply)".to_string()]);
        Box::pin(async move {
            for clause in reply {
                let _ = clauses.send(clause).await;
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {
        self.script.lock().expect("script lock").clear();
    }
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to start tokio runtime: {e}"))?;

    // Annotating the result pins the async block's error type to
    // `SkadooshError`, so the `??` below (anyhow::Error -> SkadooshError ->
    // SkadooshError) converts unambiguously.
    let ran: skadoosh::Result<()> = rt.block_on(async {
        // Stage 1 — STT: scripted transcript (the audio samples are ignored).
        let stt = MockStt::from_replies(["what is the weather in Berlin?"]);
        // Stage 2 — LLM: a scripted reply streamed clause by clause.
        let mut llm = ScriptedLlm::new().turn(&["It's sunny ", "and around 22 degrees."]);
        // Stage 3 — TTS: zero-model sine-wave mock engine.
        let mut tts = MockTts::new();

        // 1. STT: transcribe a one-second "segment" (16 kHz mono f32).
        let transcript = stt
            .transcribe(vec![0.0_f32; 16_000])
            .await
            .map_err(|_| anyhow::anyhow!("STT receiver dropped"))??;
        println!("STT  -> {transcript:?}");

        // 2. LLM: stream reply clauses for the transcript.
        let (tx, mut rx) = mpsc::channel(16);
        llm.stream_reply(&transcript, tx, CancellationToken::new())
            .await?;
        println!("LLM  -> streaming reply clauses...");

        // 3. TTS: synthesize each streamed clause into a clip.
        let mut clips = Vec::new();
        while let Some(clause) = rx.recv().await {
            let clip = tts.synthesize(&clause)?;
            println!(
                "TTS  -> clause {:?} => {} samples ({:.0} ms)",
                clause,
                clip.samples.len(),
                clip.samples.len() as f32 / clip.sample_rate as f32 * 1000.0
            );
            clips.push(clip);
        }

        println!("\nfull mock pipeline produced {} clip(s)", clips.len());

        // End-to-end assertions: every stage ran, in order.
        assert!(!transcript.is_empty(), "STT produced a transcript");
        assert_eq!(clips.len(), 2, "one MockTts clip per scripted clause");
        assert!(
            clips.iter().all(|c| c.sample_rate == 24_000),
            "MockTts emits 24 kHz"
        );
        assert!(
            clips.iter().all(|c| !c.samples.is_empty()),
            "non-empty clips"
        );

        println!("\n12_mock_pipeline: OK");
        Ok(())
    });
    ran?;
    Ok(())
}
