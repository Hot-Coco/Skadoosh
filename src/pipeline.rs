//! Orchestrator: task/channel topology (§7), barge-in (§8), shutdown
//! ordering, and the headless `--selftest` path.
//!
//! Shutdown ordering (so a clean ctrlc never exits non-zero): cancel the
//! shutdown + per-turn tokens → stop event sources (drop the mic stream /
//! close the clips sender so idle tasks wake on closed channels) → drain
//! in-flight items, treating `WorkerGone`/closed-channel errors during drain
//! as benign (never forwarded to the fatal-error mpsc) → join all tasks →
//! exit non-zero only on a real fatal error.

use std::path::Path;

use tokio_util::sync::CancellationToken;

use crate::config::Config;
use crate::error::Result;

/// Latency breakdown of one `--selftest` run (milliseconds), printed as a
/// table by the binary.
#[derive(Debug, Clone)]
pub struct SelftestReport {
    /// VAD segment close latency.
    pub segment_ms: u64,
    /// STT transcription time.
    pub stt_ms: u64,
    /// LLM time to first token.
    pub llm_ttft_ms: u64,
    /// Time to first completed clause.
    pub first_clause_ms: u64,
    /// TTS synthesis time (first clause).
    pub tts_ms: u64,
    /// End-to-end total.
    pub total_ms: u64,
    /// Transcript of the input segment.
    pub transcript: String,
}

/// The voice-agent pipeline orchestrator.
///
/// Owns the shutdown `CancellationToken` (ctrlc), the per-turn child token,
/// the fatal-error mpsc receiver, and the task join set. Barge-in: a VAD
/// `SpeechStart` while playback is active (2-frame / 64 ms hangover) cancels
/// the turn token and flushes playback; a `turn_id` tags turns so stale
/// clips/text are dropped defensively. STT is never cancelled.
#[allow(dead_code)] // fields consumed by the task-6.1 implementation
pub struct Pipeline {
    config: Config,
    shutdown: CancellationToken,
}

impl Pipeline {
    /// Creates the pipeline from a validated [`Config`]. Tasks are spawned by
    /// [`Pipeline::run`].
    pub fn new(config: Config) -> Result<Self> {
        Ok(Self {
            config,
            shutdown: CancellationToken::new(),
        })
    }

    /// Runs the full mic↔speaker topology until ctrlc or a fatal error.
    pub fn run(self) -> Result<()> {
        todo!("task 6.1: spawn 8 tasks / 9 channels, barge-in dispatch, ctrlc")
    }

    /// Headless self-test: no cpal. Loads `wav` via hound, resamples to
    /// 16 kHz, feeds the segmenter frame-by-frame, and runs the first segment
    /// through STT → LLM → TTS, writing `out_wav` and returning the latency
    /// report.
    pub fn run_selftest(self, wav: &Path, out_wav: &Path) -> Result<SelftestReport> {
        let _ = (wav, out_wav);
        todo!("task 6.1: wav-driven selftest with latency table")
    }
}
