//! whisper-rs worker: a dedicated, named std thread (`"skadoosh-stt"`) owns
//! the `WhisperContext` and processes jobs from a bounded channel, replying
//! over oneshots. A pinned thread (rather than `spawn_blocking`) keeps the
//! context warm and jobs ordered without re-serializing through the blocking
//! pool.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::Arc;
use std::thread::JoinHandle;

use tokio::sync::oneshot;

use crate::error::Result;

/// A transcription job: 16 kHz f32 samples plus the reply oneshot.
type Job = (Vec<f32>, oneshot::Sender<Result<String>>);

/// whisper-rs decode settings.
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Language code passed to whisper (`"en"`).
    pub language: String,
    /// Decode threads; defaults to `min(4, available_parallelism)`.
    pub threads: u32,
    /// Greedy sampling (no beam search) — lowest latency.
    pub greedy: bool,
    /// Suppress blank outputs.
    pub suppress_blank: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4) as u32)
            .unwrap_or(1);
        Self {
            language: "en".to_string(),
            threads,
            greedy: true,
            suppress_blank: true,
        }
    }
}

/// Handle to the whisper worker thread.
#[allow(dead_code)] // channel/handle consumed by the task-3.1 implementation
pub struct WhisperStt {
    jobs: Option<SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    dropped_jobs: Arc<AtomicU64>,
}

impl WhisperStt {
    /// Loads the model and spawns the `"skadoosh-stt"` worker thread.
    pub fn start(model_path: &Path, cfg: &SttConfig) -> Result<Self> {
        let _ = (model_path, cfg);
        todo!("task 3.1: load WhisperContext, spawn named thread")
    }

    /// Queues a transcription job; the reply arrives on the returned oneshot.
    ///
    /// The command channel is a bounded `sync_channel(2)`: when full, the
    /// oldest *queued* job is dropped (the freshest utterance wins) and the
    /// dropped-jobs counter is bumped (surfaced via `tracing`). This caps
    /// memory when speech arrives faster than tiny.en transcribes.
    pub fn transcribe(&self, samples: Vec<f32>) -> oneshot::Receiver<Result<String>> {
        let _ = samples;
        todo!("task 3.1: bounded job queue with drop-oldest policy")
    }

    /// Total jobs dropped because the queue was full.
    pub fn dropped_jobs(&self) -> u64 {
        self.dropped_jobs.load(Ordering::Relaxed)
    }

    /// Signals the worker to exit and joins the thread. Called explicitly by
    /// the orchestrator during drain (not via bare `Drop`). A closed reply
    /// channel / [`SttError::WorkerGone`](crate::error::SttError::WorkerGone)
    /// observed during shutdown drain is benign and must not reach the
    /// fatal-error mpsc.
    pub fn stop(self) {
        todo!("task 3.1: signal exit + join")
    }
}
