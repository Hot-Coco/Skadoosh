//! whisper-rs worker: a dedicated, named std thread (`"skadoosh-stt"`) owns
//! the `WhisperContext` and processes jobs from a bounded channel, replying
//! over oneshots. A pinned thread (rather than `spawn_blocking`) keeps the
//! context warm and jobs ordered without re-serializing through the blocking
//! pool.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tokio::sync::oneshot;
use tracing::{debug, warn};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::error::{Result, SttError};
use crate::stt::SttEngine;

/// A transcription job: 16 kHz f32 samples plus the reply oneshot.
type Job = (Vec<f32>, oneshot::Sender<Result<String>>);

/// Bounded job-queue capacity (plan §7: 2 commands in flight).
const JOB_QUEUE_CAPACITY: usize = 2;

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
pub struct WhisperStt {
    jobs: SyncSender<Job>,
    worker: JoinHandle<()>,
    /// Consumer end, shared so [`WhisperStt::transcribe`] can evict the
    /// oldest queued job when the bounded channel is full. The worker blocks
    /// in `recv()` while holding the lock, but eviction is only attempted
    /// when the queue is full — in which case `recv()` never blocks, so the
    /// lock is always available to the evictor.
    rx: Arc<Mutex<Receiver<Job>>>,
    dropped_jobs: Arc<AtomicU64>,
}

impl WhisperStt {
    /// Loads the model and spawns the `"skadoosh-stt"` worker thread.
    ///
    /// The model is loaded *on* the worker thread (so the context is created,
    /// owned, and used on one thread) and the load result is reported back
    /// synchronously, so `start` fails fast on a bad model path.
    pub fn start(model_path: &Path, cfg: &SttConfig) -> Result<Self> {
        let (tx, rx) = sync_channel::<Job>(JOB_QUEUE_CAPACITY);
        let shared_rx = Arc::new(Mutex::new(rx));
        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<()>>();
        let path = model_path.to_path_buf();
        let cfg = cfg.clone();
        let worker_rx = Arc::clone(&shared_rx);
        let worker = std::thread::Builder::new()
            .name("skadoosh-stt".to_string())
            .spawn(move || {
                let path_str = path.to_string_lossy().into_owned();
                let ctx = match WhisperContext::new_with_params(
                    &path_str,
                    WhisperContextParameters::default(),
                ) {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        let _ = init_tx.send(Err(SttError::ModelLoad(e.to_string()).into()));
                        return;
                    }
                };
                let mut state = match ctx.create_state() {
                    Ok(state) => state,
                    Err(e) => {
                        let _ = init_tx.send(Err(SttError::ModelLoad(e.to_string()).into()));
                        return;
                    }
                };
                let _ = init_tx.send(Ok(()));
                loop {
                    let job = {
                        let guard = lock(&worker_rx);
                        guard.recv()
                    };
                    match job {
                        Ok((samples, reply)) => {
                            let result = run_job(&mut state, &cfg, &samples);
                            // A closed reply channel during shutdown drain is
                            // benign — never fatal.
                            let _ = reply.send(result);
                        }
                        // All senders dropped (stop()): exit the loop.
                        Err(_) => break,
                    }
                }
                debug!("skadoosh-stt worker exiting");
            })
            .map_err(|e| {
                SttError::ModelLoad(format!("failed to spawn skadoosh-stt thread: {e}"))
            })?;
        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                jobs: tx,
                worker,
                rx: shared_rx,
                dropped_jobs: Arc::new(AtomicU64::new(0)),
            }),
            Ok(Err(e)) => {
                let _ = worker.join();
                Err(e)
            }
            Err(_) => {
                let _ = worker.join();
                Err(SttError::WorkerGone.into())
            }
        }
    }

    /// Queues a transcription job; the reply arrives on the returned oneshot.
    ///
    /// The command channel is a bounded `sync_channel(2)`: when full, the
    /// oldest *queued* job is dropped (the freshest utterance wins) and the
    /// dropped-jobs counter is bumped (surfaced via `tracing`). The evicted
    /// job's reply sender is dropped, so its receiver observes a closed
    /// channel. This caps memory when speech arrives faster than tiny.en
    /// transcribes.
    pub fn transcribe(&self, samples: Vec<f32>) -> oneshot::Receiver<Result<String>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let tx = &self.jobs;
        let mut job: Job = (samples, reply_tx);
        loop {
            match tx.try_send(job) {
                Ok(()) => return reply_rx,
                Err(TrySendError::Full(returned)) => {
                    // Evict the oldest *queued* job so the freshest wins.
                    // When the queue is full the worker's `recv()` cannot be
                    // blocked holding the lock, so this never deadlocks.
                    let evicted = {
                        let guard = lock(&self.rx);
                        guard.try_recv()
                    };
                    match evicted {
                        Ok(_dropped) => {
                            let n = self.dropped_jobs.fetch_add(1, Ordering::Relaxed) + 1;
                            debug!(dropped_jobs = n, "STT job queue full: dropped oldest job");
                        }
                        Err(TryRecvError::Empty) => {
                            // The worker grabbed a job between our try_send
                            // and try_recv — retry the send.
                        }
                        Err(TryRecvError::Disconnected) => {
                            let (_, reply_tx) = returned;
                            let _ = reply_tx.send(Err(SttError::WorkerGone.into()));
                            return reply_rx;
                        }
                    }
                    job = returned;
                }
                Err(TrySendError::Disconnected((_, reply_tx))) => {
                    let _ = reply_tx.send(Err(SttError::WorkerGone.into()));
                    return reply_rx;
                }
            }
        }
    }

    /// Total jobs dropped because the queue was full.
    pub fn dropped_jobs(&self) -> u64 {
        self.dropped_jobs.load(Ordering::Relaxed)
    }

    /// Signals the worker to exit and joins the thread. Called explicitly by
    /// the orchestrator during drain (not via bare `Drop`). A closed reply
    /// channel / [`SttError::WorkerGone`]
    /// observed during shutdown drain is benign and must not reach the
    /// fatal-error mpsc.
    pub fn stop(self) {
        // Dropping the last sender makes the worker's blocking recv() fail,
        // so it exits after finishing any in-flight job.
        drop(self.jobs);
        if self.worker.join().is_err() {
            warn!("skadoosh-stt worker panicked during shutdown");
        }
    }
}

impl SttEngine for WhisperStt {
    fn name(&self) -> &str {
        "whisper"
    }

    fn transcribe(&self, samples: Vec<f32>) -> oneshot::Receiver<Result<String>> {
        WhisperStt::transcribe(self, samples)
    }

    fn dropped_jobs(&self) -> u64 {
        WhisperStt::dropped_jobs(self)
    }

    fn stop(self: Box<Self>) {
        WhisperStt::stop(*self);
    }
}

/// Recovers a (theoretically) poisoned queue lock rather than panicking.
fn lock(rx: &Arc<Mutex<Receiver<Job>>>) -> std::sync::MutexGuard<'_, Receiver<Job>> {
    rx.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Runs one blocking whisper decode on the worker thread and concatenates the
/// segment texts.
fn run_job(state: &mut WhisperState, cfg: &SttConfig, samples: &[f32]) -> Result<String> {
    let mut params = FullParams::new(if cfg.greedy {
        SamplingStrategy::Greedy { best_of: 1 }
    } else {
        SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        }
    });
    params.set_language(Some(&cfg.language));
    params.set_n_threads(cfg.threads as i32);
    params.set_no_timestamps(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(cfg.suppress_blank);
    state
        .full(params, samples)
        .map_err(|e| SttError::Transcribe(e.to_string()))?;
    let n = state.full_n_segments();
    let mut text = String::new();
    for i in 0..n {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        match seg.to_str() {
            Ok(s) => text.push_str(s),
            Err(_) => text.push_str(
                &seg.to_str_lossy()
                    .map_err(|e| SttError::Transcribe(e.to_string()))?,
            ),
        }
    }
    Ok(text.trim().to_string())
}
