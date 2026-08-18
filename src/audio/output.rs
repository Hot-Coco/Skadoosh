//! Speaker playback: clips mpsc → playback std thread → SPSC ring → cpal
//! output callback.
//!
//! The playback thread owns the cpal output stream and is the *sole* pusher
//! into its own 48 000-sample (2 s @ 24 kHz) `HeapRb<f32>`: it pops
//! [`TtsClip`]s from the clips mpsc and `try_push`es them, retrying the
//! remaining slice every ~5 ms when full (natural backpressure up the clips
//! mpsc). Each retry re-checks the flush epoch, discarding the clip remainder
//! after a bump so a blocked push can never delay a barge-in flush. The RT
//! callback pulls from the ring, resamples to the device rate, and outputs
//! silence when empty.
//!
//! Flush is lock-free: [`PlaybackHandle::flush`] only bumps an epoch atomic;
//! the RT callback compares epochs each period and calls `Consumer::clear()`
//! itself — flush latency is bounded by one callback period (~5–10 ms) with
//! zero locks on the RT thread.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::Result;
use crate::tts::TtsClip;

/// Configuration for speaker playback.
#[derive(Debug, Clone, Default)]
pub struct AudioOutputConfig {
    /// Output device name; `None` selects the default output device.
    pub device_name: Option<String>,
}

/// Owning end of the playback thread. [`Playback::stop`] ends playback.
#[allow(dead_code)] // fields wired up by the task-1.3 implementation
pub struct Playback {
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Playback {
    /// Spawns the playback thread, opens the output stream, and returns the
    /// owning handle plus the cloneable [`PlaybackHandle`] used by the
    /// pipeline to queue clips and flush on barge-in.
    pub fn start(cfg: &AudioOutputConfig) -> Result<(Playback, PlaybackHandle)> {
        let _ = cfg;
        todo!("task 1.3: playback thread + SPSC ring + flush epoch")
    }

    /// Signals the playback thread to exit and joins it.
    pub fn stop(self) {
        todo!("task 1.3: join playback thread")
    }
}

/// Cloneable, `Send` handle to the running playback thread. Internally just
/// clones of the clips mpsc `Sender` plus the epoch/`is_playing` atomics — it
/// never touches the ring directly.
#[derive(Clone)]
#[allow(dead_code)] // clips_tx consumed by the task-1.3 implementation
pub struct PlaybackHandle {
    clips_tx: mpsc::Sender<TtsClip>,
    flush_epoch: Arc<AtomicU64>,
    is_playing: Arc<AtomicBool>,
    sample_rate: u32,
}

impl PlaybackHandle {
    /// Queues a clip for playback, awaiting mpsc capacity (backpressure).
    pub async fn queue_clip(&self, clip: TtsClip) -> Result<()> {
        let _ = clip;
        todo!("task 1.3: clips_tx.send(clip).await")
    }

    /// Lock-free flush: bumps the flush epoch. The RT callback clears the
    /// ring itself within one output callback period.
    pub fn flush(&self) {
        self.flush_epoch.fetch_add(1, Ordering::Release);
    }

    /// Whether non-silent samples are currently being emitted (VAD/barge-in
    /// input). Set by the RT callback, cleared when the ring runs dry.
    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Acquire)
    }

    /// Sample rate of queued clips (24 000 Hz).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
