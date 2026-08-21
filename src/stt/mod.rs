//! Speech-to-text: the pluggable [`SttEngine`] trait, the whisper-rs
//! implementation (`WhisperStt`, on a dedicated blocking std thread so the
//! synchronous whisper.cpp API stays off the async runtime — gated behind the
//! `audio` feature), and the scripted [`MockStt`] double for SDK users' tests.

#[cfg(feature = "audio")]
pub mod whisper;

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::oneshot;

#[cfg(feature = "audio")]
pub use whisper::{SttConfig, WhisperStt};

use crate::error::Result;

/// Pluggable speech-to-text engine.
///
/// Mirrors the working shape of `WhisperStt` (gated behind the `audio`
/// feature): `transcribe` hands the
/// engine one 16 kHz f32 mono segment and returns a
/// [`oneshot::Receiver`] for the reply, so the pipeline's STT bridge can
/// `await` it like any other stage. A dropped reply sender (the receiver
/// resolves to `Err`) means the job was evicted from a bounded queue or the
/// worker is gone; the bridge tells the two apart via [`dropped_jobs`].
///
/// [`dropped_jobs`]: SttEngine::dropped_jobs
///
/// The pipeline owns engine shutdown through [`stop`](SttEngine::stop)
/// (explicit, not bare `Drop`): engines holding a worker thread join it
/// there. The default implementations fit stateless engines.
pub trait SttEngine: Send {
    /// Engine name, for logs and [`crate::AgentEvent`]s.
    fn name(&self) -> &str;

    /// Queues a transcription job; the reply arrives on the returned
    /// oneshot. `samples` is 16 kHz f32 mono.
    fn transcribe(&self, samples: Vec<f32>) -> oneshot::Receiver<Result<String>>;

    /// Total jobs dropped because a bounded internal queue was full
    /// (drop-oldest policy). Used to tell an evicted job (benign) from a
    /// dead worker (fatal). Stateless engines keep the default `0`.
    fn dropped_jobs(&self) -> u64 {
        0
    }

    /// Stops the engine, joining any worker thread. Called by the pipeline
    /// during shutdown drain. The default is a no-op.
    fn stop(self: Box<Self>) {}
}

/// A scripted STT engine for tests and examples: pops canned transcripts
/// from a queue, in order.
///
/// ```no_run
/// use skadoosh::stt::{MockStt, SttEngine};
///
/// let stt = MockStt::from_replies(["hello world", "second take"]);
/// let reply = stt.transcribe(vec![0.0; 16_000]);
/// // ... await `reply` on a tokio runtime.
/// ```
///
/// Once the script is exhausted the mock answers with an empty transcript
/// (which the pipeline skips, like an empty whisper decode). A delayed
/// reply ([`MockStt::push_delayed`]) is delivered by a task
/// `tokio::spawn`ed inside `transcribe` — so the runtime requirement bites
/// at the `transcribe` CALL, which panics outside a tokio runtime context
/// (the pipeline's STT bridge and `#[tokio::test]` both qualify). Merely
/// awaiting the returned oneshot — and zero-delay replies — need no
/// runtime.
#[derive(Debug, Default)]
pub struct MockStt {
    replies: Mutex<VecDeque<(Duration, String)>>,
}

impl MockStt {
    /// An empty-script mock (answers every segment with an empty
    /// transcript).
    pub fn new() -> Self {
        Self::default()
    }

    /// A mock answering with `replies` in order, all without delay.
    pub fn from_replies<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mock = Self::new();
        for reply in replies {
            mock.push_reply(reply);
        }
        mock
    }

    /// Appends a canned transcript to the script.
    pub fn push_reply(&self, text: impl Into<String>) -> &Self {
        self.push_delayed(Duration::ZERO, text)
    }

    /// Appends a canned transcript delivered after `delay` (useful for
    /// stale-turn/barge-in tests).
    pub fn push_delayed(&self, delay: Duration, text: impl Into<String>) -> &Self {
        self.replies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push_back((delay, text.into()));
        self
    }
}

impl SttEngine for MockStt {
    fn name(&self) -> &str {
        "mock-stt"
    }

    fn transcribe(&self, _samples: Vec<f32>) -> oneshot::Receiver<Result<String>> {
        let next = self
            .replies
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop_front();
        let (delay, text) = next.unwrap_or_default();
        let (tx, rx) = oneshot::channel();
        if delay.is_zero() {
            let _ = tx.send(Ok(text));
        } else {
            // Requires a tokio runtime context (documented above).
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let _ = tx.send(Ok(text));
            });
        }
        rx
    }
}
