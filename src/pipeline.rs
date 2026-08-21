//! Orchestrator: task/channel topology (plan §7), barge-in (§8), shutdown
//! ordering (§6), per-turn latency instrumentation (§8), and the headless
//! `--selftest` path.
//!
//! # Topology (§7)
//!
//! [`Pipeline::run`] wires the eight tasks: the cpal input callback (owned by
//! [`MicCapture`]) pushes 16 kHz mono into a lock-free ring; the VAD task
//! drains it in 512-sample frames through [`SileroVad`] + [`VadSegmenter`];
//! segments cross the STT bridge to the dedicated whisper thread
//! ([`WhisperStt`]); transcripts feed the LLM task ([`LlmClient`] SSE
//! streaming); clauses feed the TTS task (`spawn_blocking` per clause);
//! clips flow into the playback thread through [`PlaybackHandle`]. The
//! orchestrator itself dispatches VAD events, mints per-turn cancellation
//! tokens, supervises barge-in, and owns the fatal-error channel.
//!
//! # Barge-in (§8)
//!
//! A gated `SpeechStart` (2-frame / 64 ms hangover applied in the VAD task,
//! which is the only place with per-frame probabilities) while playback is
//! audible cancels the current turn token and bumps the playback flush
//! epoch. In-flight TTS clips are discarded via the turn token plus a
//! `turn_id` staleness check; the playback thread drains its clips queue on
//! every flush bump. STT is never cancelled — the user's new segment is
//! already accumulating.
//!
//! The turn token outlives the LLM stream: the TTS task can still hold a
//! buffered clause backlog (`CLAUSE_CAP`) plus queued clips when the stream
//! ends, so the orchestrator keeps the token live until the next segment
//! supersedes the turn. Conversely, a `SpeechStart` during a silent gap
//! neither cancels nor flushes — a one-frame VAD false positive there
//! (rejected by the segmenter's min-length guard, so no replacement
//! utterance ever follows) must not silently kill a mid-stream reply; a
//! genuine new utterance still cancels it via the segment supersede path.
//!
//! # Shutdown ordering (§6, revised)
//!
//! Shutdown (requested through [`Pipeline::shutdown_token`] — the binary
//! bridges SIGINT onto it, see `main.rs`; keeping the process-signal
//! handler in the bin leaves embedders in control of their own signals) or
//! a fatal error on the
//! fatal mpsc → cancel the shutdown + per-turn tokens → stop event sources
//! (the VAD task drops the mic stream; senders close so idle tasks wake on
//! closed channels) → drain in-flight items (`WorkerGone` / closed-channel
//! errors during drain are benign, never forwarded to the fatal mpsc) →
//! join all tasks → exit non-zero only on a real fatal error.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use ringbuf::traits::{Consumer, Observer};
use ringbuf::HeapCons;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, info_span, warn, Instrument};

use crate::agent::{AgentEvent, EVENT_CAP};
use crate::audio::{
    resample_offline, AudioInputConfig, AudioOutputConfig, MicCapture, Playback, PlaybackHandle,
    CAPTURE_RATE,
};
use crate::config::{Config, OutputMode};
use crate::error::{LlmError, Result, SkadooshError, SttError};
use crate::llm::client::{ensure_success, SseLineBuffer, CLAUSE_MAX_LEN, CLAUSE_MIN_LEN};
use crate::llm::{parse_sse_line, ClauseSplitter, LlmBackend, LlmClient};
use crate::stt::{SttConfig, SttEngine, WhisperStt};
use crate::tts::{build_engine, concat_clip_samples, TtsClip, TtsEngine, TTS_SAMPLE_RATE};
use crate::vad::{SileroVad, VadEvent, VadSegmenter, FRAME_LEN};
use crate::watch::{WatchConfig, WatchEvent, WatchManager};
use crate::wav::write_wav16;

/// VAD-events channel capacity (§7).
const VAD_EVENTS_CAP: usize = 8;
/// Segment-forward channel capacity (§7).
const SEGMENT_CAP: usize = 4;
/// STT-text channel capacity (§7).
const TEXT_CAP: usize = 8;
/// Turn-announcement channel capacity (LLM → TTS; one entry per turn).
const TURN_CAP: usize = 8;
/// Per-turn clause channel capacity (§7).
const CLAUSE_CAP: usize = 16;
/// Turn-completion channel capacity (LLM → orchestrator).
const TURN_DONE_CAP: usize = 8;
/// Fatal-error channel capacity (§7).
const FATAL_CAP: usize = 8;

/// Poll interval while waiting for the first audible sample of a turn.
const AUDIBLE_POLL: Duration = Duration::from_millis(2);
/// Give up waiting for first-audible after this long (stalled device).
const AUDIBLE_TIMEOUT: Duration = Duration::from_secs(3);

/// Latency breakdown of one `--selftest` run (milliseconds), printed as a
/// table by the binary.
///
/// Stage offsets: `segment_ms` covers the offline VAD pass over the input
/// wav up to the first segment close; `stt_ms` is the whisper decode;
/// `llm_ttft_ms` is from the LLM request to the first SSE content token;
/// `first_clause_ms` is from the LLM request to the first completed clause;
/// `tts_ms` is the first clause's synthesis wall time; `total_ms` covers
/// wav-load → `out_wav` written.
#[derive(Debug, Clone)]
pub struct SelftestReport {
    /// VAD segment close latency (offline pass over the input wav).
    pub segment_ms: u64,
    /// STT transcription time.
    pub stt_ms: u64,
    /// LLM time to first token.
    pub llm_ttft_ms: u64,
    /// Time from the LLM request to the first completed clause.
    pub first_clause_ms: u64,
    /// TTS synthesis time (first clause).
    pub tts_ms: u64,
    /// End-to-end total (wav load → output wav written).
    pub total_ms: u64,
    /// Transcript of the input segment.
    pub transcript: String,
}

impl fmt::Display for SelftestReport {
    /// Renders the latency table printed by `skadoosh --selftest`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "skadoosh selftest — latency report")?;
        let rows = [
            ("vad segmentation", self.segment_ms),
            ("stt (whisper)", self.stt_ms),
            ("llm time-to-first-token", self.llm_ttft_ms),
            ("llm first clause", self.first_clause_ms),
            ("tts first clip", self.tts_ms),
            ("total", self.total_ms),
        ];
        for (label, ms) in rows {
            writeln!(f, "  {label:<30} {ms:>8} ms")?;
        }
        write!(f, "  transcript: {:?}", self.transcript)
    }
}

/// The voice-agent pipeline orchestrator.
///
/// Owns the shutdown `CancellationToken` (ctrlc, bridged by the binary) and
/// drives the full task/channel topology (§7). Barge-in: a gated VAD
/// `SpeechStart` while playback is active cancels the per-turn token and
/// flushes playback; a `turn_id` tags turns so stale clips/text are dropped
/// defensively. STT is never cancelled.
///
/// [`Pipeline::new`] builds every stage from the [`Config`] (the binary's
/// path); `Pipeline::from_parts` (crate-internal) additionally injects
/// engine trait objects and the event bus — that is how the
/// [`Agent`](crate::agent::Agent) SDK facade drives the same machinery.
/// In [`OutputMode::Text`] no TTS engine or playback device is built: the
/// TTS task drains clauses and surfaces them as
/// [`AgentEvent::Clause`]s instead of synthesizing audio.
pub struct Pipeline {
    config: Config,
    shutdown: CancellationToken,
    stt: Option<Box<dyn SttEngine>>,
    llm: Option<Box<dyn LlmBackend>>,
    tts: Option<Box<dyn TtsEngine>>,
    events: broadcast::Sender<AgentEvent>,
}

impl Pipeline {
    /// Creates the pipeline from a validated [`Config`]. Tasks are spawned by
    /// [`Pipeline::run`]; no devices are opened or models loaded until then.
    pub fn new(config: Config) -> Result<Self> {
        let (events, _) = broadcast::channel(EVENT_CAP);
        Ok(Self::from_parts(
            config,
            CancellationToken::new(),
            events,
            None,
            None,
            None,
        ))
    }

    /// Creates the pipeline with SDK-injected stages and shared
    /// shutdown/event handles. Any stage left `None` is built from the
    /// config exactly as [`Pipeline::new`] would at `run()` time.
    pub(crate) fn from_parts(
        config: Config,
        shutdown: CancellationToken,
        events: broadcast::Sender<AgentEvent>,
        stt: Option<Box<dyn SttEngine>>,
        llm: Option<Box<dyn LlmBackend>>,
        tts: Option<Box<dyn TtsEngine>>,
    ) -> Self {
        Self {
            config,
            shutdown,
            stt,
            llm,
            tts,
            events,
        }
    }

    /// A clone of the shutdown token. Cancelling it requests a graceful
    /// shutdown of [`Pipeline::run`] (§6 ordering) — this is the programmatic
    /// shutdown injection point; the binary cancels it on SIGINT, tests
    /// cancel it directly.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Runs the full mic↔speaker topology until shutdown or a fatal error.
    ///
    /// Opens devices first so a headless machine fails fast with
    /// [`crate::error::AudioError::NoDevice`] (no panic, no hang), then loads
    /// the models, builds a multi-threaded tokio runtime, and blocks until
    /// the orchestrator exits. Returns `Err` only on a real fatal error; a
    /// requested shutdown is `Ok(())`.
    ///
    /// In [`OutputMode::Text`] no output device is opened and no TTS engine
    /// is built — replies surface as [`AgentEvent::Clause`]s instead of
    /// audio (the binary prints them).
    pub fn run(self) -> Result<()> {
        let Self {
            config,
            shutdown,
            stt,
            llm,
            tts,
            events,
        } = self;

        // Sources first: fail fast with a clean AudioError on headless
        // machines, before paying for model loads. In audio mode that is
        // BOTH devices (the v1 ordering — the v0.2 text-mode split had
        // moved playback behind the model loads, so a missing output
        // device was only reported after two model loads).
        let (capture, cons) = MicCapture::start(&AudioInputConfig {
            device_name: config.input_device.clone(),
        })?;
        let playback = match config.output {
            OutputMode::Audio => Some(Playback::start(&AudioOutputConfig {
                device_name: config.output_device.clone(),
            })?),
            OutputMode::Text => None,
        };
        let vad = SileroVad::new(&config.vad_model)?;
        let segmenter = VadSegmenter::new(config.vad_threshold, config.silence_ms);
        let stt: Box<dyn SttEngine> = match stt {
            Some(stt) => {
                info!(engine = stt.name(), "using injected STT engine");
                stt
            }
            None => Box::new(WhisperStt::start(
                &config.whisper_model,
                &SttConfig::default(),
            )?),
        };
        let llm: Box<dyn LlmBackend> = match llm {
            Some(llm) => {
                info!(backend = llm.name(), "using injected LLM backend");
                llm
            }
            None => Box::new(LlmClient::from_config(&config)),
        };
        let mut tts = match (tts, config.output) {
            (Some(tts), OutputMode::Audio) => {
                info!(engine = "injected", "using injected TTS engine");
                Some(tts)
            }
            (Some(_), OutputMode::Text) => {
                warn!("--output text: injected TTS engine ignored (no audio is synthesized)");
                None
            }
            (None, OutputMode::Audio) => Some(build_engine(&config)?),
            (None, OutputMode::Text) => None,
        };

        // Optional agent greeting (`--agent-name`): speak — or, in text mode,
        // print — a one-line greeting before the listening loop begins. Audio
        // mode synthesizes it on the already-open output device; text mode
        // has no TTS, so the greeting surfaces as a printed clause instead.
        if let Some(name) = config
            .agent_name
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
        {
            let text = crate::agent::greeting_text(name);
            match (tts.as_mut(), playback.as_ref()) {
                (Some(engine), Some((_, handle))) => {
                    if let Err(err) = crate::agent::speak_text(&mut **engine, handle.clone(), &text)
                    {
                        warn!(error = %err, "agent greeting synthesis failed");
                    }
                }
                _ => {
                    let _ = events.send(AgentEvent::Clause(text));
                    let _ = events.send(AgentEvent::ReplyDone);
                }
            }
        }

        match (config.output, playback) {
            (OutputMode::Audio, Some((playback, handle))) => {
                let result = run_loop(
                    capture, cons, vad, segmenter, &config, stt, llm, tts, handle, shutdown, events,
                );
                // Every PlaybackHandle clone was dropped with the tasks
                // above, so the playback thread has seen the channel close
                // (stop() also sets the stop flag) and joins promptly.
                playback.stop();
                result
            }
            (OutputMode::Text, None) => run_loop(
                capture, cons, vad, segmenter, &config, stt, llm, tts, NullSink, shutdown, events,
            ),
            _ => unreachable!("playback is opened exactly in audio mode"),
        }
    }

    /// Headless self-test: no cpal. Loads `wav`, resamples to 16 kHz, feeds
    /// the real [`SileroVad`] + [`VadSegmenter`] frame-by-frame, runs the
    /// FIRST segment through whisper STT → streaming LLM
    /// (`Config::llm_url`, pointed at a mock in tests) → clause splitter →
    /// TTS engine, and writes the concatenated clips to `out_wav` at 24 kHz.
    ///
    /// Returns the per-stage latency report (§8 stamps) that the binary
    /// prints as a table.
    pub fn run_selftest(self, wav: &Path, out_wav: &Path) -> Result<SelftestReport> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| anyhow::anyhow!("failed to start tokio runtime: {err}"))?;
        runtime.block_on(self.selftest_async(wav, out_wav))
    }
}

/// `Pipeline::run`'s task/runtime body, generic over the clip sink so text
/// mode can substitute [`NullSink`] for the real [`PlaybackHandle`].
#[allow(clippy::too_many_arguments)]
fn run_loop<C: ClipSink>(
    capture: MicCapture,
    cons: HeapCons<f32>,
    vad: SileroVad,
    segmenter: VadSegmenter,
    config: &Config,
    stt: Box<dyn SttEngine>,
    llm: Box<dyn LlmBackend>,
    tts: Option<Box<dyn TtsEngine>>,
    sink: C,
    shutdown: CancellationToken,
    events: broadcast::Sender<AgentEvent>,
) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|err| anyhow::anyhow!("failed to start tokio runtime: {err}"))?;

    runtime.block_on(async move {
        let (vad_tx, vad_rx) = mpsc::channel(VAD_EVENTS_CAP);
        let (fatal_tx, fatal_rx) = mpsc::channel(FATAL_CAP);
        let vad_join: tokio::task::JoinHandle<()> = if config.push_to_talk {
            tokio::spawn(
                push_to_talk_task(
                    capture,
                    cons,
                    vad_tx.clone(),
                    fatal_tx.clone(),
                    shutdown.clone(),
                    events.clone(),
                )
                .instrument(info_span!("ptt")),
            )
        } else {
            tokio::spawn(
                vad_task(VadParts {
                    capture,
                    cons,
                    vad,
                    segmenter,
                    threshold: config.vad_threshold,
                    sink: sink.clone(),
                    events_tx: vad_tx,
                    fatal_tx: fatal_tx.clone(),
                    shutdown: shutdown.clone(),
                    events: events.clone(),
                })
                .instrument(info_span!("vad")),
            )
        };
        let hold_music_flag = llm.hold_music_flag().cloned();
        // Proactive triggers (`--watch-*`): spawn the background watchers and
        // feed their events into the orchestrator. Held across the run so the
        // watcher tasks stay alive; joined after the orchestrator exits.
        let watch_config = WatchConfig {
            files: config.watch_files.clone(),
            processes: config.watch_processes.clone(),
            timers: config.watch_timers.clone(),
        };
        let (watch_manager, watch_rx) = if watch_config.is_empty() {
            (None, None)
        } else {
            let (manager, rx) = WatchManager::start(&watch_config, shutdown.clone());
            (Some(manager), Some(rx))
        };
        let result = run_orchestrator(Topology {
            vad_events: vad_rx,
            fatal_tx,
            fatal_rx,
            stt,
            llm,
            tts_engine: tts,
            sink,
            shutdown,
            events,
            wake_word: config.wake_word.clone(),
            hold_music: hold_music_flag,
            watch_rx,
        })
        .await;
        // The orchestrator cancelled the shutdown token on its way out, so the
        // watcher tasks are already exiting; join them.
        if let Some(manager) = watch_manager {
            manager.shutdown().await;
        }
        // The orchestrator cancelled the shutdown token on its way out,
        // so the VAD task is already exiting; collect it.
        match vad_join.await {
            Ok(()) => {}
            Err(join_err) => {
                warn!(error = %join_err, "VAD task panicked");
                if result.is_ok() {
                    return Err(anyhow::anyhow!("VAD task panicked: {join_err}").into());
                }
            }
        }
        result
    })
}

/// Clip sink for [`OutputMode::Text`]: no playback device is opened, clips
/// are never queued (text mode builds no TTS engine), playback is never
/// audible — so a `SpeechStart` never flushes/cancels (the "silent-gap"
/// rule) and a fresh segment still supersedes a stale turn.
#[derive(Debug, Clone, Copy, Default)]
struct NullSink;

impl ClipSink for NullSink {
    async fn queue_clip(&self, _clip: TtsClip) -> Result<()> {
        Err(anyhow::anyhow!("NullSink received a clip; text mode synthesizes no audio").into())
    }

    fn flush(&self) {}

    fn is_playing(&self) -> bool {
        false
    }
}

/// A VAD event crossing from the VAD task to the orchestrator, with the
/// latency stamp the segmenter close carries (§8: `t_speech_end`).
///
/// Public so integration tests can inject scripted events into
/// [`run_orchestrator`]; not part of the stable embedding API.
#[doc(hidden)]
#[derive(Debug)]
pub enum VadEventMsg {
    /// Speech onset (already past the 2-frame barge-in hangover in the live
    /// pipeline; injected as-is by tests).
    SpeechStart,
    /// A complete speech segment (16 kHz f32 mono), stamped at close.
    Segment {
        /// Segment samples (preroll included), 16 kHz f32 mono.
        samples: Vec<f32>,
        /// Segmenter close instant (§8 `t_speech_end`).
        t_speech_end: Instant,
    },
}

/// Clip sink seam: playback in production, a scripted recorder in tests.
///
/// Implemented by [`PlaybackHandle`]; integration tests inject their own.
/// Public for testability; not part of the stable embedding API.
#[doc(hidden)]
pub trait ClipSink: Clone + Send + 'static {
    /// Queues a clip for playback, awaiting capacity (backpressure).
    fn queue_clip(&self, clip: TtsClip) -> impl std::future::Future<Output = Result<()>> + Send;
    /// Barge-in flush: drops all queued/pending audio.
    fn flush(&self);
    /// Whether non-silent samples are currently being emitted.
    fn is_playing(&self) -> bool;
}

impl ClipSink for PlaybackHandle {
    fn queue_clip(&self, clip: TtsClip) -> impl std::future::Future<Output = Result<()>> + Send {
        PlaybackHandle::queue_clip(self, clip)
    }

    fn flush(&self) {
        PlaybackHandle::flush(self);
    }

    fn is_playing(&self) -> bool {
        PlaybackHandle::is_playing(self)
    }
}

/// Everything [`run_orchestrator`] needs: the injected channel ends and the
/// stage implementations. Public so integration tests can drive the
/// orchestrator without cpal/whisper; not part of the stable embedding API
/// (SDK users should go through [`Agent`](crate::agent::Agent)).
///
/// `tts_engine: None` selects text mode: the TTS task synthesizes nothing
/// and surfaces clauses as [`AgentEvent::Clause`]s (pair it with a
/// null [`ClipSink`]).
#[doc(hidden)]
pub struct Topology<C: ClipSink> {
    /// VAD events (from the VAD task in production; scripted in tests).
    pub vad_events: mpsc::Receiver<VadEventMsg>,
    /// Fatal-error channel: send end (cloned into every task).
    pub fatal_tx: mpsc::Sender<SkadooshError>,
    /// Fatal-error channel: receive end (owned by the orchestrator, §6).
    pub fatal_rx: mpsc::Receiver<SkadooshError>,
    /// STT stage.
    pub stt: Box<dyn SttEngine>,
    /// LLM stage (already pointed at the serving endpoint).
    pub llm: Box<dyn LlmBackend>,
    /// TTS stage, or `None` for text mode (no synthesis).
    pub tts_engine: Option<Box<dyn TtsEngine>>,
    /// Clip sink (playback handle in production).
    pub sink: C,
    /// Shutdown token; cancelled by the orchestrator on the way out.
    pub shutdown: CancellationToken,
    /// Agent-event broadcast; every stage reports through it.
    pub events: broadcast::Sender<AgentEvent>,
    /// Optional wake word: when set, the LLM task silently drops
    /// transcripts that do not contain this word.
    pub wake_word: Option<String>,
    /// Hold-music active flag: shared between the LLM client (which toggles
    /// it during tool execution) and the hold-music feeder task.
    pub hold_music: Option<Arc<AtomicBool>>,
    /// Proactive-trigger events (`--watch-*`), or `None` when no triggers are
    /// configured. Each event is synthesized into a system-injected user turn.
    pub watch_rx: Option<mpsc::Receiver<WatchEvent>>,
}

/// A segment forwarded by the orchestrator to the STT bridge.
struct SegmentMsg {
    turn_id: u64,
    token: CancellationToken,
    samples: Vec<f32>,
    t_speech_end: Instant,
}

/// A transcript forwarded by the STT bridge to the LLM task.
struct TextMsg {
    turn_id: u64,
    token: CancellationToken,
    text: String,
    t_speech_end: Instant,
    t_text: Instant,
}

/// A turn announcement from the LLM task to the TTS task, handing over the
/// per-turn clause channel.
struct TurnMsg {
    turn_id: u64,
    token: CancellationToken,
    clauses: mpsc::Receiver<String>,
    t_speech_end: Instant,
    t_text: Instant,
}

/// §8 latency stamps carried on messages through a turn.
///
/// `t_first_token` is not separately observable in the live pipeline:
/// [`LlmClient::stream_reply`] surfaces completed clauses, not raw tokens,
/// so the first-clause stamp doubles as the (slightly late) first-token
/// stamp. `run_selftest` drives the SSE stream directly and reports a true
/// TTFT.
#[derive(Debug, Clone)]
struct TurnTiming {
    t_speech_end: Instant,
    t_text: Instant,
    t_first_clause: Option<Instant>,
    t_first_clip: Option<Instant>,
}

/// Barge-in onset gate (§8): a `SpeechStart` while playback is active is
/// only forwarded after a *second consecutive* speech frame (2 frames =
/// 64 ms hangover) to reject clicks. With no playback, onsets forward
/// immediately.
#[derive(Debug, Default)]
struct OnsetGate {
    /// A gated onset is awaiting its confirmation frame.
    pending: bool,
}

impl OnsetGate {
    /// Called on every frame. `is_start`: the segmenter fired `SpeechStart`
    /// this frame; `is_speech`: this frame is at/above threshold; `playing`:
    /// playback audible. Returns `true` when a `SpeechStart` must be
    /// forwarded to the orchestrator for *this* frame.
    fn filter(&mut self, is_start: bool, is_speech: bool, playing: bool) -> bool {
        if self.pending {
            self.pending = false;
            // Confirmed only if this frame is still speech; a click dies here.
            return is_speech;
        }
        if is_start && playing {
            self.pending = true;
            return false;
        }
        is_start
    }
}

/// Everything the VAD task needs (bundled to stay under the argument lint).
struct VadParts<C: ClipSink> {
    capture: MicCapture,
    cons: HeapCons<f32>,
    vad: SileroVad,
    segmenter: VadSegmenter,
    threshold: f32,
    sink: C,
    events_tx: mpsc::Sender<VadEventMsg>,
    fatal_tx: mpsc::Sender<SkadooshError>,
    shutdown: CancellationToken,
    events: broadcast::Sender<AgentEvent>,
}

/// VAD task (§7 task 2): drains the mic ring in 512-sample frames, runs
/// Silero + the segmenter, applies the barge-in onset gate, and forwards
/// events. When fewer than [`FRAME_LEN`] samples are available it sleeps
/// `min(time-to-fill, 10 ms)` (never below 1 ms) — no busy-spin. Keeps
/// running during playback (required for barge-in). Owns the capture stream,
/// which stops when the task exits.
async fn vad_task<C: ClipSink>(parts: VadParts<C>) {
    let VadParts {
        capture,
        mut cons,
        mut vad,
        mut segmenter,
        threshold,
        sink,
        events_tx,
        fatal_tx,
        shutdown,
        events,
    } = parts;
    // Owned here; dropping it at task exit stops the mic stream (§6 "stop
    // event sources").
    let _capture = capture;
    let mut gate = OnsetGate::default();

    loop {
        if shutdown.is_cancelled() {
            break;
        }
        let occupied = cons.occupied_len();
        if occupied < FRAME_LEN {
            // 16 samples per millisecond at 16 kHz; clamp to [1, 10] ms.
            let deficit = (FRAME_LEN - occupied) as u64;
            let wait = Duration::from_millis((deficit / 16).clamp(1, 10));
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                _ = tokio::time::sleep(wait) => {}
            }
            continue;
        }
        let mut frame = [0.0f32; FRAME_LEN];
        let popped = cons.pop_slice(&mut frame);
        debug_assert_eq!(popped, FRAME_LEN);
        let prob = match vad.process(&frame) {
            Ok(prob) => prob,
            Err(err) => {
                // Inference failures are permanent: report fatal and exit.
                emit_error(&events, &err);
                let _ = fatal_tx.send(err).await;
                break;
            }
        };
        let is_speech = prob >= threshold;
        let event = segmenter.push(&frame, prob);
        let is_start = matches!(event, Some(VadEvent::SpeechStart));
        let forward_start = gate.filter(is_start, is_speech, sink.is_playing());
        if forward_start {
            emit(&events, AgentEvent::SpeechStart);
        }
        let msg = match event {
            Some(VadEvent::Segment(samples)) => {
                vad.reset_state();
                Some(VadEventMsg::Segment {
                    samples,
                    t_speech_end: Instant::now(),
                })
            }
            _ if forward_start => Some(VadEventMsg::SpeechStart),
            _ => None,
        };
        if let Some(msg) = msg {
            let sent = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                sent = events_tx.send(msg) => sent,
            };
            if sent.is_err() {
                debug!("orchestrator gone; VAD task exiting");
                break;
            }
        }
    }
}

/// Context shared by the STT/LLM/TTS stage tasks (bundled to stay under
/// the argument lint). All fields are cheap shared-handle clones.
#[derive(Clone)]
struct StageCtx {
    /// The current (latest) turn id; stale-turn drops compare against it.
    current_turn: Arc<AtomicU64>,
    /// Pipeline-wide shutdown.
    shutdown: CancellationToken,
    /// Fatal-error channel (§6).
    fatal_tx: mpsc::Sender<SkadooshError>,
    /// Agent-event broadcast.
    events: broadcast::Sender<AgentEvent>,
    /// Optional wake word: transcripts that do NOT contain this word
    /// are silently dropped before reaching the LLM.
    wake_word: Option<String>,
}

/// STT bridge (§7 task 4): segment → `transcribe` oneshot await → text
/// mpsc. STT is never cancelled (§8); the only early-abandon is shutdown.
/// Stale turns (superseded while whisper ran) are dropped defensively; an
/// evicted job (drop-oldest) is benign, a dead worker is fatal.
async fn stt_bridge(
    mut segment_rx: mpsc::Receiver<SegmentMsg>,
    text_tx: mpsc::Sender<TextMsg>,
    stt: Box<dyn SttEngine>,
    ctx: StageCtx,
) {
    let StageCtx {
        current_turn,
        shutdown,
        fatal_tx,
        events,
        ..
    } = ctx;
    loop {
        let msg = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            msg = segment_rx.recv() => match msg {
                Some(msg) => msg,
                None => break, // orchestrator dropped the sender: shutting down
            },
        };
        let SegmentMsg {
            turn_id,
            token,
            samples,
            t_speech_end,
        } = msg;
        if turn_id != current_turn.load(Ordering::SeqCst) {
            debug!(turn_id, "dropping stale segment before transcription");
            continue;
        }
        let dropped_before = stt.dropped_jobs();
        // A closed reply channel means the job was evicted (drop-oldest) or
        // the worker is gone; `dropped_jobs` tells the two apart below.
        let result = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            reply = stt.transcribe(samples) => match reply {
                Ok(result) => result,
                Err(_) => Err(SttError::WorkerGone.into()),
            },
        };
        let text = match result {
            Ok(text) => text,
            Err(err) => {
                let evicted = stt.dropped_jobs() > dropped_before;
                if shutdown.is_cancelled() || evicted {
                    debug!(
                        turn_id,
                        evicted, "STT job dropped during drain/eviction (benign)"
                    );
                    continue;
                }
                warn!(turn_id, error = %err, "fatal STT error");
                emit_error(&events, &err);
                let _ = fatal_tx.send(err).await;
                break;
            }
        };
        if text.trim().is_empty() {
            debug!(turn_id, "empty transcript; skipping turn");
            continue;
        }
        if token.is_cancelled() || turn_id != current_turn.load(Ordering::SeqCst) {
            debug!(turn_id, "dropping stale transcript");
            continue;
        }
        let msg = TextMsg {
            turn_id,
            token,
            text,
            t_speech_end,
            t_text: Instant::now(),
        };
        let sent = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            sent = text_tx.send(msg) => sent,
        };
        if sent.is_err() {
            if shutdown.is_cancelled() {
                break;
            }
            let err: SkadooshError = anyhow::anyhow!("LLM task channel closed unexpectedly").into();
            emit_error(&events, &err);
            let _ = fatal_tx.send(err).await;
            break;
        }
    }
    // Drain: stop the STT worker (joins the thread) off the async executor.
    let stopped = tokio::task::spawn_blocking(move || stt.stop()).await;
    if let Err(join_err) = stopped {
        warn!(error = %join_err, "STT stop panicked");
    }
}

/// LLM task (§7 task 5): transcript → SSE stream with the per-turn child
/// token → per-turn clause channel handed to the TTS task. `Cancelled` is
/// benign (barge-in/shutdown); any other stream error is fatal. The
/// completed/cancelled turn is reported back so the orchestrator can mark
/// it stream-done — the token stays live, because the TTS backlog can
/// outlive the stream and barge-in must still be able to cancel it.
async fn llm_task(
    mut text_rx: mpsc::Receiver<TextMsg>,
    turn_tx: mpsc::Sender<TurnMsg>,
    turn_done_tx: mpsc::Sender<u64>,
    mut client: Box<dyn LlmBackend>,
    ctx: StageCtx,
) {
    let StageCtx {
        current_turn,
        shutdown,
        fatal_tx,
        events,
        wake_word,
    } = ctx;
    loop {
        let msg = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            msg = text_rx.recv() => match msg {
                Some(msg) => msg,
                None => break,
            },
        };
        let TextMsg {
            turn_id,
            token,
            text,
            t_speech_end,
            t_text,
        } = msg;
        if token.is_cancelled() || turn_id != current_turn.load(Ordering::SeqCst) {
            debug!(turn_id, "dropping stale transcript before LLM request");
            continue;
        }
        // Wake word gating: when set, discard transcripts that do not
        // contain the wake word (case-insensitive substring match).
        if let Some(ref ww) = wake_word {
            if !text.to_lowercase().contains(&ww.to_lowercase()) {
                debug!(turn_id, %text, wake_word = %ww, "transcript missing wake word; skipping turn");
                continue;
            }
        }
        let (clause_tx, clause_rx) = mpsc::channel(CLAUSE_CAP);
        let turn = TurnMsg {
            turn_id,
            token: token.clone(),
            clauses: clause_rx,
            t_speech_end,
            t_text,
        };
        let sent = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            sent = turn_tx.send(turn) => sent,
        };
        if sent.is_err() {
            if shutdown.is_cancelled() {
                break;
            }
            let err: SkadooshError = anyhow::anyhow!("TTS task channel closed unexpectedly").into();
            emit_error(&events, &err);
            let _ = fatal_tx.send(err).await;
            break;
        }
        debug!(turn_id, %text, "LLM turn started");
        emit(&events, AgentEvent::Transcript(text.clone()));
        let result = client.stream_reply(&text, clause_tx, token).await;
        // Best effort: the orchestrator marks the turn stream-done (keeping
        // the token live for barge-in against the TTS backlog); a lost
        // notification is benign (the next segment supersedes it anyway).
        let _ = turn_done_tx.try_send(turn_id);
        match result {
            Ok(()) => {}
            Err(SkadooshError::Llm(LlmError::Cancelled)) => {
                debug!(turn_id, "LLM stream cancelled (barge-in or shutdown)");
            }
            Err(err) if shutdown.is_cancelled() => {
                debug!(turn_id, error = %err, "LLM stream error during shutdown (benign)");
            }
            Err(err) => {
                warn!(turn_id, error = %err, "fatal LLM error");
                emit_error(&events, &err);
                let _ = fatal_tx.send(err).await;
                break;
            }
        }
    }
}

/// TTS task (§7 task 6): clause → `engine.synthesize` on the blocking pool →
/// clip → playback sink. The turn token is checked between clauses and again
/// after each synthesis, so a cancelled turn emits no further clips; stale
/// `turn_id`s are dropped defensively. The first queued clip spawns the
/// first-audible watcher that logs the per-turn latency summary (§8).
///
/// Text mode ([`OutputMode::Text`], `engine == None`): nothing is
/// synthesized and the sink is never touched — each clause is surfaced as
/// an [`AgentEvent::Clause`] (the binary prints them) and the per-turn
/// [`AgentEvent::StageLatency`] is emitted when the stream ends (with
/// `tts_ms`/`playback_ms` zeroed: those stages do not exist in text mode).
async fn tts_task<C: ClipSink>(
    mut turn_rx: mpsc::Receiver<TurnMsg>,
    mut engine: Option<Box<dyn TtsEngine>>,
    sink: C,
    ctx: StageCtx,
) {
    let StageCtx {
        current_turn,
        shutdown,
        fatal_tx,
        events,
        ..
    } = ctx;
    'outer: loop {
        let turn = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            turn = turn_rx.recv() => match turn {
                Some(turn) => turn,
                None => break,
            },
        };
        let TurnMsg {
            turn_id,
            token,
            mut clauses,
            t_speech_end,
            t_text,
        } = turn;
        let mut timing = TurnTiming {
            t_speech_end,
            t_text,
            t_first_clause: None,
            t_first_clip: None,
        };
        'turn: loop {
            let clause = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break 'outer,
                _ = token.cancelled() => break 'turn,
                clause = clauses.recv() => match clause {
                    Some(clause) => clause,
                    None => break 'turn, // LLM stream ended for this turn
                },
            };
            let t_clause = Instant::now(); // first-clause (≈ first-token) stamp
            if token.is_cancelled() || turn_id != current_turn.load(Ordering::SeqCst) {
                debug!(turn_id, "dropping stale clause");
                continue;
            }
            timing.t_first_clause.get_or_insert(t_clause);
            emit(&events, AgentEvent::Clause(clause.clone()));
            // Text mode: no synthesis; the clause event above is the whole
            // output path for it. (On the panic path below `engine` stays
            // `None` — but the task exits anyway, so no turn can observe it.)
            let Some(owned) = engine.take() else {
                continue;
            };
            let (returned_engine, result) = match synthesize_clause(owned, clause).await {
                Ok(pair) => pair,
                Err(join_err) => {
                    // The engine unwound with the panic; always fatal.
                    if !shutdown.is_cancelled() {
                        warn!(turn_id, error = %join_err, "TTS synthesis panicked");
                        let err: SkadooshError =
                            anyhow::anyhow!("TTS synthesis panicked: {join_err}").into();
                        emit_error(&events, &err);
                        let _ = fatal_tx.send(err).await;
                    }
                    break 'outer;
                }
            };
            engine = Some(returned_engine);
            let clip = match result {
                Ok(clip) => clip,
                Err(err) => {
                    if shutdown.is_cancelled() || token.is_cancelled() {
                        break 'turn; // unwind in progress: benign
                    }
                    warn!(turn_id, error = %err, "fatal TTS error");
                    emit_error(&events, &err);
                    let _ = fatal_tx.send(err).await;
                    break 'outer;
                }
            };
            if token.is_cancelled() || turn_id != current_turn.load(Ordering::SeqCst) {
                debug!(turn_id, "discarding clip synthesized after cancel");
                continue;
            }
            let queued = tokio::select! {
                biased;
                _ = shutdown.cancelled() => break 'outer,
                _ = token.cancelled() => break 'turn,
                queued = sink.queue_clip(clip) => queued,
            };
            if let Err(err) = queued {
                if shutdown.is_cancelled() || token.is_cancelled() {
                    break 'turn;
                }
                warn!(turn_id, error = %err, "fatal playback error");
                emit_error(&events, &err);
                let _ = fatal_tx.send(err).await;
                break 'outer;
            }
            if timing.t_first_clip.is_none() {
                timing.t_first_clip = Some(Instant::now());
                tokio::spawn(
                    audible_watcher(
                        sink.clone(),
                        token.clone(),
                        turn_id,
                        timing.clone(),
                        events.clone(),
                    )
                    .instrument(info_span!("playback")),
                );
            }
        }
        // Turn ended: if the clause channel closed cleanly (LLM stream
        // done, not cancelled), every Clause event for the turn has been
        // emitted — only NOW is the reply done (ordered after the clauses
        // and the latency summary).
        if !token.is_cancelled() && !shutdown.is_cancelled() {
            // Text mode: report the per-turn latency at stream end (the
            // audible_watcher reports it in audio mode).
            if engine.is_none() {
                if let Some(t_clause) = timing.t_first_clause {
                    let now = Instant::now();
                    let stt_ms = millis(timing.t_text - timing.t_speech_end);
                    let llm_ms = millis(t_clause - timing.t_text);
                    let total_ms = millis(now - timing.t_speech_end);
                    info!(
                        turn_id,
                        stt_ms,
                        llm_ms,
                        tts_ms = 0,
                        playback_ms = 0,
                        total_ms,
                        "turn latency: speech-end → reply stream end (text mode)"
                    );
                    emit(
                        &events,
                        AgentEvent::StageLatency {
                            stt_ms,
                            llm_ms,
                            tts_ms: 0,
                            playback_ms: 0,
                            total_ms,
                        },
                    );
                }
            }
            emit(&events, AgentEvent::ReplyDone);
            // Ready for the next utterance. Emitted here — after the
            // turn's whole clause backlog drained through this task — and
            // not in the orchestrator's stream-done handler, so Listening
            // can never race ahead of the turn's Clause/ReplyDone events.
            emit(&events, AgentEvent::Listening);
        }
    }
}

/// Synthesizes one clause on the blocking pool. The engine round-trips
/// through the closure (a `Box<dyn TtsEngine>` is not `Clone`) and always
/// comes back — unless the closure panicked, in which case it unwound with
/// the panic and only the [`tokio::task::JoinError`] is returned.
async fn synthesize_clause(
    engine: Box<dyn TtsEngine>,
    clause: String,
) -> std::result::Result<(Box<dyn TtsEngine>, Result<TtsClip>), tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let mut engine = engine;
        let result = engine.synthesize(&clause);
        (engine, result)
    })
    .await
}

/// Selftest helper: synthesizes one clause on the blocking pool, appending
/// the clip + clause text and stamping the first-clause/first-clip marks.
/// Returns the engine for reuse (a `Box<dyn TtsEngine>` is not `Clone`).
async fn synth_selftest_clause(
    engine: Box<dyn TtsEngine>,
    clause: String,
    clips: &mut Vec<TtsClip>,
    clause_texts: &mut Vec<String>,
    t_first_clause: &mut Option<Instant>,
    t_first_clip: &mut Option<Instant>,
) -> Result<Box<dyn TtsEngine>> {
    t_first_clause.get_or_insert_with(Instant::now);
    let (engine, result) = synthesize_clause(engine, clause.clone())
        .await
        .map_err(|err| anyhow::anyhow!("TTS synthesis panicked: {err}"))?;
    let clip = result?;
    t_first_clip.get_or_insert_with(Instant::now);
    clause_texts.push(clause);
    clips.push(clip);
    Ok(engine)
}

/// First-audible watcher (§8 `t_first_audible`): polls `is_playing` until
/// the playback callback leaves silence, then logs the one-per-turn latency
/// summary line and emits [`AgentEvent::StageLatency`]. Exits quietly on
/// turn cancel or timeout.
async fn audible_watcher<C: ClipSink>(
    sink: C,
    token: CancellationToken,
    turn_id: u64,
    timing: TurnTiming,
    events: broadcast::Sender<AgentEvent>,
) {
    let started = Instant::now();
    loop {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                debug!(turn_id, "turn cancelled before first audible sample");
                return;
            }
            _ = tokio::time::sleep(AUDIBLE_POLL) => {
                if sink.is_playing() {
                    let t_audible = Instant::now();
                    let (Some(t_clause), Some(t_clip)) =
                        (timing.t_first_clause, timing.t_first_clip)
                    else {
                        return; // unreachable: the watcher spawns after the first clip
                    };
                    let stt_ms = millis(timing.t_text - timing.t_speech_end);
                    let llm_ms = millis(t_clause - timing.t_text);
                    let tts_ms = millis(t_clip - t_clause);
                    let playback_ms = millis(t_audible - t_clip);
                    let total_ms = millis(t_audible - timing.t_speech_end);
                    info!(
                        turn_id,
                        stt_ms,
                        llm_ms,
                        tts_ms,
                        playback_ms,
                        total_ms,
                        "turn latency: speech-end → first audible sample"
                    );
                    emit(
                        &events,
                        AgentEvent::StageLatency {
                            stt_ms,
                            llm_ms,
                            tts_ms,
                            playback_ms,
                            total_ms,
                        },
                    );
                    return;
                }
                if started.elapsed() > AUDIBLE_TIMEOUT {
                    debug!(turn_id, "first-audible wait timed out (stalled device?)");
                    return;
                }
            }
        }
    }
}

/// Broadcasts an event, ignoring the "no subscribers" error (a plain
/// [`Pipeline::new`] run has none).
fn emit(events: &broadcast::Sender<AgentEvent>, event: AgentEvent) {
    let _ = events.send(event);
}

/// Broadcasts an [`AgentEvent::Error`] with the full `anyhow` cause chain
/// (the first line alone is often too terse to act on).
fn emit_error(events: &broadcast::Sender<AgentEvent>, err: &SkadooshError) {
    emit(events, AgentEvent::Error(format!("{err:?}")));
}

fn millis(d: Duration) -> u64 {
    d.as_millis() as u64
}

/// Push-to-talk recording task: replaces the VAD task when `--push-to-talk`
/// is set. A keyboard listener on a dedicated OS thread toggles a recording
/// flag on Enter; the async loop drains the mic ring buffer, accumulates
/// samples while recording, and emits `Segment` on release.
async fn push_to_talk_task(
    capture: MicCapture,
    mut cons: HeapCons<f32>,
    vad_tx: mpsc::Sender<VadEventMsg>,
    _fatal_tx: mpsc::Sender<SkadooshError>,
    shutdown: CancellationToken,
    _events: broadcast::Sender<AgentEvent>,
) {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // MicCapture is already started by the caller; we hold it to keep
    // the stream alive. Samples arrive via `cons`.
    let _capture = capture;

    let recording = Arc::new(AtomicBool::new(false));
    let rec = recording.clone();

    std::thread::spawn(move || {
        let _raw = crossterm::terminal::enable_raw_mode();
        loop {
            if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    if matches!(key.code, KeyCode::Enter | KeyCode::Char(' ')) {
                        rec.store(!rec.load(Ordering::SeqCst), Ordering::SeqCst);
                    } else if matches!(key.code, KeyCode::Esc) {
                        rec.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
        }
        let _ = crossterm::terminal::disable_raw_mode();
    });

    let mut turn_id: u64 = 0;
    let mut accumulating: Vec<f32> = Vec::new();
    let mut was_recording = false;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            _ = tokio::time::sleep(Duration::from_millis(30)) => {
                // Drain the ring buffer.
                while let Some(s) = cons.try_pop() {
                    accumulating.push(s);
                }

                let is_recording = recording.load(Ordering::SeqCst);

                if is_recording && !was_recording {
                    // Recording just started.
                    turn_id = turn_id.wrapping_add(1);
                    accumulating.clear();
                    let _ = vad_tx.send(VadEventMsg::SpeechStart).await;
                }

                if !is_recording && was_recording && !accumulating.is_empty() {
                    // Recording just stopped — send segment.
                    let samples = std::mem::take(&mut accumulating);
                    let t_speech_end = Instant::now();
                    let _ = vad_tx
                        .send(VadEventMsg::Segment {
                            samples,
                            t_speech_end,
                        })
                        .await;
                }

                was_recording = is_recording;
            }
        }
    }
}

/// Orchestrator-side state for the in-flight turn. The entry survives the
/// end of the LLM stream (`llm_done`): the TTS task can still hold a
/// buffered clause backlog ([`CLAUSE_CAP`]) plus queued clips, so barge-in
/// must keep its cancel capability until the turn is barged in, superseded
/// by the next segment, or shut down.
struct ActiveTurn {
    turn_id: u64,
    token: CancellationToken,
    /// The LLM stream ended; the TTS backlog may still be draining.
    llm_done: bool,
}

/// The orchestrator core (§7 task 8): spawns the STT bridge, LLM, and TTS
/// tasks; dispatches VAD events (barge-in + turn minting); owns the fatal
/// channel; unwinds in the §6 order. Exposed (hidden) for integration tests
/// — [`Pipeline::run`] is the production entry point.
#[doc(hidden)]
pub async fn run_orchestrator<C: ClipSink>(topology: Topology<C>) -> Result<()> {
    let Topology {
        vad_events: mut vad_rx,
        fatal_tx,
        mut fatal_rx,
        stt,
        llm,
        tts_engine,
        sink,
        shutdown,
        events,
        wake_word,
        hold_music: hold_music_flag,
        mut watch_rx,
    } = topology;

    let current_turn = Arc::new(AtomicU64::new(0));
    let (segment_tx, segment_rx) = mpsc::channel(SEGMENT_CAP);
    let (text_tx, text_rx) = mpsc::channel(TEXT_CAP);
    let (turn_tx, turn_rx) = mpsc::channel(TURN_CAP);
    let (turn_done_tx, mut turn_done_rx) = mpsc::channel(TURN_DONE_CAP);
    // Clone kept by the orchestrator to inject watch events straight into the
    // LLM task (as system-injected user turns), bypassing STT.
    let watch_text_tx = text_tx.clone();

    let ctx = StageCtx {
        current_turn,
        shutdown: shutdown.clone(),
        fatal_tx: fatal_tx.clone(),
        // Cloned: the orchestrator itself still emits Listening/TurnCancelled.
        events: events.clone(),
        wake_word,
    };
    let mut tasks = tokio::task::JoinSet::new();
    tasks.spawn(stt_bridge(segment_rx, text_tx, stt, ctx.clone()).instrument(info_span!("stt")));
    tasks.spawn(
        llm_task(text_rx, turn_tx, turn_done_tx, llm, ctx.clone()).instrument(info_span!("llm")),
    );
    tasks.spawn(
        tts_task(turn_rx, tts_engine, sink.clone(), ctx.clone()).instrument(info_span!("tts")),
    );

    // If hold music is enabled, spawn a feeder that generates and queues
    // procedural music through the sink when the flag is active (during
    // tool execution).
    if let Some(ref hm_flag) = hold_music_flag {
        let hm_flag = Arc::clone(hm_flag);
        let hm_sink = sink.clone();
        let hm_shutdown = shutdown.clone();
        tasks.spawn(
            async move {
                hold_music_feeder(hm_flag, hm_sink, hm_shutdown).await;
            }
            .instrument(info_span!("hold-music")),
        );
    }

    // The orchestrator itself dispatches on `ctx.current_turn`.
    let current_turn = ctx.current_turn;

    let mut active_turn: Option<ActiveTurn> = None;
    let mut fatal: Option<SkadooshError> = None;
    // Latches closed once the watch channel closes (all watchers done) so the
    // watch select branch stops firing — avoiding a busy loop on `recv` None.
    let watch_done = AtomicBool::new(false);

    // The topology is up; the agent is waiting for speech.
    emit(&events, AgentEvent::Listening);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                debug!("shutdown requested");
                break;
            }
            err = fatal_rx.recv() => {
                match err {
                    Some(err) => {
                        warn!(error = %err, "fatal error; shutting down");
                        fatal = Some(err);
                    }
                    None => {
                        // Every sender gone: all tasks exited already.
                        debug!("fatal channel closed; tasks exited");
                    }
                }
                break;
            }
            done = turn_done_rx.recv() => {
                if let Some(done_id) = done {
                    // Mark the stream done but KEEP the token: the TTS task
                    // may still hold a buffered clause backlog + queued
                    // clips, and barge-in must stay able to cancel them.
                    // The entry is cleared by barge-in, by the next
                    // segment's supersede, or at shutdown.
                    if let Some(turn) = &mut active_turn {
                        if turn.turn_id == done_id {
                            turn.llm_done = true;
                            // The turn-end `Listening` event is the TTS
                            // task's job (after the clause backlog drains
                            // and ReplyDone fires) — emitting it here would
                            // race ahead of the turn's events.
                        }
                    }
                }
            }
            event = vad_rx.recv() => {
                let Some(event) = event else {
                    if shutdown.is_cancelled() {
                        break;
                    }
                    fatal = Some(
                        anyhow::anyhow!("VAD event stream closed unexpectedly").into(),
                    );
                    break;
                };
                match event {
                    VadEventMsg::SpeechStart => {
                        // Barge-in only while playback is audible (§8): a
                        // SpeechStart in a silent gap must NOT cancel — a
                        // one-frame VAD false positive there is rejected by
                        // the segmenter's min-length guard, so no
                        // replacement utterance ever arrives and a
                        // mid-stream reply would silently die. A genuine new
                        // utterance still cancels the turn via the supersede
                        // below.
                        if sink.is_playing() {
                            sink.flush();
                            if let Some(turn) = active_turn.take() {
                                info!(
                                    turn_id = turn.turn_id,
                                    llm_done = turn.llm_done,
                                    "barge-in: cancelled turn, flushed playback"
                                );
                                turn.token.cancel();
                                emit(&events, AgentEvent::TurnCancelled);
                            } else {
                                debug!("barge-in flush with no active LLM turn");
                            }
                        }
                    }
                    VadEventMsg::Segment { samples, t_speech_end } => {
                        // Defensive: a fresh utterance supersedes any
                        // in-flight turn (normally barge-in already cancelled
                        // it at SpeechStart).
                        if let Some(turn) = active_turn.take() {
                            debug!(
                                turn_id = turn.turn_id,
                                llm_done = turn.llm_done,
                                "superseding in-flight turn"
                            );
                            turn.token.cancel();
                        }
                        let turn_id = current_turn.fetch_add(1, Ordering::SeqCst) + 1;
                        let token = shutdown.child_token();
                        active_turn = Some(ActiveTurn {
                            turn_id,
                            token: token.clone(),
                            llm_done: false,
                        });
                        info!(
                            turn_id,
                            audio_ms = samples.len() as u64 * 1000 / u64::from(CAPTURE_RATE),
                            "speech segment captured"
                        );
                        let msg = SegmentMsg {
                            turn_id,
                            token,
                            samples,
                            t_speech_end,
                        };
                        let sent = tokio::select! {
                            biased;
                            _ = shutdown.cancelled() => break,
                            sent = segment_tx.send(msg) => sent,
                        };
                        if sent.is_err() {
                            if shutdown.is_cancelled() {
                                break;
                            }
                            fatal = Some(
                                anyhow::anyhow!("STT bridge channel closed unexpectedly").into(),
                            );
                            break;
                        }
                    }
                }
            }
            ev = async {
                if watch_done.load(Ordering::Relaxed) {
                    std::future::pending::<Option<WatchEvent>>().await
                } else {
                    match watch_rx.as_mut() {
                        Some(rx) => rx.recv().await,
                        None => std::future::pending::<Option<WatchEvent>>().await,
                    }
                }
            } => {
                match ev {
                    Some(event) => {
                        // Treat the trigger like a user-initiated turn:
                        // supersede any in-flight turn, mint a fresh one, and
                        // inject the notification straight into the LLM task
                        // (bypassing STT — the text is already known). The LLM
                        // task emits the Transcript event, so the notification
                        // surfaces like any other user utterance.
                        if let Some(turn) = active_turn.take() {
                            debug!(
                                turn_id = turn.turn_id,
                                llm_done = turn.llm_done,
                                "superseding in-flight turn for watch event"
                            );
                            turn.token.cancel();
                        }
                        let turn_id = current_turn.fetch_add(1, Ordering::SeqCst) + 1;
                        let token = shutdown.child_token();
                        active_turn = Some(ActiveTurn {
                            turn_id,
                            token: token.clone(),
                            llm_done: false,
                        });
                        let text = format!("NOTIFICATION: {}", event.message());
                        info!(turn_id, %text, "watch event injected as user turn");
                        let msg = TextMsg {
                            turn_id,
                            token,
                            text,
                            t_speech_end: Instant::now(),
                            t_text: Instant::now(),
                        };
                        let sent = tokio::select! {
                            biased;
                            _ = shutdown.cancelled() => break,
                            sent = watch_text_tx.send(msg) => sent,
                        };
                        if sent.is_err() {
                            if shutdown.is_cancelled() {
                                break;
                            }
                            fatal = Some(
                                anyhow::anyhow!("LLM task channel closed unexpectedly").into(),
                            );
                            break;
                        }
                    }
                    None => {
                        // Watch channel closed (all watchers finished); latch
                        // so this branch stops firing instead of busy-looping
                        // on repeated `recv` None.
                        watch_done.store(true, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    // §6 shutdown ordering: cancel tokens → stop sources (the senders this
    // scope owns close on drop; the VAD task sees the cancelled token and
    // drops the mic stream) → drain → join.
    shutdown.cancel();
    if let Some(turn) = active_turn.take() {
        turn.token.cancel();
    }
    drop(segment_tx);
    drop(fatal_tx);
    drop(watch_text_tx);
    while let Some(joined) = tasks.join_next().await {
        if let Err(join_err) = joined {
            warn!(error = %join_err, "pipeline task panicked");
            if fatal.is_none() {
                fatal = Some(anyhow::anyhow!("pipeline task panicked: {join_err}").into());
            }
        }
    }

    match fatal {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// `Pipeline::run_selftest` body, split out so the sync wrapper can own
/// runtime construction.
impl Pipeline {
    async fn selftest_async(self, wav: &Path, out_wav: &Path) -> Result<SelftestReport> {
        let t_start = Instant::now();

        // Stage 0: wav in → 16 kHz mono f32.
        let (mono, src_rate) = read_wav(wav)?;
        let samples = resample_offline(&mono, src_rate, CAPTURE_RATE);
        let t_loaded = Instant::now();

        // Stage 1: VAD segmentation (real Silero + segmenter), first segment
        // only. Trailing silence forces the endpoint to close, like a live
        // stream would.
        let mut vad = SileroVad::new(&self.config.vad_model)?;
        let mut segmenter = VadSegmenter::new(self.config.vad_threshold, self.config.silence_ms);
        let silence_frames = (self.config.silence_ms / 32 + 2) as usize;
        let mut segment = None;
        let mut feed = samples;
        feed.extend(std::iter::repeat_n(0.0, silence_frames * FRAME_LEN));
        for frame in feed.as_chunks::<FRAME_LEN>().0 {
            let prob = vad.process(frame)?;
            if let Some(VadEvent::Segment(audio)) = segmenter.push(frame, prob) {
                segment = Some(audio);
                break;
            }
        }
        let segment = segment.ok_or_else(|| {
            anyhow::anyhow!(
                "no speech segment detected in {} (needs audible speech followed by \
                 > {} ms of silence)",
                wav.display(),
                self.config.silence_ms
            )
        })?;
        let t_segment = Instant::now();

        // Stage 2: whisper STT on the dedicated worker thread.
        let stt = WhisperStt::start(&self.config.whisper_model, &SttConfig::default())?;
        let transcript = stt
            .transcribe(segment)
            .await
            .map_err(|_| SttError::WorkerGone)??;
        stt.stop();
        let t_text = Instant::now();
        if transcript.trim().is_empty() {
            return Err(anyhow::anyhow!("STT produced an empty transcript").into());
        }

        // Stage 3: LLM SSE stream. Driven directly (rather than via
        // LlmClient::stream_reply, which only surfaces completed clauses) so
        // the report gets a true time-to-first-token; parsing reuses the
        // client's tolerant `parse_sse_line`.
        let http = reqwest::Client::new();
        let url = format!(
            "{}/chat/completions",
            self.config.llm_url.trim_end_matches('/')
        );
        let body = serde_json::json!({
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": transcript},
            ],
            "stream": true,
        });
        let t_llm = Instant::now();
        let mut request = http.post(&url).json(&body);
        if let Some(key) = &self.config.api_key {
            // Same bearer wiring as LlmClient: keyed providers 401 without it.
            request = request.bearer_auth(key);
        }
        let resp = request.send().await.map_err(LlmError::Http)?;
        let resp = ensure_success(resp).await?;

        // Stages 3+4 interleaved: clause-split the token stream and
        // synthesize each clause as it completes.
        let mut engine = build_engine(&self.config)?;
        let mut splitter = ClauseSplitter::new(CLAUSE_MIN_LEN, CLAUSE_MAX_LEN);
        let mut clips: Vec<TtsClip> = Vec::new();
        let mut clause_texts: Vec<String> = Vec::new();
        let mut t_first_token: Option<Instant> = None;
        let mut t_first_clause: Option<Instant> = None;
        let mut t_first_clip: Option<Instant> = None;
        let mut stream = resp.bytes_stream();
        let mut lines = SseLineBuffer::default();
        let mut done = false;
        let mut eof = false;
        // Same shape as `LlmClient::stream_reply`: at a clean connection
        // close (with or without `data: [DONE]`) `close()` makes
        // `next_line` yield any unterminated final line once, so a server
        // that omits the trailing `\n` loses no content.
        while !done && !eof {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    if !lines.feed(&bytes) {
                        return Err(LlmError::Sse("SSE line exceeded maximum size".into()).into());
                    }
                }
                Some(Err(err)) => return Err(LlmError::Http(err).into()),
                None => {
                    lines.close();
                    eof = true;
                }
            }
            while let Some(line) = lines.next_line() {
                match parse_sse_line(&line) {
                    None => {}
                    Some(Ok(None)) => {
                        // data: [DONE]
                        done = true;
                        break;
                    }
                    Some(Ok(Some(token))) => {
                        t_first_token.get_or_insert_with(Instant::now);
                        for clause in splitter.push(&token) {
                            engine = synth_selftest_clause(
                                engine,
                                clause,
                                &mut clips,
                                &mut clause_texts,
                                &mut t_first_clause,
                                &mut t_first_clip,
                            )
                            .await?;
                        }
                    }
                    Some(Err(err)) => {
                        warn!(error = %err, "skipping malformed SSE data line");
                    }
                }
            }
        }
        if let Some(rest) = splitter.flush() {
            // Stream over: the returned engine is dropped with the report.
            synth_selftest_clause(
                engine,
                rest,
                &mut clips,
                &mut clause_texts,
                &mut t_first_clause,
                &mut t_first_clip,
            )
            .await?;
        }
        if clips.is_empty() {
            return Err(anyhow::anyhow!("LLM reply produced no clauses").into());
        }
        let t_llm_done = Instant::now();

        // Stage 5: concatenate clips → 24 kHz 16-bit PCM wav.
        let pcm = concat_clip_samples(&clips);
        write_wav16(out_wav, &pcm, TTS_SAMPLE_RATE)?;
        let total_ms = millis(t_start.elapsed());
        info!(
            clauses = clause_texts.len(),
            audio_ms = pcm.len() as u64 * 1000 / u64::from(TTS_SAMPLE_RATE),
            out_wav = %out_wav.display(),
            "selftest complete"
        );

        Ok(SelftestReport {
            segment_ms: millis(t_segment - t_loaded),
            stt_ms: millis(t_text - t_segment),
            llm_ttft_ms: millis(t_first_token.unwrap_or(t_llm_done) - t_llm),
            first_clause_ms: millis(t_first_clause.unwrap_or(t_llm_done) - t_llm),
            tts_ms: millis(
                t_first_clip.unwrap_or(t_llm_done) - t_first_clause.unwrap_or(t_llm_done),
            ),
            total_ms,
            transcript,
        })
    }
}

/// Minimal WAV reader (PCM 8/16/24/32-bit int and 32-bit float, any channel
/// count, mixed down to mono f32).
///
/// `hound` is a dev-dependency of this crate, so library code parses the
/// RIFF container itself; the `--selftest` input contract is "16-bit PCM,
/// any rate" and this accepts a superset of that.
fn read_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)
        .map_err(|err| anyhow::anyhow!("failed to read {}: {err}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(anyhow::anyhow!("{} is not a RIFF/WAVE file", path.display()).into());
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size =
            u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().expect("4 bytes")) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(bytes.len());
        match id {
            // Guard on the bytes actually present, not the declared size:
            // a truncated file must not panic the slice indexing below.
            b"fmt " if body_end - body_start >= 16 => {
                let b = &bytes[body_start..body_end];
                let format = u16::from_le_bytes(b[0..2].try_into().expect("2 bytes"));
                let channels = u16::from_le_bytes(b[2..4].try_into().expect("2 bytes"));
                let rate = u32::from_le_bytes(b[4..8].try_into().expect("4 bytes"));
                let bits = u16::from_le_bytes(b[14..16].try_into().expect("2 bytes"));
                fmt = Some((format, channels, rate, bits));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        // Chunks are padded to even sizes.
        pos = body_start + size + (size & 1);
    }

    let (format, channels, rate, bits) =
        fmt.ok_or_else(|| anyhow::anyhow!("{}: missing fmt chunk", path.display()))?;
    let data = data.ok_or_else(|| anyhow::anyhow!("{}: missing data chunk", path.display()))?;
    let channels = usize::from(channels);
    if channels == 0 || rate == 0 {
        return Err(anyhow::anyhow!("{}: bad fmt chunk", path.display()).into());
    }

    let per_sample = |b: &[u8]| -> Result<f32> {
        match (format, bits) {
            (1, 8) => Ok((f32::from(b[0]) - 128.0) / 128.0),
            (1, 16) => {
                Ok(f32::from(i16::from_le_bytes(b[0..2].try_into().expect("2 bytes"))) / 32768.0)
            }
            (1, 24) => {
                let v =
                    i32::from_le_bytes([b[0], b[1], b[2], if b[2] & 0x80 != 0 { 0xFF } else { 0 }]);
                Ok(v as f32 / 8_388_608.0)
            }
            (1, 32) => Ok(
                i32::from_le_bytes(b[0..4].try_into().expect("4 bytes")) as f32 / 2_147_483_648.0,
            ),
            (3, 32) => Ok(f32::from_le_bytes(b[0..4].try_into().expect("4 bytes"))),
            (format, bits) => Err(anyhow::anyhow!(
                "{}: unsupported wav format (format {format}, {bits} bits); \
                 supported: PCM 8/16/24/32-bit and 32-bit float",
                path.display()
            )
            .into()),
        }
    };

    let sample_bytes = usize::from(bits / 8);
    let frame_bytes = sample_bytes * channels;
    if frame_bytes == 0 {
        return Err(anyhow::anyhow!("{}: bad fmt chunk", path.display()).into());
    }
    let frames = data.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * frame_bytes;
        let mut acc = 0.0f32;
        for ch in 0..channels {
            acc += per_sample(&data[base + ch * sample_bytes..])?;
        }
        mono.push(acc / channels as f32);
    }
    Ok((mono, rate))
}

/// Background task: watches the hold-music active flag and feeds
/// procedural music through the clip sink when the flag is true (i.e.
/// during tool execution). Polls every ~100 ms to keep latency low.
async fn hold_music_feeder<C: ClipSink>(
    flag: Arc<AtomicBool>,
    sink: C,
    shutdown: CancellationToken,
) {
    let mut music = crate::audio::HoldMusic::new(Arc::clone(&flag));
    let poll_interval = std::time::Duration::from_millis(100);
    let chunk_samples = ((TTS_SAMPLE_RATE as f64) * 0.1) as usize; // ~100 ms

    loop {
        tokio::select! {
            _ = shutdown.cancelled() => break,
            _ = tokio::time::sleep(poll_interval) => {},
        }

        if !flag.load(Ordering::Relaxed) || sink.is_playing() {
            continue;
        }

        let samples = music.generate(chunk_samples);
        if samples.is_empty() {
            continue;
        }

        let clip = TtsClip {
            samples,
            sample_rate: TTS_SAMPLE_RATE,
        };
        if sink.queue_clip(clip).await.is_err() {
            break;
        }
    }
}
