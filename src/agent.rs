//! The SDK facade: one [`Agent`] type that assembles the voice pipeline
//! from a [`Config`] plus optional engine trait objects, broadcasts
//! [`AgentEvent`]s, and offers the three entry points an embedding
//! application needs — `run` (full audio loop, behind the `audio` feature),
//! [`text_turn`](Agent::text_turn) / [`repl`](Agent::repl) (text-in), and
//! `say` (behind the `audio` feature) / [`say_to_wav`](Agent::say_to_wav)
//! (speech out).
//!
//! # Quickstart
//!
//! ```no_run
//! use skadoosh::{Agent, Config, Result};
//!
//! fn main() -> Result<()> {
//!     let agent = Agent::builder().config(Config::default()).build()?;
//!     let mut events = agent.events();
//!     std::thread::spawn(move || while let Ok(_event) = events.blocking_recv() {});
//!     agent.run() // blocks until `shutdown()` or a fatal error
//! }
//! ```
//!
//! # Pluggability
//!
//! Any stage can be swapped for a custom implementation:
//! [`AgentBuilder::stt`] takes a [`SttEngine`],
//! [`AgentBuilder::llm`] an [`LlmBackend`], and
//! [`AgentBuilder::tts`] a [`TtsEngine`]. Stages
//! left unset are built from the [`Config`] exactly like the binary does
//! (`WhisperStt` → [`LlmClient`] →
//! [`build_engine`]). Only the stages a mode
//! actually uses are built: [`text_turn`](Agent::text_turn) never touches
//! STT, and [`repl`](Agent::repl) never touches STT or TTS — so a text-only
//! agent needs no model files at all.
//!
//! # Threading model
//!
//! The facade is synchronous from the caller's side: `run` blocks the
//! calling thread (driving an internal multi-thread tokio runtime), and the
//! per-call entry points (`text_turn`, `repl`, `say`) drive their async
//! stages on a scoped worker thread with a small private runtime. That
//! keeps them safe to call from any context — inside or outside an existing
//! tokio runtime — at the cost of one thread spawn per call, which is
//! negligible next to an LLM round-trip.

use std::io::{BufRead, Write};
use std::path::Path;

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

#[cfg(feature = "audio")]
use crate::audio::{AudioOutputConfig, Playback, PlaybackHandle};
use crate::config::{Config, OutputMode};
use crate::error::Result;
use crate::llm::client::{CLAUSE_MAX_LEN, CLAUSE_MIN_LEN};
use crate::llm::{ClauseSplitter, LlmBackend, LlmClient};
#[cfg(feature = "audio")]
use crate::pipeline::Pipeline;
use crate::stt::SttEngine;
use crate::tts::{build_engine, concat_clip_samples, TtsClip, TtsEngine, TTS_SAMPLE_RATE};
use crate::wav::write_wav16;

/// The concrete type queued clips are handed to. With the `audio` feature it
/// is [`Playback`] / [`PlaybackHandle`] (cpal playback); without it the
/// text/REPL paths still compile by discarding clips onto this no-op `()`
/// sink — they never receive a clip in practice (no playback field exists,
/// so `text_turn`/`repl` always pass `None`).
#[cfg(feature = "audio")]
type ClipSink = PlaybackHandle;
#[cfg(not(feature = "audio"))]
type ClipSink = ();

/// Capacity of the [`AgentEvent`] broadcast channel (subscribers that lag
/// more than this see `RecvError::Lagged`). Shared with
/// `Pipeline::new` (in the `pipeline` module, behind the `audio` feature),
/// which builds the same bus for non-SDK runs.
pub(crate) const EVENT_CAP: usize = 64;

/// Capacity of the per-turn clause channel between the LLM backend and the
/// text-turn driver.
const TEXT_TURN_CLAUSE_CAP: usize = 16;

/// Events broadcast by a running [`Agent`] (subscribe with
/// [`Agent::events`]).
///
/// Events are emitted from the orchestrator and stage tasks at the points
/// described on each variant. The stream is best-effort observability: with
/// no subscribers nothing is buffered, and slow subscribers may observe
/// `Lagged` gaps.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// The agent can accept the next utterance: emitted when the pipeline
    /// finishes starting, and again right after each completed turn's
    /// [`ReplyDone`](AgentEvent::ReplyDone) — never between a turn's
    /// [`Transcript`](AgentEvent::Transcript) and its `ReplyDone`. (In
    /// audio mode the reply's queued audio may still be playing when it
    /// fires.) A cancelled turn does not re-emit it.
    Listening,
    /// Speech onset detected by the VAD (past the 2-frame barge-in
    /// hangover). Full audio pipeline (`Agent::run`, behind the `audio`
    /// feature) only.
    SpeechStart,
    /// A user utterance was transcribed (STT output, LLM input).
    Transcript(String),
    /// One reply clause completed out of the LLM stream (TTS input in
    /// audio mode; the printed unit in text mode and the repl).
    Clause(String),
    /// A tool/function call was requested by the model (tool-calling
    /// support). The caller should execute the function and feed the
    /// result back — without an executor, the LLM is told to respond with
    /// text instead.
    ToolCall {
        /// Tool/function name.
        name: String,
        /// JSON arguments string.
        arguments: String,
    },
    /// The LLM reply stream for the current turn completed; every
    /// [`Clause`](AgentEvent::Clause) of the turn precedes it, and
    /// [`Listening`](AgentEvent::Listening) follows it immediately.
    ///
    /// Ordering against [`StageLatency`](AgentEvent::StageLatency) is
    /// mode-dependent: only text mode guarantees the latency event
    /// precedes `ReplyDone` (see its note). A turn cancelled mid-stream
    /// emits no `ReplyDone` (it ends with
    /// [`TurnCancelled`](AgentEvent::TurnCancelled)); a barge-in landing
    /// after the stream completed — while the unplayed audio backlog was
    /// still draining — emits `ReplyDone` first, then `TurnCancelled`
    /// when the flush hits.
    ReplyDone,
    /// The in-flight turn was cancelled by a barge-in (the pipeline's
    /// playback was audible and the user started speaking). When the
    /// reply stream had already completed, it follows
    /// [`ReplyDone`](AgentEvent::ReplyDone) — the cancellation then
    /// discards only the unplayed audio backlog.
    TurnCancelled,
    /// Per-turn latency breakdown (milliseconds), speech end → first
    /// audible sample.
    ///
    /// Ordering and presence are mode-dependent. In [`OutputMode::Text`]
    /// the TTS/playback stages do not exist: `tts_ms`/`playback_ms` are 0
    /// and the event fires at stream end — after the turn's clauses and
    /// before [`ReplyDone`](AgentEvent::ReplyDone). In
    /// [`OutputMode::Audio`] it is emitted by the first-audible watcher
    /// when playback actually starts: NOT ordered against `ReplyDone`,
    /// and absent entirely when the turn is cancelled before audibility
    /// or the first-audible wait times out (stalled device).
    StageLatency {
        /// Speech segment close → transcript.
        stt_ms: u64,
        /// Transcript → first reply clause.
        llm_ms: u64,
        /// First clause → first synthesized clip.
        tts_ms: u64,
        /// First clip queued → first audible sample.
        playback_ms: u64,
        /// Speech end → first audible sample (or stream end, in text mode).
        total_ms: u64,
    },
    /// A stage failed fatally (the pipeline is shutting down; the
    /// originating error is also returned by `Agent::run`, behind the
    /// `audio` feature).
    Error(String),
}

/// The SDK agent facade. See the [module docs](self) for the overview.
///
/// All entry points are blocking; see the module-level threading note.
/// `Agent` is `Send` (every field is), so it can be moved to a worker
/// thread freely.
pub struct Agent {
    config: Config,
    /// Injected STT engine; only consumed by `run` (behind the `audio`
    /// feature), so it is stored-but-unused in a no-`audio` build.
    #[cfg_attr(not(feature = "audio"), allow(dead_code))]
    stt: Option<Box<dyn SttEngine>>,
    llm: Option<Box<dyn LlmBackend>>,
    tts: Option<Box<dyn TtsEngine>>,
    events: broadcast::Sender<AgentEvent>,
    shutdown: CancellationToken,
    /// Lazily started playback for the audio-producing text paths
    /// (`text_turn` in audio mode, `say`). Only present with the `audio`
    /// feature (cpal playback).
    #[cfg(feature = "audio")]
    playback: Option<(Playback, PlaybackHandle)>,
}

/// Builder for [`Agent`] (start with [`Agent::builder`]).
pub struct AgentBuilder {
    config: Config,
    stt: Option<Box<dyn SttEngine>>,
    llm: Option<Box<dyn LlmBackend>>,
    tts: Option<Box<dyn TtsEngine>>,
}

impl Agent {
    /// Starts building an agent from [`Config::default`]; chain
    /// [`AgentBuilder::config`] to supply your own.
    pub fn builder() -> AgentBuilder {
        AgentBuilder {
            config: Config::default(),
            stt: None,
            llm: None,
            tts: None,
        }
    }

    /// Subscribes to the agent's event stream. Call before
    /// `run` (behind the `audio` feature) (or any text entry point) —
    /// receivers only see events emitted after they subscribe.
    pub fn events(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Runs the full audio pipeline (mic → VAD → STT → LLM → TTS →
    /// playback with barge-in), reusing [`Pipeline`] internally and
    /// broadcasting [`AgentEvent`]s. Blocks until
    /// [`shutdown`](Agent::shutdown) is requested or a stage fails fatally.
    ///
    /// In [`OutputMode::Text`] no output device is opened and no TTS engine
    /// is built: transcripts and reply clauses arrive as events instead of
    /// audio.
    ///
    /// Stage panics surface as `Err` (the orchestrator joins its tasks and
    /// converts them) — unlike the per-call entry points
    /// ([`text_turn`](Agent::text_turn), [`repl`](Agent::repl),
    /// [`say`](Agent::say)), which re-raise engine panics on the caller.
    #[cfg(feature = "audio")]
    pub fn run(mut self) -> Result<()> {
        Pipeline::from_parts(
            self.config.clone(),
            self.shutdown.clone(),
            self.events.clone(),
            self.stt.take(),
            self.llm.take(),
            self.tts.take(),
        )
        .run()
    }

    /// One text-in turn through the LLM, returning the full reply text.
    ///
    /// [`AgentEvent::Clause`] fires per streamed clause and
    /// [`AgentEvent::ReplyDone`] at the end. In [`OutputMode::Audio`] each
    /// clause is also synthesized with the TTS engine and played (the
    /// output device is opened lazily on the first audio-producing call and
    /// reused afterwards); before returning, the call waits until every
    /// clip has been handed to the playback ring — up to the ring's
    /// capacity (~2 s) of audio may still be playing when the call
    /// returns, and dropping the `Agent` right afterwards truncates that
    /// tail ([`Drop`] stops playback abruptly; use `say` (behind the
    /// `audio` feature) when the speech must have finished). In
    /// [`OutputMode::Text`] no TTS runs.
    ///
    /// Conversation history accumulates across calls (bounded by
    /// `Config::max_history_turns`); see the [`LlmBackend`] history note.
    ///
    /// # Panics
    ///
    /// A panic in the LLM backend or TTS engine is re-raised on the caller
    /// (the scoped worker's panic propagates via `resume_unwind`) —
    /// unlike `run` (behind the `audio` feature), which converts stage
    /// panics into `Err`s.
    pub fn text_turn(&mut self, input: &str) -> Result<String> {
        // Split borrows: each field is borrowed disjointly, so the lazily
        // built config-default stages can sit next to the shared handles.
        let Agent {
            llm,
            tts,
            events,
            config,
            ..
        } = self;
        let llm = ensure_llm(llm, config);
        let tts: Option<&mut dyn TtsEngine> = match config.output {
            OutputMode::Audio => Some(ensure_tts(tts, config)?),
            OutputMode::Text => None,
        };
        // Lazily start playback only when audio output is possible. Without
        // the `audio` feature there is no playback field, so no clips are
        // ever queued (`text_turn_async` receives `None`).
        #[cfg(feature = "audio")]
        let playback: Option<ClipSink> = if tts.is_some() {
            Some(lazy_playback(&mut self.playback, config)?.1.clone())
        } else {
            None
        };
        #[cfg(not(feature = "audio"))]
        let playback: Option<ClipSink> = None;
        let events = events.clone();
        run_scoped(move || async move {
            let mut tts = tts;
            text_turn_async(llm, &mut tts, playback, &events, input, |_| Ok(())).await
        })
    }

    /// Interactive text-in/text-out loop: reads user lines from `input`,
    /// streams each reply's clauses to `output` as they arrive, until EOF
    /// or a `/quit` line. No audio: no VAD/STT/TTS, only the (shared,
    /// history-keeping) LLM backend.
    ///
    /// The abstract `BufRead`/`Write` signature lets tests drive the loop
    /// fully in-memory; the binary wires stdin/stdout.
    ///
    /// ```no_run
    /// # use skadoosh::{Agent, Result};
    /// # fn main() -> Result<()> {
    /// let mut agent = Agent::builder().build()?;
    /// // (`StdinLock`/`StdoutLock` are !Send and `Stdin` is not `BufRead`,
    /// // hence the unlocked, buffered handles.)
    /// agent.repl(
    ///     std::io::BufReader::new(std::io::stdin()),
    ///     std::io::stdout(),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// A panic in the LLM backend is re-raised on the caller (see
    /// [`text_turn`](Agent::text_turn)'s note).
    pub fn repl(
        &mut self,
        input: impl BufRead + Send,
        mut output: impl Write + Send,
    ) -> Result<()> {
        let events = self.events.clone();
        let llm = ensure_llm(&mut self.llm, &self.config);
        writeln!(output, "skadoosh repl — type a line, /quit to exit").map_err(repl_io_error)?;
        run_scoped(move || async move {
            for line in input.lines() {
                let line = line.map_err(repl_io_error)?;
                let text = line.trim();
                if text == "/quit" {
                    break;
                }
                if text.is_empty() {
                    continue;
                }
                write!(output, "bot> ").map_err(repl_io_error)?;
                output.flush().map_err(repl_io_error)?;
                let mut first_clause = true;
                text_turn_async(&mut *llm, &mut None, None, &events, text, |clause| {
                    if !first_clause {
                        write!(output, " ").map_err(repl_io_error)?;
                    }
                    first_clause = false;
                    write!(output, "{}", clause.trim()).map_err(repl_io_error)?;
                    output.flush().map_err(repl_io_error)
                })
                .await?;
                writeln!(output).map_err(repl_io_error)?;
            }
            writeln!(output, "bye").map_err(repl_io_error)?;
            Ok(())
        })
    }

    /// One-shot text→speech: splits `text` into clauses, synthesizes each
    /// with the TTS engine, and plays them on the output device (opened
    /// lazily on first use). Only the TTS stage is involved — no other
    /// models are needed.
    ///
    /// Blocks until playback has finished: after queueing, the call waits
    /// for every clip to be pushed to the playback ring and every sample
    /// consumed by the output callback (a graceful drain — the playback
    /// thread is NOT stopped and the device stays open for reuse).
    /// Dropping the agent after `say` returns therefore truncates nothing;
    /// the binary relies on this (it drops the agent immediately after).
    /// Unspeakable text is rejected BEFORE the output device is opened.
    ///
    /// # Panics
    ///
    /// A panic in the TTS engine is re-raised on the caller (see
    /// [`text_turn`](Agent::text_turn)'s note).
    #[cfg(feature = "audio")]
    pub fn say(&mut self, text: &str) -> Result<()> {
        // Synthesize first: unspeakable text must be rejected before the
        // output device is touched.
        let clips = self.synthesize_clips(text)?;
        let handle = lazy_playback(&mut self.playback, &self.config)?.1.clone();
        run_scoped(move || async move {
            for clip in clips {
                handle.queue_clip(clip).await?;
            }
            // Graceful finish: return only once the queued audio has
            // actually played (dropping the agent right after `say` —
            // what the binary does — must not truncate it).
            handle.wait_drained().await;
            Ok(())
        })
    }

    /// One-shot text→speech to a file: like `say` (behind the `audio`
    /// feature), but writes a 24 kHz 16-bit mono wav instead of playing —
    /// no audio device needed.
    pub fn say_to_wav(&mut self, text: &str, path: &Path) -> Result<()> {
        let clips = self.synthesize_clips(text)?;
        write_wav16(path, &concat_clip_samples(&clips), TTS_SAMPLE_RATE)
    }

    /// Requests a graceful shutdown of `run` (behind the `audio` feature;
    /// the binary bridges SIGINT onto this).
    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }

    /// A clone of the agent's shutdown token (what
    /// [`shutdown`](Agent::shutdown) cancels) — for integrators that bridge
    /// their own signal handling, as the binary does with SIGINT.
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    /// Clause-splits `text` and synthesizes each clause with the (lazily
    /// built) TTS engine.
    fn synthesize_clips(&mut self, text: &str) -> Result<Vec<TtsClip>> {
        let engine = ensure_tts(&mut self.tts, &self.config)?;
        let mut splitter = ClauseSplitter::new(CLAUSE_MIN_LEN, CLAUSE_MAX_LEN);
        let mut clips = Vec::new();
        for clause in splitter.push(text).into_iter().chain(splitter.flush()) {
            clips.push(engine.synthesize(&clause)?);
        }
        if clips.is_empty() {
            return Err(anyhow::anyhow!("the text produced no speakable clauses").into());
        }
        Ok(clips)
    }
}

impl AgentBuilder {
    /// The configuration every unset stage is built from.
    pub fn config(mut self, config: Config) -> Self {
        self.config = config;
        self
    }

    /// Injects a custom speech-to-text engine (used by `Agent::run`, behind
    /// the `audio` feature, only).
    pub fn stt(mut self, engine: Box<dyn SttEngine>) -> Self {
        self.stt = Some(engine);
        self
    }

    /// Injects a custom LLM backend (used by every text path).
    pub fn llm(mut self, backend: Box<dyn LlmBackend>) -> Self {
        self.llm = Some(backend);
        self
    }

    /// Injects a custom TTS engine (used by the audio output paths:
    /// `Agent::run` in audio mode, `Agent::say`, and audio-mode
    /// [`Agent::text_turn`](Agent::text_turn)). `run` and `say` are behind
    /// the `audio` feature.
    pub fn tts(mut self, engine: Box<dyn TtsEngine>) -> Self {
        self.tts = Some(engine);
        self
    }

    /// Assembles the agent. No devices are opened and no models are loaded
    /// here — that happens in the entry points (or not at all, for modes
    /// that skip a stage), so a text-only agent never pays for whisper/VAD
    /// model loads.
    pub fn build(self) -> Result<Agent> {
        let (events, _) = broadcast::channel(EVENT_CAP);
        Ok(Agent {
            config: self.config,
            stt: self.stt,
            llm: self.llm,
            tts: self.tts,
            events,
            shutdown: CancellationToken::new(),
            #[cfg(feature = "audio")]
            playback: None,
        })
    }
}

impl Drop for Agent {
    /// Stops the lazily-started playback thread abruptly
    /// (`Playback::stop`, behind the `audio` feature): any audio still
    /// queued or sitting in the playback ring is discarded — dropping an
    /// `Agent` right after queueing audio truncates playback. The
    /// audio-producing entry points bound that by construction: `say`
    /// returns only after playback has fully drained, and audio-mode
    /// [`text_turn`](Agent::text_turn) only after every clip is in the
    /// ring (≤ ~2 s of audio in flight). Without the `audio` feature the
    /// `Agent` owns no playback thread, so `drop` is a no-op.
    fn drop(&mut self) {
        // Join the playback thread if text_turn/say started one.
        #[cfg(feature = "audio")]
        if let Some((playback, _)) = self.playback.take() {
            playback.stop();
        }
    }
}

/// The LLM backend, lazily built from the config when not injected.
fn ensure_llm<'a>(
    slot: &'a mut Option<Box<dyn LlmBackend>>,
    config: &Config,
) -> &'a mut dyn LlmBackend {
    if slot.is_none() {
        *slot = Some(Box::new(LlmClient::from_config(config)));
    }
    &mut **slot.as_mut().expect("filled above")
}

/// The TTS engine, lazily built from the config when not injected.
fn ensure_tts<'a>(
    slot: &'a mut Option<Box<dyn TtsEngine>>,
    config: &Config,
) -> Result<&'a mut dyn TtsEngine> {
    if slot.is_none() {
        *slot = Some(build_engine(config)?);
    }
    Ok(&mut **slot.as_mut().expect("filled above"))
}

/// Starts playback on first use, returning the (now guaranteed) pair.
#[cfg(feature = "audio")]
fn lazy_playback<'a>(
    slot: &'a mut Option<(Playback, PlaybackHandle)>,
    config: &Config,
) -> Result<&'a (Playback, PlaybackHandle)> {
    if slot.is_none() {
        *slot = Some(Playback::start(&AudioOutputConfig {
            device_name: config.output_device.clone(),
        })?);
    }
    Ok(slot.as_ref().expect("filled above"))
}

/// Drives one LLM turn: streams clauses out of `llm`, emitting
/// [`AgentEvent::Clause`] per clause and [`AgentEvent::ReplyDone`] at the
/// end; calls `on_clause` per clause (repl printing), and — when `tts` +
/// `playback` are present (audio mode) — synthesizes and plays each clause.
/// Returns the concatenated reply text.
async fn text_turn_async(
    llm: &mut dyn LlmBackend,
    tts: &mut Option<&mut dyn TtsEngine>,
    playback: Option<ClipSink>,
    events: &broadcast::Sender<AgentEvent>,
    input: &str,
    mut on_clause: impl FnMut(&str) -> Result<()>,
) -> Result<String> {
    let (clause_tx, mut clause_rx) = mpsc::channel::<String>(TEXT_TURN_CLAUSE_CAP);
    let token = CancellationToken::new();
    let turn = llm.stream_reply(input, clause_tx, token);
    tokio::pin!(turn);

    let mut reply = String::new();
    let mut stream_result: Option<Result<()>> = None;
    loop {
        tokio::select! {
            biased;
            result = &mut turn => {
                stream_result = Some(result);
                // The future owns (and has now dropped) the clause sender:
                // drain anything still buffered.
                while let Ok(clause) = clause_rx.try_recv() {
                    handle_clause(
                        &mut reply, clause, events, &mut on_clause, tts, &playback,
                    ).await?;
                }
                break;
            }
            clause = clause_rx.recv() => {
                match clause {
                    Some(clause) => handle_clause(
                        &mut reply, clause, events, &mut on_clause, tts, &playback,
                    ).await?,
                    // The sender dropped without `turn` resolving first is
                    // unreachable (the future owns it), but guard anyway.
                    None => break,
                }
            }
        }
    }

    match stream_result {
        Some(Ok(())) | None => {
            let _ = events.send(AgentEvent::ReplyDone);
            // Audio mode: before returning, wait until the turn's clips
            // have all been handed to the playback ring, so the unplayed
            // tail is bounded by the ring capacity (see the text_turn
            // docs for the drop-truncation caveat).
            #[cfg(feature = "audio")]
            if let Some(handle) = &playback {
                handle.wait_buffered().await;
            }
            Ok(reply)
        }
        // A mid-stream failure surfaces the error; the partial reply is
        // dropped (same discard rule as the pipeline's cancel path).
        Some(Err(err)) => Err(err),
    }
}

/// One clause of a text turn: accumulate into the reply, broadcast the
/// event, run the caller's callback, and optionally synthesize + play.
/// Clause synthesis blocks the (private, current-thread) runtime briefly —
/// acceptable for the sync facade; `Agent::run` (behind the `audio`
/// feature) is the latency-critical path and synthesizes on the blocking
/// pool.
async fn handle_clause(
    reply: &mut String,
    clause: String,
    events: &broadcast::Sender<AgentEvent>,
    on_clause: &mut impl FnMut(&str) -> Result<()>,
    tts: &mut Option<&mut dyn TtsEngine>,
    playback: &Option<ClipSink>,
) -> Result<()> {
    reply.push_str(&clause);
    let _ = events.send(AgentEvent::Clause(clause.clone()));
    on_clause(&clause)?;
    if let (Some(engine), Some(handle)) = (tts.as_deref_mut(), playback) {
        let clip = engine.synthesize(&clause)?;
        #[cfg(feature = "audio")]
        {
            handle.queue_clip(clip).await?;
        }
        #[cfg(not(feature = "audio"))]
        {
            // No `audio` feature: no playback device exists, so the
            // synthesized clip is dropped. (Unreachable in practice — the
            // callers always pass `None` — but keeps the body compiling
            // without the cpal stack.)
            let _ = (clip, handle);
        }
    }
    Ok(())
}

/// Runs `f` on a scoped worker thread with a fresh current-thread tokio
/// runtime, blocking the caller until it completes. Safe from any caller
/// context (inside or outside an existing runtime).
fn run_scoped<F, Fut, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Fut + Send,
    Fut: std::future::Future<Output = Result<T>>,
    T: Send,
{
    std::thread::scope(|scope| {
        let handle = scope.spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| anyhow::anyhow!("failed to start tokio runtime: {err}"))?;
            runtime.block_on(f())
        });
        match handle.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// I/O failures inside the repl become plain errors (the `Display` of
/// `io::Error` reads well in the binary's error chain).
fn repl_io_error(err: std::io::Error) -> crate::error::SkadooshError {
    anyhow::anyhow!("repl I/O failed: {err}").into()
}

/// The spoken greeting the agent uses on first [`Agent::run`] (behind the
/// `audio` feature) when `--agent-name` is set:
/// `"Hi, I'm {name}. What's your name?"`. Returns an empty string when
/// `agent_name` is empty or whitespace (no name configured → no greeting),
/// so callers can gate on the result.
pub fn greeting_text(agent_name: &str) -> String {
    let name = agent_name.trim();
    if name.is_empty() {
        String::new()
    } else {
        format!("Hi, I'm {name}. What's your name?")
    }
}

/// Synthesizes `text` clause-by-clause and plays it through `handle`,
/// blocking until every clip has played (a graceful drain, like
/// [`Agent::say`]). Used by the pipeline to speak the startup greeting on
/// the already-open output device before the listening loop begins.
#[cfg(feature = "audio")]
pub(crate) fn speak_text(
    engine: &mut dyn TtsEngine,
    handle: PlaybackHandle,
    text: &str,
) -> Result<()> {
    let mut splitter = ClauseSplitter::new(CLAUSE_MIN_LEN, CLAUSE_MAX_LEN);
    let mut clips = Vec::new();
    for clause in splitter.push(text).into_iter().chain(splitter.flush()) {
        clips.push(engine.synthesize(&clause)?);
    }
    if clips.is_empty() {
        return Err(anyhow::anyhow!("the text produced no speakable clauses").into());
    }
    run_scoped(move || async move {
        for clip in clips {
            handle.queue_clip(clip).await?;
        }
        handle.wait_drained().await;
        Ok(())
    })
}
