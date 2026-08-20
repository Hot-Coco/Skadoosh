//! End-to-end `--selftest` and wired-orchestrator tests (plan §10 tasks
//! 14/15/16 acceptance).
//!
//! Everything here is headless: the LLM side is the in-process mock SSE
//! server (`tests/common/mock_openai.rs`), TTS is [`MockTts`], VAD/STT use
//! the real models (the selftest skips with a printed reason when fixtures
//! are absent), and no cpal device is needed — the orchestrator core is
//! driven through injected channels ([`run_orchestrator`]), while
//! `Pipeline::run`'s device-open path is exercised adaptively: a clean
//! `AudioError` on a device-less box, or a started pipeline shut down via
//! [`Pipeline::shutdown_token`] on a box with a usable ALSA device.

// Not every mock knob (serve_many/serve_error) is used by this suite.
#[allow(dead_code)]
#[path = "common/mock_openai.rs"]
mod mock_openai;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mock_openai::{done_line, token_line, Chunk, MockOpenAi};
use skadoosh::agent::AgentEvent;
use skadoosh::config::{Config, OutputMode};
use skadoosh::error::SkadooshError;
use skadoosh::llm::LlmClient;
use skadoosh::pipeline::{run_orchestrator, ClipSink, Topology, VadEventMsg};
use skadoosh::stt::MockStt;
use skadoosh::tts::{MockTts, TtsClip};
use skadoosh::{Pipeline, Result};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

const VAD_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
const WHISPER_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-tiny.en.bin");
const JFK_WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jfk.wav");

fn fixtures_present() -> bool {
    let present = Path::new(VAD_MODEL).is_file()
        && Path::new(WHISPER_MODEL).is_file()
        && Path::new(JFK_WAV).is_file();
    if !present {
        eprintln!(
            "skipping: {VAD_MODEL}, {WHISPER_MODEL} or {JFK_WAV} missing \
             (run scripts/download_models.sh)"
        );
    }
    present
}

fn base_config(llm_url: String) -> Config {
    Config {
        images: Vec::new(),
        llm_url,
        llm_model: "mock-model".to_string(),
        api_key: None,
        system_prompt: "You are a test bot.".to_string(),
        max_history_turns: 8,
        whisper_model: PathBuf::from(WHISPER_MODEL),
        vad_model: PathBuf::from(VAD_MODEL),
        tts_model: None,
        tts_voices: None,
        vad_threshold: 0.5,
        silence_ms: 300,
        input_device: None,
        output_device: None,
        list_devices: false,
        mock_tts: true,
        selftest: None,
        repl: false,
        say: None,
        output: OutputMode::Audio,
        out_wav: None,
        tools_file: None,
        max_tool_rounds: 5,
        tts_voice: "af".to_string(),
        tts_speed: 1.0,
        wake_word: None,
        push_to_talk: false,
    }
}

/// Clip-sink double: records queued clips, counts flushes (the flush-epoch
/// bump), and scripts `is_playing`. An optional gate parks every
/// `queue_clip` until opened, so a test can hold the TTS task mid-turn
/// while the clause backlog piles up behind it (a cancelled turn's
/// `select!` simply drops the parked future).
///
/// The STT double is the crate's [`MockStt`]: a scripted queue of
/// `(delay, transcript)` replies.
#[derive(Clone)]
struct RecordingSink {
    clips: mpsc::UnboundedSender<TtsClip>,
    flushes: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
    gate: Option<Arc<ClipGate>>,
}

/// Test-controlled `queue_clip` gate: starts closed (calls park on a short
/// poll), [`ClipGate::open`] releases all current and future calls.
struct ClipGate {
    open: AtomicBool,
}

impl ClipGate {
    fn open(&self) {
        self.open.store(true, Ordering::SeqCst);
    }
}

impl ClipSink for RecordingSink {
    async fn queue_clip(&self, clip: TtsClip) -> Result<()> {
        if let Some(gate) = &self.gate {
            while !gate.open.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        self.clips
            .send(clip)
            .map_err(|_| anyhow::anyhow!("test sink closed").into())
    }

    fn flush(&self) {
        self.flushes.fetch_add(1, Ordering::SeqCst);
    }

    fn is_playing(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }
}

/// A running orchestrator plus the test ends of its injected channels.
struct Harness {
    vad_tx: mpsc::Sender<VadEventMsg>,
    fatal_tx: mpsc::Sender<SkadooshError>,
    shutdown: CancellationToken,
    clips_rx: mpsc::UnboundedReceiver<TtsClip>,
    flushes: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
    gate: Option<Arc<ClipGate>>,
    join: tokio::task::JoinHandle<Result<()>>,
}

fn spawn_orchestrator(llm_url: &str, stt_replies: Vec<(Duration, &str)>) -> Harness {
    spawn_orchestrator_full(llm_url, stt_replies, false)
}

/// Full harness constructor. `gated` parks the sink's `queue_clip` on a
/// test-opened [`ClipGate`] (the barge-in-after-stream-end regression test
/// uses it to hold a TTS backlog behind the first clip).
fn spawn_orchestrator_full(
    llm_url: &str,
    stt_replies: Vec<(Duration, &str)>,
    gated: bool,
) -> Harness {
    let (vad_tx, vad_rx) = mpsc::channel(8);
    let (fatal_tx, fatal_rx) = mpsc::channel(8);
    let (clips_tx, clips_rx) = mpsc::unbounded_channel();
    let flushes = Arc::new(AtomicU64::new(0));
    let playing = Arc::new(AtomicBool::new(false));
    let gate = gated.then(|| {
        Arc::new(ClipGate {
            open: AtomicBool::new(false),
        })
    });
    let sink = RecordingSink {
        clips: clips_tx,
        flushes: Arc::clone(&flushes),
        playing: Arc::clone(&playing),
        gate: gate.clone(),
    };
    let stt = MockStt::new();
    for (delay, text) in stt_replies {
        stt.push_delayed(delay, text);
    }
    let llm = LlmClient::new(llm_url, "mock-model", "You are a test bot.", 8, None);
    let shutdown = CancellationToken::new();
    let (events, _) = broadcast::channel(64);
    let join = tokio::spawn(run_orchestrator(Topology {
        vad_events: vad_rx,
        fatal_tx: fatal_tx.clone(),
        fatal_rx,
        stt: Box::new(stt),
        llm: Box::new(llm),
        tts_engine: Some(Box::new(MockTts::new())),
        sink,
        shutdown: shutdown.clone(),
        events,
        wake_word: None,
    }));
    Harness {
        vad_tx,
        fatal_tx,
        shutdown,
        clips_rx,
        flushes,
        playing,
        gate,
        join,
    }
}

/// A scripted one-second "segment" (contents are ignored by the STT double).
fn segment() -> VadEventMsg {
    VadEventMsg::Segment {
        samples: vec![0.0; 16_000],
        t_speech_end: Instant::now(),
    }
}

/// MockTts sample count for a clause: `clamp(chars * 55 ms, 250..2500 ms)`
/// at 24 kHz.
fn mock_clip_samples(clause: &str) -> usize {
    let ms = (clause.chars().count() as f32 * 55.0).clamp(250.0, 2_500.0);
    (24_000.0 * ms / 1000.0).round() as usize
}

/// Joins the orchestrator after cancelling shutdown, asserting a clean
/// (non-fatal) exit.
async fn shutdown_cleanly(h: Harness) {
    h.shutdown.cancel();
    let result = tokio::time::timeout(Duration::from_secs(5), h.join)
        .await
        .expect("orchestrator hung on shutdown")
        .expect("orchestrator task panicked");
    result.expect("shutdown must be a clean Ok(())");
}

/// Task 15 acceptance: the full `--selftest` path against the mock LLM —
/// real VAD + whisper, mock SSE server, MockTts, wav output.
#[tokio::test]
async fn selftest_end_to_end_with_mock_llm() {
    if !fixtures_present() {
        return;
    }
    // Token boundaries chosen so the two clauses each span several tokens.
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Hello ")),
        Chunk::now(token_line("there. ")),
        Chunk::now(token_line("This is ")),
        Chunk::now(token_line("the mock ")),
        Chunk::now(token_line("speaking.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut config = base_config(server.url());
    // jfk.wav has a ~1 s rhetorical pause after "…my fellow Americans," —
    // with the 300 ms default endpoint the *first* segment closes before
    // the "ask not" phrase. Widen the window so one segment covers the
    // whole sentence (the test builds its Config programmatically; the CLI
    // default is unchanged).
    config.silence_ms = 1_500;
    let out_wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/selftest_test_out.wav");
    let out = out_wav.clone();

    let report = tokio::time::timeout(
        Duration::from_secs(60),
        tokio::task::spawn_blocking(move || {
            Pipeline::new(config).and_then(|p| p.run_selftest(Path::new(JFK_WAV), &out))
        }),
    )
    .await
    .expect("selftest exceeded 60 s")
    .expect("selftest thread panicked")
    .expect("run_selftest failed");

    // The latency table (printed like `skadoosh --selftest` does).
    println!("{report}");

    // Transcript: the JFK phrase, punctuation/casing-insensitive.
    let normalized = report
        .transcript
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        normalized.contains("ask not what your country"),
        "transcript missing the JFK phrase: {:?}",
        report.transcript
    );

    // Sane report fields: STT did real work, total covers the stages, and
    // the LLM stamps are ordered (first token before first clause).
    assert!(report.stt_ms > 0, "stt_ms must be > 0: {report:?}");
    assert!(report.segment_ms > 0, "segment_ms must be > 0: {report:?}");
    assert!(
        report.llm_ttft_ms <= report.first_clause_ms,
        "ttft must not exceed first-clause: {report:?}"
    );
    assert!(
        report.total_ms >= report.stt_ms && report.total_ms >= report.segment_ms,
        "total must cover the stages: {report:?}"
    );

    // Output wav: exists, 24 kHz 16-bit mono, non-silent, duration ≈ the
    // sum of the two clause durations ("Hello there." 660 ms + " This is
    // the mock speaking." 1485 ms = 51 480 samples).
    let reader = hound::WavReader::open(&out_wav).expect("failed to open selftest output wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 24_000, "output must be 24 kHz");
    assert_eq!(spec.channels, 1, "output must be mono");
    assert_eq!(spec.bits_per_sample, 16, "output must be 16-bit PCM");
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("sample read"))
        .collect();
    let expected =
        mock_clip_samples("Hello there.") + mock_clip_samples(" This is the mock speaking.");
    let tol = expected / 20;
    assert!(
        samples.len().abs_diff(expected) <= tol,
        "wav has {} samples, expected ≈ {expected} (sum of clause durations)",
        samples.len()
    );
    let peak = samples.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
    assert!(
        peak > 3_000,
        "output wav is implausibly quiet (peak {peak})"
    );

    let _ = std::fs::remove_file(&out_wav);
}

/// `--selftest` with `--api-key` sends the bearer header (regression: the
/// selftest drives its own reqwest POST — not `LlmClient` — and used to
/// skip the key, so keyed providers answered 401).
#[tokio::test]
async fn selftest_sends_bearer_auth_with_api_key() {
    if !fixtures_present() {
        return;
    }
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Keyed reply.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut config = base_config(server.url());
    config.api_key = Some("sk-selftest-secret".to_string());
    config.silence_ms = 1_500;
    let out_wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/selftest_auth_out.wav");
    let out = out_wav.clone();

    tokio::time::timeout(
        Duration::from_secs(60),
        tokio::task::spawn_blocking(move || {
            Pipeline::new(config).and_then(|p| p.run_selftest(Path::new(JFK_WAV), &out))
        }),
    )
    .await
    .expect("selftest exceeded 60 s")
    .expect("selftest thread panicked")
    .expect("run_selftest failed");

    let req = server.captured_request().expect("request captured");
    assert!(
        req.to_lowercase()
            .contains("authorization: bearer sk-selftest-secret"),
        "selftest request must carry the bearer token: {req}"
    );

    let _ = std::fs::remove_file(&out_wav);
}

/// EOF-flush adoption: a server that closes the SSE stream with an
/// UNTERMINATED final `data:` line (no trailing `\n`, no `[DONE]`) must
/// lose no content — `SseLineBuffer::close` + drain yields the remainder.
/// The reply "Hello there." only completes if the unterminated second line
/// is parsed: dropped, the wav would hold just "Hello " (7 920 samples).
#[tokio::test]
async fn selftest_reads_unterminated_final_sse_line() {
    if !fixtures_present() {
        return;
    }
    let trailing = format!(
        "data: {}",
        serde_json::json!({
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "mock-model",
            "choices": [{"index": 0, "delta": {"content": "there."}, "finish_reason": null}],
        })
    );
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Hello ")),
        Chunk::now(trailing), // no trailing "\n\n"; the mock half-closes after it
    ])
    .await;
    let mut config = base_config(server.url());
    config.silence_ms = 1_500; // one segment for the whole JFK sentence
    let out_wav =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/selftest_unterminated_out.wav");
    let out = out_wav.clone();

    tokio::time::timeout(
        Duration::from_secs(60),
        tokio::task::spawn_blocking(move || {
            Pipeline::new(config).and_then(|p| p.run_selftest(Path::new(JFK_WAV), &out))
        }),
    )
    .await
    .expect("selftest exceeded 60 s")
    .expect("selftest thread panicked")
    .expect("run_selftest failed");

    let reader = hound::WavReader::open(&out_wav).expect("failed to open selftest output wav");
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("sample read"))
        .collect();
    let expected = mock_clip_samples("Hello there.");
    let tol = expected / 20;
    assert!(
        samples.len().abs_diff(expected) <= tol,
        "wav has {} samples, expected ≈ {expected} — the unterminated final \
         SSE line must be parsed (dropped, it would be ≈ {})",
        samples.len(),
        mock_clip_samples("Hello "),
    );

    let _ = std::fs::remove_file(&out_wav);
}

/// Task 14 acceptance, wiring: a scripted Segment produces text → clauses →
/// clips in order.
#[tokio::test]
async fn wired_segment_flows_text_clauses_clips_in_order() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Alpha. ")),
        Chunk::now(token_line("Beta. ")),
        Chunk::now(token_line("Gamma.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut h = spawn_orchestrator(&server.url(), vec![(Duration::ZERO, "say the alphabet")]);

    h.vad_tx.send(segment()).await.expect("vad channel open");

    // Clauses "Alpha.", " Beta.", " Gamma." → clips in order with the
    // deterministic MockTts durations.
    for (i, clause) in ["Alpha.", " Beta.", " Gamma."].iter().enumerate() {
        let clip = tokio::time::timeout(Duration::from_secs(5), h.clips_rx.recv())
            .await
            .unwrap_or_else(|_| panic!("clip {i} ({clause:?}) timed out"))
            .expect("clips channel closed early");
        assert_eq!(clip.sample_rate, 24_000);
        assert_eq!(
            clip.samples.len(),
            mock_clip_samples(clause),
            "clip {i} must be clause {clause:?}"
        );
    }
    // No fourth clause.
    let extra = tokio::time::timeout(Duration::from_millis(300), h.clips_rx.recv()).await;
    assert!(extra.is_err(), "unexpected extra clip: {extra:?}");

    shutdown_cleanly(h).await;
}

/// Task 14 acceptance, barge-in: a SpeechStart while clips are queued and
/// playing cancels the turn — the mock LLM observes the dropped connection,
/// no further clips arrive, the flush epoch is bumped — and the pipeline
/// still shuts down cleanly.
#[tokio::test]
async fn barge_in_cancels_turn_flushes_and_stops_clips() {
    let mut script = vec![Chunk::now(token_line("One. "))];
    // A long, slow tail: cancellation must cut the stream short.
    for _ in 0..100 {
        script.push(Chunk::after(
            Duration::from_millis(30),
            token_line("still talking "),
        ));
    }
    let server = MockOpenAi::serve(script).await;
    let mut h = spawn_orchestrator(&server.url(), vec![(Duration::ZERO, "interrupt me")]);

    h.vad_tx.send(segment()).await.expect("vad channel open");

    // The first clause ("One.", clamped to 250 ms) is synthesized and queued.
    let clip = tokio::time::timeout(Duration::from_secs(5), h.clips_rx.recv())
        .await
        .expect("first clip timed out")
        .expect("clips channel closed");
    assert_eq!(clip.samples.len(), mock_clip_samples("One."));

    // The speaker is mid-clip; the user starts talking (the VAD task's
    // 2-frame hangover lives upstream of this injection point).
    h.playing.store(true, Ordering::SeqCst);
    h.vad_tx
        .send(VadEventMsg::SpeechStart)
        .await
        .expect("vad channel open");

    // The flush epoch is bumped exactly once.
    let deadline = Instant::now() + Duration::from_secs(2);
    while h.flushes.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        h.flushes.load(Ordering::SeqCst),
        1,
        "barge-in must flush playback exactly once"
    );

    // The mock LLM observes the cancelled (dropped) SSE connection.
    assert!(
        server.wait_peer_gone(Duration::from_secs(2)).await,
        "cancelling the turn must drop the LLM connection"
    );

    // No further clips from the cancelled turn: queued-but-unsynthesized
    // clauses and any in-flight synthesis are discarded (stale-turn drop).
    let extra = tokio::time::timeout(Duration::from_millis(400), h.clips_rx.recv()).await;
    assert!(
        extra.is_err(),
        "no clips may arrive after the cancel; got {extra:?}"
    );

    shutdown_cleanly(h).await;
}

/// Regression (barge-in after stream end): the LLM stream has ENDED (the
/// turn-done notification processed) while the TTS task still holds a
/// clause backlog — here parked inside the gated `queue_clip` on clip #1
/// with the remaining clauses buffered in the clause channel. A SpeechStart
/// while playback is audible must STILL cancel the turn token: the
/// turn-done notification must not disarm barge-in, or the backlog would
/// resume talking right after the flush.
#[tokio::test]
async fn barge_in_after_llm_stream_end_still_cancels_tts_backlog() {
    // Six clauses written at once + [DONE]: the whole reply streams down
    // while the TTS task is parked on the first clip's queue_clip.
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("One. ")),
        Chunk::now(token_line("Two. ")),
        Chunk::now(token_line("Three. ")),
        Chunk::now(token_line("Four. ")),
        Chunk::now(token_line("Five. ")),
        Chunk::now(token_line("Six.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut h = spawn_orchestrator_full(&server.url(), vec![(Duration::ZERO, "talk")], true);
    let gate = h.gate.clone().expect("gated harness");

    h.vad_tx.send(segment()).await.expect("vad channel open");

    // The LLM stream runs to completion: the mock observes the client
    // closing the connection after [DONE]. Clauses 2..6 sit buffered in
    // the clause channel behind the parked queue_clip.
    assert!(
        server.wait_peer_gone(Duration::from_secs(5)).await,
        "the LLM stream must run to completion"
    );
    // Let the orchestrator process the turn-done notification (it must
    // mark the turn stream-done but KEEP the cancel token). The margin is
    // generous: the orchestrator is idle and try_send immediately follows
    // stream_reply's return.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The speaker is (still) audible; the user starts talking.
    h.playing.store(true, Ordering::SeqCst);
    h.vad_tx
        .send(VadEventMsg::SpeechStart)
        .await
        .expect("vad channel open");

    // The flush epoch is bumped exactly once...
    let deadline = Instant::now() + Duration::from_secs(2);
    while h.flushes.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        h.flushes.load(Ordering::SeqCst),
        1,
        "barge-in must flush playback exactly once"
    );

    // ...and the turn token is cancelled: the parked queue_clip future is
    // dropped and the buffered clause backlog abandoned, so opening the
    // gate releases NOTHING — no clip ever reaches the sink.
    gate.open();
    let extra = tokio::time::timeout(Duration::from_millis(400), h.clips_rx.recv()).await;
    assert!(
        extra.is_err(),
        "no clips may arrive once the post-stream-end barge-in cancelled the turn; got {extra:?}"
    );

    shutdown_cleanly(h).await;
}

/// Regression (silent-gap SpeechStart): a SpeechStart while playback is
/// NOT audible must neither flush nor cancel — a one-frame VAD false
/// positive (cough/click) in a silent gap is rejected by the segmenter's
/// min-length guard, so no replacement utterance ever follows; cancelling
/// here would silently kill a mid-stream reply.
#[tokio::test]
async fn speech_start_in_silent_gap_does_not_cancel_turn() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Alpha. ")),
        Chunk::after(Duration::from_millis(150), token_line("Beta. ")),
        Chunk::after(Duration::from_millis(150), token_line("Gamma.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut h = spawn_orchestrator(&server.url(), vec![(Duration::ZERO, "keep going")]);

    h.vad_tx.send(segment()).await.expect("vad channel open");

    // Clip #1 queued: the turn is mid-stream (" Beta." is still delayed
    // server-side) and the speaker is SILENT (playback has not ramped, or
    // a momentary dry gap). A one-frame VAD blip fires SpeechStart.
    let clip = tokio::time::timeout(Duration::from_secs(5), h.clips_rx.recv())
        .await
        .expect("first clip timed out")
        .expect("clips channel closed");
    assert_eq!(clip.samples.len(), mock_clip_samples("Alpha."));
    h.vad_tx
        .send(VadEventMsg::SpeechStart)
        .await
        .expect("vad channel open");

    // The blip is ignored: no flush, no cancel — the turn runs to
    // completion and the remaining clauses arrive as clips, in order.
    for (i, clause) in [" Beta.", " Gamma."].iter().enumerate() {
        let clip = tokio::time::timeout(Duration::from_secs(5), h.clips_rx.recv())
            .await
            .unwrap_or_else(|_| panic!("clip {} ({clause:?}) timed out", i + 1))
            .expect("clips channel closed early");
        assert_eq!(
            clip.samples.len(),
            mock_clip_samples(clause),
            "clip {} must be clause {clause:?}",
            i + 1
        );
    }
    let extra = tokio::time::timeout(Duration::from_millis(300), h.clips_rx.recv()).await;
    assert!(extra.is_err(), "unexpected extra clip: {extra:?}");
    assert_eq!(
        h.flushes.load(Ordering::SeqCst),
        0,
        "a silent-gap SpeechStart must not flush playback"
    );

    shutdown_cleanly(h).await;
}

/// Task 14 acceptance, stale-turn defense: a segment superseded while its
/// transcription is in flight never reaches the LLM.
#[tokio::test]
async fn stale_segment_transcript_is_dropped() {
    let server = MockOpenAi::serve(vec![
        Chunk::now(token_line("Fresh reply.")),
        Chunk::now(done_line()),
    ])
    .await;
    let mut h = spawn_orchestrator(
        &server.url(),
        vec![
            (Duration::from_millis(300), "stale utterance"),
            (Duration::ZERO, "fresh utterance"),
        ],
    );

    // Turn 1 transcribes slowly; turn 2 supersedes it mid-flight.
    h.vad_tx.send(segment()).await.expect("vad channel open");
    tokio::time::sleep(Duration::from_millis(50)).await;
    h.vad_tx.send(segment()).await.expect("vad channel open");

    // Only turn 2's reply is ever synthesized ("Fresh reply." → 660 ms).
    let clip = tokio::time::timeout(Duration::from_secs(5), h.clips_rx.recv())
        .await
        .expect("clip timed out")
        .expect("clips channel closed");
    assert_eq!(clip.samples.len(), mock_clip_samples("Fresh reply."));
    let extra = tokio::time::timeout(Duration::from_millis(300), h.clips_rx.recv()).await;
    assert!(extra.is_err(), "stale turn produced audio: {extra:?}");

    // The mock served exactly one request, carrying only the fresh text.
    let req = server.captured_request().expect("LLM request captured");
    assert!(req.contains("fresh utterance"), "request: {req}");
    assert!(!req.contains("stale utterance"), "request: {req}");

    shutdown_cleanly(h).await;
}

/// Task 14 acceptance, fatal propagation: an LLM task failure (connection
/// refused) shuts the pipeline down with a non-zero `Result`, and a fatal
/// error injected directly on the fatal mpsc does the same.
#[tokio::test]
async fn fatal_error_from_task_shuts_pipeline_down() {
    // Nothing listens on port 1: the LLM request fails fast and fatally.
    let h = spawn_orchestrator("http://127.0.0.1:1", vec![(Duration::ZERO, "hi")]);
    h.vad_tx.send(segment()).await.expect("vad channel open");
    let result = tokio::time::timeout(Duration::from_secs(10), h.join)
        .await
        .expect("orchestrator hung after fatal error")
        .expect("orchestrator task panicked");
    assert!(
        matches!(result, Err(SkadooshError::Llm(_))),
        "expected a fatal LLM error, got {result:?}"
    );

    // Direct fatal-mpsc injection takes the same shutdown path.
    let h = spawn_orchestrator("http://127.0.0.1:1", vec![]);
    h.fatal_tx
        .send(anyhow::anyhow!("injected boom").into())
        .await
        .expect("fatal channel open");
    let result = tokio::time::timeout(Duration::from_secs(5), h.join)
        .await
        .expect("orchestrator hung after injected fatal")
        .expect("orchestrator task panicked");
    match result {
        Err(err) => assert!(err.to_string().contains("injected boom"), "got {err:?}"),
        Ok(()) => panic!("injected fatal must surface as Err"),
    }
}

/// v0.2: `--output text` voice turn — wav in (real Silero VAD + segmenter),
/// MockStt transcript, mock-LLM reply, and NO TTS/playback: the reply
/// surfaces as `Clause`/`ReplyDone` events (what the binary prints) and the
/// sink never sees a clip. Mirrors the orchestrator half of
/// `Pipeline::run`'s text mode (`NullSink` + `tts_engine: None`).
#[tokio::test]
async fn output_text_voice_turn_streams_reply_events() {
    if !fixtures_present() {
        return;
    }

    let (events_tx, mut events_rx) = broadcast::channel::<AgentEvent>(64);
    let (result_tx, result_rx) = std::sync::mpsc::channel();

    // The orchestrator runs on its own runtime thread; this test thread
    // collects events concurrently.
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async move {
            let server = MockOpenAi::serve(vec![
                Chunk::now(token_line("Alpha. ")),
                Chunk::now(token_line("Beta.")),
                Chunk::now(done_line()),
            ])
            .await;

            let (vad_tx, vad_rx) = mpsc::channel(8);
            let (fatal_tx, fatal_rx) = mpsc::channel(8);
            let (clips_tx, _clips_rx) = mpsc::unbounded_channel();
            let sink = RecordingSink {
                clips: clips_tx,
                flushes: Arc::new(AtomicU64::new(0)),
                playing: Arc::new(AtomicBool::new(false)),
                gate: None,
            };
            let stt = MockStt::from_replies(["ask not what your country can do for you"]);
            let llm = LlmClient::new(&server.url(), "mock-model", "You are a test bot.", 8, None);
            let shutdown = CancellationToken::new();
            let join = tokio::spawn(run_orchestrator(Topology {
                vad_events: vad_rx,
                fatal_tx,
                fatal_rx,
                stt: Box::new(stt),
                llm: Box::new(llm),
                tts_engine: None, // text mode: no synthesis
                sink,
                shutdown: shutdown.clone(),
                events: events_tx,
                wake_word: None,
            }));

            // Feed jfk.wav through the real VAD, like the VAD task would.
            let (mono, rate) = read_wav_samples(Path::new(JFK_WAV));
            let samples = skadoosh::audio::resample_offline(&mono, rate, 16_000);
            let mut vad = skadoosh::vad::SileroVad::new(Path::new(VAD_MODEL)).expect("vad loads");
            let mut segmenter = skadoosh::vad::VadSegmenter::new(0.5, 1_500); // one segment for the whole sentence
                                                                              // The file has no trailing silence to close the segment against
                                                                              // the widened window — append some, like a live stream would.
            let mut feed = samples;
            feed.extend(std::iter::repeat_n(
                0.0,
                (1_500 / 32 + 2) * skadoosh::vad::FRAME_LEN,
            ));
            let mut sent = false;
            for chunk in feed.chunks_exact(skadoosh::vad::FRAME_LEN) {
                let frame: &[f32; skadoosh::vad::FRAME_LEN] =
                    chunk.try_into().expect("chunks_exact");
                let prob = vad.process(frame).expect("vad inference");
                if let Some(skadoosh::vad::VadEvent::Segment(audio)) = segmenter.push(frame, prob) {
                    vad_tx
                        .send(VadEventMsg::Segment {
                            samples: audio,
                            t_speech_end: Instant::now(),
                        })
                        .await
                        .expect("vad channel open");
                    sent = true;
                    break;
                }
            }
            assert!(sent, "jfk.wav must produce a segment");

            // The turn completes (ReplyDone observed by the collector); give
            // it a bounded moment, then shut down.
            tokio::time::sleep(Duration::from_secs(3)).await;
            shutdown.cancel();
            let result = tokio::time::timeout(Duration::from_secs(5), join)
                .await
                .expect("orchestrator hung")
                .expect("orchestrator panicked");
            let _ = result_tx.send(result);
        });
    });

    // Collect the full event sequence until the turn-end Listening (or a
    // generous deadline): ORDER matters, not just presence — a turn-end
    // Listening must never land between the turn's Transcript and its
    // ReplyDone (it used to be emitted by the orchestrator at LLM
    // stream-done, racing the TTS task's clause drain).
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut order: Vec<&'static str> = Vec::new();
    let mut transcripts = Vec::new();
    let mut clauses = Vec::new();
    let mut latency = None;
    while Instant::now() < deadline {
        if order.len() > 1 && order.last() == Some(&"Listening") {
            break; // the turn-end Listening (after ReplyDone)
        }
        match tokio::time::timeout(Duration::from_secs(5), events_rx.recv()).await {
            Ok(Ok(event)) => {
                let kind = match &event {
                    AgentEvent::Listening => "Listening",
                    AgentEvent::SpeechStart => "SpeechStart",
                    AgentEvent::Transcript(t) => {
                        transcripts.push(t.clone());
                        "Transcript"
                    }
                    AgentEvent::Clause(c) => {
                        clauses.push(c.clone());
                        "Clause"
                    }
                    AgentEvent::ReplyDone => "ReplyDone",
                    AgentEvent::TurnCancelled => "TurnCancelled",
                    AgentEvent::StageLatency {
                        tts_ms,
                        playback_ms,
                        ..
                    } => {
                        latency = Some((*tts_ms, *playback_ms));
                        "StageLatency"
                    }
                    AgentEvent::Error(_) => "Error",
                    AgentEvent::ToolCall { .. } => "ToolCall",
                };
                order.push(kind);
            }
            Ok(Err(_)) => {} // lagged: keep going
            Err(_) => break, // timeout
        }
    }

    assert_eq!(
        order,
        vec![
            "Listening",
            "Transcript",
            "Clause",
            "Clause",
            "StageLatency",
            "ReplyDone",
            "Listening"
        ],
        "text-mode turn event order"
    );
    assert!(
        transcripts
            .iter()
            .any(|t| t.contains("ask not what your country")),
        "transcript event: {transcripts:?}"
    );
    assert_eq!(clauses, vec!["Alpha.".to_string(), " Beta.".to_string()]);
    assert_eq!(
        latency,
        Some((0, 0)),
        "text mode has no tts/playback stage: {latency:?}"
    );

    let result = result_rx
        .recv_timeout(Duration::from_secs(15))
        .expect("orchestrator result");
    result.expect("clean shutdown");
}

/// Minimal wav decode for the test above (16-bit PCM mono, like jfk.wav).
fn read_wav_samples(path: &Path) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let rate = spec.sample_rate;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| f32::from(s.expect("sample")) / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("sample"))
            .collect(),
    };
    (samples, rate)
}

/// Task 14 acceptance, headless `Pipeline::run`: on a device-less box the
/// device-open must fail cleanly (`AudioError`, no panic/hang); on a box
/// with a usable ALSA device (like this one's null plugin) the pipeline
/// starts and must shut down cleanly when the shutdown token — the same
/// injection point the binary's SIGINT bridge cancels — fires. Hard 20 s
/// bound either way.
#[test]
fn run_headless_no_device_or_clean_shutdown() {
    if !fixtures_present() {
        return;
    }
    let pipeline =
        Pipeline::new(base_config("http://127.0.0.1:1".to_string())).expect("Pipeline::new");
    let token = pipeline.shutdown_token();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(pipeline.run());
    });

    match done_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Err(err)) => {
            assert!(
                matches!(err, SkadooshError::Audio(_)),
                "device-open failure must be a clean AudioError, got {err:?}"
            );
            eprintln!("headless branch: run() failed cleanly with {err}");
        }
        Ok(Ok(())) => panic!("run() returned Ok without a shutdown request"),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("pipeline thread died without returning a result")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("pipeline started (usable ALSA device); cancelling via shutdown token");
            token.cancel();
            match done_rx.recv_timeout(Duration::from_secs(15)) {
                Ok(Ok(())) => eprintln!("clean shutdown after token cancel"),
                Ok(Err(err)) => panic!("shutdown after cancel must be clean Ok, got {err:?}"),
                Err(err) => panic!("pipeline hung after shutdown request: {err}"),
            }
        }
    }
}
