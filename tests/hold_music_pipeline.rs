//! Integration check for the hold-music feeder task in the pipeline.
//!
//! Verifies that `hold_music_feeder` is spawned when `Topology.hold_music`
//! is `Some`, and that it queues procedural audio clips when the active flag
//! is set to true, while remaining silent when the flag is false.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use skadoosh::error::SkadooshError;
use skadoosh::llm::LlmClient;
use skadoosh::pipeline::{run_orchestrator, ClipSink, Topology, VadEventMsg};
use skadoosh::stt::MockStt;
use skadoosh::tts::{MockTts, TtsClip};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
struct RecordingSink {
    clips: mpsc::UnboundedSender<TtsClip>,
    playing: Arc<AtomicBool>,
}

impl ClipSink for RecordingSink {
    async fn queue_clip(&self, clip: TtsClip) -> Result<(), SkadooshError> {
        self.clips
            .send(clip)
            .map_err(|_| anyhow::anyhow!("test sink closed").into())
    }

    fn flush(&self) {}

    fn is_playing(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }
}

type HoldMusicTestHarness = (
    mpsc::UnboundedReceiver<TtsClip>,
    mpsc::Sender<VadEventMsg>,
    CancellationToken,
    tokio::task::JoinHandle<Result<(), SkadooshError>>,
);

/// Spawns the orchestrator with a live hold-music flag and a mock LLM that
/// is never triggered (no VAD segments are injected).
fn spawn_with_hold_music(flag: Arc<AtomicBool>, playing_initially: bool) -> HoldMusicTestHarness {
    let (vad_tx, vad_rx) = mpsc::channel(8);
    let (fatal_tx, fatal_rx) = mpsc::channel(8);
    let (clips_tx, clips_rx) = mpsc::unbounded_channel();
    let playing = Arc::new(AtomicBool::new(playing_initially));
    let sink = RecordingSink {
        clips: clips_tx,
        playing: Arc::clone(&playing),
    };
    let llm = LlmClient::new(
        "http://localhost:0",
        "mock-model",
        "You are a test bot.",
        8,
        None,
    );
    let shutdown = CancellationToken::new();
    let (events, _) = broadcast::channel(64);
    let join = tokio::spawn(run_orchestrator(Topology {
        vad_events: vad_rx,
        fatal_tx: fatal_tx.clone(),
        fatal_rx,
        stt: Box::new(MockStt::new()),
        llm: Box::new(llm),
        tts_engine: Some(Box::new(MockTts::new())),
        sink,
        shutdown: shutdown.clone(),
        events,
        wake_word: None,
        hold_music: Some(flag),
        watch_rx: None,
    }));

    (clips_rx, vad_tx, shutdown, join)
}

#[tokio::test]
async fn hold_music_feeder_queues_clips_when_active() {
    let flag = Arc::new(AtomicBool::new(false));
    let (mut clips_rx, _vad_tx, shutdown, join) = spawn_with_hold_music(Arc::clone(&flag), false);

    // Not active: no clips should arrive within a short window.
    let no_clip = tokio::time::timeout(Duration::from_millis(200), clips_rx.recv()).await;
    assert!(
        no_clip.is_err(),
        "hold music must not queue clips while the active flag is false"
    );

    // Activate hold music.
    flag.store(true, Ordering::Relaxed);

    // A clip should arrive within a few polling periods (100 ms poll).
    let clip = tokio::time::timeout(Duration::from_millis(500), clips_rx.recv())
        .await
        .expect("hold music should queue a clip after the flag is set")
        .expect("clips channel closed unexpectedly");
    assert_eq!(clip.sample_rate, 24_000);
    assert!(
        !clip.samples.is_empty(),
        "queued hold-music clip must contain samples"
    );
    let energy: f32 = clip.samples.iter().map(|s| s * s).sum();
    assert!(energy > 0.0, "hold-music clip must contain non-zero audio");

    // Deactivate: no further clips.
    flag.store(false, Ordering::Relaxed);
    let no_more = tokio::time::timeout(Duration::from_millis(250), clips_rx.recv()).await;
    assert!(
        no_more.is_err(),
        "hold music must stop queuing clips after the flag is cleared"
    );

    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), join)
        .await
        .expect("orchestrator hung on shutdown")
        .expect("orchestrator task panicked")
        .expect("orchestrator must exit cleanly");
}

#[tokio::test]
async fn hold_music_feeder_is_silent_when_sink_reports_playing() {
    // If the sink says something is already playing, the feeder should skip
    // generation so it does not stomp on TTS output.
    let flag = Arc::new(AtomicBool::new(true));
    let (mut clips_rx, _vad_tx, shutdown, join) = spawn_with_hold_music(Arc::clone(&flag), true);

    // Because the sink reports it is already playing, the feeder must
    // skip generating hold music clips.
    let no_clip = tokio::time::timeout(Duration::from_millis(300), clips_rx.recv()).await;
    assert!(
        no_clip.is_err(),
        "hold music must not queue clips when the sink is already playing"
    );

    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), join)
        .await
        .expect("orchestrator hung on shutdown")
        .expect("orchestrator task panicked")
        .expect("orchestrator must exit cleanly");
}
