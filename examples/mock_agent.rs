//! The plugin story end-to-end with zero models, zero servers, zero audio
//! devices: an agent turn driven through the real orchestrator
//! ([`run_orchestrator`], the pipeline's injection seam) with
//!
//! * STT: [`MockStt`] (scripted transcripts),
//! * LLM: `FakeLlm` (a scripted [`LlmBackend`] written right here),
//! * TTS: [`MockTts`] (sine wave, duration ∝ clause length),
//! * playback: a recording [`ClipSink`] (defined here).
//!
//! Run it:
//!
//! ```sh
//! cargo run --example mock_agent
//! ```
//!
//! It is also run as a smoke test — see
//! `tests/sdk_agent.rs::mock_agent_example_runs_green`.
//!
//! Caveat: [`run_orchestrator`], [`Topology`], [`ClipSink`], and
//! [`VadEventMsg`] are doc-hidden internals (the pipeline's test seam), not
//! the stable SDK surface — they may change between releases. The stable
//! embedding API is [`Agent`](skadoosh::Agent).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use skadoosh::error::Result;
use skadoosh::llm::LlmBackend;
use skadoosh::pipeline::{run_orchestrator, ClipSink, Topology, VadEventMsg};
use skadoosh::stt::MockStt;
use skadoosh::tts::{MockTts, TtsClip};
use skadoosh::AgentEvent;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

/// A scripted LLM backend: replies with canned clauses, one queue entry per
/// user turn. This is the trait an SDK user implements to plug in their own
/// model/serving stack.
struct FakeLlm {
    script: Mutex<VecDeque<Vec<String>>>,
}

impl LlmBackend for FakeLlm {
    fn name(&self) -> &str {
        "fake-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        user: &'a str,
        clauses: mpsc::Sender<String>,
        _cancel: CancellationToken,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let script = self
                .script
                .lock()
                .expect("script lock")
                .pop_front()
                .unwrap_or_else(|| vec![format!("(no scripted reply for {user:?})")]);
            for clause in script {
                clauses
                    .send(clause)
                    .await
                    .map_err(|_| anyhow::anyhow!("clause consumer gone"))?;
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {
        // Stateless: nothing to clear.
    }
}

/// A clip sink that records instead of playing (the audio-free stand-in for
/// [`PlaybackHandle`](skadoosh::audio::PlaybackHandle)).
#[derive(Clone, Default)]
struct RecordingSink {
    clips: Arc<Mutex<Vec<TtsClip>>>,
    flushes: Arc<Mutex<u64>>,
}

impl ClipSink for RecordingSink {
    async fn queue_clip(&self, clip: TtsClip) -> Result<()> {
        self.clips.lock().expect("clips lock").push(clip);
        Ok(())
    }

    fn flush(&self) {
        *self.flushes.lock().expect("flushes lock") += 1;
    }

    fn is_playing(&self) -> bool {
        false
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // The scripted turn: user segment → MockStt transcript → FakeLlm reply.
    let stt = MockStt::from_replies(["what is the airspeed of an unladen swallow?"]);
    let llm = FakeLlm {
        script: Mutex::new(VecDeque::from(vec![vec![
            "African or European?".to_string(),
            " I don't know that.".to_string(),
        ]])),
    };
    let sink = RecordingSink::default();

    let (vad_tx, vad_rx) = mpsc::channel(8);
    let (fatal_tx, fatal_rx) = mpsc::channel(8);
    let (events, mut events_rx) = broadcast::channel(64);
    let shutdown = CancellationToken::new();

    // Observe the agent's event stream.
    let observer = tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Ok(event) = events_rx.recv().await {
            println!("event: {event:?}");
            seen.push(event);
        }
        seen
    });

    let orchestrator = tokio::spawn(run_orchestrator(Topology {
        vad_events: vad_rx,
        fatal_tx,
        fatal_rx,
        stt: Box::new(stt),
        llm: Box::new(llm),
        tts_engine: Some(Box::new(MockTts::new())),
        sink: sink.clone(),
        shutdown: shutdown.clone(),
        events,
    }));

    // One scripted voice turn: a one-second "segment" (contents are ignored
    // by MockStt; the real pipeline's VAD produces these from mic audio).
    vad_tx
        .send(VadEventMsg::Segment {
            samples: vec![0.0; 16_000],
            t_speech_end: Instant::now(),
        })
        .await
        .map_err(|_| anyhow::anyhow!("orchestrator gone"))?;

    // Wait for the turn to complete (ReplyDone), with a generous bound.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let clips = sink.clips.lock().expect("clips lock").len();
        if clips >= 2 {
            break;
        }
        assert!(Instant::now() < deadline, "timed out waiting for clips");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Shut down cleanly and collect the outcomes.
    shutdown.cancel();
    tokio::time::timeout(Duration::from_secs(5), orchestrator)
        .await
        .expect("orchestrator hung on shutdown")
        .expect("orchestrator panicked")?;

    drop(vad_tx); // close every event-sender path, ending the observer
    let seen = tokio::time::timeout(Duration::from_secs(5), observer)
        .await
        .expect("observer hung")
        .expect("observer panicked");

    // Assertions: the whole plugin chain ran, in order, via events + clips.
    let clips = sink.clips.lock().expect("clips lock");
    assert_eq!(clips.len(), 2, "one MockTts clip per FakeLlm clause");
    assert!(clips.iter().all(|c| c.sample_rate == 24_000));
    assert!(clips.iter().all(|c| !c.samples.is_empty()));
    assert_eq!(
        *sink.flushes.lock().expect("flushes lock"),
        0,
        "no barge-in"
    );
    assert!(
        seen.iter()
            .any(|e| matches!(e, AgentEvent::Transcript(t) if t.contains("unladen swallow"))),
        "MockStt's scripted transcript flowed through"
    );
    assert!(
        seen.iter()
            .any(|e| matches!(e, AgentEvent::Clause(c) if c == "African or European?")),
        "FakeLlm's first clause flowed through"
    );

    // ORDER, not just presence: the startup Listening brackets the turn
    // with the turn-end Listening, and no Listening may land between a
    // turn's Transcript and its ReplyDone (that race existed when the
    // orchestrator emitted Listening at LLM stream-done, before the TTS
    // task had drained the clause backlog).
    let order: Vec<&'static str> = seen
        .iter()
        .map(|e| match e {
            AgentEvent::Listening => "Listening",
            AgentEvent::SpeechStart => "SpeechStart",
            AgentEvent::Transcript(_) => "Transcript",
            AgentEvent::Clause(_) => "Clause",
            AgentEvent::ReplyDone => "ReplyDone",
            AgentEvent::TurnCancelled => "TurnCancelled",
            AgentEvent::StageLatency { .. } => "StageLatency",
            AgentEvent::Error(_) => "Error",
        })
        .collect();
    assert_eq!(
        order,
        vec![
            "Listening",
            "Transcript",
            "Clause",
            "Clause",
            "ReplyDone",
            "Listening"
        ],
        "full-turn event order (seen: {order:?})"
    );

    println!("mock_agent: full plugin turn OK (2 clauses → 2 clips)");
    Ok(())
}
