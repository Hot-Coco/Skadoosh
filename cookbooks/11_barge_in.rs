//! Cookbook 11 — Barge-in cancellation.
//!
//! Simulates barge-in: while the agent is replying, the user starts speaking
//! again. The VAD detects the fresh speech onset ([`VadEvent::SpeechStart`]),
//! which fires the in-flight turn's [`CancellationToken`]; the streaming
//! reply is then cut short and returns [`LlmError::Cancelled`], and the
//! partial reply is discarded (the user never heard it).
//!
//! Uses a slow scripted [`LlmBackend`] that honors the cancel token — the
//! same contract the real [`LlmClient`] implements. No server, no models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 11_barge_in
//! ```

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use skadoosh::error::{LlmError, SkadooshError};
use skadoosh::llm::LlmBackend;
use skadoosh::vad::{VadEvent, VadSegmenter, FRAME_LEN};
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A slow scripted backend: streams one clause per `delay_ms`, and honors the
/// cancel token — returning [`LlmError::Cancelled`] mid-stream (the barge-in
/// contract: the partial reply is discarded).
struct SlowLlm {
    clauses: Vec<String>,
    delay_ms: u64,
}

impl LlmBackend for SlowLlm {
    fn name(&self) -> &str {
        "slow-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        _user: &'a str,
        clauses: mpsc::Sender<String>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let delay = Duration::from_millis(self.delay_ms);
        let script = self.clauses.clone();
        Box::pin(async move {
            for clause in script {
                // Wait before emitting each clause; bail out on barge-in.
                tokio::select! {
                    _ = cancel.cancelled() => return Err(LlmError::Cancelled.into()),
                    _ = tokio::time::sleep(delay) => {}
                }
                if clauses.send(clause).await.is_err() {
                    return Ok(()); // consumer gone
                }
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {}
}

fn main() -> Result<()> {
    // 1. The barge-in trigger: a VAD segmenter detects a fresh speech onset
    //    (loud frames) — the signal that fires the turn's cancel token.
    let mut vad = VadSegmenter::new(0.5, 300);
    let mut barge_in_detected = false;
    for _ in 0..3 {
        let frame = [0.9_f32; FRAME_LEN]; // loud "speech" burst
        if matches!(vad.push(&frame, 0.9), Some(VadEvent::SpeechStart)) {
            barge_in_detected = true;
        }
    }
    assert!(barge_in_detected, "VAD detects the barge-in speech onset");
    println!("VAD: barge-in speech onset (SpeechStart) detected -> firing cancel token");

    // 2. Start a slow streaming reply, then cancel it mid-stream.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to start tokio runtime: {e}"))?;
    let (received, outcome) = rt.block_on(async {
        let mut llm = SlowLlm {
            clauses: vec![
                "First. ".to_string(),
                "Second. ".to_string(),
                "Third. ".to_string(),
                "Fourth. ".to_string(),
                "Fifth.".to_string(),
            ],
            delay_ms: 60,
        };
        let token = CancellationToken::new();
        let (tx, mut rx) = mpsc::channel(16);
        let turn = llm.stream_reply("give me a long answer", tx, token.clone());
        tokio::pin!(turn);

        // Cancellation driver: simulate the barge-in by cancelling the turn's
        // token shortly after the first clause lands.
        let canceller = {
            let t = token.clone();
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(150)).await;
                t.cancel();
            })
        };

        let mut got = Vec::new();
        let mut result: Option<Result<()>> = None;
        loop {
            tokio::select! {
                clause = rx.recv() => match clause {
                    Some(c) => got.push(c),
                    None => if result.is_some() { break; },
                },
                res = &mut turn => {
                    result = Some(res);
                    while let Some(c) = rx.recv().await { got.push(c); } // drain remainder
                    break;
                }
            }
        }
        let _ = canceller.await;
        (got, result)
    });

    let n = received.len();
    println!("received {n} of 5 clauses before barge-in cut the stream");
    let outcome = outcome.expect("stream resolved");
    match &outcome {
        Err(SkadooshError::Llm(LlmError::Cancelled)) => {
            println!("stream_reply returned LlmError::Cancelled (barge-in honored)");
        }
        other => println!("stream_reply returned: {other:?}"),
    }

    assert!(
        matches!(outcome, Err(SkadooshError::Llm(LlmError::Cancelled))),
        "barge-in must cancel the turn with LlmError::Cancelled"
    );
    assert!(n < 5, "stream was cut short (got {n} clauses)");
    assert!(!received.is_empty(), "at least the first clause arrived");

    println!("\n11_barge_in: OK");
    Ok(())
}
