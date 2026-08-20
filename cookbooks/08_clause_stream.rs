//! Cookbook 08 — Clause streaming from a mock LLM.
//!
//! Drives the [`LlmBackend::stream_reply`] contract directly (the lower-level
//! streaming API under [`Agent::text_turn`]): a scripted backend streams its
//! reply one clause at a time over an `mpsc` channel, and we print each
//! clause *as it arrives* — the same clause stream the pipeline feeds into
//! TTS (audio mode) or prints (text mode). No server, no models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 08_clause_stream
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// A scripted LLM backend that streams a queued reply clause by clause,
/// yielding between clauses so the streaming arrival order is observable.
struct StreamingLlm {
    script: Mutex<VecDeque<Vec<String>>>,
}

impl StreamingLlm {
    fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
        }
    }

    fn turn(self, clauses: &[&str]) -> Self {
        self.script
            .lock()
            .expect("script lock")
            .push_back(clauses.iter().map(|s| s.to_string()).collect());
        self
    }
}

impl LlmBackend for StreamingLlm {
    fn name(&self) -> &str {
        "streaming-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        _user: &'a str,
        clauses: mpsc::Sender<String>,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let reply: Vec<String> = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .unwrap_or_else(|| vec!["(no scripted reply)".to_string()]);
        Box::pin(async move {
            for clause in reply {
                // Yield between clauses: emulates token-arrival latency so
                // each clause is received separately (as in a real stream).
                tokio::task::yield_now().await;
                if clauses.send(clause).await.is_err() {
                    // Consumer gone (e.g. barge-in): stop streaming quietly.
                    return Ok(());
                }
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {
        self.script.lock().expect("script lock").clear();
    }
}

fn main() -> Result<()> {
    // A private current-thread runtime: stream_reply is async, so we need a
    // driver. (Agent::text_turn hides this; here we exercise the seam.)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to start tokio runtime: {e}"))?;
    let received: Vec<String> = rt.block_on(async {
        let mut llm = StreamingLlm::new().turn(&[
            "Skadoosh ",
            "streams replies ",
            "clause by clause, ",
            "low latency!",
        ]);

        let (tx, mut rx) = mpsc::channel(16);
        let token = CancellationToken::new();
        // Drive the stream future while receiving clauses as they arrive.
        let turn = llm.stream_reply("tell me about streaming", tx, token);
        tokio::pin!(turn);

        let mut got = Vec::new();
        loop {
            tokio::select! {
                clause = rx.recv() => match clause {
                    Some(c) => {
                        println!("clause #{:>2}: {:?}", got.len() + 1, c);
                        got.push(c);
                    }
                    None => break, // sender dropped → stream finished
                },
                _ = &mut turn => {
                    // Stream future resolved: drain anything still buffered.
                    while let Some(c) = rx.recv().await {
                        println!("clause #{:>2}: {:?}", got.len() + 1, c);
                        got.push(c);
                    }
                    break;
                }
            }
        }
        got
    });

    println!("\nreceived {} clause(s)", received.len());
    let reassembled: String = received.concat();
    println!("reassembled reply: {reassembled}");

    assert_eq!(received.len(), 4, "four clauses streamed one at a time");
    assert_eq!(
        reassembled,
        "Skadoosh streams replies clause by clause, low latency!"
    );

    println!("\n08_clause_stream: OK");
    Ok(())
}
