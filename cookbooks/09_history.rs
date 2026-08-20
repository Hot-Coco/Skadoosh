//! Cookbook 09 — History truncation across turns.
//!
//! Demonstrates the real [`LlmClient`]'s bounded conversation history:
//! after each turn only the system prompt plus the last
//! `max_history_turns` user/assistant turns are kept, so long sessions
//! cannot overflow a small model's context.
//!
//! To exercise the genuine HTTP streaming path headlessly (no external
//! server), this cookbook reuses the crate's in-process mock OpenAI SSE
//! server from the test suite (`tests/common/mock_openai.rs`). Three turns
//! are run with `max_history_turns = 2`; the first turn's messages fall off
//! the end.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 09_history
//! ```

// Reuse the test suite's in-process mock OpenAI-compatible SSE server.
// `#[allow(dead_code)]` silences unused-item warnings from the helper (it
// exposes more knobs than this cookbook exercises) — same as tests/sdk_agent.rs.
#[allow(dead_code)]
#[path = "../tests/common/mock_openai.rs"]
mod mock_openai;

use mock_openai::{done_line, token_line, Chunk, MockOpenAi};

use skadoosh::llm::LlmClient;
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const SYSTEM: &str = "You are a test bot.";

fn main() -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to start tokio runtime: {e}"))?;

    // Annotating the result pins the async block's error type to
    // `SkadooshError` so the inner `?` conversions resolve unambiguously.
    let ran: skadoosh::Result<()> = rt.block_on(async {
        // One scripted SSE response per turn: a single token then [DONE].
        let script = |reply: &str| vec![Chunk::now(token_line(reply)), Chunk::now(done_line())];
        let server =
            MockOpenAi::serve_many(vec![script("A one."), script("A two."), script("A three.")])
                .await;

        // max_history_turns = 2 → keep the system prompt + last 2 turns
        // (4 non-system messages) max.
        let mut client = LlmClient::new(&server.url(), "mock-model", SYSTEM, 2, None);

        for (i, user) in ["u1", "u2", "u3"].iter().enumerate() {
            let (tx, _rx) = mpsc::channel(16);
            client
                .stream_reply(user, tx, CancellationToken::new())
                .await?;
            println!(
                "after turn {} ({}): history len = {}",
                i + 1,
                user,
                client.history().len()
            );
        }

        let h = client.history();
        println!("\nfinal history (system + last 2 turns):");
        for m in h {
            println!(
                "  {:<9} {}",
                m.role,
                m.content.as_text().unwrap_or("<multimodal>"),
            );
        }

        // system + u2/a2 + u3/a3 == 5 messages; u1/a1 truncated off.
        assert_eq!(h.len(), 5, "bounded to system + 2 turns: {h:?}");
        assert_eq!(h[0].role, "system");
        assert_eq!(h[0].content.as_text(), Some(SYSTEM));
        assert_eq!(h[1].content.as_text(), Some("u2"));
        assert_eq!(h[2].content.as_text(), Some("A two."));
        assert_eq!(h[3].content.as_text(), Some("u3"));
        assert_eq!(h[4].content.as_text(), Some("A three."));
        assert!(
            !h.iter().any(|m| m.content.as_text() == Some("u1")),
            "u1 must have been truncated"
        );

        println!("\n09_history: OK");
        Ok(())
    });
    ran?;
    Ok(())
}
