#![allow(dead_code)]
//! Integration test for the `docbot` example: it loads and chunks a fixture
//! document set, then answers one query through a scripted LLM backend,
//! verifying the load → chunk → retrieve → inject → answer path with no model
//! files and no LLM server.

#[path = "../examples/docbot.rs"]
mod docbot;

use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use skadoosh::llm::LlmBackend;
use skadoosh::rag::Embedder;
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Temp fixture directory (best-effort cleanup on drop).
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "skadoosh-docbot-{}-{label}-{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&p).expect("create temp dir");
        TempDir(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_doc(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("write doc");
}

// ---------------------------------------------------------------------------
// Deterministic bag-of-words embedder (no ONNX model needed).
//
// Mirrors `tests/rag.rs::BowEmbedder`: FNV-1a hash of each word into a fixed
// vector, L2-normalized. Cosine similarity is word overlap, so a query sharing
// words with a chunk ranks it first — enough to exercise the RagStore search
// and prompt-injection logic without a real embedding model.
// ---------------------------------------------------------------------------

struct BowEmbedder {
    dim: usize,
}

impl BowEmbedder {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn vec_for(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        for word in text.to_lowercase().split_whitespace() {
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if word.is_empty() {
                continue;
            }
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in word.as_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            v[(h as usize) % self.dim] += 1.0;
        }
        let mut norm = 0.0_f32;
        for &x in &v {
            norm += x * x;
        }
        norm = norm.sqrt();
        if norm > 1e-12 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Embedder for BowEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.vec_for(t)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// Scripted LLM backend (the cookbook 02 pattern) that also records the prompt
// it received, so the test can assert RAG context was injected.
// ---------------------------------------------------------------------------

struct ScriptedLlm {
    script: Mutex<VecDeque<Vec<String>>>,
    last_prompt: Mutex<Option<String>>,
}

impl ScriptedLlm {
    fn new() -> Self {
        Self {
            script: Mutex::new(VecDeque::new()),
            last_prompt: Mutex::new(None),
        }
    }

    /// Queues one turn's reply (a list of clauses streamed in order).
    fn turn(self, clauses: &[&str]) -> Self {
        self.script
            .lock()
            .expect("script lock")
            .push_back(clauses.iter().map(|s| s.to_string()).collect());
        self
    }

    /// The full user-turn prompt received by the most recent `stream_reply`.
    fn last_prompt(&self) -> String {
        self.last_prompt
            .lock()
            .expect("last_prompt lock")
            .clone()
            .unwrap_or_default()
    }
}

impl LlmBackend for ScriptedLlm {
    fn name(&self) -> &str {
        "scripted-llm"
    }

    fn stream_reply<'a>(
        &'a mut self,
        user: &'a str,
        clauses: mpsc::Sender<String>,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        // Capture the prompt and drain the script before any await so the
        // MutexGuards never cross an await point.
        *self.last_prompt.lock().expect("last_prompt lock") = Some(user.to_string());
        let reply: Vec<String> = self
            .script
            .lock()
            .expect("script lock")
            .pop_front()
            .unwrap_or_else(|| vec!["(no scripted reply)".to_string()]);
        Box::pin(async move {
            for clause in reply {
                let _ = clauses.send(clause).await;
            }
            Ok(())
        })
    }

    fn clear_history(&mut self) {
        self.script.lock().expect("script lock").clear();
    }
}

// ---------------------------------------------------------------------------
// The test.
// ---------------------------------------------------------------------------

/// Builds the fixture document set: a one-paragraph cat-facts `.txt` and a
/// three-paragraph recipe `.md`, so chunking yields more chunks than files.
fn fixtures() -> TempDir {
    let dir = TempDir::new("fixtures");
    write_doc(
        dir.path(),
        "cats.txt",
        "Cats are popular pets. They eat fish and drink water.",
    );
    write_doc(
        dir.path(),
        "recipe.md",
        "# Chocolate Cake\n\n\
         A chocolate cake recipe uses flour, eggs, and sugar.\n\n\
         Bake at 180 degrees Celsius for thirty minutes.",
    );
    dir
}

#[tokio::test]
async fn docbot_loads_chunks_and_answers_via_scripted_llm() {
    let dir = fixtures();

    // 1. Loading + chunking: 2 files -> 4 chunks (1 + 3 paragraphs).
    let chunks = skadoosh::rag::load_documents(dir.path()).expect("load_documents");
    assert_eq!(
        chunks.len(),
        4,
        "2 files should chunk into 4 paragraphs: {chunks:?}"
    );
    assert!(
        chunks.iter().any(|c| c.contains("fish")),
        "cats chunk present"
    );
    assert!(
        chunks.iter().any(|c| c.contains("Chocolate Cake")),
        "recipe title chunk present"
    );

    // 2. Index the same docs through the example's `build_store` (mock embedder).
    let mut store =
        docbot::build_store(dir.path(), Box::new(BowEmbedder::new(4096)), 1).expect("build_store");
    assert_eq!(store.len(), 4, "store indexed all 4 chunks");

    // 3. Answer one query with a scripted LLM (top_k = 1).
    let mut llm =
        ScriptedLlm::new().turn(&["Based on the documents, ", "cats eat fish and drink water."]);
    let query = "What do cats eat and drink water";
    let reply = docbot::answer(&mut store, query, &mut llm)
        .await
        .expect("answer");

    // The reply came from the scripted LLM (reassembled from its clauses).
    assert!(reply.contains("Based on the documents"), "reply: {reply:?}");
    assert!(reply.contains("fish"), "reply: {reply:?}");

    // 4. RAG injection: the prompt the LLM received carries the retrieved
    //    cats chunk and the context block, but not the irrelevant cake chunk
    //    (top_k = 1).
    let prompt = llm.last_prompt();
    assert!(prompt.contains("Relevant context"), "prompt: {prompt:?}");
    assert!(
        prompt.contains("fish"),
        "retrieved chunk injected into prompt: {prompt:?}"
    );
    assert!(
        prompt.contains(query),
        "the question is in the prompt: {prompt:?}"
    );
    assert!(
        !prompt.to_lowercase().contains("chocolate"),
        "irrelevant chunk must not leak into the prompt: {prompt:?}"
    );
}
