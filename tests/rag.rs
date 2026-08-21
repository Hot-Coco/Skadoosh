#![allow(dead_code)]
//! RAG pipeline integration tests: document chunking, ONNX embedding, cosine
//! search, and system-prompt context injection.
//!
//! Chunking and the mock-embedder search/injection tests need no model files
//! and run everywhere. The real-embedding and real-search tests skip with a
//! printed reason when `all-MiniLM-L6-v2` is absent (both fetched by
//! `scripts/download_models.sh --with-rag`).

#[path = "common/mock_openai.rs"]
mod mock_openai;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use skadoosh::rag::{Embedder, OnnxEmbedder, RagStore, DEFAULT_MAX_SEQ_LEN};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use mock_openai::{done_line, token_line, Chunk, MockOpenAi};

const RAG_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/all-MiniLM-L6-v2.onnx");

/// `true` when the embedding model and its companion vocab are present.
fn rag_model_present() -> bool {
    let model = Path::new(RAG_MODEL);
    let present = model.is_file() && OnnxEmbedder::companion_vocab(model).is_file();
    if !present {
        eprintln!(
            "skipping RAG model test: {RAG_MODEL} (or its vocab) missing \
             (run scripts/download_models.sh --with-rag)"
        );
    }
    present
}

/// A unique temp directory per test (process id + monotonic counter) so
/// parallel runs never collide. [`TempDir`] removes it on drop (best-effort).
fn temp_dir(label: &str) -> (PathBuf, TempDir) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!("skadoosh-rag-{}-{label}-{n}", std::process::id()));
    std::fs::create_dir_all(&p).expect("create temp dir");
    (p.clone(), TempDir(p))
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Writes `name` (a `.txt`/`.md` file) with `contents` under `dir`.
fn write_doc(dir: &Path, name: &str, contents: &str) {
    std::fs::write(dir.join(name), contents).expect("write doc");
}

/// Deterministic bag-of-words embedder (FNV-1a hash of each word into a fixed
/// vector, L2-normalized). No model needed; cosine similarity is word overlap,
/// so a query sharing words with a chunk ranks it first. Used to test the
/// [`RagStore`] search and prompt-injection logic without ONNX.
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
    fn embed(&mut self, texts: &[&str]) -> skadoosh::error::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.vec_for(t)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Builds a [`RagStore`] from a temp dir of docs using [`BowEmbedder`].
fn bow_store(dir: &Path, top_k: usize) -> RagStore {
    let embedder = BowEmbedder::new(4096);
    RagStore::build(dir, Box::new(embedder), top_k).expect("build store")
}

// ---------------------------------------------------------------------------
// Chunk loading.
// ---------------------------------------------------------------------------

#[test]
fn load_documents_chunks_by_paragraph_and_caps_size() {
    let (dir, _g) = temp_dir("chunks");
    write_doc(
        &dir,
        "notes.txt",
        "First paragraph here.\nStill the first one.\n\n\
         Second paragraph is short.\n\n\
         Third.",
    );
    let chunks = skadoosh::rag::load_documents(&dir).expect("load");
    assert_eq!(
        chunks,
        vec![
            "First paragraph here. Still the first one.".to_string(),
            "Second paragraph is short.".to_string(),
            "Third.".to_string(),
        ]
    );
}

#[test]
fn load_documents_reads_txt_and_md_recursively() {
    let (dir, _g) = temp_dir("recurse");
    write_doc(&dir, "a.md", "# Title\n\nMarkdown body.\n");
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    write_doc(&dir.join("sub"), "b.txt", "Nested doc.\n");
    // Ignored: non-text extension.
    write_doc(&dir, "ignore.log", "not indexed\n");
    let chunks = skadoosh::rag::load_documents(&dir).expect("load");
    // a.md -> "# Title" + "Markdown body." (two paragraphs); b.txt -> "Nested doc."
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().any(|c| c.contains("Markdown body")));
    assert!(chunks.iter().any(|c| c.contains("Nested doc")));
    // The non-text .log file must be ignored.
    assert!(!chunks.iter().any(|c| c.contains("not indexed")));
}

#[test]
fn load_documents_splits_oversized_paragraphs_under_max() {
    let (dir, _g) = temp_dir("oversize");
    // One long paragraph with many short words -> multiple packed chunks.
    let para = (0..400)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    write_doc(&dir, "big.txt", &para);
    let chunks = skadoosh::rag::load_documents(&dir).expect("load");
    assert!(chunks.len() > 1);
    assert!(
        chunks
            .iter()
            .all(|c| c.chars().count() <= skadoosh::rag::MAX_CHUNK_CHARS),
        "every chunk must be within the cap"
    );
    // No words lost (join with spaces: `concat` would fuse boundary words).
    assert_eq!(chunks.join(" ").split_whitespace().count(), 400);
}

#[test]
fn load_documents_missing_dir_errors() {
    let nope = std::env::temp_dir().join("skadoosh-rag-does-not-exist-xyz");
    assert!(skadoosh::rag::load_documents(&nope).is_err());
}

// ---------------------------------------------------------------------------
// Search (mock embedder — no model required).
// ---------------------------------------------------------------------------

#[test]
fn search_ranks_lexically_overlapping_chunk_first() {
    let (dir, _g) = temp_dir("search");
    write_doc(&dir, "cats.txt", "cats eat fish and drink water");
    write_doc(&dir, "paris.txt", "the eiffel tower is in paris france");
    write_doc(
        &dir,
        "plants.txt",
        "photosynthesis converts sunlight into energy",
    );
    let mut store = bow_store(&dir, 3);
    assert_eq!(store.len(), 3);

    let top = store.search("what do cats eat and drink", 1);
    assert_eq!(top.len(), 1);
    assert!(
        top[0].contains("cats"),
        "top result should be the cats chunk, got: {:?}",
        top[0]
    );

    // top_k bounds the result count, most-similar first.
    let two = store.search("what do cats eat and drink", 2);
    assert_eq!(two.len(), 2);
    assert!(two[0].contains("cats"));
}

#[test]
fn search_empty_store_returns_empty() {
    let (dir, _g) = temp_dir("empty");
    // No docs -> empty index.
    let mut store = bow_store(&dir, 3);
    assert!(store.is_empty());
    assert!(store.search("anything", 3).is_empty());
    // An empty embedder result also yields no chunks.
    let _ = store; // store with zero chunks stays usable.
}

// ---------------------------------------------------------------------------
// Embedding (real model — skipped when absent).
// ---------------------------------------------------------------------------

#[test]
fn onnx_embedder_produces_normalized_384d_vectors() {
    if !rag_model_present() {
        return;
    }
    let model = Path::new(RAG_MODEL);
    let vocab = OnnxEmbedder::companion_vocab(model);
    let mut emb = OnnxEmbedder::load(model, &vocab, DEFAULT_MAX_SEQ_LEN).expect("load embedder");

    let out = emb
        .embed(&["a cat sat on the mat", "dogs are loyal pets"])
        .expect("embed");
    assert_eq!(out.len(), 2);
    for v in &out {
        assert_eq!(v.len(), 384, "all-MiniLM-L6-v2 hidden size is 384");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "embedding must be L2-normalized, got norm {norm}"
        );
    }
    // Empty input -> empty output (no inference, no panic).
    assert!(emb.embed(&[]).expect("embed empty").is_empty());
}

#[test]
fn onnx_embedder_related_sentences_are_more_similar() {
    if !rag_model_present() {
        return;
    }
    let model = Path::new(RAG_MODEL);
    let vocab = OnnxEmbedder::companion_vocab(model);
    let mut emb = OnnxEmbedder::load(model, &vocab, DEFAULT_MAX_SEQ_LEN).expect("load embedder");

    let v = emb
        .embed(&[
            "Cats are popular pets that love to sleep.",
            "Dogs are loyal companions and good pets.",
            "A chocolate cake recipe uses flour and sugar.",
        ])
        .expect("embed");
    let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let cat_dog = cos(&v[0], &v[1]);
    let cat_cake = cos(&v[0], &v[2]);
    // Related (cat/dog, both pets) must beat unrelated (cat/cake).
    assert!(
        cat_dog > cat_cake,
        "cos(cat,dog)={cat_dog:.3} should exceed cos(cat,cake)={cat_cake:.3}"
    );
}

#[test]
fn rag_store_real_search_retrieves_relevant_chunk() {
    if !rag_model_present() {
        return;
    }
    let (dir, _g) = temp_dir("realsearch");
    write_doc(
        &dir,
        "cats.txt",
        "Cats are popular pets that love to sleep and play with toys.",
    );
    write_doc(
        &dir,
        "cake.txt",
        "A chocolate cake recipe uses flour, eggs, and sugar.",
    );
    write_doc(
        &dir,
        "paris.txt",
        "The Eiffel Tower is a famous landmark in Paris, France.",
    );

    let model = Path::new(RAG_MODEL);
    let vocab = OnnxEmbedder::companion_vocab(model);
    let embedder = OnnxEmbedder::load(model, &vocab, DEFAULT_MAX_SEQ_LEN).expect("load");
    let mut store = RagStore::build(&dir, Box::new(embedder), 3).expect("build");
    assert_eq!(store.len(), 3);

    let top = store.search("What do cats eat?", 1);
    assert_eq!(top.len(), 1);
    assert!(
        top[0].to_lowercase().contains("cats"),
        "query about cats should retrieve the cats chunk, got: {:?}",
        top[0]
    );
}

// ---------------------------------------------------------------------------
// Pipeline integration: RAG context is injected into the system prompt.
// ---------------------------------------------------------------------------

/// Drains the clause channel after `stream_reply` returns (its `Sender` is
/// dropped, so `recv` terminates).
async fn drain(rx: &mut mpsc::Receiver<String>) -> Vec<String> {
    let mut got = Vec::new();
    while let Some(c) = rx.recv().await {
        got.push(c);
    }
    got
}

#[tokio::test]
async fn rag_context_is_injected_into_system_prompt() {
    let (dir, _g) = temp_dir("inject");
    write_doc(&dir, "cats.txt", "cats eat fish and drink water");
    write_doc(&dir, "paris.txt", "the eiffel tower is in paris france");
    // top_k = 1 so only the single most-relevant chunk is injected.
    let store = bow_store(&dir, 1);

    let server =
        MockOpenAi::serve(vec![Chunk::now(token_line("Ok.")), Chunk::now(done_line())]).await;

    let mut client =
        skadoosh::llm::LlmClient::new(&server.url(), "mock-model", "You are a test bot.", 8, None)
            .with_rag(store);
    let (tx, mut rx) = mpsc::channel(16);
    client
        .stream_reply("what do cats eat and drink", tx, CancellationToken::new())
        .await
        .expect("stream_reply should succeed");
    let _ = drain(&mut rx).await;

    let req = server.captured_request().expect("request captured");
    // The base system prompt is preserved ...
    assert!(req.contains("You are a test bot."), "request: {req}");
    // ... and the RAG block + the relevant chunk are injected into it.
    assert!(req.contains("Relevant context"), "request: {req}");
    assert!(
        req.contains("Answer using this context if helpful."),
        "request: {req}"
    );
    assert!(
        req.contains("cats"),
        "request must carry the cats chunk: {req}"
    );
    // The irrelevant chunk is not injected at top_k = 1.
    assert!(
        !req.contains("eiffel"),
        "irrelevant chunk must not leak into the prompt: {req}"
    );
}
