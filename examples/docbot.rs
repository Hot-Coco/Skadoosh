//! Example — `docbot`: a retrieval-augmented generation (RAG) demo.
//!
//! Loads every `.txt`/`.md` document under a directory, chunks and embeds
//! them with [`skadoosh::rag::RagStore`], then answers a question by
//! retrieving the top-k most relevant chunks and injecting them into the LLM
//! prompt. Text-only — no audio devices, no Whisper/VAD/TTS.
//!
//! This is a focused demo of the RAG *API* (`RagStore::build` →
//! [`RagStore::search`] → prompt injection → [`LlmBackend::stream_reply`]),
//! not a multi-turn chatbot: each `Ask>` turn is independent and keeps no
//! conversation history, so the docbot is a sequence of single-query turns.
//!
//! # Setup
//!
//! 1. Run a local OpenAI-compatible LLM server (Ollama is the default
//!    endpoint) and pull a model:
//!    ```sh
//!    ollama pull llama3.2
//!    ollama serve            # listens on http://localhost:11434
//!    ```
//! 2. Fetch the embedding model used to index the documents:
//!    ```sh
//!    ./scripts/download_models.sh --with-rag
//!    ```
//!    This places `models/all-MiniLM-L6-v2.onnx` and its companion
//!    `models/all-MiniLM-L6-v2-vocab.txt`.
//! 3. Point `SKADOOSH_RAG_DIR` at a directory of `.txt`/`.md` files, e.g.:
//!    ```sh
//!    export SKADOOSH_RAG_DIR=$PWD/docs
//!    ```
//!
//! # Running
//!
//! ```sh
//! cargo run --example docbot
//! ```
//!
//! Type a question at the `Ask>` prompt. `/docs` lists the indexed documents
//! and `/quit` (or Ctrl-D / EOF) exits.
//!
//! # Environment
//!
//! * `SKADOOSH_RAG_DIR` *(required)* — directory of `.txt`/`.md` documents to
//!   index (walked recursively).
//! * `SKADOOSH_BASE_URL` *(optional, default `http://localhost:11434/v1`)* —
//!   the OpenAI-compatible LLM endpoint.
//! * `SKADOOSH_MODEL` *(optional, default `llama3.2`)* — model name requested
//!   from the endpoint.
//! * `SKADOOSH_RAG_MODEL` *(optional, default
//!   `models/all-MiniLM-L6-v2.onnx`)* — path to the embedding ONNX model.
//! * `SKADOOSH_RAG_TOP_K` *(optional, default `3`)* — chunks retrieved per
//!   query.
//! * `SKADOOSH_API_KEY` *(optional)* — bearer token for hosted providers;
//!   local Ollama needs none.
//!
//! # Testing
//!
//! `tests/examples.rs` drives [`build_store`] and [`answer`] against a
//! deterministic mock embedder and a scripted LLM backend (no model files, no
//! server) to verify the load → chunk → retrieve → inject → answer path.

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use skadoosh::llm::{LlmBackend, LlmClient};
use skadoosh::rag::{Embedder, OnnxEmbedder, RagStore, DEFAULT_MAX_SEQ_LEN, DEFAULT_RAG_MODEL};
use skadoosh::Result;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Capacity of the clause channel between the LLM stream and the collector.
/// Matches the agent's text-turn channel; the `select!` in [`answer`] drains
/// concurrently so a long reply never blocks on it.
const CLAUSE_CAP: usize = 16;

/// Default chunks retrieved per query when `SKADOOSH_RAG_TOP_K` is unset.
const DEFAULT_TOP_K: usize = 3;

/// Persona carried by the LLM client's *system* message. The per-turn context
/// (retrieved chunks) is assembled separately by [`build_prompt`] into the
/// *user* message, so this never needs to repeat the documents.
const SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Answer the user's question using the provided \
     context when it is relevant; say so if the context does not contain an answer.";

/// Builds the user-turn prompt for one RAG query: the retrieved context block
/// (top-k chunks) followed by the question. With no chunks the question is
/// returned bare — retrieval must never block a turn.
///
/// Only the per-turn user message is assembled here (the persona lives in the
/// client's system prompt, set in `main`), so this works with any
/// [`LlmBackend`] — a real [`LlmClient`] in `main` or a scripted mock in the
/// test.
fn build_prompt(query: &str, chunks: &[String]) -> String {
    if chunks.is_empty() {
        return format!("Question: {query}");
    }
    format!(
        "Relevant context:\n{}\n\nAnswer using this context if helpful.\n\nQuestion: {query}",
        chunks.join("\n")
    )
}

/// Loads and embeds every `.txt`/`.md` document under `rag_dir`, returning a
/// ready [`RagStore`] keyed by `embedder`. Thin wrapper over
/// [`RagStore::build`] that logs the indexed chunk count — shared by `main`
/// (a real [`OnnxEmbedder`]) and the test (a deterministic mock embedder).
pub fn build_store(rag_dir: &Path, embedder: Box<dyn Embedder>, top_k: usize) -> Result<RagStore> {
    let store = RagStore::build(rag_dir, embedder, top_k)?;
    eprintln!(
        "docbot: indexed {} chunk(s) from {}",
        store.len(),
        rag_dir.display()
    );
    Ok(store)
}

/// One RAG turn: retrieves the top-k chunks for `query` from `store`, injects
/// them into the prompt, streams the reply from `llm`, and returns the full
/// reply text.
///
/// This is the testable core — `main` passes a real [`LlmClient`]; the test
/// passes a scripted [`LlmBackend`] (see `tests/examples.rs`). Each call is
/// stateless: it searches the index and asks one question, with no carry-over
/// conversation history.
pub async fn answer(store: &mut RagStore, query: &str, llm: &mut dyn LlmBackend) -> Result<String> {
    let top_k = store.top_k;
    let chunks = store.search(query, top_k);
    let prompt = build_prompt(query, &chunks);

    let (tx, mut rx) = mpsc::channel::<String>(CLAUSE_CAP);
    let cancel = CancellationToken::new();
    let turn = llm.stream_reply(&prompt, tx, cancel);
    tokio::pin!(turn);

    // Drive the stream and drain clauses concurrently (select!) so a long
    // reply can never block on the bounded channel; once the stream resolves,
    // drain anything still buffered.
    let mut reply = String::new();
    let mut stream_result: Option<Result<()>> = None;
    loop {
        tokio::select! {
            biased;
            result = &mut turn => {
                stream_result = Some(result);
                while let Ok(clause) = rx.try_recv() {
                    reply.push_str(&clause);
                }
                break;
            }
            clause = rx.recv() => match clause {
                Some(clause) => reply.push_str(&clause),
                // The sender dropped before `turn` resolved (unreachable in
                // practice — the future owns it — but guard anyway).
                None => break,
            },
        }
    }
    match stream_result {
        Some(Ok(())) | None => Ok(reply),
        Some(Err(err)) => Err(err),
    }
}

/// Recursively lists the `.txt`/`.md` files under `rag_dir` (sorted), for the
/// `/docs` command. The [`RagStore`] keeps chunks, not file paths, so this is
/// a cheap directory walk mirroring the loader's own discovery rules
/// (`.txt`/`.md`, recursive, sorted, no symlinks).
fn list_documents(rag_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_docs(rag_dir, &mut out);
    out.sort();
    out
}

fn walk_docs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_dir() {
            walk_docs(&path, out);
        } else if ft.is_file() && is_doc(&path) {
            out.push(path);
        }
    }
}

/// `true` for `.txt`/`.md` extensions (case-insensitive) — matches
/// `skadoosh::rag`'s own document filter.
fn is_doc(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("txt") | Some("md")
    )
}

/// The interactive loop. Each input line is one independent RAG query turn
/// (no conversation history is kept): `/docs` lists the indexed documents,
/// `/quit` exits, and any other non-empty line is answered from the store.
async fn repl(store: &mut RagStore, llm: &mut LlmClient, rag_dir: &Path) -> Result<()> {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("Ask> ");
        let _ = io::stdout().flush();
        let line = match lines.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                return Err(anyhow::anyhow!("stdin read error: {e}").into());
            }
            None => break, // EOF (Ctrl-D)
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" {
            break;
        }
        if line == "/docs" {
            let docs = list_documents(rag_dir);
            if docs.is_empty() {
                println!("(no .txt/.md documents found in {})", rag_dir.display());
            } else {
                println!("Indexed documents ({}):", docs.len());
                for d in &docs {
                    println!("  - {}", d.display());
                }
            }
            continue;
        }
        match answer(store, line, llm).await {
            Ok(reply) => println!("Answer> {}\n", reply.trim()),
            Err(e) => eprintln!("error: {e}"),
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    // --- environment --------------------------------------------------------
    let rag_dir = std::env::var("SKADOOSH_RAG_DIR").map_err(|_| {
        anyhow::anyhow!(
            "SKADOOSH_RAG_DIR is required: point it at a directory of .txt/.md documents to index"
        )
    })?;
    let base_url = std::env::var("SKADOOSH_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let model = std::env::var("SKADOOSH_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
    let api_key = std::env::var("SKADOOSH_API_KEY").ok();
    let rag_model =
        std::env::var("SKADOOSH_RAG_MODEL").unwrap_or_else(|_| DEFAULT_RAG_MODEL.to_string());
    let top_k = std::env::var("SKADOOSH_RAG_TOP_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_TOP_K);

    let rag_dir = PathBuf::from(&rag_dir);
    let rag_model = PathBuf::from(&rag_model);

    // --- index the documents (real ONNX embedder) ---------------------------
    let vocab = OnnxEmbedder::companion_vocab(&rag_model);
    if !rag_model.is_file() || !vocab.is_file() {
        return Err(anyhow::anyhow!(
            "RAG embedding model not found at {} (or its companion vocab {}). \
             Run `./scripts/download_models.sh --with-rag` to fetch all-MiniLM-L6-v2.",
            rag_model.display(),
            vocab.display(),
        )
        .into());
    }
    let embedder = OnnxEmbedder::load(&rag_model, &vocab, DEFAULT_MAX_SEQ_LEN)?;
    let mut store = build_store(&rag_dir, Box::new(embedder), top_k)?;

    // --- real LLM client (OpenAI-compatible) --------------------------------
    let mut llm = LlmClient::new(&base_url, &model, SYSTEM_PROMPT, 8, api_key);

    // --- REPL: one independent single-query turn per line -------------------
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("failed to start runtime: {e}"))?;
    rt.block_on(repl(&mut store, &mut llm, &rag_dir))
}
