//! Retrieval-augmented generation (RAG): load `.txt`/`.md` documents from a
//! directory, chunk them by paragraph, embed each chunk with a small ONNX
//! sentence-embedding model ([`OnnxEmbedder`], all-MiniLM-L6-v2 via `ort`),
//! and retrieve the top-k chunks for a query by cosine similarity.
//!
//! The embedder ports the standard sentence-transformers pipeline to pure
//! Rust: a BERT WordPiece tokenizer (no `tokenizers` crate) feeds
//! `input_ids`/`attention_mask`/`token_type_ids` to the model, the
//! `last_hidden_state` is mean-pooled (masked) and L2-normalized so that a
//! dot product *is* the cosine similarity. The LLM client
//! (`LlmClient::from_config`) loads the docs once on startup and
//! injects the top-k chunks for the current query into the system prompt.
//!
//! # Limitations
//!
//! The tokenizer lowercases and splits on ASCII punctuation but does not strip
//! accents (no `unicode-normalization` dependency), so accented characters fall
//! back to `[UNK]` subwords — fine for English prose, slightly lower quality
//! for heavy diacritics.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::error::{RagError, Result};

/// Default path for the embedding ONNX model
/// (`--rag-model` / `SKADOOSH_RAG_MODEL`).
pub const DEFAULT_RAG_MODEL: &str = "models/all-MiniLM-L6-v2.onnx";

/// Default maximum sequence length the embedder pads/truncates to.
/// all-MiniLM-L6-v2 was exported with a dynamic sequence dimension; 128 keeps
/// batch inference fast while covering typical document sentences.
pub const DEFAULT_MAX_SEQ_LEN: usize = 128;

/// Maximum characters per chunk (~512, per `--rag-dir` chunking).
pub const MAX_CHUNK_CHARS: usize = 512;

/// Special token ids for the BERT uncased vocabulary used by all-MiniLM-L6-v2
/// (`[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102).
const PAD_ID: i64 = 0;
const UNK_ID: i64 = 100;
const CLS_ID: i64 = 101;
const SEP_ID: i64 = 102;

/// Maximum characters per word before WordPiece gives up and returns `[UNK]`
/// (matches HuggingFace's `BasicTokenizer` default).
const MAX_INPUT_CHARS_PER_WORD: usize = 100;

/// Mini-batch size for embedding inference (bounds peak memory: one
/// `batch * max_seq_len * 384` f32 output tensor at a time).
const EMBED_BATCH: usize = 32;

/// Pluggable text embedder: batch-convert texts into fixed-dim, L2-normalized
/// vectors. Implemented by [`OnnxEmbedder`]; tests inject a deterministic mock.
/// Requiring `Send` lets a [`RagStore`] live inside the `Send`
/// [`crate::llm::LlmClient`].
pub trait Embedder: Send {
    /// Embeds `texts` (in order), returning one vector per input.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// The embedding dimension (384 for all-MiniLM-L6-v2).
    fn dim(&self) -> usize;
}

/// ONNX Runtime sentence embedder for all-MiniLM-L6-v2 (384-dim, mean-pooled +
/// L2-normalized). Loads the model and its companion BERT vocab (see
/// [`OnnxEmbedder::companion_vocab`]).
pub struct OnnxEmbedder {
    session: Session,
    vocab: HashMap<String, i64>,
    max_seq_len: usize,
    /// Detected embedding dimension (set on the first successful
    /// [`Embedder::embed`] call; `384` before then).
    dim: usize,
}

impl OnnxEmbedder {
    /// Loads the ONNX model and vocab. `max_seq_len` is the padded sequence
    /// length (use [`DEFAULT_MAX_SEQ_LEN`] unless the model was exported with a
    /// different fixed length).
    pub fn load(model: &Path, vocab: &Path, max_seq_len: usize) -> Result<Self> {
        let vocab = load_vocab(vocab)?;
        let builder = Session::builder().map_err(|e| RagError::ModelLoad(e.to_string()))?;
        let mut builder = crate::gpu::apply_gpu_ep(builder)
            .map_err(|e| RagError::ModelLoad(format!("GPU EP: {e}")))?;
        let session = builder
            .commit_from_file(model)
            .map_err(|e| RagError::ModelLoad(format!("{}: {e}", model.display())))?;
        Ok(Self {
            session,
            vocab,
            max_seq_len,
            dim: 0,
        })
    }

    /// Derives the companion BERT vocab path from the model path:
    /// `<file_stem>-vocab.txt` in the same directory. For the default model
    /// `models/all-MiniLM-L6-v2.onnx` this is
    /// `models/all-MiniLM-L6-v2-vocab.txt` (what
    /// `scripts/download_models.sh --with-rag` fetches).
    pub fn companion_vocab(model: &Path) -> PathBuf {
        let stem = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        model.with_file_name(format!("{stem}-vocab.txt"))
    }

    /// Tokenizes `text` into `(input_ids, attention_mask)` padded to
    /// `max_seq_len`: `[CLS] <wordpieces> [SEP] [PAD]...`, mask `1` for real
    /// tokens and `0` for padding. Mirrors the HuggingFace BERT uncased
    /// tokenizer (see the module docs for the accent limitation).
    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        tokenize(text, &self.vocab, self.max_seq_len)
    }

    /// Runs one mini-batch through the model and pools the output into
    /// `batch` L2-normalized vectors. `ids`/`mask` are flattened row-major
    /// (`batch * max_seq_len`).
    fn run_batch(&mut self, ids: &[i64], mask: &[i64], batch: usize) -> Result<Vec<Vec<f32>>> {
        let seq = self.max_seq_len;
        let input_ids = Tensor::from_array((vec![batch as i64, seq as i64], ids.to_vec()))
            .map_err(|e| RagError::Inference(format!("input_ids tensor: {e}")))?;
        let attention = Tensor::from_array((vec![batch as i64, seq as i64], mask.to_vec()))
            .map_err(|e| RagError::Inference(format!("attention_mask tensor: {e}")))?;

        let mut inputs: Vec<(String, Tensor<i64>)> = Vec::with_capacity(3);
        inputs.push(("input_ids".to_string(), input_ids));
        inputs.push(("attention_mask".to_string(), attention));
        // Some exports drop token_type_ids; feed zeros only when declared.
        if self
            .session
            .inputs()
            .iter()
            .any(|o| o.name() == "token_type_ids")
        {
            let token_type =
                Tensor::from_array((vec![batch as i64, seq as i64], vec![0_i64; batch * seq]))
                    .map_err(|e| RagError::Inference(format!("token_type_ids tensor: {e}")))?;
            inputs.push(("token_type_ids".to_string(), token_type));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| RagError::Inference(e.to_string()))?;
        // The first output is the embedding tensor — either
        // `last_hidden_state` [B, S, D] (mean-pool below) or a pre-pooled
        // [B, D] (use directly). A valid embedding model always has ≥1
        // output, so `outputs[0]` is safe here (mirrors the TTS engine).
        let out_val = &outputs[0];
        let (shape, data) = out_val
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Inference(format!("extract embedding tensor: {e}")))?;
        match shape.len() {
            3 => {
                let dim = shape[2] as usize;
                mean_pool(data, mask, batch, seq, dim)
            }
            2 => {
                let dim = shape[1] as usize;
                copy_rows(data, batch, dim)
            }
            n => Err(RagError::Inference(format!(
                "unexpected embedding output rank {n} (expected 2 or 3)"
            ))
            .into()),
        }
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for batch_texts in texts.chunks(EMBED_BATCH) {
            let batch = batch_texts.len();
            let mut ids = Vec::with_capacity(batch * self.max_seq_len);
            let mut mask = Vec::with_capacity(batch * self.max_seq_len);
            for t in batch_texts {
                let (i, m) = self.tokenize(t);
                ids.extend_from_slice(&i);
                mask.extend_from_slice(&m);
            }
            let mut embs = self.run_batch(&ids, &mask, batch)?;
            if self.dim == 0 {
                if let Some(first) = embs.first() {
                    self.dim = first.len();
                }
            }
            out.append(&mut embs);
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        if self.dim != 0 {
            self.dim
        } else {
            384
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer (pure, unit-tested; no `tokenizers` crate).
// ---------------------------------------------------------------------------

/// Loads a BERT `vocab.txt` (one token per line) into a `token -> id` map,
/// where `id` is the 0-based line index (`[PAD]`=0, `[UNK]`=100, ...).
fn load_vocab(path: &Path) -> Result<HashMap<String, i64>> {
    let bytes = std::fs::read(path)
        .map_err(|e| RagError::ModelLoad(format!("cannot read vocab {}: {e}", path.display())))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| RagError::ModelLoad(format!("vocab is not utf-8: {e}")))?;
    let mut vocab = HashMap::new();
    for (i, line) in text.lines().enumerate() {
        vocab.insert(line.to_string(), i as i64);
    }
    Ok(vocab)
}

/// Basic tokenization: clean → lowercase → whitespace split → split on
/// punctuation (each punctuation char becomes its own token).
fn basic_tokenize(text: &str) -> Vec<String> {
    let lowered = clean_text(text).to_lowercase();
    let mut tokens = Vec::new();
    for ws_tok in lowered.split_whitespace() {
        let mut cur = String::new();
        for ch in ws_tok.chars() {
            if is_punctuation(ch) {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
                tokens.push(ch.to_string());
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            tokens.push(cur);
        }
    }
    tokens
}

/// Tokenizes `text` into `(input_ids, attention_mask)` padded to `max_seq_len`:
/// `[CLS] <wordpieces> [SEP] [PAD]...`, mask `1` for real tokens and `0` for
/// padding. (Free function so the pure tokenization path is unit-testable
/// without an ONNX session.)
fn tokenize(text: &str, vocab: &HashMap<String, i64>, max_seq_len: usize) -> (Vec<i64>, Vec<i64>) {
    let max_content = max_seq_len.saturating_sub(3); // room for [CLS], [SEP], and ≥1 [PAD]
    let mut ids: Vec<i64> = Vec::new();
    'outer: for tok in basic_tokenize(text) {
        for id in wordpiece(&tok, vocab) {
            if ids.len() >= max_content {
                break 'outer;
            }
            ids.push(id);
        }
    }
    let mut full = Vec::with_capacity(max_seq_len);
    full.push(CLS_ID);
    full.extend_from_slice(&ids);
    full.push(SEP_ID);
    let real = full.len().min(max_seq_len);
    let mut mask = vec![0_i64; max_seq_len];
    for item in mask.iter_mut().take(real) {
        *item = 1;
    }
    full.resize(max_seq_len, PAD_ID);
    (full, mask)
}

/// WordPiece greedy longest-match: the first subtoken has no prefix, later
/// ones get `##`; a char with no match yields `[UNK]` for the whole word.
fn wordpiece(token: &str, vocab: &HashMap<String, i64>) -> Vec<i64> {
    if token.chars().count() > MAX_INPUT_CHARS_PER_WORD {
        return vec![UNK_ID];
    }
    let chars: Vec<char> = token.chars().collect();
    let mut sub: Vec<i64> = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = chars.len();
        let mut found_id: Option<i64> = None;
        while start < end {
            let substr: String = chars[start..end].iter().collect();
            let key = if start == 0 {
                substr
            } else {
                format!("##{substr}")
            };
            if let Some(&id) = vocab.get(&key) {
                found_id = Some(id);
                break;
            }
            end -= 1;
        }
        match found_id {
            Some(id) => {
                sub.push(id);
                start = end;
            }
            None => return vec![UNK_ID],
        }
    }
    sub
}

/// ASCII punctuation (HuggingFace `_is_punctuation` ASCII branch). Non-ASCII
/// punctuation is not split — see the module docs.
fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
}

/// Removes control chars / the replacement char and maps all whitespace to a
/// single space (HuggingFace `_clean_text`).
fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_was_ws = false;
    for ch in text.chars() {
        let cp = ch as u32;
        let is_control = ch != '\t' && ch != '\n' && ch != '\r' && ch.is_control();
        if cp == 0 || cp == 0xFFFD || is_control {
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_ws {
                out.push(' ');
                last_was_ws = true;
            }
        } else {
            out.push(ch);
            last_was_ws = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Chunking + document loading.
// ---------------------------------------------------------------------------

/// In-memory RAG index: the loaded chunks, their embeddings, the embedder, and
/// the default `top_k`. Built once at startup
/// (`LlmClient::from_config`) and queried per turn
/// ([`RagStore::search`]).
pub struct RagStore {
    chunks: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    embedder: Box<dyn Embedder>,
    /// Default number of chunks to retrieve (overridable per `search` call).
    pub top_k: usize,
}

impl RagStore {
    /// Creates an empty store holding `embedder` and the default `top_k`.
    pub fn new(embedder: Box<dyn Embedder>, top_k: usize) -> Self {
        Self {
            chunks: Vec::new(),
            embeddings: Vec::new(),
            embedder,
            top_k,
        }
    }

    /// Loads the documents in `dir` ([`load_documents`]), embeds every chunk,
    /// and returns the ready store.
    pub fn build(dir: &Path, embedder: Box<dyn Embedder>, top_k: usize) -> Result<Self> {
        let chunks = load_documents(dir)?;
        let mut store = Self::new(embedder, top_k);
        if chunks.is_empty() {
            tracing::warn!(
                dir = %dir.display(),
                "RAG: no .txt/.md documents found; retrieval disabled for this store"
            );
            return Ok(store);
        }
        tracing::info!(chunks = chunks.len(), "RAG: embedding indexed chunks");
        let mut embeddings = store.embed_chunks(&chunks)?;
        store.chunks = chunks;
        store.embeddings.append(&mut embeddings);
        Ok(store)
    }

    /// Batch-embeds `chunks`, returning one L2-normalized vector per chunk.
    pub fn embed_chunks(&mut self, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        self.embedder.embed(&refs)
    }

    /// Embeds `query` and returns the `top_k` chunks with the highest cosine
    /// similarity, most-similar first. Returns an empty vec (with a logged
    /// warning) if the index is empty or the query embedding fails — RAG must
    /// never break the LLM turn.
    pub fn search(&mut self, query: &str, top_k: usize) -> Vec<String> {
        if self.chunks.is_empty() {
            return Vec::new();
        }
        let query_emb = match self.embedder.embed(&[query]) {
            Ok(mut e) => e.pop().unwrap_or_default(),
            Err(e) => {
                tracing::warn!(error = %e, "RAG query embedding failed; skipping retrieval");
                return Vec::new();
            }
        };
        if query_emb.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine(&query_emb, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(i, _)| self.chunks[i].clone())
            .collect()
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the index has no chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Walks `dir` recursively, reads every `.txt`/`.md` file (UTF-8), and chunks
/// each by paragraph (≤ [`MAX_CHUNK_CHARS`] chars; long paragraphs are packed
/// word-by-word, oversized words hard-split). Files are visited in sorted
/// order for deterministic indexing.
pub fn load_documents(dir: &Path) -> Result<Vec<String>> {
    let mut paths = collect_text_files(dir)?;
    paths.sort();
    let mut chunks = Vec::new();
    for path in paths {
        match std::fs::read_to_string(&path) {
            Ok(text) => chunks.extend(chunk_text(&text)),
            Err(e) => tracing::warn!(
                path = %path.display(),
                error = %e,
                "RAG: skipping unreadable document"
            ),
        }
    }
    Ok(chunks)
}

/// Recursively collects `.txt`/`.md` file paths under `dir`.
fn collect_text_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    Ok(out)
}

/// Recursive directory walk (no external crate, no symlinks followed).
fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| RagError::Io(format!("cannot read {}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry.map_err(|e| RagError::Io(format!("dir entry: {e}")))?;
        let path = entry.path();
        let ft = entry
            .file_type()
            .map_err(|e| RagError::Io(format!("file type: {e}")))?;
        if ft.is_dir() {
            walk(&path, out)?;
        } else if ft.is_file() && is_text_doc(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// `true` for `.txt`/`.md` extensions (case-insensitive).
fn is_text_doc(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("txt") | Some("md")
    )
}

/// Splits `text` into paragraph chunks, then packs long paragraphs into
/// ≤ [`MAX_CHUNK_CHARS`]-char pieces.
fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    for para in split_paragraphs(text) {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.chars().count() <= MAX_CHUNK_CHARS {
            chunks.push(para.to_string());
        } else {
            chunks.extend(pack_words(para, MAX_CHUNK_CHARS));
        }
    }
    chunks
}

/// Splits on blank lines, joining wrapped lines within a paragraph with a
/// space. (`"a\nb\n\nc\nd"` → `["a b", "c d"]`.)
fn split_paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(line.trim());
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Packs whitespace-separated words into ≤ `max`-char chunks (by char count);
/// a single word longer than `max` is hard-split at `max` chars.
fn pack_words(text: &str, max: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    for word in text.split_whitespace() {
        let wlen = word.chars().count();
        let add = if cur.is_empty() { 0 } else { 1 }; // joining space
        if cur_len + add + wlen <= max {
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(word);
            cur_len += add + wlen;
        } else {
            if !cur.is_empty() {
                chunks.push(std::mem::take(&mut cur));
                cur_len = 0;
            }
            if wlen <= max {
                cur.push_str(word);
                cur_len = wlen;
            } else {
                // Hard-split an oversized word at `max` chars.
                let mut buf = String::new();
                for c in word.chars() {
                    buf.push(c);
                    if buf.chars().count() >= max {
                        chunks.push(std::mem::take(&mut buf));
                    }
                }
                if !buf.is_empty() {
                    cur = buf;
                    cur_len = cur.chars().count();
                }
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

// ---------------------------------------------------------------------------
// Vector math (cosine over pre-normalized vectors).
// ---------------------------------------------------------------------------

/// Cosine similarity of two L2-normalized vectors (their dot product). If the
/// lengths differ, only the overlapping prefix contributes.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0_f32;
    for i in 0..n {
        dot += a[i] * b[i];
    }
    dot
}

/// L2-normalizes `v` in place (no-op for a zero vector).
fn l2_normalize(v: &mut [f32]) {
    let mut norm = 0.0_f32;
    for &x in v.iter() {
        norm += x * x;
    }
    norm = norm.sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Mean-pools `last_hidden_state` (`data`, `[batch, seq, dim]`) over the
/// sequence axis, weighted by `mask`, then L2-normalizes each row.
fn mean_pool(
    data: &[f32],
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut acc = vec![0.0_f32; dim];
        let mut count = 0.0_f32;
        for s in 0..seq {
            let m = mask[b * seq + s] as f32;
            if m == 0.0 {
                continue;
            }
            count += m;
            let off = (b * seq + s) * dim;
            for d in 0..dim {
                acc[d] += data[off + d] * m;
            }
        }
        let n = if count > 0.0 { count } else { 1.0 };
        for item in acc.iter_mut().take(dim) {
            *item /= n;
        }
        l2_normalize(&mut acc);
        out.push(acc);
    }
    Ok(out)
}

/// Copies a pre-pooled `[batch, dim]` output, L2-normalizing each row.
fn copy_rows(data: &[f32], batch: usize, dim: usize) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = data[b * dim..(b + 1) * dim].to_vec();
        l2_normalize(&mut row);
        out.push(row);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure tokenizer, chunking, and vector math — no model
    //! required. End-to-end embedding/search tests live in `tests/rag.rs`.

    use super::*;

    fn vocab(pairs: &[(&str, i64)]) -> HashMap<String, i64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn clean_text_replaces_whitespace_and_drops_controls() {
        assert_eq!(clean_text("a\tb\nc\r\nd"), "a b c d");
        assert_eq!(clean_text("a\u{0000}b"), "ab");
        assert_eq!(clean_text("a\u{FFFD}b"), "ab");
        // Non-breaking space and ideographic space are whitespace.
        assert_eq!(clean_text("a\u{00A0}b"), "a b");
    }

    #[test]
    fn basic_tokenize_lowercases_and_splits_punctuation() {
        let toks = basic_tokenize("The CAT sat, on the mat.");
        assert_eq!(
            toks,
            vec!["the", "cat", "sat", ",", "on", "the", "mat", "."]
        );
    }

    #[test]
    fn wordpiece_greedy_longest_match_and_unk() {
        let v = vocab(&[
            ("the", 1),
            ("cat", 2),
            ("##s", 3),
            ("play", 4),
            ("##ing", 5),
            ("[UNK]", 100),
        ]);
        assert_eq!(wordpiece("the", &v), vec![1]);
        assert_eq!(wordpiece("cats", &v), vec![2, 3]); // cat + ##s
        assert_eq!(wordpiece("playing", &v), vec![4, 5]); // play + ##ing
        assert_eq!(wordpiece("zzz", &v), vec![100]); // no match -> [UNK]
                                                     // Oversized word -> [UNK].
        let big = "x".repeat(MAX_INPUT_CHARS_PER_WORD + 1);
        assert_eq!(wordpiece(&big, &v), vec![100]);
    }

    #[test]
    fn tokenize_pads_truncates_and_masks() {
        // Vocab reproduces real BERT ids for the probe sentence.
        let mut v = HashMap::new();
        for (tok, id) in [
            ("[PAD]", 0),
            ("[UNK]", 100),
            ("[CLS]", 101),
            ("[SEP]", 102),
            ("the", 1996),
            ("cat", 4937),
            ("sat", 2938),
            ("on", 2006),
            ("mat", 13523),
            (".", 1012),
        ] {
            v.insert(tok.to_string(), id);
        }
        let (ids, mask) = tokenize("The cat sat on the mat.", &v, 16);
        assert_eq!(ids.len(), 16);
        assert_eq!(mask.len(), 16);
        // [CLS] the cat sat on the mat . [SEP] then padding.
        assert_eq!(
            &ids[..9],
            &[101, 1996, 4937, 2938, 2006, 1996, 13523, 1012, 102]
        );
        assert_eq!(&mask[..9], &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(&ids[9..], &[0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&mask[9..], &[0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn tokenize_truncates_to_max_seq_len() {
        let v = vocab(&[("[PAD]", 0), ("[UNK]", 100), ("[CLS]", 101), ("[SEP]", 102)]);
        // Every word -> [UNK]; only [CLS] [UNK] [SEP] fit (max_content = 2).
        let (ids, mask) = tokenize("one two three four", &v, 4);
        assert_eq!(ids, vec![101, 100, 102, 0]);
        assert_eq!(mask, vec![1, 1, 1, 0]);
    }

    #[test]
    fn split_paragraphs_joins_wrapped_lines() {
        let ps = split_paragraphs("a\nb\n\nc\n\nd");
        assert_eq!(
            ps,
            vec!["a b".to_string(), "c".to_string(), "d".to_string()]
        );
        // Leading/trailing blank lines are ignored.
        assert_eq!(split_paragraphs("\n\nhi\n\n"), vec!["hi".to_string()]);
    }

    #[test]
    fn chunk_text_caps_at_max_chars() {
        let para = (0..600).map(|_| "word").collect::<Vec<_>>().join(" ");
        let chunks = chunk_text(&para);
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX_CHUNK_CHARS));
        assert!(chunks.len() > 1);
        // Every input word survives in exactly one chunk.
        assert_eq!(
            chunks.iter().flat_map(|c| c.split_whitespace()).count(),
            600
        );
    }

    #[test]
    fn chunk_text_hard_splits_oversized_words() {
        let big = "a".repeat(MAX_CHUNK_CHARS * 2 + 7);
        let chunks = chunk_text(&big);
        assert!(chunks.iter().all(|c| c.chars().count() <= MAX_CHUNK_CHARS));
        assert_eq!(chunks.concat().chars().count(), big.chars().count());
    }

    #[test]
    fn cosine_and_l2_normalize() {
        let mut a = vec![3.0_f32, 4.0];
        l2_normalize(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6 && (a[1] - 0.8).abs() < 1e-6);
        let mut b = vec![3.0_f32, 4.0];
        l2_normalize(&mut b);
        // Same direction -> cosine 1.0; orthogonal -> 0.0.
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        let mut o = vec![1.0_f32, 0.0];
        l2_normalize(&mut o);
        let mut p = vec![0.0_f32, 1.0];
        l2_normalize(&mut p);
        assert!(cosine(&o, &p).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_masks_padding() {
        // batch=1, seq=4, dim=2; token rows: [1,1],[2,2],[3,3]; last is pad.
        let data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 9.0, 9.0];
        let mask = vec![1_i64, 1, 1, 0];
        let out = mean_pool(&data, &mask, 1, 4, 2).unwrap();
        // mean of [1,1],[2,2],[3,3] = [2,2]; normalized -> [1/sqrt2, 1/sqrt2].
        assert_eq!(out.len(), 1);
        assert!((out[0][0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert!((out[0][1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }
}
