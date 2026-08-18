//! Clause boundary detection for streaming TTS (pure, unit-tested).

/// Splits an LLM token stream into speakable clauses.
///
/// Emits a clause on `.` `?` `!` `,` once the buffer is at least `min_len`;
/// hard-flushes at `max_len` at the last whitespace (or a hard cut when none
/// exists). Char-boundary safe: iterates `char_indices`, never slices
/// mid-UTF-8.
#[allow(dead_code)] // buffer consumed by the task-4.1 implementation
pub struct ClauseSplitter {
    min_len: usize,
    max_len: usize,
    buf: String,
}

impl ClauseSplitter {
    /// Creates a splitter; typical values are `min_len ≈ 4`, `max_len ≈ 160`.
    pub fn new(min_len: usize, max_len: usize) -> Self {
        Self {
            min_len,
            max_len,
            buf: String::new(),
        }
    }

    /// Pushes a text chunk; returns any completed clauses, in order.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        let _ = text;
        todo!("task 4.1: boundary chars + max-len flush, UTF-8 safe")
    }

    /// Drains the remainder at stream end; `None` when the buffer is empty.
    pub fn flush(&mut self) -> Option<String> {
        todo!("task 4.1: drain remainder exactly once")
    }
}
