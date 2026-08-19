//! Clause boundary detection for streaming TTS (pure, unit-tested).

/// Splits an LLM token stream into speakable clauses.
///
/// Emits a clause on `.` `?` `!` `,` once the buffer is at least `min_len`;
/// hard-flushes at `max_len` at the last whitespace (or a hard cut when none
/// exists). Char-boundary safe: iterates `char_indices`, never slices
/// mid-UTF-8.
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
    ///
    /// Emitted clauses keep their original text verbatim (boundary
    /// punctuation included), so concatenating every clause reproduces the
    /// input except at hard `max_len` flushes, where the single whitespace
    /// the clause was broken on is consumed.
    pub fn push(&mut self, text: &str) -> Vec<String> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        // Each cut consumes at least one char, so this loop always terminates.
        while let Some((emit_end, consume_end)) = self.next_cut() {
            let drained: String = self.buf.drain(..consume_end).collect();
            debug_assert!(emit_end <= drained.len());
            let clause = &drained[..emit_end];
            if !clause.is_empty() {
                out.push(clause.to_string());
            }
        }
        out
    }

    /// Drains the remainder at stream end; `None` when the buffer is empty or
    /// holds only whitespace (a whitespace-only clause is not speakable).
    pub fn flush(&mut self) -> Option<String> {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }

    /// Finds the next cut as `(emit_end, consume_end)` byte offsets into
    /// `self.buf`: the clause is `buf[..emit_end]` and `buf[..consume_end]`
    /// is removed from the buffer (they differ only for a `max_len` flush at
    /// whitespace, where the whitespace itself is dropped).
    fn next_cut(&self) -> Option<(usize, usize)> {
        let mut char_count = 0usize;
        let mut boundary_cut: Option<usize> = None;
        // Track the last whitespace at or before the first `max_len` chars
        // for the hard flush below.
        let mut last_ws: Option<(usize, usize)> = None; // (byte_idx, len_utf8)
        for (byte_idx, ch) in self.buf.char_indices() {
            char_count += 1;
            if boundary_cut.is_none()
                && matches!(ch, '.' | '?' | '!' | ',')
                && char_count >= self.min_len
            {
                boundary_cut = Some(byte_idx + ch.len_utf8());
            }
            if char_count <= self.max_len && ch.is_whitespace() {
                last_ws = Some((byte_idx, ch.len_utf8()));
            }
        }
        // Rule 1: emit at the first boundary char whose clause is >= min_len.
        if let Some(end) = boundary_cut {
            return Some((end, end));
        }
        // Rule 2: no boundary and the buffer is over max_len — hard flush at
        // the last whitespace inside the first max_len chars (dropping that
        // whitespace), or a hard cut exactly at the max_len-th char boundary.
        if char_count > self.max_len {
            if let Some((ws_idx, ws_len)) = last_ws {
                return Some((ws_idx, ws_idx + ws_len));
            }
            let hard = self
                .buf
                .char_indices()
                .nth(self.max_len)
                .map_or(self.buf.len(), |(idx, _)| idx);
            return Some((hard, hard));
        }
        None
    }
}
