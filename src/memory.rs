//! Conversation memory: a JSON-file-backed store of user preferences and
//! recent conversation summaries.
//!
//! [`MemoryStore`] is intentionally simple — one JSON file (e.g.
//! `~/.skadoosh/memory.json`, or the path from `--memory-file`), no SQLite.
//! It auto-loads existing contents on [`open`](MemoryStore::open) and
//! auto-saves after every mutation, so callers never manage persistence
//! themselves.
//!
//! It is wired in by `LlmClient::from_config` when
//! `--memory-file` is set: remembered preferences are injected into the
//! system prompt so the agent recalls them across runs, and each completed
//! turn is summarized back into the store.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Maximum number of recent conversation summaries retained on disk.
const MAX_SUMMARIES: usize = 10;
/// Rough per-turn summary cap (characters), keeping the file small.
const SUMMARY_MAX_CHARS: usize = 280;

/// On-disk JSON shape of [`MemoryStore`]. New fields default in via
/// `#[serde(default)]` so an older file (or one missing a field) loads
/// cleanly instead of erroring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct MemoryData {
    /// Key→value preferences the user has stated ([`MemoryStore::remember`]).
    #[serde(default)]
    preferences: BTreeMap<String, String>,
    /// Recent conversation summaries, oldest first
    /// ([`MemoryStore::summarize_turn`]).
    #[serde(default)]
    summaries: VecDeque<String>,
}

/// A simple JSON-file-backed conversation memory.
///
/// Stores two things:
/// * **preferences** — arbitrary key/value pairs the user has expressed
///   ([`remember`](Self::remember) / [`recall`](Self::recall)), injected into
///   the LLM system prompt so the agent remembers them across runs; and
/// * **summaries** — a rolling list of recent turn summaries
///   ([`summarize_turn`](Self::summarize_turn)), capped at
///   `MAX_SUMMARIES` entries.
///
/// Persistence is automatic: [`open`](Self::open) loads any existing file,
/// and every mutator writes it back. All I/O is best-effort — a read failure
/// yields an empty store (with a warning), and a write failure is warned but
/// never returned, so a bad memory path can never break the voice loop.
pub struct MemoryStore {
    path: PathBuf,
    data: MemoryData,
}

impl MemoryStore {
    /// Opens (or creates) the store at `path`, loading any existing
    /// contents. A missing file is normal (first run); an unreadable or
    /// malformed file yields an empty store with a warning rather than an
    /// error — memory is best-effort, never a hard dependency.
    pub fn open(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let data = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<MemoryData>(&bytes).unwrap_or_else(|err| {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "memory file unreadable; starting with empty memory"
                );
                MemoryData::default()
            }),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => MemoryData::default(),
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "memory file unreadable; starting with empty memory"
                );
                MemoryData::default()
            }
        };
        Self { path, data }
    }

    /// Remembers a preference (`key` → `value`), overwriting any prior value
    /// for `key`, then saves.
    pub fn remember(&mut self, key: &str, value: &str) {
        self.data
            .preferences
            .insert(key.to_string(), value.to_string());
        self.save();
    }

    /// Recalls a previously remembered preference, or `None` if `key` was
    /// never stored.
    pub fn recall(&self, key: &str) -> Option<String> {
        self.data.preferences.get(key).cloned()
    }

    /// Searches preferences for `query` in both keys and values
    /// (case-insensitive substring match). Returns matching `(key, value)`
    /// pairs.
    pub fn search_preferences(&self, query: &str) -> Vec<(String, String)> {
        let q = query.to_lowercase();
        self.data
            .preferences
            .iter()
            .filter(|(k, v)| k.to_lowercase().contains(&q) || v.to_lowercase().contains(&q))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Returns the number of stored preferences.
    pub fn preference_count(&self) -> usize {
        self.data.preferences.len()
    }

    /// Records a one-line summary of a completed turn (`user_text` in,
    /// `agent_reply` out), appends it to the rolling list, trims to the last
    /// `MAX_SUMMARIES` entries, then saves.
    pub fn summarize_turn(&mut self, user_text: &str, agent_reply: &str) {
        let summary = format_turn_summary(user_text, agent_reply);
        self.data.summaries.push_back(summary);
        while self.data.summaries.len() > MAX_SUMMARIES {
            self.data.summaries.pop_front();
        }
        self.save();
    }

    /// Formats all remembered preferences as `key: value; key: value` for
    /// system-prompt injection, or `None` when there are none.
    pub fn preferences_summary(&self) -> Option<String> {
        if self.data.preferences.is_empty() {
            return None;
        }
        Some(
            self.data
                .preferences
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join("; "),
        )
    }

    /// Returns the recent conversation summaries, oldest first (cloned for
    /// inspection/tests).
    pub fn summaries(&self) -> Vec<String> {
        self.data.summaries.iter().cloned().collect()
    }

    /// The on-disk path this store reads and writes.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes the store to its JSON file, creating parent directories as
    /// needed. Best-effort: errors are warned, not returned, so a bad path
    /// never fails a turn.
    fn save(&self) {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    tracing::warn!(
                        path = %self.path.display(),
                        error = %err,
                        "memory: could not create directory"
                    );
                    return;
                }
            }
        }
        match serde_json::to_vec_pretty(&self.data) {
            Ok(bytes) => {
                if let Err(err) = fs::write(&self.path, bytes) {
                    tracing::warn!(
                        path = %self.path.display(),
                        error = %err,
                        "memory: save failed"
                    );
                }
            }
            Err(err) => tracing::warn!(error = %err, "memory: serialize failed"),
        }
    }
}

/// Builds a compact `User: … | Assistant: …` summary, truncating each side
/// so the rolling list stays small.
fn format_turn_summary(user_text: &str, agent_reply: &str) -> String {
    let half = SUMMARY_MAX_CHARS / 2;
    let user = truncate(user_text.trim(), half);
    let agent = truncate(agent_reply.trim(), half);
    format!("User: {user} | Assistant: {agent}")
}

/// Truncates `s` to at most `max` bytes, backing up to a UTF-8 char boundary.
fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut idx = max;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    &s[..idx]
}
