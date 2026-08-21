//! Conversation-memory integration tests: store/load/recall/remember round-trip
//! and per-turn summaries, against a real (temp) JSON file — no models, no
//! network, no audio.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use skadoosh::memory::MemoryStore;

/// A unique temp path per test (process id + monotonic counter) so parallel
/// test runs never collide. The returned [`TempFile`] removes it on drop.
fn temp_path(label: &str) -> (PathBuf, TempFile) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "skadoosh-memory-{}-{label}-{n}.json",
        std::process::id()
    ));
    (p.clone(), TempFile(p))
}

/// Removes the path on drop (best-effort).
struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Opening a path with no existing file starts empty (first-run behavior).
#[test]
fn open_missing_file_starts_empty() {
    let (path, _guard) = temp_path("missing");
    let store = MemoryStore::open(&path);
    assert!(store.recall("anything").is_none());
    assert!(store.preferences_summary().is_none());
    assert!(store.summaries().is_empty());
}

/// `remember` then `recall` round-trips in-process, and the file written by
/// `remember` is auto-loaded on a fresh `open` (persistence across runs).
#[test]
fn remember_and_recall_round_trip() {
    let (path, _guard) = temp_path("roundtrip");
    {
        let mut store = MemoryStore::open(&path);
        assert!(store.recall("name").is_none(), "nothing remembered yet");
        store.remember("name", "Alice");
        store.remember("city", "Paris");
        assert_eq!(store.recall("name").as_deref(), Some("Alice"));
        assert_eq!(store.recall("city").as_deref(), Some("Paris"));
        assert!(store.recall("unknown").is_none());
    }
    // Re-open: auto-loaded from the JSON file written by remember().
    let store = MemoryStore::open(&path);
    assert_eq!(
        store.recall("name").as_deref(),
        Some("Alice"),
        "persisted across open"
    );
    assert_eq!(store.recall("city").as_deref(), Some("Paris"));
}

/// Re-`remember`ing a key overwrites the prior value.
#[test]
fn remember_overwrites_existing_key() {
    let (path, _guard) = temp_path("overwrite");
    let mut store = MemoryStore::open(&path);
    store.remember("color", "red");
    store.remember("color", "blue");
    assert_eq!(store.recall("color").as_deref(), Some("blue"));
}

/// `preferences_summary` is `None` when empty and formats known pairs when set.
#[test]
fn preferences_summary_formats_known_pairs() {
    let (path, _guard) = temp_path("prefs");
    let mut store = MemoryStore::open(&path);
    assert!(store.preferences_summary().is_none(), "empty → no summary");
    store.remember("name", "Alice");
    store.remember("city", "Paris");
    let summary = store.preferences_summary().expect("non-empty");
    assert!(summary.contains("name: Alice"), "{summary}");
    assert!(summary.contains("city: Paris"), "{summary}");
}

/// `summarize_turn` appends a `User: … | Assistant: …` line and persists it.
#[test]
fn summarize_turn_appends_and_persists() {
    let (path, _guard) = temp_path("summaries");
    {
        let mut store = MemoryStore::open(&path);
        store.summarize_turn("hi there", "hello!");
        store.summarize_turn("how are you?", "great");
        let summaries = store.summaries();
        assert_eq!(summaries.len(), 2);
        assert!(summaries[0].contains("hi there"), "{summaries:?}");
        assert!(summaries[0].contains("hello!"), "{summaries:?}");
        assert!(summaries[1].contains("how are you?"), "{summaries:?}");
        assert!(summaries[1].contains("great"), "{summaries:?}");
    }
    let store = MemoryStore::open(&path);
    assert_eq!(
        store.summaries().len(),
        2,
        "summaries persisted across open"
    );
}

/// The rolling summary list is capped (oldest entries evicted).
#[test]
fn summarize_turn_caps_rolling_list() {
    let (path, _guard) = temp_path("cap");
    let mut store = MemoryStore::open(&path);
    for i in 0..15 {
        store.summarize_turn(&format!("u{i}"), &format!("a{i}"));
    }
    let summaries = store.summaries();
    assert_eq!(summaries.len(), 10, "rolling list capped at 10");
    // Entries 0..4 were evicted; the oldest kept is u5, newest is u14.
    assert!(
        summaries[0].contains("u5"),
        "oldest kept: {:?}",
        summaries[0]
    );
    assert!(
        summaries[9].contains("u14"),
        "newest kept: {:?}",
        summaries[9]
    );
}

/// A malformed memory file does not panic: the store starts empty, and a
/// later `remember` overwrites the junk with valid JSON.
#[test]
fn malformed_file_starts_empty_without_panicking() {
    let (path, _guard) = temp_path("malformed");
    std::fs::write(&path, b"{ this is not valid json").expect("write junk");
    let mut store = MemoryStore::open(&path);
    assert!(store.recall("x").is_none());
    assert!(store.summaries().is_empty());
    store.remember("k", "v");
    let store = MemoryStore::open(&path);
    assert_eq!(
        store.recall("k").as_deref(),
        Some("v"),
        "recovered after overwrite"
    );
}
