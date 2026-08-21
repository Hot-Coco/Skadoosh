//! Integration checks for tool execution: parallel and streaming.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use skadoosh::tools::{execute_parallel, ShellExecutor};

/// The shell tool name. `ShellExecutor` feeds the script through stdin, not
/// as command-line arguments, so the script body is the `arguments` string.
const TOOL_SH: &str = "sh";

#[tokio::test]
async fn execute_parallel_runs_concurrently_and_returns_results_by_call_id() {
    // Three short sleeps. If run serially they would take ~300 ms; in
    // parallel they should finish in ~100 ms plus spawn overhead.
    let calls = vec![
        (
            TOOL_SH.to_string(),
            "sleep 0.1; echo one".to_string(),
            "call_1".to_string(),
        ),
        (
            TOOL_SH.to_string(),
            "sleep 0.1; echo two".to_string(),
            "call_2".to_string(),
        ),
        (
            TOOL_SH.to_string(),
            "sleep 0.1; echo three".to_string(),
            "call_3".to_string(),
        ),
    ];

    let started = std::time::Instant::now();
    let results = execute_parallel(calls).await;
    let elapsed = started.elapsed();

    assert_eq!(results.len(), 3, "one result per call id must be returned");

    let ids: BTreeSet<_> = results.keys().cloned().collect();
    assert_eq!(
        ids,
        BTreeSet::from([
            "call_1".to_string(),
            "call_2".to_string(),
            "call_3".to_string()
        ])
    );

    for (id, expected) in [
        ("call_1", "one\n"),
        ("call_2", "two\n"),
        ("call_3", "three\n"),
    ] {
        let result = results
            .get(id)
            .unwrap_or_else(|| panic!("missing result for {id}"))
            .as_ref()
            .unwrap_or_else(|e| panic!("{id} failed: {e}"));
        assert_eq!(result.trim(), expected.trim(), "wrong output for {id}");
    }

    // Parallelism heuristic: three 100 ms sleeps should complete in well
    // under 500 ms (serial would be ~300 ms plus three shell startups, so
    // this is loose enough for a shared sandbox while still rejecting truly
    // serial execution).
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "parallel execution was too slow: {elapsed:?}"
    );
}

#[tokio::test]
async fn execute_parallel_preserves_error_results() {
    let calls = vec![
        ("sh".to_string(), "exit 0".to_string(), "ok".to_string()),
        ("sh".to_string(), "exit 7".to_string(), "err".to_string()),
    ];

    let results = execute_parallel(calls).await;
    assert!(results.get("ok").unwrap().is_ok(), "ok call must succeed");
    assert!(results.get("err").unwrap().is_err(), "err call must fail");
}

#[tokio::test]
async fn execute_streaming_calls_back_per_line() {
    let lines = Arc::new(Mutex::new(Vec::new()));
    let lines_clone = Arc::clone(&lines);

    let (output, _duration) =
        ShellExecutor::execute_streaming("sh", "printf 'line1\nline2\nline3\n'", move |line| {
            lines_clone.lock().unwrap().push(line);
        })
        .await
        .expect("streaming execution should succeed");

    let captured = lines.lock().unwrap();
    assert_eq!(
        captured.as_slice(),
        &["line1", "line2", "line3"],
        "callback must receive each stdout line exactly once"
    );
    assert!(
        output.contains("line1\nline2\nline3\n"),
        "full output must contain all lines"
    );
}

#[tokio::test]
async fn execute_streaming_callback_order_is_line_order() {
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = Arc::clone(&counter);

    ShellExecutor::execute_streaming("sh", "for i in 1 2 3 4 5; do echo $i; done", move |line| {
        let idx = counter_clone.fetch_add(1, Ordering::SeqCst);
        let expected = (idx + 1).to_string();
        assert_eq!(
            line, expected,
            "line callback order mismatch at index {idx}"
        );
    })
    .await
    .expect("streaming execution should succeed");
}
