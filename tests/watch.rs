//! Proactive-trigger (`--watch`) tests: the [`WatchManager`] emits the correct
//! [`WatchEvent`] variant for each trigger type, timers fire, and an empty
//! config emits nothing.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use skadoosh::watch::{WatchConfig, WatchEvent, WatchManager};
use tokio_util::sync::CancellationToken;

/// A unique temp-file path scoped to this process + a per-call counter, so
/// parallel test invocations never collide.
fn temp_file(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "skadoosh-watch-test-{}-{}-{}.txt",
        std::process::id(),
        label,
        n,
    ));
    p
}

/// `WatchEvent::message` renders the notification text for each variant.
#[test]
fn watch_event_renders_notification_text() {
    let file = WatchEvent::FileChanged(PathBuf::from("Cargo.toml"));
    assert_eq!(file.message(), "The file Cargo.toml has changed.");

    let proc_ = WatchEvent::ProcessExited(4242);
    assert_eq!(proc_.message(), "Process 4242 has exited.");

    let timer = WatchEvent::TimerElapsed("30".to_string());
    assert_eq!(timer.message(), "Your 30-second timer is up.");
}

/// The timer fires exactly once after N seconds, carrying the seconds label.
#[tokio::test]
async fn timer_fires_timer_elapsed_event() {
    let shutdown = CancellationToken::new();
    let cfg = WatchConfig {
        files: vec![],
        processes: vec![],
        timers: vec![1],
    };
    let (manager, mut rx) = WatchManager::start(&cfg, shutdown.clone());

    let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .expect("timer should fire within 3s")
        .expect("watch channel closed before the timer fired");
    assert_eq!(ev, WatchEvent::TimerElapsed("1".to_string()));
    assert_eq!(ev.message(), "Your 1-second timer is up.");

    // A one-shot timer fires once, then the channel closes.
    let closed = tokio::time::timeout(Duration::from_millis(200), rx.recv())
        .await
        .expect("channel should close promptly after the timer fires");
    assert!(closed.is_none(), "timer must not fire more than once");

    manager.shutdown().await;
}

/// A file watcher emits `FileChanged` (carrying the watched path) when the
/// file is modified.
#[tokio::test]
async fn file_watcher_emits_file_changed() {
    let path = temp_file("file");
    std::fs::write(&path, b"initial").unwrap();
    let shutdown = CancellationToken::new();
    let cfg = WatchConfig {
        files: vec![path.clone()],
        processes: vec![],
        timers: vec![],
    };
    let (manager, mut rx) = WatchManager::start(&cfg, shutdown.clone());

    // Give the watcher a moment to register its inotify watch.
    tokio::time::sleep(Duration::from_millis(200)).await;
    std::fs::write(&path, b"modified").unwrap();

    let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
        .await
        .expect("file change should be detected within 3s")
        .expect("watch channel closed before the file change arrived");
    assert_eq!(ev, WatchEvent::FileChanged(path.clone()));
    assert!(ev.message().contains("has changed"));

    let _ = std::fs::remove_file(&path);
    manager.shutdown().await;
}

/// The process watcher emits `ProcessExited` when a watched PID disappears
/// from `/proc`. Linux-specific; skipped where `/proc` is absent. Runs on a
/// multi-thread runtime because reaping the child (`Child::wait`) blocks, and
/// the watcher task must keep polling `/proc` concurrently on another worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_watcher_emits_process_exited() {
    if !PathBuf::from("/proc").exists() {
        eprintln!("skipping: /proc not present (non-Linux)");
        return;
    }
    // Spawn a short-lived child and grab its PID.
    let mut child = std::process::Command::new("sleep")
        .arg("2")
        .spawn()
        .expect("spawn `sleep`");
    let pid = child.id();

    let shutdown = CancellationToken::new();
    let cfg = WatchConfig {
        files: vec![],
        processes: vec![pid],
        timers: vec![],
    };
    let (manager, mut rx) = WatchManager::start(&cfg, shutdown.clone());

    // Reap the child so /proc/<pid> disappears. (Blocking — hence the
    // multi-thread runtime so the watcher polls concurrently.)
    let _ = child.wait().expect("wait for child");

    let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("process exit should be detected within 5s")
        .expect("watch channel closed before the process exit arrived");
    assert_eq!(ev, WatchEvent::ProcessExited(pid));
    assert!(ev.message().contains("has exited"));

    manager.shutdown().await;
}

/// `WatchConfig::is_empty` is true for the default config, and starting it
/// yields no events (the channel closes immediately with no watchers).
#[tokio::test]
async fn empty_config_emits_no_events() {
    let shutdown = CancellationToken::new();
    let cfg = WatchConfig::default();
    assert!(cfg.is_empty());

    let (manager, mut rx) = WatchManager::start(&cfg, shutdown.clone());
    // Either the channel closes (Ok(None)) or the wait times out (Err) —
    // neither is a `Some(_)` event.
    let res = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
    assert!(
        !matches!(res, Ok(Some(_))),
        "no events should fire for an empty config, got {res:?}",
    );

    manager.shutdown().await;
}

/// `WatchManager::shutdown` cancels a pending timer so it never fires.
#[tokio::test]
async fn shutdown_cancels_pending_timer() {
    let shutdown = CancellationToken::new();
    let cfg = WatchConfig {
        files: vec![],
        processes: vec![],
        timers: vec![30],
    };
    let (manager, mut rx) = WatchManager::start(&cfg, shutdown.clone());

    // Cancel well before the 30s timer would fire.
    manager.shutdown().await;

    let res = tokio::time::timeout(Duration::from_millis(300), rx.recv()).await;
    assert!(
        !matches!(res, Ok(Some(_))),
        "a cancelled timer must not fire, got {res:?}",
    );
}
