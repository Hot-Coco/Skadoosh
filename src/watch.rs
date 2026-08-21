//! Proactive triggers (`--watch`): background file/process/timer watchers
//! that inject notifications into the conversation as if the user said them.
//!
//! [`WatchManager::start`] spawns one tokio task per configured trigger. Each
//! task emits [`WatchEvent`]s through an [`mpsc`] channel; the pipeline
//! orchestrator drains that channel and turns each event into a
//! system-injected user turn (see `pipeline`).
//!
//! * **File watcher** (`--watch-file`): the [`notify`] crate watches the
//!   file's parent directory (non-recursive) and filters events to the target
//!   path, so edits, atomic saves, and (re)creation are all caught. Emits
//!   [`WatchEvent::FileChanged`].
//! * **Process watcher** (`--watch-process`): polls `/proc/<pid>` (Linux) and
//!   emits [`WatchEvent::ProcessExited`] when the entry vanishes.
//! * **Timer** (`--watch-timer`): sleeps for N seconds, then emits
//!   [`WatchEvent::TimerElapsed`].

use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::error::{Result, SkadooshError};

/// Capacity of the watch-event channel (watchers → orchestrator).
const WATCH_CAP: usize = 16;
/// Process-exit poll interval.
const PROCESS_POLL: Duration = Duration::from_millis(200);

/// A proactive trigger event, surfaced to the pipeline as a user turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEvent {
    /// A watched file was modified (`--watch-file`). Carries the watched path.
    FileChanged(PathBuf),
    /// A watched process exited (`--watch-process`). Carries the PID.
    ProcessExited(u32),
    /// A timer elapsed (`--watch-timer`). Carries the original seconds.
    TimerElapsed(String),
}

impl WatchEvent {
    /// Renders the event as the user-facing notification text (without the
    /// `NOTIFICATION:` prefix the pipeline adds).
    pub fn message(&self) -> String {
        match self {
            WatchEvent::FileChanged(path) => {
                format!("The file {} has changed.", path.display())
            }
            WatchEvent::ProcessExited(pid) => format!("Process {pid} has exited."),
            WatchEvent::TimerElapsed(secs) => format!("Your {secs}-second timer is up."),
        }
    }
}

/// Configuration for the proactive triggers, mirroring the `--watch-*` flags.
#[derive(Debug, Clone, Default)]
pub struct WatchConfig {
    /// Files to watch for changes (`--watch-file` / `SKADOOSH_WATCH_FILE`).
    pub files: Vec<PathBuf>,
    /// Process IDs to watch for exit (`--watch-process` / `SKADOOSH_WATCH_PROCESS`).
    pub processes: Vec<u32>,
    /// Timer durations in seconds (`--watch-timer` / `SKADOOSH_WATCH_TIMER`).
    pub timers: Vec<u64>,
}

impl WatchConfig {
    /// Whether any trigger is configured.
    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.processes.is_empty() && self.timers.is_empty()
    }
}

/// Manages proactive-trigger background tasks.
///
/// [`WatchManager::start`] spawns a tokio task per configured trigger that
/// emits [`WatchEvent`]s through the returned [`mpsc::Receiver`]. The channel
/// closes (receivers yield `None`) once every watcher task has finished, so the
/// orchestrator can treat closure as "no more watch events". Call
/// [`WatchManager::shutdown`] (or drop) to cancel outstanding watchers.
pub struct WatchManager {
    tasks: JoinSet<()>,
    shutdown: CancellationToken,
}

impl WatchManager {
    /// Spawns a watcher task for every configured trigger and returns the
    /// manager plus the receiver to drain [`WatchEvent`]s from.
    ///
    /// With an empty [`WatchConfig`] no tasks are spawned and the returned
    /// receiver is immediately closed.
    pub fn start(
        config: &WatchConfig,
        shutdown: CancellationToken,
    ) -> (Self, mpsc::Receiver<WatchEvent>) {
        let (tx, rx) = mpsc::channel(WATCH_CAP);
        let mut tasks = JoinSet::new();

        for path in &config.files {
            let tx = tx.clone();
            let path = path.clone();
            let shutdown = shutdown.clone();
            tasks.spawn(async move {
                if let Err(err) = watch_file(path, tx, shutdown).await {
                    warn!(error = %err, "file watcher exited with error");
                }
            });
        }
        for &pid in &config.processes {
            let tx = tx.clone();
            let shutdown = shutdown.clone();
            tasks.spawn(async move {
                watch_process(pid, tx, shutdown).await;
            });
        }
        for &secs in &config.timers {
            let tx = tx.clone();
            let shutdown = shutdown.clone();
            let label = secs.to_string();
            tasks.spawn(async move {
                watch_timer(secs, label, tx, shutdown).await;
            });
        }

        // Drop the original sender so the channel closes once every watcher
        // task finishes (the orchestrator treats closure as end-of-watch).
        drop(tx);

        (Self { tasks, shutdown }, rx)
    }

    /// Cancels all watchers and joins them.
    pub async fn shutdown(mut self) {
        self.shutdown.cancel();
        while let Some(res) = self.tasks.join_next().await {
            if let Err(err) = res {
                warn!(error = %err, "watch task panicked");
            }
        }
    }
}

impl Drop for WatchManager {
    fn drop(&mut self) {
        // Best-effort: cancel so background tasks exit even if `shutdown`
        // wasn't awaited (the runtime aborts them on drop otherwise).
        self.shutdown.cancel();
    }
}

/// File watcher (`--watch-file`): watches the file's parent directory and
/// filters events to the target path, emitting [`WatchEvent::FileChanged`] on
/// any modify/create/remove. Watching the directory (rather than the inode)
/// survives editor atomic saves that replace the file.
async fn watch_file(
    path: PathBuf,
    tx: mpsc::Sender<WatchEvent>,
    shutdown: CancellationToken,
) -> Result<()> {
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

    // notify's callback runs on an internal thread; bridge to async via an
    // unbounded channel (sync `send`, never blocks that thread).
    let (ev_tx, mut ev_rx) =
        mpsc::unbounded_channel::<std::result::Result<notify::Event, notify::Error>>();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            let _ = ev_tx.send(res);
        },
        notify::Config::default(),
    )
    .map_err(|e| SkadooshError::Other(anyhow::anyhow!("file watcher init: {e}")))?;

    // Watch the parent directory when there is one (robust to file
    // replacement); otherwise watch the path itself.
    let parent = path.parent();
    let (watch_target, filter): (PathBuf, bool) = match parent {
        Some(p) if !p.as_os_str().is_empty() => (p.to_path_buf(), true),
        _ => (path.clone(), false),
    };
    watcher
        .watch(&watch_target, RecursiveMode::NonRecursive)
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("file watch start: {e}")))?;
    info!(path = %path.display(), "watching file for changes");

    loop {
        let res = tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            res = ev_rx.recv() => match res {
                Some(res) => res,
                None => break, // watcher dropped / channel closed
            },
        };
        let event = match res {
            Ok(event) => event,
            Err(err) => {
                warn!(path = %path.display(), error = %err, "file watcher error");
                continue;
            }
        };
        // React to content/metadata writes and life-cycle events, not access.
        let is_change = matches!(
            event.kind,
            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
        );
        if !is_change {
            continue;
        }
        // When watching the parent dir, only fire for our file.
        if filter && !event.paths.iter().any(|p| p.as_path() == path.as_path()) {
            continue;
        }
        let _ = tx.send(WatchEvent::FileChanged(path.clone())).await;
    }
    Ok(())
}

/// Process watcher (`--watch-process`): polls `/proc/<pid>` (Linux) and emits
/// [`WatchEvent::ProcessExited`] when the entry disappears. If the process is
/// not present at start (already exited, or not on Linux), nothing is emitted.
async fn watch_process(pid: u32, tx: mpsc::Sender<WatchEvent>, shutdown: CancellationToken) {
    let proc_path = PathBuf::from(format!("/proc/{pid}"));
    if !proc_path.exists() {
        warn!(pid, "watched process not found in /proc; not watching");
        return;
    }
    info!(pid, "watching process for exit");
    loop {
        // Check before sleeping so a fast exit is caught promptly.
        if !proc_path.exists() {
            info!(pid, "watched process exited");
            let _ = tx.send(WatchEvent::ProcessExited(pid)).await;
            return;
        }
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => (),
            _ = tokio::time::sleep(PROCESS_POLL) => {}
        }
    }
}

/// Timer (`--watch-timer`): fires after `secs`, emitting
/// [`WatchEvent::TimerElapsed`] with the seconds as its label.
async fn watch_timer(
    secs: u64,
    label: String,
    tx: mpsc::Sender<WatchEvent>,
    shutdown: CancellationToken,
) {
    info!(secs, "watch timer armed");
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => (),
        _ = tokio::time::sleep(Duration::from_secs(secs)) => {
            let _ = tx.send(WatchEvent::TimerElapsed(label)).await;
        }
    }
}
