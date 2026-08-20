//! `skadoosh` binary: parse CLI → init tracing → dispatch.
//!
//! Exit codes: `0` on a clean run (including a clean SIGINT shutdown), `1`
//! on any error (the error chain is printed to stderr).
//!
//! # SIGINT handling
//!
//! The library exposes [`Pipeline::shutdown_token`] as the shutdown
//! injection point; this binary bridges SIGINT onto it with a dedicated
//! thread running a current-thread tokio runtime that awaits
//! `tokio::signal::ctrl_c()`. A second SIGINT force-exits with the
//! conventional 128+SIGINT status, so a wedged shutdown can always be
//! killed.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use skadoosh::audio::input::list_devices;
use skadoosh::{Agent, AgentEvent, Config, OutputMode, Pipeline, SkadooshError};
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    let config = Config::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match dispatch(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            print_error_chain(&err);
            ExitCode::FAILURE
        }
    }
}

fn dispatch(config: Config) -> skadoosh::Result<()> {
    config.validate()?;

    if config.list_devices {
        for name in list_devices()? {
            println!("{name}");
        }
        return Ok(());
    }

    if let Some(wav) = config.selftest.clone() {
        let pipeline = Pipeline::new(config)?;
        let report = pipeline.run_selftest(&wav, Path::new("selftest_out.wav"))?;
        println!("{report}");
        return Ok(());
    }

    let mut agent = Agent::builder().config(config.clone()).build()?;

    if config.repl {
        // Interactive text↔text loop: no audio, no models. (`StdinLock`/
        // `StdoutLock` are !Send and `Stdin` is not `BufRead`, hence the
        // unlocked, buffered handles.)
        return agent.repl(std::io::BufReader::new(std::io::stdin()), std::io::stdout());
    }

    if let Some(text) = &config.say {
        return match &config.out_wav {
            // No audio device needed.
            Some(path) => agent.say_to_wav(text, path),
            None => agent.say(text),
        };
    }

    // The voice loop. In --output text mode, print transcripts and streamed
    // reply clauses instead of playing audio.
    if config.output == OutputMode::Text {
        let events = agent.events();
        let out: Box<dyn Write + Send> = Box::new(std::io::stdout());
        std::thread::spawn(move || print_text_mode(events, out));
    }

    // Bridge SIGINT onto the shutdown token so a clean ctrlc exits 0.
    let token = agent.shutdown_token();
    let bridge = sigint::install(token.clone());
    let result = agent.run();
    // Signal the bridge that the pipeline is done (so its thread exits
    // instead of waiting for a second SIGINT), and cover the case where
    // `run` returned before touching the token (e.g. AudioError::NoDevice).
    bridge.done();
    token.cancel();
    if let Some(handle) = bridge.join {
        let _ = handle.join();
    }
    result
}

/// `--output text` printer: `you: <transcript>`, then the reply's clauses
/// as they stream on one `bot: ...` line per turn.
fn print_text_mode(
    mut events: tokio::sync::broadcast::Receiver<AgentEvent>,
    mut out: Box<dyn Write + Send>,
) {
    let mut mid_reply = false;
    loop {
        match events.blocking_recv() {
            Ok(AgentEvent::Transcript(text)) => {
                if mid_reply {
                    let _ = writeln!(out);
                    mid_reply = false;
                }
                if writeln!(out, "you: {}", text.trim()).is_err() {
                    return;
                }
            }
            Ok(AgentEvent::Clause(clause)) => {
                if mid_reply {
                    let _ = write!(out, " ");
                } else {
                    let _ = write!(out, "bot: ");
                    mid_reply = true;
                }
                if write!(out, "{}", clause.trim()).is_err() || out.flush().is_err() {
                    return;
                }
            }
            Ok(AgentEvent::ReplyDone) => {
                if mid_reply {
                    mid_reply = false;
                    if writeln!(out).is_err() {
                        return;
                    }
                }
            }
            Ok(AgentEvent::TurnCancelled) => {
                if mid_reply {
                    mid_reply = false;
                    if writeln!(out).is_err() {
                        return;
                    }
                }
                if writeln!(out, "  [interrupted]").is_err() {
                    return;
                }
            }
            Ok(AgentEvent::ToolCall { name, arguments }) => {
                if writeln!(out, "  [tool: {name}({arguments})]").is_err() {
                    return;
                }
            }
            Ok(AgentEvent::Error(err)) => {
                if writeln!(out, "error: {err}").is_err() {
                    return;
                }
            }
            Ok(_) => {}
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                let _ = writeln!(out, "  [... {n} events dropped ...]");
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
        }
    }
}

/// Prints the error and its sources, one `caused by:` line per level.
fn print_error_chain(err: &SkadooshError) {
    eprintln!("error: {err}");
    let mut source = std::error::Error::source(err);
    while let Some(err) = source {
        eprintln!("caused by: {err}");
        source = err.source();
    }
}

/// SIGINT → shutdown-token bridge (see the module-level note). Spawns a
/// thread with a small current-thread runtime so the safe
/// `tokio::signal::ctrl_c()` future can drive the token; no unsafe code.
mod sigint {
    use std::thread::JoinHandle;

    use tokio_util::sync::CancellationToken;

    /// Handle to the bridge thread. `done()` tells the thread the pipeline
    /// has exited (so it stops waiting for SIGINT and returns); `join`
    /// reaps it.
    pub struct SigintBridge {
        /// Cancelled by `done()` when the pipeline run is over.
        done: CancellationToken,
        /// The bridge thread's join handle.
        pub join: Option<JoinHandle<()>>,
    }

    impl SigintBridge {
        /// Signals the bridge thread to exit. Without this the thread would
        /// keep waiting for a second SIGINT after the pipeline finished,
        /// hanging a `join()` — and a force-quit after a *clean* shutdown
        /// would wrongly exit 130.
        pub fn done(&self) {
            self.done.cancel();
        }
    }

    /// Spawns the bridge thread, which cancels `token` on the first SIGINT.
    /// Afterwards it races a second SIGINT (force-exit 130 = 128+SIGINT,
    /// for a wedged shutdown) against `done` (clean pipeline exit). The
    /// thread also exits when `token` is cancelled for any other reason
    /// (e.g. a fatal-error shutdown), so it never outlives the pipeline.
    /// Returns a bridge with `join: None` if the thread could not be
    /// spawned (degraded: SIGINT then kills the process with the default
    /// disposition).
    pub fn install(token: CancellationToken) -> SigintBridge {
        let done = CancellationToken::new();
        let thread_done = done.clone();
        let join = std::thread::Builder::new()
            .name("skadoosh-sigint".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(err) => {
                        tracing::warn!(%err, "failed to build SIGINT runtime; ctrlc will kill the process");
                        return;
                    }
                };
                runtime.block_on(async move {
                    tokio::select! {
                        first = tokio::signal::ctrl_c() => {
                            if first.is_err() {
                                tracing::warn!("failed to listen for SIGINT; ctrlc will kill the process");
                                return;
                            }
                            tracing::info!("SIGINT received; shutting down (press ctrl-c again to force)");
                            token.cancel();
                            tokio::select! {
                                // A second SIGINT force-exits a wedged shutdown.
                                _ = tokio::signal::ctrl_c() => std::process::exit(128 + 2),
                                // Pipeline finished cleanly — no force-quit watch needed.
                                _ = thread_done.cancelled() => {}
                            }
                        }
                        _ = token.cancelled() => {}
                        _ = thread_done.cancelled() => {}
                    }
                });
            })
            .map_err(|err| {
                tracing::warn!(%err, "failed to spawn SIGINT bridge thread");
                err
            })
            .ok();
        SigintBridge { done, join }
    }
}
