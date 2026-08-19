//! Binary-level SIGINT regression tests.
//!
//! The orchestrator tests inject `Pipeline::shutdown_token` directly and
//! never exercise the binary's SIGINT bridge — which is how the "ctrl-c
//! once hangs forever" deadlock shipped. These tests spawn the real
//! `skadoosh` binary on the ALSA `null` device (no audio hardware needed),
//! send real SIGINTs, and assert on the exit status.
//!
//! Skips (with a printed reason) when the VAD/whisper models are absent or
//! no null-like ALSA device exists on the machine.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// ALSA's null PCM describes itself as "Discard all samples (playback) or
/// generate zero samples (capture)"; some setups name it plainly "null".
const NULL_DEVICE_HINTS: [&str; 2] = ["discard all samples", "null"];

fn models_present() -> bool {
    std::path::Path::new("models/silero_vad.onnx").exists()
        && std::path::Path::new("models/ggml-tiny.en.bin").exists()
}

/// Finds an input/output device pair backed by the ALSA null PCM by asking
/// the binary itself, or `None` (test skips) when unavailable.
fn null_device_name() -> Option<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_skadoosh"))
        .arg("--list-devices")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .filter_map(|line| line.strip_prefix("input: "))
        .find(|name| {
            let lower = name.to_lowercase();
            NULL_DEVICE_HINTS.iter().any(|hint| lower.contains(hint))
        })
        .map(str::to_owned)
}

fn spawn_agent(device: &str) -> Child {
    Command::new(env!("CARGO_BIN_EXE_skadoosh"))
        .args([
            // No speech ever arrives from the null device, so the LLM is
            // never contacted — any URL works.
            "--llm-url",
            "http://127.0.0.1:9/v1",
            "--mock-tts",
            "--input-device",
            device,
            "--output-device",
            device,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn skadoosh binary")
}

fn send_sigint(child: &Child) {
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("run kill");
    assert!(status.success(), "kill -INT failed: {status}");
}

/// Waits up to `timeout` for the child to exit, returning its status.
/// On timeout the child is SIGKILLed and the test fails.
fn wait_exit(child: &mut Child, timeout: Duration, context: &str) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if start.elapsed() > timeout {
            let _ = Command::new("kill")
                .args(["-KILL", &child.id().to_string()])
                .status();
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                let _ = pipe.read_to_string(&mut stderr);
            }
            panic!(
                "{context}: process did not exit within {timeout:?} (SIGKILLed).\nstderr tail:\n{}",
                &stderr[stderr.len().saturating_sub(4000)..]
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Regression: one SIGINT must shut the pipeline down cleanly with exit
/// code 0 — the original bug parked the bridge thread waiting for a second
/// SIGINT and hung the process forever.
#[test]
fn sigint_once_exits_zero() {
    if !models_present() {
        eprintln!("skipping: models absent (run scripts/download_models.sh)");
        return;
    }
    let Some(device) = null_device_name() else {
        eprintln!("skipping: no null-like ALSA device on this machine");
        return;
    };

    let mut child = spawn_agent(&device);
    // Model load + stream startup. VAD/whisper loads take ~1–2 s here.
    std::thread::sleep(Duration::from_secs(6));
    assert!(
        child.try_wait().expect("try_wait").is_none(),
        "agent exited before SIGINT (startup failure)"
    );

    send_sigint(&child);
    let status = wait_exit(&mut child, Duration::from_secs(15), "single SIGINT");
    assert!(
        status.success(),
        "expected clean exit 0 after one SIGINT, got {status}"
    );
}

/// Two SIGINTs in quick succession exercise the force-exit path. The race
/// is inherent: if shutdown completes before the second signal lands, the
/// process exits 0; if the second signal wins, it force-exits 130. Either
/// way it must exit promptly — the assertion that matters is "no hang".
#[test]
fn sigint_twice_never_hangs() {
    if !models_present() {
        eprintln!("skipping: models absent (run scripts/download_models.sh)");
        return;
    }
    let Some(device) = null_device_name() else {
        eprintln!("skipping: no null-like ALSA device on this machine");
        return;
    };

    let mut child = spawn_agent(&device);
    std::thread::sleep(Duration::from_secs(6));
    assert!(child.try_wait().expect("try_wait").is_none());

    send_sigint(&child);
    // Land the second SIGINT as deep inside the shutdown window as we can.
    std::thread::sleep(Duration::from_millis(50));
    send_sigint(&child);
    let status = wait_exit(&mut child, Duration::from_secs(15), "double SIGINT");
    assert!(
        status.success() || status.code() == Some(130),
        "expected clean exit 0 or force-exit 130 after two SIGINTs, got {status}"
    );
}
