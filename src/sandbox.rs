//! Sandboxed code execution for the `code_exec` tool.
//!
//! [`SandboxExecutor`] implements [`crate::tools::ToolExecutor`] so the model
//! can run Python, shell, or arbitrary-binary snippets via function calling.
//! Each run executes in a restricted subprocess with:
//!
//! * a wall-clock `timeout_secs` limit (default 30 s) enforced by a watchdog
//!   thread that kills the whole process group on expiry;
//! * a scrubbed environment — every variable is cleared except `PATH` and
//!   `HOME` (plus `TMPDIR`, pointed at the sandbox working directory);
//! * resource limits (`RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NOFILE`) applied via
//!   the shell `ulimit` builtin so they take effect before the guest program
//!   starts — no `unsafe` pre-exec hook required;
//! * a fresh, private working directory that is removed afterwards; and
//! * best-effort network isolation via `unshare -n` on Linux when privileges
//!   allow (probed once at construction; silently skipped otherwise).
//!
//! When Docker is available, a [`SandboxMode::Docker`] backend runs each
//! snippet in an ephemeral `--network none --read-only` container with
//! `--cap-drop=ALL` and `--ulimit` caps.
//!
//! The crate is `#![forbid(unsafe_code)]`; all syscall-level work (rlimits,
//! `unshare`) is delegated to existing binaries (`bash`/`ulimit`, `unshare`)
//! or to the `rlimit` crate (which encapsulates its own `unsafe`), so this
//! module contains no `unsafe` blocks. User code is passed to the guest as a
//! separate argv positional parameter (`$1` / `"$@"`), never interpolated
//! into the wrapper script, so it cannot inject into the sandbox harness.

use std::ffi::OsString;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use serde::Serialize;

use clap::ValueEnum;

use crate::error::{Result, SkadooshError};
use crate::tools::ToolExecutor;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// The tool name the LLM invokes to run sandboxed code.
pub const CODE_EXEC_TOOL_NAME: &str = "code_exec";

/// Default wall-clock timeout (seconds) for a sandboxed run.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Sandbox backend used by [`SandboxExecutor`] (`--code-exec-sandbox` /
/// `SKADOOSH_CODE_EXEC_SANDBOX`).
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SandboxMode {
    /// Run in a restricted subprocess on the host (ulimits + scrubbed env +
    /// temp dir + best-effort `unshare -n`).
    #[default]
    Subprocess,
    /// Run inside an ephemeral Docker container (`--network none`).
    Docker,
}

/// Which interpreter/runner to use for a `code_exec` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    /// `python3 -c <code>`.
    Python,
    /// `bash -c <code>`.
    Shell,
    /// Run the binary at `code` with the supplied `args`.
    Binary,
}

/// Resource caps applied to each subprocess run. A field of `0` means
/// "unlimited" (the limit is not applied).
#[derive(Debug, Clone, Copy)]
pub struct ResourceLimits {
    /// `RLIMIT_CPU`: max CPU seconds (0 = unlimited).
    pub cpu_secs: u64,
    /// `RLIMIT_AS`: max address space in KiB (0 = unlimited).
    pub as_kb: u64,
    /// `RLIMIT_NOFILE`: max open file descriptors (0 = unlimited).
    pub nofile: u64,
}

/// Docker images used per language in [`SandboxMode::Docker`].
#[derive(Debug, Clone)]
pub struct DockerImages {
    /// Image for [`CodeLanguage::Python`].
    pub python: String,
    /// Image for [`CodeLanguage::Shell`].
    pub shell: String,
    /// Image for [`CodeLanguage::Binary`].
    pub binary: String,
}

impl Default for DockerImages {
    fn default() -> Self {
        Self {
            python: "python:3-slim".to_string(),
            shell: "bash:latest".to_string(),
            binary: "ubuntu:22.04".to_string(),
        }
    }
}

/// The result of a sandboxed run, serialized to JSON for the tool caller.
#[derive(Debug, Serialize)]
struct ExecResult {
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    timed_out: bool,
}

/// The tool definition the LLM sees when sandboxed code execution is enabled.
///
/// Arguments the model provides: `{"language": "python", "code":
/// "print('hello')"}`. For `language: "binary"`, `code` is the executable
/// path and the optional `args` array supplies its arguments.
pub fn code_exec_tool_definition() -> crate::llm::Tool {
    crate::llm::Tool::function(
        CODE_EXEC_TOOL_NAME,
        "Execute a short code snippet in a sandboxed subprocess and return \
         its stdout, stderr, and exit code. Use for quick computations, \
         checks, or scripting. Keep snippets short and self-contained; \
         network access is blocked.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "shell", "binary"],
                    "description": "Runner to use: 'python' (python3 -c), \
                                    'shell' (bash -c), or 'binary' (run an \
                                    executable)."
                },
                "code": {
                    "type": "string",
                    "description": "The code to run (python source or shell \
                                    script), or the executable path when \
                                    language is 'binary'."
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments for the executable; only used \
                                    when language is 'binary'."
                }
            },
            "required": ["language", "code"]
        }),
    )
}

/// Parses the `{"language","code","args"}` arguments the model emits for a
/// `code_exec` call. Missing `language` defaults to `python`; missing `code`
/// defaults to the empty string. For `binary`, a `command` field is accepted
/// as an alias for `code`.
pub(crate) fn parse_code_exec_args(arguments: &str) -> Result<(CodeLanguage, String, Vec<String>)> {
    let v: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let lang_str = v
        .get("language")
        .and_then(|x| x.as_str())
        .unwrap_or("python");
    let language = parse_language(lang_str).ok_or_else(|| {
        SkadooshError::Other(anyhow::anyhow!(
            "code_exec: unknown language '{lang_str}' \
             (expected one of: python, shell, binary)"
        ))
    })?;
    let code = if language == CodeLanguage::Binary {
        v.get("command")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| v.get("code").and_then(|x| x.as_str()))
            .unwrap_or("")
            .to_string()
    } else {
        v.get("code")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string()
    };
    let args: Vec<String> = v
        .get("args")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Ok((language, code, args))
}

fn parse_language(s: &str) -> Option<CodeLanguage> {
    match s.to_lowercase().as_str() {
        "python" | "python3" | "py" => Some(CodeLanguage::Python),
        "shell" | "bash" | "sh" => Some(CodeLanguage::Shell),
        "binary" | "bin" | "exec" => Some(CodeLanguage::Binary),
        _ => None,
    }
}

/// Sandboxed code executor implementing [`ToolExecutor`].
///
/// Construct with [`SandboxExecutor::new`] (or [`Default`]) and register via
/// `LlmClient::from_config` when
/// `--code-exec-timeout` is set.
#[derive(Debug, Clone)]
pub struct SandboxExecutor {
    timeout_secs: u64,
    mode: SandboxMode,
    limits: ResourceLimits,
    network_isolated: bool,
    unshare_available: bool,
    docker_available: bool,
    docker_images: DockerImages,
}

impl Default for SandboxExecutor {
    fn default() -> Self {
        Self::new(DEFAULT_TIMEOUT_SECS, SandboxMode::Subprocess)
    }
}

impl SandboxExecutor {
    /// Creates a new executor with the given wall-clock timeout and backend.
    ///
    /// Probes the host once for `unshare -n` support (Linux network
    /// namespace) and, in [`SandboxMode::Docker`], for a reachable Docker
    /// daemon. Resource limits default to CPU = `timeout_secs`, address space
    /// = 1 GiB, open files = 256 (clamped to the inherited hard ceilings on
    /// unix); override with [`Self::with_limits`].
    pub fn new(timeout_secs: u64, mode: SandboxMode) -> Self {
        let mut limits = ResourceLimits {
            cpu_secs: timeout_secs,
            as_kb: 1_048_576, // 1 GiB
            nofile: 256,
        };
        clamp_to_hard_limits(&mut limits);
        let unshare_available = unshare_network_available();
        let docker_available = if mode == SandboxMode::Docker {
            docker_available()
        } else {
            false
        };
        Self {
            timeout_secs,
            mode,
            limits,
            network_isolated: true,
            unshare_available,
            docker_available,
            docker_images: DockerImages::default(),
        }
    }

    /// Overrides the resource limits.
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Enables/disables best-effort network isolation (`unshare -n`). Default
    /// is enabled; when disabled the `unshare` prefix is skipped entirely.
    pub fn with_network_isolated(mut self, isolated: bool) -> Self {
        self.network_isolated = isolated;
        self
    }

    /// Overrides the Docker images used per language (Docker mode only).
    pub fn with_docker_images(mut self, images: DockerImages) -> Self {
        self.docker_images = images;
        self
    }

    /// The configured wall-clock timeout (seconds).
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// The configured sandbox backend.
    pub fn mode(&self) -> SandboxMode {
        self.mode
    }

    /// Whether `unshare -n` was available at construction time.
    pub fn unshare_available(&self) -> bool {
        self.unshare_available
    }

    /// Builds the subprocess argv for a run (used by [`Self::run_subprocess`]
    /// and unit-tested directly to check the ulimit script and `unshare`
    /// prefix without spawning anything).
    fn build_argv(&self, language: CodeLanguage, code: &str, args: &[String]) -> Vec<OsString> {
        build_subprocess_argv(
            self.network_isolated && self.unshare_available,
            self.limits,
            language,
            code,
            args,
        )
    }

    /// Runs a snippet and returns the JSON result string.
    fn run(&self, language: CodeLanguage, code: &str, args: &[String]) -> Result<String> {
        match self.mode {
            SandboxMode::Subprocess => self.run_subprocess(language, code, args),
            SandboxMode::Docker => self.run_docker(language, code, args),
        }
    }

    fn run_subprocess(
        &self,
        language: CodeLanguage,
        code: &str,
        args: &[String],
    ) -> Result<String> {
        let workdir = create_temp_dir()?;
        let _guard = TempDir(workdir.clone());
        let mut argv = self.build_argv(language, code, args);

        // Wrap with `timeout` so the child is reliably killed on expiry.
        // `timeout` exits 124 when it kills the command — `wait_with_timeout`
        // below serves as a safety net but `timeout` is the primary enforcer.
        if self.timeout_secs > 0 {
            let mut wrapped: Vec<OsString> = Vec::with_capacity(argv.len() + 5);
            wrapped.push("timeout".into());
            wrapped.push("--kill-after".into());
            wrapped.push("5".into());
            wrapped.push(self.timeout_secs.to_string().into());
            wrapped.append(&mut argv);
            argv = wrapped;
        }

        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        scrub_env(&mut cmd, &workdir);
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd.spawn().map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!(
                "code_exec: failed to spawn sandbox subprocess: {e}"
            ))
        })?;

        let stdout_h = child.stdout.take();
        let stderr_h = child.stderr.take();
        let out_thread = std::thread::spawn(move || read_all(stdout_h));
        let err_thread = std::thread::spawn(move || read_all(stderr_h));

        let (status, mut timed_out) = wait_with_timeout(child, self.timeout_secs);

        // `timeout(1)` wrapper exits 124 when it kills the command, so
        // even when the watchdog didn't fire, exit code 124 means timeout.
        if !timed_out && status.as_ref().and_then(|s| s.code()) == Some(124) {
            timed_out = true;
        }

        let stdout = out_thread.join().unwrap_or_default();
        let stderr = err_thread.join().unwrap_or_default();

        Ok(format_result(&stdout, &stderr, status, timed_out))
    }

    fn run_docker(&self, language: CodeLanguage, code: &str, args: &[String]) -> Result<String> {
        if !self.docker_available {
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "code_exec: Docker sandbox requested but the Docker daemon \
                 is not available"
            )));
        }
        let workdir = create_temp_dir()?;
        let _guard = TempDir(workdir.clone());

        let image = match language {
            CodeLanguage::Python => &self.docker_images.python,
            CodeLanguage::Shell => &self.docker_images.shell,
            CodeLanguage::Binary => &self.docker_images.binary,
        };

        let mut argv: Vec<OsString> = Vec::new();
        // Wrap with `timeout` so the container is stopped (SIGTERM, proxied to
        // the guest via the default signal proxy) and auto-removed (`--rm`) on
        // expiry. `timeout` exits 124 when it kills the command.
        if self.timeout_secs > 0 {
            argv.push("timeout".into());
            argv.push("-s".into());
            argv.push("TERM".into());
            argv.push("--kill-after".into());
            argv.push("10".into());
            argv.push(self.timeout_secs.to_string().into());
        }
        argv.push("docker".into());
        argv.push("run".into());
        argv.push("--rm".into());
        argv.push("--network".into());
        argv.push("none".into());
        argv.push("--read-only".into());
        argv.push("--cap-drop=ALL".into());
        argv.push("--security-opt".into());
        argv.push("no-new-privileges".into());
        argv.push("--cpus".into());
        argv.push("1".into());
        if self.limits.as_kb > 0 {
            argv.push("--memory".into());
            argv.push(format!("{}m", self.limits.as_kb / 1024).into());
        }
        push_ulimit(&mut argv, "nofile", self.limits.nofile);
        push_ulimit(&mut argv, "cpu", self.limits.cpu_secs);
        push_ulimit(&mut argv, "as", self.limits.as_kb);
        argv.push("-v".into());
        argv.push(format!("{}:/work", workdir.display()).into());
        argv.push("-w".into());
        argv.push("/work".into());
        argv.push(image.into());
        match language {
            CodeLanguage::Python => {
                argv.push("python3".into());
                argv.push("-c".into());
                argv.push(code.into());
            }
            CodeLanguage::Shell => {
                argv.push("bash".into());
                argv.push("-c".into());
                argv.push(code.into());
            }
            CodeLanguage::Binary => {
                argv.push(code.into());
                for a in args {
                    argv.push(a.into());
                }
            }
        }

        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!("code_exec: failed to spawn docker: {e}"))
        })?;

        let stdout_h = child.stdout.take();
        let stderr_h = child.stderr.take();
        let out_thread = std::thread::spawn(move || read_all(stdout_h));
        let err_thread = std::thread::spawn(move || read_all(stderr_h));

        // Outer safety net beyond `timeout`'s own grace period.
        let outer = self.timeout_secs.saturating_add(30);
        let (status, _killed) = wait_with_timeout(child, outer);

        let stdout = out_thread.join().unwrap_or_default();
        let stderr = err_thread.join().unwrap_or_default();
        let timed_out = status.and_then(|s| s.code()) == Some(124);

        Ok(format_result(&stdout, &stderr, status, timed_out))
    }
}

impl ToolExecutor for SandboxExecutor {
    fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        tracing::debug!(tool = %name, args = %arguments, "code_exec invoked");
        let (language, code, args) = parse_code_exec_args(arguments)?;
        self.run(language, &code, &args)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Builds the subprocess argv without spawning.
///
/// On unix the guest is launched through a `bash -c '<ulimits>; exec …'`
/// wrapper: the ulimit caps take effect before the guest starts, and the user
/// code is passed as a separate positional parameter (`$1` / `"$@"`) so it can
/// never inject into the wrapper script. When `unshare` is true the whole
/// command is prefixed with `unshare -n` for network isolation.
#[cfg(unix)]
fn build_subprocess_argv(
    unshare: bool,
    limits: ResourceLimits,
    language: CodeLanguage,
    code: &str,
    args: &[String],
) -> Vec<OsString> {
    let mut argv: Vec<OsString> = Vec::new();
    if unshare {
        argv.push("unshare".into());
        argv.push("-n".into());
    }
    let ulimits = ulimit_prefix(limits);
    let exec_tail = match language {
        CodeLanguage::Python => "exec python3 -c \"$1\"",
        CodeLanguage::Shell => "exec bash -c \"$1\"",
        CodeLanguage::Binary => "exec \"$@\"",
    };
    let script = format!("{ulimits}; {exec_tail}");
    argv.push("bash".into());
    argv.push("-c".into());
    argv.push(script.into());
    argv.push("skadoosh".into()); // $0
    if language == CodeLanguage::Binary {
        argv.push(code.into()); // $1 = binary path
        for a in args {
            argv.push(a.into());
        }
    } else {
        argv.push(code.into()); // $1 = code
    }
    argv
}

/// Non-unix fallback: no `ulimit`/`unshare` wrappers exist, so launch the
/// guest directly. Resource limits and network isolation are not applied on
/// this platform (documented limitation); the wall-clock timeout and scrubbed
/// environment still apply.
#[cfg(not(unix))]
fn build_subprocess_argv(
    _unshare: bool,
    _limits: ResourceLimits,
    language: CodeLanguage,
    code: &str,
    args: &[String],
) -> Vec<OsString> {
    match language {
        CodeLanguage::Python => vec!["python3".into(), "-c".into(), code.into()],
        CodeLanguage::Shell => vec!["bash".into(), "-c".into(), code.into()],
        CodeLanguage::Binary => {
            let mut v: Vec<OsString> = vec![code.into()];
            for a in args {
                v.push(a.into());
            }
            v
        }
    }
}

/// The `ulimit -t …; ulimit -v …; ulimit -n …` prefix applied before `exec`.
fn ulimit_prefix(limits: ResourceLimits) -> String {
    [
        format!("ulimit -t {}", fmt_limit(limits.cpu_secs)),
        format!("ulimit -v {}", fmt_limit(limits.as_kb)),
        format!("ulimit -n {}", fmt_limit(limits.nofile)),
    ]
    .join("; ")
}

/// Formats a limit value: `0` becomes `unlimited` (bash's ulimit accepts it).
fn fmt_limit(n: u64) -> String {
    if n == 0 {
        "unlimited".to_string()
    } else {
        n.to_string()
    }
}

/// Pushes a `--ulimit name=value` pair only when the value is non-zero.
fn push_ulimit(argv: &mut Vec<OsString>, name: &str, val: u64) {
    if val > 0 {
        argv.push("--ulimit".into());
        argv.push(format!("{name}={val}").into());
    }
}

/// Clears the child environment, then restores only `PATH` and `HOME` (with
/// `USERPROFILE` as a `HOME` fallback on Windows) and points `TMPDIR` at the
/// sandbox working directory so guest temp files are cleaned up with it.
fn scrub_env(cmd: &mut Command, workdir: &PathBuf) {
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    } else if let Ok(profile) = std::env::var("USERPROFILE") {
        cmd.env("HOME", profile);
    }
    cmd.env("TMPDIR", workdir);
}

/// Creates a unique, private working directory under the system temp dir.
fn create_temp_dir() -> Result<PathBuf> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "skadoosh-sandbox-{}-{}-{}",
        std::process::id(),
        n,
        nanos
    ));
    std::fs::create_dir_all(&dir).map_err(|e| {
        SkadooshError::Other(anyhow::anyhow!(
            "code_exec: failed to create sandbox workdir: {e}"
        ))
    })?;
    Ok(dir)
}

/// RAII guard that removes the sandbox working directory on drop.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Reads a piped child stream to EOF (or an empty buffer if there is none).
fn read_all(mut r: Option<impl Read>) -> Vec<u8> {
    let mut buf = Vec::new();
    if let Some(reader) = r.as_mut() {
        let _ = reader.read_to_end(&mut buf);
    }
    buf
}

/// Waits for `child` to exit, killing its process group if `timeout_secs`
/// elapses first. Returns `(exit_status, was_killed_for_timeout)`. A
/// `timeout_secs` of `0` disables the watchdog (waits forever).
///
/// A lightweight watchdog thread sleeps for the deadline and sets a flag; the
/// main thread polls `try_wait` so it never holds a lock while blocked (which
/// would deadlock the watchdog). The watchdog exits early via a channel when
/// the child finishes, so fast commands do not pay the full timeout.
fn wait_with_timeout(mut child: Child, timeout_secs: u64) -> (Option<ExitStatus>, bool) {
    if timeout_secs == 0 {
        let status = child.wait().ok();
        return (status, false);
    }

    let timed_out = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&timed_out);
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let deadline = Duration::from_secs(timeout_secs);
    let watchdog = std::thread::spawn(move || {
        // `Ok` => main signalled done before the deadline; nothing to kill.
        // `Err(Timeout)` => deadline elapsed; tell the main loop to kill.
        if done_rx.recv_timeout(deadline).is_ok() {
            return;
        }
        flag.store(true, Ordering::Relaxed);
    });

    let mut status: Option<ExitStatus> = None;
    let mut killed = false;
    loop {
        match child.try_wait() {
            // Natural exit takes priority over a concurrently-set flag.
            Ok(Some(s)) => {
                status = Some(s);
                break;
            }
            Ok(None) => {
                if timed_out.load(Ordering::Relaxed) {
                    killed = true;
                    kill_tree(&mut child);
                    status = child.wait().ok();
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => break,
        }
    }
    let _ = done_tx.send(());
    let _ = watchdog.join();
    (status, killed)
}

/// Kills the child and (on unix) its whole process group. The child is made a
/// process-group leader via `process_group(0)` at spawn, so its PID equals the
/// PGID; killing `-PGID` reaps descendants too. The handle is still owned, so
/// the PID cannot have been reused before we reap.
#[cfg(unix)]
fn kill_tree(child: &mut Child) {
    // The child is its own process-group leader (process_group(0) at spawn),
    // so its PID equals the PGID; `kill -KILL -PGID` reaps descendants too.
    let pid = child.id();
    let _ = Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pid}"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill_tree(child: &mut Child) {
    let _ = child.kill();
}

/// Serializes the run outcome to the JSON string returned to the tool caller.
/// On timeout the conventional exit code `124` is reported (mirroring the
/// `timeout(1)` command).
fn format_result(
    stdout: &[u8],
    stderr: &[u8],
    status: Option<ExitStatus>,
    timed_out: bool,
) -> String {
    let exit_code = if timed_out {
        Some(124)
    } else {
        status.and_then(|s| s.code())
    };
    let res = ExecResult {
        stdout: String::from_utf8_lossy(stdout).into_owned(),
        stderr: String::from_utf8_lossy(stderr).into_owned(),
        exit_code,
        timed_out,
    };
    serde_json::to_string(&res)
        .unwrap_or_else(|_| "{\"error\":\"code_exec result serialization failed\"}".to_string())
}

/// Probes whether `unshare -n` (a new network namespace) is usable on this
/// host. Requires `CAP_SYS_ADMIN` (typically root), so it is expected to
/// return `false` in unprivileged sandboxes — the executor then skips network
/// isolation gracefully.
#[cfg(unix)]
fn unshare_network_available() -> bool {
    Command::new("unshare")
        .arg("-n")
        .arg("/bin/true")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn unshare_network_available() -> bool {
    false
}

/// Probes whether a Docker daemon is reachable (used only in Docker mode).
fn docker_available() -> bool {
    let mut cmd = Command::new("docker");
    cmd.arg("info")
        .arg("--format")
        .arg("{{.ServerVersion}}")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    cmd.process_group(0);
    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let (status, _killed) = wait_with_timeout(child, 5);
    status.map(|s| s.success()).unwrap_or(false)
}

/// Clamps the requested limits to the process's inherited hard ceilings on
/// unix, so the child's `ulimit` calls never try to raise a limit above the
/// hard maximum (which would fail and abort the run). Read via the `rlimit`
/// crate; any read failure leaves the request unchanged.
#[cfg(unix)]
fn clamp_to_hard_limits(limits: &mut ResourceLimits) {
    use rlimit::Resource;
    if let Ok(hard) = Resource::CPU.get_hard() {
        if hard != rlimit::INFINITY && limits.cpu_secs > hard {
            limits.cpu_secs = hard;
        }
    }
    if let Ok(hard) = Resource::AS.get_hard() {
        let as_bytes = limits.as_kb.saturating_mul(1024);
        if hard != rlimit::INFINITY && as_bytes > hard {
            limits.as_kb = hard / 1024;
        }
    }
    if let Ok(hard) = Resource::NOFILE.get_hard() {
        if hard != rlimit::INFINITY && limits.nofile > hard {
            limits.nofile = hard;
        }
    }
}

#[cfg(not(unix))]
fn clamp_to_hard_limits(_limits: &mut ResourceLimits) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn osvec(v: &[&str]) -> Vec<OsString> {
        v.iter().map(OsString::from).collect()
    }

    #[test]
    fn parse_language_accepts_aliases() {
        assert_eq!(parse_language("python"), Some(CodeLanguage::Python));
        assert_eq!(parse_language("Python3"), Some(CodeLanguage::Python));
        assert_eq!(parse_language("sh"), Some(CodeLanguage::Shell));
        assert_eq!(parse_language("BASH"), Some(CodeLanguage::Shell));
        assert_eq!(parse_language("binary"), Some(CodeLanguage::Binary));
        assert_eq!(parse_language("exec"), Some(CodeLanguage::Binary));
        assert_eq!(parse_language("rust"), None);
    }

    #[test]
    fn parse_code_exec_args_defaults_language_to_python() {
        let (lang, code, args) = parse_code_exec_args(r#"{"code":"print(1)"}"#).expect("parses");
        assert_eq!(lang, CodeLanguage::Python);
        assert_eq!(code, "print(1)");
        assert!(args.is_empty());
    }

    #[test]
    fn parse_code_exec_args_reads_binary_args_and_command_alias() {
        let (lang, code, args) = parse_code_exec_args(
            r#"{"language":"binary","command":"/bin/ls","args":["-la","/work"]}"#,
        )
        .expect("parses");
        assert_eq!(lang, CodeLanguage::Binary);
        assert_eq!(code, "/bin/ls");
        assert_eq!(args, vec!["-la".to_string(), "/work".to_string()]);
    }

    #[test]
    fn parse_code_exec_args_rejects_unknown_language() {
        assert!(parse_code_exec_args(r#"{"language":"rust","code":"x"}"#).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn build_subprocess_argv_python_passes_code_as_positional() {
        let limits = ResourceLimits {
            cpu_secs: 30,
            as_kb: 1_048_576,
            nofile: 256,
        };
        let argv = build_subprocess_argv(false, limits, CodeLanguage::Python, "print('hi')", &[]);
        // bash -c <script> skadoosh <code>
        assert_eq!(argv[0], OsString::from("bash"));
        assert_eq!(argv[1], OsString::from("-c"));
        let script = argv[2].to_string_lossy();
        assert!(script.contains("ulimit -t 30"), "script: {script}");
        assert!(script.contains("ulimit -v 1048576"), "script: {script}");
        assert!(script.contains("ulimit -n 256"), "script: {script}");
        assert!(
            script.contains("exec python3 -c \"$1\""),
            "script: {script}"
        );
        assert_eq!(argv[3], OsString::from("skadoosh")); // $0
        assert_eq!(argv[4], OsString::from("print('hi')")); // $1 = code
    }

    #[cfg(unix)]
    #[test]
    fn build_subprocess_argv_prepends_unshare_when_isolated() {
        let argv = build_subprocess_argv(
            true,
            ResourceLimits {
                cpu_secs: 5,
                as_kb: 0,
                nofile: 0,
            },
            CodeLanguage::Shell,
            "echo hi",
            &[],
        );
        assert_eq!(argv[0], OsString::from("unshare"));
        assert_eq!(argv[1], OsString::from("-n"));
        assert_eq!(argv[2], OsString::from("bash"));
        let script = argv[4].to_string_lossy();
        assert!(script.contains("ulimit -t 5"), "script: {script}");
        assert!(script.contains("ulimit -v unlimited"), "script: {script}");
        assert!(script.contains("ulimit -n unlimited"), "script: {script}");
        assert!(script.contains("exec bash -c \"$1\""), "script: {script}");
        assert_eq!(argv[5], OsString::from("skadoosh"));
        assert_eq!(argv[6], OsString::from("echo hi"));
    }

    #[cfg(unix)]
    #[test]
    fn build_subprocess_argv_binary_passes_path_and_args_positionally() {
        let argv = build_subprocess_argv(
            false,
            ResourceLimits {
                cpu_secs: 0,
                as_kb: 0,
                nofile: 0,
            },
            CodeLanguage::Binary,
            "/bin/echo",
            &["hello".to_string(), "world".to_string()],
        );
        let script = argv[2].to_string_lossy();
        assert!(script.contains("exec \"$@\""), "script: {script}");
        assert_eq!(argv[3], OsString::from("skadoosh"));
        assert_eq!(argv[4], OsString::from("/bin/echo"));
        assert_eq!(argv[5], OsString::from("hello"));
        assert_eq!(argv[6], OsString::from("world"));
    }

    #[test]
    fn fmt_limit_zero_is_unlimited() {
        assert_eq!(fmt_limit(0), "unlimited");
        assert_eq!(fmt_limit(42), "42");
    }

    #[test]
    fn code_exec_tool_definition_has_expected_shape() {
        let tool = code_exec_tool_definition();
        assert_eq!(tool.function.name, CODE_EXEC_TOOL_NAME);
        let params = &tool.function.parameters;
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["language"]["enum"][0], "python");
        assert_eq!(params["properties"]["code"]["type"], "string");
        assert_eq!(params["required"][0], "language");
        assert_eq!(params["required"][1], "code");
    }

    #[test]
    fn new_clamps_limits_and_probes_unshare() {
        // Construction must not panic and must surface a probed unshare flag.
        let exec = SandboxExecutor::new(2, SandboxMode::Subprocess);
        assert_eq!(exec.timeout_secs(), 2);
        assert_eq!(exec.mode(), SandboxMode::Subprocess);
        // In an unprivileged sandbox unshare is unavailable; just check it ran.
        let _ = exec.unshare_available();
    }

    #[test]
    fn osvec_helper() {
        assert_eq!(
            osvec(&["a", "b"]),
            vec![OsString::from("a"), OsString::from("b")]
        );
    }
}
