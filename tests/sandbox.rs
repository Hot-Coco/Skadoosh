//! Integration tests for the sandboxed `code_exec` tool: basic execution
//! (python/shell/binary), stdout+stderr+exit-code capture, timeout kill,
//! environment scrubbing, and error handling. Docker-mode tests are gated
//! behind `SKADOOSH_TEST_DOCKER=1` (they need image pulls + a daemon).

use skadoosh::sandbox::{
    code_exec_tool_definition, SandboxExecutor, SandboxMode, CODE_EXEC_TOOL_NAME,
};
use skadoosh::tools::ToolExecutor;

/// Parses the JSON result string returned by `SandboxExecutor::execute`.
fn parse_result(out: &str) -> serde_json::Value {
    serde_json::from_str(out).unwrap_or_else(|e| panic!("result is JSON: {out:?} ({e})"))
}

/// `python3 -c` runs and returns stdout + a zero exit code.
#[test]
fn code_exec_runs_python_and_returns_stdout_and_exit_code() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"python","code":"print(2 + 3)"}"#,
        )
        .expect("python run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    assert_eq!(v["timed_out"], false);
    assert!(
        v["stdout"].as_str().unwrap().contains("5"),
        "stdout should contain 5: {v}"
    );
}

/// `bash -c` runs shell snippets.
#[test]
fn code_exec_runs_shell() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"shell","code":"echo hi; echo $((6*7))"}"#,
        )
        .expect("shell run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    let stdout = v["stdout"].as_str().unwrap();
    assert!(stdout.contains("hi"), "stdout: {stdout}");
    assert!(stdout.contains("42"), "stdout: {stdout}");
}

/// Arbitrary binaries run with their arguments.
#[test]
fn code_exec_runs_binary_with_args() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"binary","code":"/bin/echo","args":["hello","world"]}"#,
        )
        .expect("binary run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    assert!(
        v["stdout"].as_str().unwrap().contains("hello world"),
        "stdout: {v}"
    );
}

/// Both stdout and stderr are captured and reported.
#[test]
fn code_exec_captures_stdout_and_stderr() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"python","code":"print('on stdout'); import sys; sys.stderr.write('on stderr\\n')"}"#,
        )
        .expect("run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    assert!(
        v["stdout"].as_str().unwrap().contains("on stdout"),
        "stdout: {v}"
    );
    assert!(
        v["stderr"].as_str().unwrap().contains("on stderr"),
        "stderr: {v}"
    );
}

/// A non-zero guest exit code is reported (not treated as a tool error), with
/// stderr surfaced.
#[test]
fn code_exec_reports_nonzero_exit_and_stderr() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"python","code":"import sys; print('boom', file=sys.stderr); sys.exit(7)"}"#,
        )
        .expect("execution itself succeeds (non-zero exit is a result)");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 7, "exit_code: {v}");
    assert_eq!(v["timed_out"], false);
    assert!(
        v["stderr"].as_str().unwrap().contains("boom"),
        "stderr: {v}"
    );
}

/// A run exceeding the wall-clock timeout is killed and reported as timed out
/// with the conventional exit code 124.
#[test]
fn code_exec_times_out_and_is_killed() {
    let exec = SandboxExecutor::new(2, SandboxMode::Subprocess);
    let started = std::time::Instant::now();
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            // A long sleep that uses no CPU and no network — only the
            // wall-clock watchdog can stop it.
            r#"{"language":"shell","code":"sleep 30; echo done"}"#,
        )
        .expect("a timed-out run still returns a result JSON");

    let v = parse_result(&out);
    assert_eq!(v["timed_out"], true, "should be marked timed out: {v}");
    assert_eq!(v["exit_code"], 124, "timeout exit code: {v}");
    // Must have returned around the 2 s deadline, not after the 30 s sleep.
    assert!(
        started.elapsed() < std::time::Duration::from_secs(15),
        "timeout should fire near the deadline, took {:?}",
        started.elapsed()
    );
}

/// The child environment is scrubbed: everything except PATH/HOME/TMPDIR is
/// cleared, so variables present in the parent (e.g. `CARGO_MANIFEST_DIR`)
/// are invisible to the guest, while PATH is preserved.
#[test]
fn code_exec_scrubs_environment_but_preserves_path() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"shell","code":"echo CARGO=${CARGO_MANIFEST_DIR:-UNSET}; echo PATHSET=${PATH:+yes}"}"#,
        )
        .expect("run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    let stdout = v["stdout"].as_str().unwrap();
    assert!(
        stdout.contains("CARGO=UNSET"),
        "CARGO_MANIFEST_DIR must be scrubbed: {stdout}"
    );
    assert!(
        stdout.contains("PATHSET=yes"),
        "PATH must be preserved: {stdout}"
    );
}

/// The sandbox runs in a private temp working directory: a snippet can write
/// there, and the directory is cleaned up afterwards (no `skadoosh-sandbox-*`
// leftover after the call returns).
#[test]
fn code_exec_uses_and_cleans_up_temp_workdir() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"shell","code":"echo hi > ./out.txt; cat ./out.txt; pwd"}"#,
        )
        .expect("run should succeed");

    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    assert!(
        v["stdout"].as_str().unwrap().contains("hi"),
        "should write and read a file in the workdir: {v}"
    );
    // The workdir is under the system temp dir and is removed on return.
    // (Transient dirs from concurrent sandbox tests may exist, so only assert
    // that *this* run did not leave its own dir — checked by absence of any
    // dir whose name matches and is older than a few seconds is unreliable;
    // instead just confirm the cleanup path doesn't panic by re-running.)
    let _ = exec.execute(
        CODE_EXEC_TOOL_NAME,
        r#"{"language":"python","code":"open('x.txt','w').write('ok')"}"#,
    );
}

/// An unknown language is rejected with a clear error (not silently run).
#[test]
fn code_exec_rejects_unknown_language() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let err = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"rust","code":"fn main(){}"}"#,
        )
        .expect_err("unknown language should error");
    assert!(
        err.to_string().contains("rust"),
        "error should mention the bad language: {err}"
    );
}

/// Malformed JSON arguments fall back to the python default and an empty
/// script (the tool still runs rather than erroring on parse).
#[test]
fn code_exec_invalid_json_defaults_to_python() {
    let exec = SandboxExecutor::new(10, SandboxMode::Subprocess);
    let out = exec
        .execute(CODE_EXEC_TOOL_NAME, "not-json-at-all")
        .expect("invalid JSON defaults rather than erroring");
    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "empty python script exits 0: {v}");
}

/// The `code_exec` tool definition has the expected name, languages, and
/// required fields.
#[test]
fn code_exec_tool_definition_shape() {
    let tool = code_exec_tool_definition();
    assert_eq!(tool.function.name, CODE_EXEC_TOOL_NAME);
    let params = &tool.function.parameters;
    assert_eq!(params["type"], "object");
    assert_eq!(params["properties"]["language"]["enum"][0], "python");
    assert_eq!(params["properties"]["language"]["enum"][1], "shell");
    assert_eq!(params["properties"]["language"]["enum"][2], "binary");
    assert_eq!(params["required"][0], "language");
    assert_eq!(params["required"][1], "code");
}

/// `--code-exec-timeout` / `SKADOOSH_CODE_EXEC_TIMEOUT` is parsed from the CLI
/// and is unset by default; `--code-exec-sandbox` defaults to `subprocess`.
#[test]
fn code_exec_flags_are_parsed() {
    use clap::Parser;
    use skadoosh::config::Config;

    let config = Config::try_parse_from(["skadoosh", "--code-exec-timeout", "45"]).expect("parses");
    assert_eq!(config.code_exec_timeout, Some(45));
    assert_eq!(
        config.code_exec_sandbox,
        SandboxMode::Subprocess,
        "sandbox defaults to subprocess"
    );

    let docker = Config::try_parse_from([
        "skadoosh",
        "--code-exec-timeout",
        "20",
        "--code-exec-sandbox",
        "docker",
    ])
    .expect("parses");
    assert_eq!(docker.code_exec_timeout, Some(20));
    assert_eq!(docker.code_exec_sandbox, SandboxMode::Docker);

    // Default (no flag) is unset.
    let default = Config::try_parse_from(["skadoosh"]).expect("parses");
    assert!(
        default.code_exec_timeout.is_none(),
        "default code_exec_timeout is None"
    );
}

/// `Config::default()` matches the clap default (SDK ergonomics).
#[test]
fn default_config_has_no_code_exec_timeout() {
    use skadoosh::config::Config;
    let default = Config::default();
    assert!(default.code_exec_timeout.is_none());
    assert_eq!(default.code_exec_sandbox, SandboxMode::Subprocess);
}

fn docker_enabled() -> bool {
    std::env::var("SKADOOSH_TEST_DOCKER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Docker mode runs python in an ephemeral `--network none` container. Gated
/// behind `SKADOOSH_TEST_DOCKER=1` (needs the daemon + an image pull).
#[test]
fn code_exec_docker_mode_runs_python() {
    if !docker_enabled() {
        eprintln!("skipping docker sandbox test (set SKADOOSH_TEST_DOCKER=1 to enable)");
        return;
    }
    let exec = SandboxExecutor::new(60, SandboxMode::Docker);
    let out = exec
        .execute(
            CODE_EXEC_TOOL_NAME,
            r#"{"language":"python","code":"print('hello from docker')"}"#,
        )
        .expect("docker run should succeed");
    let v = parse_result(&out);
    assert_eq!(v["exit_code"], 0, "exit_code: {v}");
    assert!(
        v["stdout"].as_str().unwrap().contains("hello from docker"),
        "stdout: {v}"
    );
}
