//! Integration tests for the WASM skill/plugin system (`--plugins-dir`).
//!
//! A minimal plugin is synthesized in-memory from WAT via the `wat` crate (no
//! toolchain required), written into a temp directory, loaded by
//! [`PluginManager`], and exercised end-to-end: manifest parsing, tool
//! definition generation, the `ToolExecutor` trait, error handling, and the
//! sandbox memory round-trip.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use skadoosh::plugins::{PluginManager, PluginManifest};
use skadoosh::tools::ToolExecutor;

/// A unique temp directory per test (process id + monotonic counter) so
/// parallel runs never collide. [`TempDir`] removes it on drop (best-effort).
fn temp_dir(label: &str) -> (PathBuf, TempDir) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "skadoosh-plugins-{}-{label}-{n}",
        std::process::id()
    ));
    std::fs::create_dir_all(&p).expect("create temp dir");
    (p.clone(), TempDir(p))
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Builds a WAT module implementing the plugin ABI: a bump allocator, a no-op
/// `dealloc`, a `manifest` pointing at a length-prefixed static manifest
/// string, and an `execute` that echoes the input bytes (length-prefixed).
fn echo_wat(manifest: &str) -> String {
    let manifest_len = manifest.len() as u32;
    let lb = manifest_len.to_le_bytes();
    let manifest_offset: u32 = 8;
    let heap_start: u32 = manifest_offset + 4 + manifest_len;
    let len_escape = format!(
        "\\{:02x}\\{:02x}\\{:02x}\\{:02x}",
        lb[0], lb[1], lb[2], lb[3]
    );
    // WAT-escape the manifest text: backslash then double-quote.
    let manifest_escaped = manifest.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        r#"(module
  (memory (export "memory") 1)
  (global $heap (mut i32) (i32.const {heap_start}))
  (data (i32.const {manifest_offset}) "{len_escape}" "{manifest_escaped}")
  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (local.get $size)))
    (local.get $ptr))
  (func (export "dealloc") (param $ptr i32) (param $len i32))
  (func (export "manifest") (result i32)
    (i32.const {manifest_offset}))
  (func (export "execute") (param $ptr i32) (param $len i32) (result i32)
    (local $out i32)
    (local $i i32)
    (local.set $out (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (i32.add (local.get $len) (i32.const 4))))
    (i32.store (local.get $out) (local.get $len))
    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
        (i32.store8
          (i32.add (i32.add (local.get $out) (i32.const 4)) (local.get $i))
          (i32.load8_u (i32.add (local.get $ptr) (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $out))
)
"#
    )
}

const ECHO_MANIFEST: &str =
    r#"{"name":"echo","description":"echoes its input json","version":"0.1.0"}"#;

/// Compiles `manifest`'s WAT to wasm bytes and writes them as `fname` under
/// `dir`, returning the path.
fn write_plugin(dir: &Path, fname: &str, manifest: &str) -> PathBuf {
    let path = dir.join(fname);
    let wasm = wat::parse_str(echo_wat(manifest)).expect("parse wat");
    std::fs::write(&path, wasm).expect("write wasm");
    path
}

#[test]
fn loads_plugin_and_echoes_input() {
    let (dir, _guard) = temp_dir("echo");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    assert_eq!(mgr.len(), 1, "one plugin loaded");
    assert!(mgr.has("echo"));
    assert!(!mgr.has("nope"));

    let manifests: Vec<&PluginManifest> = mgr.manifests();
    assert_eq!(manifests.len(), 1);
    assert_eq!(manifests[0].name, "echo");
    assert_eq!(manifests[0].version, "0.1.0");
    assert_eq!(manifests[0].description, "echoes its input json");

    let input = r#"{"msg":"hello world","n":42}"#;
    let out = mgr.run("echo", input).expect("run");
    assert_eq!(out, input, "echo plugin must return its input verbatim");
}

#[test]
fn tool_definitions_register_one_tool_per_plugin() {
    let (dir, _guard) = temp_dir("tools");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    let tools = mgr.tool_definitions();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "echo");
    assert_eq!(tools[0].tool_type, "function");
    // Permissive default schema when the manifest omits `parameters`.
    assert_eq!(
        tools[0].function.parameters,
        serde_json::json!({"type": "object"})
    );
}

#[test]
fn manifest_parameters_override_default_tool_schema() {
    let manifest = r#"{"name":"add","description":"adds two numbers","version":"1.0.0","parameters":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}"#;
    let (dir, _guard) = temp_dir("params");
    write_plugin(&dir, "add.wasm", manifest);

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    let tools = mgr.tool_definitions();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "add");
    assert_eq!(
        tools[0].function.parameters["properties"]["a"]["type"],
        "number"
    );
}

#[test]
fn tool_executor_trait_runs_plugin() {
    let (dir, _guard) = temp_dir("trait");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    let input = r#"{"k":"v"}"#;
    // Exercise the ToolExecutor impl, not the inherent run() method.
    let out = ToolExecutor::execute(&mgr, "echo", input).expect("trait execute");
    assert_eq!(out, input);
}

#[test]
fn unknown_plugin_name_errors() {
    let (dir, _guard) = temp_dir("unknown");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);
    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    let err = mgr.run("does-not-exist", "{}").unwrap_err();
    assert!(err.to_string().contains("unknown plugin"), "got {err:?}");
}

#[test]
fn empty_input_round_trips() {
    let (dir, _guard) = temp_dir("empty");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);
    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    let out = mgr.run("echo", "").expect("run empty");
    assert_eq!(out, "", "empty input must echo back empty");
}

#[test]
fn non_wasm_files_are_ignored() {
    let (dir, _guard) = temp_dir("mixed");
    write_plugin(&dir, "echo.wasm", ECHO_MANIFEST);
    // Stray non-wasm files must not break loading.
    std::fs::write(dir.join("README.txt"), "not a plugin").unwrap();
    std::fs::write(dir.join("echo.wat"), echo_wat(ECHO_MANIFEST)).unwrap();

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    assert_eq!(mgr.len(), 1, "only .wasm files load");
}

#[test]
fn duplicate_plugin_names_keep_first() {
    let (dir, _guard) = temp_dir("dup");
    write_plugin(&dir, "a.wasm", ECHO_MANIFEST);
    write_plugin(&dir, "b.wasm", ECHO_MANIFEST); // same name "echo"

    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    assert_eq!(mgr.len(), 1, "duplicate name skipped");
}

#[test]
fn missing_dir_errors() {
    // `PluginManager` isn't `Debug` (it owns wasmtime stores), so use `.err()`
    // rather than `.unwrap_err()` to inspect the failure.
    let err = PluginManager::load_dir(Path::new("/nonexistent/skadoosh/plugins"))
        .err()
        .expect("missing dir should error");
    assert!(
        err.to_string().contains("plugins dir not found"),
        "got {err:?}"
    );
}

#[test]
fn empty_dir_loads_nothing() {
    let (dir, _guard) = temp_dir("emptydir");
    let mgr = PluginManager::load_dir(&dir).expect("load_dir");
    assert!(mgr.is_empty());
    assert!(mgr.tool_definitions().is_empty());
}
