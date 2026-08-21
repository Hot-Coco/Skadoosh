//! WASM skill/plugin system: loads `.wasm` plugins from a directory, runs each
//! in a [`wasmtime`] sandbox, and exposes every plugin as a
//! `ToolExecutor`-compatible tool that is
//! auto-registered with the LLM for function calling.
//!
//! # Sandbox
//!
//! Plugins are compiled and run with [`wasmtime`]. By default a plugin gets:
//!
//! * **No filesystem and no network** — no WASI imports are linked, so the
//!   module has no `fd_*`, `path_*`, or socket imports to call. Granting
//!   those capabilities later means explicitly linking a WASI instance, which
//!   the manager never does.
//! * **A fuel budget** ([`DEFAULT_FUEL`]) — wasm execution consumes fuel per
//!   instruction; running out traps the call instead of looping forever. Fuel
//!   is refreshed before every execution.
//! * **A linear-memory cap** ([`DEFAULT_MAX_MEMORY_BYTES`]) and a table cap
//!   ([`DEFAULT_MAX_TABLE_ELEMENTS`]) enforced through wasmtime's
//!   [`ResourceLimiter`].
//!
//! # Plugin ABI
//!
//! A plugin is a standalone `*.wasm` module that exports four things (see
//! [`PLUGINS.md`] in the repo root for the full authoring guide):
//!
//! * `memory` — the linear memory backing the allocator.
//! * `alloc(size: i32) -> i32` — bump-allocate `size` bytes, return the ptr.
//! * `dealloc(ptr: i32, len: i32)` — free a prior allocation (may be a no-op).
//! * `manifest() -> i32` — return a pointer to the manifest, a
//!   **length-prefixed** string: `[len: u32 little-endian][manifest JSON]`.
//!   The manifest JSON is `{"name","description","version"[,"parameters"]}`.
//! * `execute(ptr: i32, len: i32) -> i32` — receive `len` bytes of input JSON
//!   at `ptr`, return a pointer to a **length-prefixed** result string
//!   (`[len: u32 LE][result JSON]`).
//!
//! The host writes input into plugin memory via `alloc`, calls `execute`, then
//! reads the length-prefixed result back out and `dealloc`s both buffers.
//!
//! [`PLUGINS.md`]: https://github.com/Hot-Coco/Skadoosh/blob/main/PLUGINS.md

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use wasmtime::{
    Config, Engine, Extern, Func, Instance, Linker, Memory, Module, ResourceLimiter, Store, Val,
};

use crate::error::{Result, SkadooshError};
use crate::llm::Tool;
use crate::tools::ToolExecutor;

/// Fuel budget granted to each plugin execution (wasmtime fuel units). Bounds
/// how much computation one call may do before it traps. Refreshed before
/// every [`PluginManager::run`] call.
pub const DEFAULT_FUEL: u64 = 1_000_000;

/// Maximum linear memory a plugin may grow to (16 MiB). Enforced through
/// wasmtime's [`ResourceLimiter`]; a `memory.grow` past this returns -1.
pub const DEFAULT_MAX_MEMORY_BYTES: usize = 16 * 1024 * 1024;

/// Maximum table elements a plugin may grow to. Enforced through
/// [`ResourceLimiter`].
pub const DEFAULT_MAX_TABLE_ELEMENTS: usize = 1024;

/// Per-store resource limits enforced inside the wasmtime sandbox.
#[derive(Debug, Clone, Copy)]
struct PluginLimits {
    max_memory_bytes: usize,
    max_table_elements: usize,
}

/// wasmtime store host state carrying the plugin's [`ResourceLimiter`].
struct PluginCtx {
    limits: PluginLimits,
}

impl ResourceLimiter for PluginCtx {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(desired <= self.limits.max_memory_bytes)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(desired <= self.limits.max_table_elements)
    }
}

/// Plugin metadata returned by the plugin's `manifest()` export.
///
/// `name` becomes the tool name the LLM invokes; `description` is shown to the
/// model; `version` is for logging only. An optional `parameters` JSON Schema
/// overrides the default permissive `{"type":"object"}` tool schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Tool name the LLM uses to invoke this plugin. Must be non-empty and
    /// unique across the loaded directory; duplicates are skipped at load time.
    pub name: String,
    /// Human-readable description shown to the LLM as the tool's purpose.
    pub description: String,
    /// Semantic version string (free-form, logged on load).
    pub version: String,
    /// Optional JSON Schema for the tool's parameters. When absent the tool
    /// accepts any JSON object and the raw arguments string is passed verbatim
    /// to the plugin's `execute`.
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// The wasmtime runtime handles for one loaded plugin, guarded by a mutex:
/// invoking exported functions needs `&mut Store`, while
/// [`ToolExecutor::execute`] only provides `&self`.
struct PluginRuntime {
    store: Store<PluginCtx>,
    memory: Memory,
    alloc: Func,
    dealloc: Func,
    execute: Func,
}

/// One loaded WASM plugin: its parsed [`PluginManifest`], source path, and
/// sandboxed runtime.
pub struct LoadedPlugin {
    manifest: PluginManifest,
    path: PathBuf,
    rt: Mutex<PluginRuntime>,
}

impl LoadedPlugin {
    /// The parsed manifest (name, description, version, optional parameters).
    pub fn manifest(&self) -> &PluginManifest {
        &self.manifest
    }

    /// Filesystem path the plugin was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Compiles and instantiates the `.wasm` at `path` against `engine`,
    /// reads its manifest, and returns the loaded plugin. `limits` and `fuel`
    /// bound the sandbox.
    fn load(engine: &Engine, path: &Path, limits: PluginLimits, fuel: u64) -> Result<Self> {
        let wasm = fs::read(path).map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!("read plugin {}: {e}", path.display()))
        })?;
        // Module::new compiles wasm bytes (the `.wasm` binary format). WAT
        // text is rejected here — plugins ship as compiled `.wasm`.
        let module = Module::new(engine, &wasm).map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!("compile plugin {}: {e}", path.display()))
        })?;
        let mut store = Store::new(engine, PluginCtx { limits });
        store
            .set_fuel(fuel)
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("set fuel: {e}")))?;
        store.limiter(|ctx: &mut PluginCtx| ctx as &mut dyn ResourceLimiter);

        let linker = Linker::new(engine);
        let instance = linker.instantiate(&mut store, &module).map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!(
                "instantiate plugin {}: {e}",
                path.display()
            ))
        })?;

        let memory = required_memory(&instance, &mut store, "memory", path)?;
        let alloc = required_func(&instance, &mut store, "alloc", path)?;
        let dealloc = required_func(&instance, &mut store, "dealloc", path)?;
        let manifest_func = required_func(&instance, &mut store, "manifest", path)?;
        let execute = required_func(&instance, &mut store, "execute", path)?;

        // Read the manifest once, up front, so tool definitions can be built
        // without ever entering the sandbox again.
        let mptr = call_i32(&manifest_func, &mut store, &[])?;
        let mjson = read_prefixed(&memory, &mut store, mptr)?;
        let manifest: PluginManifest = serde_json::from_str(&mjson).map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!(
                "plugin {} manifest JSON parse: {e}",
                path.display()
            ))
        })?;
        if manifest.name.is_empty() {
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "plugin {} manifest has empty name",
                path.display()
            )));
        }
        // Best-effort cleanup of the manifest buffer the plugin allocated.
        let _ = dealloc.call(
            &mut store,
            &[Val::I32(mptr), Val::I32(4 + mjson.len() as i32)],
            &mut [],
        );

        Ok(Self {
            manifest,
            path: path.to_path_buf(),
            rt: Mutex::new(PluginRuntime {
                store,
                memory,
                alloc,
                dealloc,
                execute,
            }),
        })
    }
}

/// Loads `.wasm` plugins from a directory and runs them in wasmtime sandboxes.
///
/// Each loaded plugin is exposed as an LLM tool via
/// [`PluginManager::tool_definitions`] and is executable through the
/// [`ToolExecutor`] impl (so it slots into the existing function-calling loop).
pub struct PluginManager {
    engine: Engine,
    plugins: Vec<LoadedPlugin>,
    by_name: HashMap<String, usize>,
    fuel: u64,
    limits: PluginLimits,
}

impl PluginManager {
    /// Creates an empty manager with the default sandbox limits
    /// ([`DEFAULT_FUEL`], [`DEFAULT_MAX_MEMORY_BYTES`],
    /// [`DEFAULT_MAX_TABLE_ELEMENTS`]).
    pub fn new() -> Result<Self> {
        Self::with_limits(
            DEFAULT_FUEL,
            DEFAULT_MAX_MEMORY_BYTES,
            DEFAULT_MAX_TABLE_ELEMENTS,
        )
    }

    /// Creates an empty manager with explicit sandbox limits.
    pub fn with_limits(
        fuel: u64,
        max_memory_bytes: usize,
        max_table_elements: usize,
    ) -> Result<Self> {
        let mut config = Config::new();
        // Fuel caps per-call computation; refreshed before each run.
        config.consume_fuel(true);
        let engine = Engine::new(&config)
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("wasmtime engine init: {e}")))?;
        Ok(Self {
            engine,
            plugins: Vec::new(),
            by_name: HashMap::new(),
            fuel,
            limits: PluginLimits {
                max_memory_bytes,
                max_table_elements,
            },
        })
    }

    /// Creates a manager and loads every `*.wasm` file in `dir`. Files are
    /// visited in sorted order; a plugin that fails to load or has a
    /// duplicate name is skipped with a warning rather than aborting the lot.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut mgr = Self::new()?;
        mgr.load_directory(dir)?;
        Ok(mgr)
    }

    /// Loads every `*.wasm` file in `dir` into this manager. The directory
    /// must exist.
    pub fn load_directory(&mut self, dir: &Path) -> Result<()> {
        if !dir.exists() {
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "plugins dir not found: {}",
                dir.display()
            )));
        }
        let entries = fs::read_dir(dir).map_err(|e| {
            SkadooshError::Other(anyhow::anyhow!("read plugins dir {}: {e}", dir.display()))
        })?;
        let mut paths = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugins dir entry: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("wasm") && path.is_file() {
                paths.push(path);
            }
        }
        paths.sort();
        for path in paths {
            match LoadedPlugin::load(&self.engine, &path, self.limits, self.fuel) {
                Ok(plugin) => {
                    let name = plugin.manifest.name.clone();
                    if self.by_name.contains_key(&name) {
                        tracing::warn!(
                            plugin = %name,
                            path = %path.display(),
                            "duplicate plugin name; skipping"
                        );
                        continue;
                    }
                    tracing::info!(
                        plugin = %plugin.manifest.name,
                        version = %plugin.manifest.version,
                        path = %path.display(),
                        "loaded plugin"
                    );
                    self.by_name.insert(name, self.plugins.len());
                    self.plugins.push(plugin);
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to load plugin; skipping"
                    );
                }
            }
        }
        Ok(())
    }

    /// Compiles and loads a single `.wasm` file, registering it by manifest
    /// name. Returns the just-loaded plugin's manifest. A duplicate name is an
    /// error here (unlike [`PluginManager::load_directory`], which skips).
    pub fn load_path(&mut self, path: &Path) -> Result<&PluginManifest> {
        let plugin = LoadedPlugin::load(&self.engine, path, self.limits, self.fuel)?;
        let name = plugin.manifest.name.clone();
        if self.by_name.contains_key(&name) {
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "duplicate plugin name '{name}'"
            )));
        }
        self.by_name.insert(name.clone(), self.plugins.len());
        self.plugins.push(plugin);
        Ok(&self.plugins.last().expect("just pushed").manifest)
    }

    /// Number of loaded plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether any plugins are loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Whether a plugin with tool name `name` is loaded.
    pub fn has(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Manifests of all loaded plugins, in load order.
    pub fn manifests(&self) -> Vec<&PluginManifest> {
        self.plugins.iter().map(|p| &p.manifest).collect()
    }

    /// One [`Tool`] definition per loaded plugin, ready to register with the
    /// LLM. The tool name is the manifest `name`; the parameters schema is the
    /// manifest `parameters` when present, otherwise a permissive
    /// `{"type":"object"}`.
    pub fn tool_definitions(&self) -> Vec<Tool> {
        self.plugins
            .iter()
            .map(|p| plugin_tool_definition(&p.manifest))
            .collect()
    }

    /// Runs the plugin named `name`, passing `input` (a JSON string) to its
    /// `execute` export and returning its JSON result string. Fuel is
    /// refreshed before the call so prior runs cannot exhaust the budget.
    pub fn run(&self, name: &str, input: &str) -> Result<String> {
        let idx = *self
            .by_name
            .get(name)
            .ok_or_else(|| SkadooshError::Other(anyhow::anyhow!("unknown plugin '{name}'")))?;
        let plugin = &self.plugins[idx];
        let mut rt = plugin
            .rt
            .lock()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin mutex poisoned: {e}")))?;
        rt.store
            .set_fuel(self.fuel)
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("set fuel: {e}")))?;
        run_plugin(&mut rt, input)
    }
}

impl ToolExecutor for PluginManager {
    fn execute(&self, name: &str, arguments: &str) -> Result<String> {
        // The model's JSON arguments string is passed verbatim as the plugin's
        // input_json — the plugin is responsible for parsing its own schema.
        self.run(name, arguments)
    }
}

/// Builds the LLM [`Tool`] definition for one plugin from its manifest.
fn plugin_tool_definition(manifest: &PluginManifest) -> Tool {
    let parameters = manifest.parameters.clone().unwrap_or_else(|| {
        // Permissive default: the plugin declares its own schema via the
        // manifest `parameters` field; without it, accept any JSON object.
        serde_json::json!({"type": "object"})
    });
    Tool::function(&manifest.name, &manifest.description, parameters)
}

/// Calls a wasm function that takes `params` and returns one `i32`.
fn call_i32(func: &Func, store: &mut Store<PluginCtx>, params: &[Val]) -> Result<i32> {
    let mut out = [Val::I32(0)];
    func.call(store, params, &mut out)
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin function call trapped: {e}")))?;
    match out[0] {
        Val::I32(v) => Ok(v),
        _ => Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin function returned non-i32 result"
        ))),
    }
}

/// Runs a plugin's `execute` export: writes `input` into plugin memory via
/// `alloc`, calls `execute(ptr, len)`, reads the length-prefixed result, and
/// `dealloc`s both buffers (best-effort).
fn run_plugin(rt: &mut PluginRuntime, input: &str) -> Result<String> {
    let in_len = input.len();
    let in_ptr = call_i32(&rt.alloc, &mut rt.store, &[Val::I32(in_len as i32)])?;
    if in_ptr < 0 {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin alloc returned negative pointer"
        )));
    }
    let in_ptr_u = in_ptr as usize;
    let mem_size = rt.memory.data_size(&rt.store);
    if in_ptr_u.saturating_add(in_len) > mem_size {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin alloc returned out-of-range pointer"
        )));
    }
    rt.memory
        .write(&mut rt.store, in_ptr_u, input.as_bytes())
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin memory write: {e}")))?;

    let rptr = call_i32(
        &rt.execute,
        &mut rt.store,
        &[Val::I32(in_ptr), Val::I32(in_len as i32)],
    )?;

    // Free the input buffer (best-effort — a bump allocator may no-op).
    let _ = rt.dealloc.call(
        &mut rt.store,
        &[Val::I32(in_ptr), Val::I32(in_len as i32)],
        &mut [],
    );

    let result = read_prefixed(&rt.memory, &mut rt.store, rptr)?;

    // Free the result buffer (4-byte length prefix + body).
    let total = 4usize
        .checked_add(result.len())
        .ok_or_else(|| SkadooshError::Other(anyhow::anyhow!("plugin result length overflow")))?;
    let _ = rt.dealloc.call(
        &mut rt.store,
        &[Val::I32(rptr), Val::I32(total as i32)],
        &mut [],
    );

    Ok(result)
}

/// Reads a length-prefixed string (`[len: u32 LE][bytes]`) at `ptr` out of
/// plugin memory, bounds-checking both the prefix and the body against the
/// current memory size so a malformed pointer cannot trigger a huge
/// allocation.
fn read_prefixed(memory: &Memory, store: &mut Store<PluginCtx>, ptr: i32) -> Result<String> {
    if ptr < 0 {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin returned negative pointer"
        )));
    }
    let ptr = ptr as usize;
    let size = memory.data_size(&*store);
    if ptr.checked_add(4).map(|end| end > size).unwrap_or(true) {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin result pointer out of bounds"
        )));
    }
    let mut len_bytes = [0u8; 4];
    memory
        .read(&*store, ptr, &mut len_bytes)
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin result length read: {e}")))?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if ptr
        .checked_add(4)
        .and_then(|p| p.checked_add(len))
        .map(|end| end > size)
        .unwrap_or(true)
    {
        return Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin result length {len} exceeds memory"
        )));
    }
    let mut buf = vec![0u8; len];
    if len > 0 {
        memory
            .read(&*store, ptr + 4, &mut buf)
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin result body read: {e}")))?;
    }
    String::from_utf8(buf)
        .map_err(|e| SkadooshError::Other(anyhow::anyhow!("plugin returned non-utf8 result: {e}")))
}

/// Fetches a required exported memory, or an error naming the plugin and the
/// missing export.
fn required_memory(
    instance: &Instance,
    store: &mut Store<PluginCtx>,
    name: &str,
    path: &Path,
) -> Result<Memory> {
    match instance.get_export(&mut *store, name) {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin {} missing exported memory '{name}'",
            path.display()
        ))),
    }
}

/// Fetches a required exported function, or an error naming the plugin and the
/// missing export.
fn required_func(
    instance: &Instance,
    store: &mut Store<PluginCtx>,
    name: &str,
    path: &Path,
) -> Result<Func> {
    match instance.get_export(&mut *store, name) {
        Some(Extern::Func(f)) => Ok(f),
        _ => Err(SkadooshError::Other(anyhow::anyhow!(
            "plugin {} missing required export '{name}'",
            path.display()
        ))),
    }
}

/// Resolves the default plugins directory (`~/.skadoosh/plugins/`) from the
/// `HOME` environment variable. Returns `None` when `HOME` is unset. Callers
/// treat a non-existent default directory as "no plugins" (silently), unlike
/// an explicitly-configured `--plugins-dir`, which warns when missing.
pub fn default_plugins_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".skadoosh").join("plugins"))
}
