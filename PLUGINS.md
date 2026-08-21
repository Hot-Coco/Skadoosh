# Skadoosh Plugins (WASM Skills)

Skadoosh can load WebAssembly (`.wasm`) plugins and expose each one to the LLM
as a function-calling tool. Plugins run inside a [wasmtime] sandbox with **no
filesystem, no network, a fuel-bounded CPU budget, and a capped linear memory**,
so they are safe to let the model invoke.

## Enabling plugins

```
# CLI flag
skadoosh --plugins-dir ./my-plugins --repl

# or environment variable
SKADOOSH_PLUGINS_DIR=./my-plugins skadoosh --repl
```

When `--plugins-dir` is not set, Skadoosh looks in the default directory
`~/.skadoosh/plugins/`. If that directory does not exist, no plugins are loaded
(silently). If you explicitly pass a directory that is missing, Skadoosh logs a
warning and continues without plugins.

Every `*.wasm` file in the directory is loaded at startup. Files are visited in
sorted order; a plugin that fails to load, or whose manifest `name` duplicates
an already-loaded plugin, is skipped with a warning.

Each loaded plugin is automatically registered with the LLM as a tool whose
name is the manifest `name`. When the model calls that tool, its JSON arguments
string is passed verbatim to the plugin's `execute` and the plugin's JSON
result is returned to the model as the tool output.

## Sandbox / resource limits

| Limit | Default | Constant |
| --- | --- | --- |
| CPU fuel per execution | 1,000,000 units | `plugins::DEFAULT_FUEL` |
| Max linear memory | 16 MiB | `plugins::DEFAULT_MAX_MEMORY_BYTES` |
| Max table elements | 1024 | `plugins::DEFAULT_MAX_TABLE_ELEMENTS` |
| Filesystem | **none** | — |
| Network | **none** | — |

No WASI imports are linked, so a plugin has no `fd_*`, `path_*`, or socket
imports to call. Fuel is refreshed before every execution, so one runaway call
traps (and returns an error to the model) instead of hanging forever. Memory
and table growth past the cap causes `memory.grow`/`table.grow` to fail. Limits
can be customized with `PluginManager::with_limits`.

## Plugin ABI

A plugin is a standalone `*.wasm` module that exports five things:

| Export | Signature | Purpose |
| --- | --- | --- |
| `memory` | linear memory | backs the allocator below |
| `alloc` | `(size: i32) -> i32` | allocate `size` bytes, return pointer |
| `dealloc` | `(ptr: i32, len: i32) -> ()` | free a prior allocation (may be a no-op) |
| `manifest` | `() -> i32` | return pointer to a length-prefixed manifest string |
| `execute` | `(ptr: i32, len: i32) -> i32` | run the plugin on `len` bytes of input at `ptr`; return pointer to a length-prefixed result string |

### Length-prefixed strings

Strings cross the host/wasm boundary as **length-prefixed** byte buffers:

```
[ len: u32 little-endian ][ len bytes of UTF-8 string ]
```

`manifest()` and `execute()` both return a pointer to such a buffer. The host
reads the 4-byte little-endian length, then reads that many bytes of body,
then `dealloc`s the buffer.

### Input

The host calls `alloc(input.len())`, writes the input bytes into plugin memory
at the returned pointer, then calls `execute(ptr, input.len())`. The input is a
JSON string — the model's tool-call arguments, passed verbatim. The plugin is
responsible for parsing its own schema.

### Manifest JSON

`manifest()` returns a length-prefixed JSON string of the form:

```json
{
  "name": "echo",
  "description": "Echoes its input JSON back to the model.",
  "version": "0.1.0",
  "parameters": { "type": "object", "properties": { "input": { "type": "string" } }, "required": ["input"] }
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `name` | yes | Tool name the LLM invokes. Non-empty, unique across the directory. |
| `description` | yes | Shown to the model as the tool's purpose. |
| `version` | yes | Free-form; logged on load. |
| `parameters` | no | JSON Schema for the tool's parameters. When omitted, the tool accepts any JSON object (`{"type":"object"}`). |

### Result

`execute()` returns a pointer to a length-prefixed JSON string. Whatever the
plugin returns is handed back to the model as the tool result, so it should be
valid JSON (e.g. `{"result": ...}` or `{"error": "..."}`).

## A minimal reference plugin (WAT)

The repository's `tests/plugins.rs` builds a complete echo plugin from WAT. The
essence: a bump allocator, a no-op `dealloc`, a `manifest` pointing at a static
length-prefixed manifest, and an `execute` that copies the input bytes into a
fresh length-prefixed buffer and returns it:

```wat
(module
  (memory (export "memory") 1)
  (global $heap (mut i32) (i32.const 64))

  ;; manifest = '{"name":"echo","description":"echo","version":"0.1.0"}'
  ;; stored length-prefixed at offset 8.
  (data (i32.const 8) "\25\00\00\00" "{\"name\":\"echo\",\"description\":\"echo\",\"version\":\"0.1.0\"}")

  (func (export "alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (local.get $size)))
    (local.get $ptr))
  (func (export "dealloc") (param $ptr i32) (param $len i32))
  (func (export "manifest") (result i32) (i32.const 8))
  (func (export "execute") (param $ptr i32) (param $len i32) (result i32)
    ;; allocate 4 + len, write the length prefix, copy the input, return ptr.
    ;; (see tests/plugins.rs for the full copy loop)
    unreachable)
)
```

## A minimal reference plugin (Rust)

Target `wasm32-unknown-unknown` with `#![no_std]`. Provide a bump allocator
and the four exports. Build with:

```
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/my_plugin.wasm ~/.skadoosh/plugins/
```

```rust
#![no_std]

use core::ptr;

static mut HEAP: usize = 64; // bump pointer, starts past any data

#[no_mangle]
pub extern "C" fn alloc(size: i32) -> i32 {
    unsafe {
        let p = HEAP;
        HEAP += size as usize;
        p as i32
    }
}

#[no_mangle]
pub extern "C" fn dealloc(_ptr: i32, _len: i32) {}

// manifest bytes: [len: u32 LE] + JSON. In real code, build this at startup
// and return a pointer to it.
#[no_mangle]
pub extern "C" fn manifest() -> i32 {
    // (return pointer to a length-prefixed manifest string you prepared)
    0
}

#[no_mangle]
pub extern "C" fn execute(ptr: i32, len: i32) -> i32 {
    // 1. alloc(4 + len)
    // 2. write `len` as u32 LE at the new pointer
    // 3. copy `len` bytes from `ptr` to new_ptr + 4
    // 4. return new_ptr  (here: echo the input as JSON)
    unsafe {
        let out = alloc(len + 4);
        let dst = out as *mut u8;
        *(dst as *mut u32) = len as u32;
        ptr::copy_nonoverlapping(ptr as *const u8, dst.add(4), len as usize);
        out
    }
}

// `memory` is exported automatically by the linker for a `wasm32-unknown-unknown`
// binary that uses a heap; if your toolchain does not export it, add
// `(export "memory" (memory 0))` via a .wat linker step or `wasm-merge`.
```

> A Rust plugin that returns a JSON result from `execute` must build that JSON
> by hand (no `serde` in `#![no_std]` without `alloc`) or by formatting into a
> preallocated buffer. The `manifest` string is easiest to prepare as a static
> byte array with a 4-byte length prefix.

## Registering plugins programmatically (SDK)

```rust
use skadoosh::plugins::PluginManager;
use skadoosh::tools::ToolExecutor;

let mut plugins = PluginManager::new()?;
plugins.load_directory("./my-plugins".as_ref())?;

// Each plugin is a tool:
let tools = plugins.tool_definitions();

// Run a plugin directly (also used by the LLM tool-call loop):
let result = plugins.run("echo", r#"{"hello":"world"}"#)?;
```

`PluginManager` implements `skadoosh::tools::ToolExecutor`, so it slots directly
into the existing function-calling machinery — when constructed via
`LlmClient::from_config`, plugin tool calls are routed into the sandbox
automatically.

[wasmtime]: https://wasmtime.dev/
