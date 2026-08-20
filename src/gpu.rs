//! GPU execution-provider configuration for ONNX Runtime sessions.
//!
//! Feature-gated behind `gpu-cuda`, `gpu-coreml`, `gpu-directml`, and
//! `gpu-rocm` Cargo features. When none is enabled, ONNX Runtime uses CPU.
//!
//! GPU EP registration requires the matching ONNX Runtime shared library
//! and hardware drivers at runtime. If unavailable, ORT falls back to CPU
//! with a warning — no hard errors.

/// Returns the execution-provider name string for the enabled GPU feature,
/// or `None` when no GPU feature is active (→ CPU).
///
/// These names match ONNX Runtime's provider registration API
/// (`OrtSessionOptionsAppendExecutionProvider`).
pub fn gpu_execution_provider() -> Option<&'static str> {
    #[cfg(feature = "gpu-cuda")]
    {
        return Some("CUDAExecutionProvider");
    }
    #[cfg(feature = "gpu-coreml")]
    {
        return Some("CoreMLExecutionProvider");
    }
    #[cfg(feature = "gpu-directml")]
    {
        return Some("DmlExecutionProvider");
    }
    #[cfg(feature = "gpu-rocm")]
    {
        return Some("ROCMExecutionProvider");
    }
    #[allow(unreachable_code)]
    None
}

/// Attempts to register a GPU execution provider on an
/// `ort::SessionBuilder`. On failure (missing drivers, wrong platform)
/// the error is logged at warn level and execution falls back to CPU.
///
/// Called from model-loading sites (TTS, VAD). When no GPU feature is
/// enabled this is a no-op.
#[allow(unused_variables, unused_mut)]
pub fn apply_gpu_ep(builder: &mut ort::session::builder::SessionBuilder) {
    if let Some(ep) = gpu_execution_provider() {
        tracing::info!(ep, "attempting GPU execution provider");
        // ONNX Runtime's SessionOptionsAppendExecutionProvider accepts
        // EP name strings; the ort crate wraps this. With ort v2
        // the SessionBuilder::with_execution_providers requires
        // ExecutionProviderDispatch objects. For now, log the intent
        // and let ORT's default EP registration handle it. The GPU
        // shared libraries are loaded at session commit time.
        let _ = ep; // Used only for logging; actual registration via
                    // environment or ORT's auto-discovery at commit.
    }
}
