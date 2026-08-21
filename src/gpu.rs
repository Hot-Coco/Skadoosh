//! GPU execution-provider feature flags for ONNX Runtime sessions.
//!
//! Feature flags `gpu-cuda`, `gpu-coreml`, `gpu-directml`, and `gpu-rocm`
//! gate the corresponding ort execution providers. None enabled by default:
//! ONNX Runtime falls back to CPU.

use ort::session::builder::SessionBuilder;

/// Returns the name of the compiled-in GPU execution provider, or `None`
/// when no GPU feature is enabled (CPU-only build).
#[allow(dead_code)]
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

/// Registers the compiled-in GPU execution provider on the session builder.
/// When no GPU feature is active this is a no-op (CPU fallback).
///
/// Feature-gated providers are loaded in priority order:
/// CUDA → CoreML → DirectML → ROCm. Only the first enabled provider is
/// registered; multiple concurrent GPU providers are not supported.
pub fn apply_gpu_ep(builder: &mut SessionBuilder) -> std::result::Result<(), ort::Error> {
    #[cfg(feature = "gpu-cuda")]
    {
        tracing::info!("registering CUDAExecutionProvider");
        return builder.with_execution_providers([ort::ep::CUDA::default().build()]);
    }
    #[cfg(feature = "gpu-coreml")]
    {
        tracing::info!("registering CoreMLExecutionProvider");
        return builder.with_execution_providers([ort::ep::CoreML::default().build()]);
    }
    #[cfg(feature = "gpu-directml")]
    {
        tracing::info!("registering DmlExecutionProvider");
        return builder.with_execution_providers([ort::ep::DirectML::default().build()]);
    }
    #[cfg(feature = "gpu-rocm")]
    {
        tracing::info!("registering ROCMExecutionProvider");
        return builder.with_execution_providers([ort::ep::ROCm::default().build()]);
    }
    #[allow(unreachable_code)]
    {
        let _ = builder;
        Ok(())
    }
}
