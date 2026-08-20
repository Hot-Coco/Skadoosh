//! GPU execution-provider feature flags for ONNX Runtime sessions.

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

#[allow(unused_variables, unused_mut)]
pub fn apply_gpu_ep(builder: &mut ort::session::builder::SessionBuilder) {
    if let Some(ep) = gpu_execution_provider() {
        tracing::info!(ep, "attempting GPU execution provider");
        let _ = ep;
    }
}
