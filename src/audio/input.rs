//! Microphone capture: cpal input → mono mix → resample to 16 kHz → ring.
//!
//! The cpal callback runs on a real-time thread: no allocations, no locks —
//! it mixes to mono by iterating-and-summing and pushes into a
//! `ringbuf::HeapRb<f32>` producer (~30 s capacity). Overflow policy: drop the
//! incoming block and bump a shared dropped-samples counter (surfaced via
//! `tracing`), implemented by the pure [`push_block_drop_count`] helper so it
//! is unit-testable without audio hardware.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ringbuf::{HeapCons, HeapProd};

use crate::error::Result;

/// Configuration for microphone capture.
#[derive(Debug, Clone, Default)]
pub struct AudioInputConfig {
    /// Input device name; `None` selects the default input device.
    pub device_name: Option<String>,
}

/// Handle to a running capture stream. [`MicCapture::stop`] ends capture.
#[allow(dead_code)] // fields wired up by the task-1.2 implementation
pub struct MicCapture {
    stream: Option<cpal::Stream>,
    dropped_samples: Arc<AtomicU64>,
}

impl MicCapture {
    /// Picks the device (named or default), negotiates the closest config
    /// (f32 preferred, any channel count), and starts capture. Returns the
    /// capture handle and the 16 kHz ring consumer drained by the VAD task.
    pub fn start(cfg: &AudioInputConfig) -> Result<(MicCapture, HeapCons<f32>)> {
        let _ = cfg;
        todo!("task 1.2: cpal capture → mono mix → resample → HeapRb")
    }

    /// Total samples dropped because the ring was full.
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    /// Stops capture by dropping the cpal stream cleanly.
    pub fn stop(self) {
        todo!("task 1.2: drop stream")
    }
}

/// Pushes a resampled block into the mic ring. When the ring lacks capacity
/// for the whole block, the block is dropped entirely and `dropped` is
/// incremented by `block.len()`. Pure helper — the cpal callback delegates to
/// it and unit tests drive it directly with a small ring.
pub fn push_block_drop_count(prod: &mut HeapProd<f32>, block: &[f32], dropped: &AtomicU64) {
    let _ = (prod, block, dropped);
    todo!("task 1.2: all-or-nothing push with drop counter")
}

/// Lists input and output device names, for `--list-devices`. Headless-safe:
/// returns an empty list (never panics) when no devices exist.
pub fn list_devices() -> Result<Vec<String>> {
    todo!("task 1.2: enumerate cpal devices")
}
