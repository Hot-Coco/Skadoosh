//! Microphone capture: cpal input → mono mix → resample to 16 kHz → ring.
//!
//! The cpal callback runs on a real-time thread: no allocations, no locks —
//! it mixes to mono by iterating-and-summing into pre-reserved scratch and
//! pushes into a `ringbuf::HeapRb<f32>` producer (~30 s capacity). Overflow
//! policy: drop the incoming block and bump a shared dropped-samples counter
//! (surfaced via `tracing`), implemented by the pure [`push_block_drop_count`]
//! helper so it is unit-testable without audio hardware.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, StreamConfig};
use ringbuf::traits::{Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

use crate::error::{AudioError, Result};

use super::resample::LinearResampler;
use super::{config_ranges, device_name, negotiate_config, pick_device};

/// Internal capture rate: the ring, VAD, and STT all run at 16 kHz mono.
pub const CAPTURE_RATE: u32 = 16_000;

/// Mic ring capacity: 30 s at 16 kHz (plan §7).
const MIC_RING_CAPACITY: usize = 480_000;

/// Bound on mono frames mixed per resampler call inside the RT callback.
/// Input streams run with `BufferSize::Default`, so the backend picks the
/// period and could in principle deliver arbitrarily large blocks; the
/// callback therefore processes input in bounded sub-chunks and the
/// preallocated scratch can never grow on the RT thread.
const CALLBACK_CHUNK_FRAMES: usize = 1024;

/// Capacity for the resampler output scratch: [`LinearResampler::process`]
/// reserves `len * CAPTURE_RATE / device_rate + 2` per call, and calls never
/// exceed [`CALLBACK_CHUNK_FRAMES`] input samples.
fn out_scratch_capacity(device_rate: u32) -> usize {
    CALLBACK_CHUNK_FRAMES * CAPTURE_RATE as usize / device_rate.max(1) as usize + 8
}

/// Configuration for microphone capture.
#[derive(Debug, Clone, Default)]
pub struct AudioInputConfig {
    /// Input device name; `None` selects the default input device.
    pub device_name: Option<String>,
}

/// Handle to a running capture stream. [`MicCapture::stop`] ends capture.
pub struct MicCapture {
    stream: cpal::Stream,
    dropped_samples: Arc<AtomicU64>,
}

impl MicCapture {
    /// Picks the device (named or default), negotiates the closest config
    /// (f32 preferred, any channel count), and starts capture. Returns the
    /// capture handle and the 16 kHz ring consumer drained by the VAD task.
    pub fn start(cfg: &AudioInputConfig) -> Result<(MicCapture, HeapCons<f32>)> {
        let host = cpal::default_host();
        let device = pick_device(
            cfg.device_name.as_deref(),
            host.input_devices(),
            host.default_input_device(),
        )?;
        tracing::info!(
            device = device_name(&device).as_deref().unwrap_or("<unknown>"),
            "capturing from input device"
        );

        let ranges = config_ranges("input", device.supported_input_configs())?;
        let negotiated = negotiate_config(&ranges, CAPTURE_RATE).ok_or_else(|| {
            AudioError::StreamConfig("device offers no f32/i16/u16 input configuration".to_string())
        })?;
        let sample_format = negotiated.sample_format();
        let config = negotiated.config();
        let channels = config.channels as usize;
        let device_rate = config.sample_rate;
        tracing::debug!(
            ?sample_format,
            channels,
            device_rate,
            "negotiated input stream config"
        );

        let (prod, cons) = HeapRb::new(MIC_RING_CAPACITY).split();
        let dropped = Arc::new(AtomicU64::new(0));
        let state = CallbackState {
            prod,
            resampler: LinearResampler::new(device_rate, CAPTURE_RATE),
            mono: Vec::with_capacity(CALLBACK_CHUNK_FRAMES),
            out: Vec::with_capacity(out_scratch_capacity(device_rate)),
            channels,
            dropped: Arc::clone(&dropped),
        };

        let stream = match sample_format {
            SampleFormat::F32 => build_input::<f32>(&device, config, state, |s| s),
            SampleFormat::I16 => {
                build_input::<i16>(&device, config, state, |s| f32::from(s) / 32768.0)
            }
            SampleFormat::U16 => build_input::<u16>(&device, config, state, |s| {
                (f32::from(s) - 32768.0) / 32768.0
            }),
            other => {
                return Err(AudioError::StreamConfig(format!(
                    "unsupported input sample format {other:?}"
                ))
                .into());
            }
        }?;
        // cpal 0.18 does not auto-start streams.
        stream
            .play()
            .map_err(|err| AudioError::StreamBuild(err.to_string()))?;

        Ok((
            MicCapture {
                stream,
                dropped_samples: dropped,
            },
            cons,
        ))
    }

    /// Total samples dropped because the ring was full.
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    /// Stops capture by dropping the cpal stream cleanly.
    pub fn stop(self) {
        drop(self.stream);
    }
}

/// Pushes a resampled block into the mic ring. When the ring lacks capacity
/// for the whole block, the block is dropped entirely and `dropped` is
/// incremented by `block.len()`. Pure helper — the cpal callback delegates to
/// it and unit tests drive it directly with a small ring.
pub fn push_block_drop_count(prod: &mut HeapProd<f32>, block: &[f32], dropped: &AtomicU64) {
    if block.is_empty() {
        return;
    }
    if prod.vacant_len() < block.len() {
        dropped.fetch_add(block.len() as u64, Ordering::Relaxed);
        return;
    }
    let pushed = prod.push_slice(block);
    debug_assert_eq!(pushed, block.len());
}

/// Lists input and output device names, for `--list-devices`. Headless-safe:
/// returns an empty list (never panics) when no devices exist.
pub fn list_devices() -> Result<Vec<String>> {
    let host = cpal::default_host();
    let mut names = Vec::new();
    if let Ok(devices) = host.input_devices() {
        for device in devices {
            names.push(format!(
                "input: {}",
                device_name(&device).unwrap_or_else(|| "<unknown>".to_string())
            ));
        }
    }
    if let Ok(devices) = host.output_devices() {
        for device in devices {
            names.push(format!(
                "output: {}",
                device_name(&device).unwrap_or_else(|| "<unknown>".to_string())
            ));
        }
    }
    Ok(names)
}

/// Mutable state owned by the RT input callback. All buffers are reserved at
/// setup for the bounded sub-chunk size ([`CALLBACK_CHUNK_FRAMES`]), so
/// callbacks perform no heap allocation no matter how large a block the
/// backend delivers.
struct CallbackState {
    prod: HeapProd<f32>,
    resampler: LinearResampler,
    /// Interleaved→mono mixdown scratch (device-rate mono samples).
    mono: Vec<f32>,
    /// Resampler output scratch (16 kHz mono samples).
    out: Vec<f32>,
    channels: usize,
    dropped: Arc<AtomicU64>,
}

/// Builds the typed input stream for one sample format, wiring the RT
/// callback: convert to f32 → mono mix → resample to 16 kHz → ring push.
fn build_input<T>(
    device: &cpal::Device,
    config: StreamConfig,
    mut state: CallbackState,
    convert: fn(T) -> f32,
) -> Result<cpal::Stream>
where
    T: SizedSample + 'static,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                process_input_block(&mut state, data, convert);
            },
            move |err| {
                tracing::warn!(%err, "mic input stream error");
            },
            None,
        )
        .map_err(|err| AudioError::StreamBuild(err.to_string()).into())
}

/// One RT callback period: convert + mix to mono without allocation, resample
/// to 16 kHz, push into the ring (drop-and-count on overflow). Processing
/// runs in bounded sub-chunks, so a backend delivering a block larger than
/// the preallocated scratch still cannot trigger a heap allocation here (the
/// resampler's persistent phase keeps sub-chunk boundaries click-free).
fn process_input_block<T: Copy>(state: &mut CallbackState, data: &[T], convert: fn(T) -> f32) {
    let channels = state.channels.max(1);
    for chunk in data.chunks(channels * CALLBACK_CHUNK_FRAMES) {
        state.mono.clear();
        let frames = chunk.len() / channels;
        debug_assert!(frames <= CALLBACK_CHUNK_FRAMES);
        if channels == 1 {
            state.mono.extend(chunk.iter().map(|&s| convert(s)));
        } else {
            state.mono.extend((0..frames).map(|frame| {
                let mut acc = 0.0f32;
                for &sample in &chunk[frame * channels..(frame + 1) * channels] {
                    acc += convert(sample);
                }
                acc / channels as f32
            }));
        }
        state.resampler.process(&state.mono, &mut state.out);
        push_block_drop_count(&mut state.prod, &state.out, &state.dropped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringbuf::traits::Observer;

    fn test_state(device_rate: u32, channels: usize) -> CallbackState {
        let (prod, _cons) = HeapRb::<f32>::new(MIC_RING_CAPACITY).split();
        CallbackState {
            prod,
            resampler: LinearResampler::new(device_rate, CAPTURE_RATE),
            mono: Vec::with_capacity(CALLBACK_CHUNK_FRAMES),
            out: Vec::with_capacity(out_scratch_capacity(device_rate)),
            channels,
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A callback block far larger than the scratch (50 000 stereo frames ≫
    /// 8192) must not grow the preallocated scratch Vecs: the RT path is
    /// allocation-free for any block size.
    #[test]
    fn huge_callback_block_does_not_grow_scratch() {
        let mut state = test_state(48_000, 2);
        let mono_cap = state.mono.capacity();
        let out_cap = state.out.capacity();
        let block = vec![0.1f32; 100_000]; // 50 000 stereo frames
        process_input_block(&mut state, &block, |s| s * 2.0);
        process_input_block(&mut state, &block, |s| s * 2.0);
        assert_eq!(state.mono.capacity(), mono_cap, "mono scratch grew");
        assert_eq!(state.out.capacity(), out_cap, "resampler scratch grew");
        // 100 000 stereo frames → 100 000 mono → 16 kHz ≈ 33 333 samples.
        let pushed = state.prod.occupied_len();
        assert!(
            (pushed as isize - 33_333).abs() <= 8,
            "expected ≈33 333 ringed samples, got {pushed}"
        );
        assert_eq!(state.dropped.load(Ordering::Relaxed), 0);
    }

    /// Mono fast path: same guarantee, and the f32 conversion is applied.
    #[test]
    fn huge_mono_block_does_not_grow_scratch() {
        let mut state = test_state(16_000, 1);
        let mono_cap = state.mono.capacity();
        let out_cap = state.out.capacity();
        let block = vec![0.25f32; 100_001]; // odd length, identity ratio
        process_input_block(&mut state, &block, |s| s);
        assert_eq!(state.mono.capacity(), mono_cap);
        assert_eq!(state.out.capacity(), out_cap);
        assert_eq!(state.prod.occupied_len(), 100_001);
    }
}
