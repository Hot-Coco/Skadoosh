//! Audio I/O: cpal capture/playback plus a zero-dependency resampler.
//!
//! Capture converts device-rate input to 16 kHz mono and pushes it into a
//! lock-free ring buffer; playback pops 24 kHz [`crate::tts::TtsClip`]s and
//! resamples to the device rate. Real-time callbacks never allocate or lock.
//!
//! The cpal capture/playback modules (`input`, `output`) and the
//! `aec` echo canceller are gated behind the `audio` feature (they need the
//! system ALSA/speexdsp libraries); the pure-Rust [`resample`]r and
//! [`hold_music`] generator are always available.

#[cfg(feature = "audio")]
pub mod aec;
pub mod hold_music;
#[cfg(feature = "audio")]
pub mod input;
#[cfg(feature = "audio")]
pub mod output;
pub mod resample;

pub use hold_music::HoldMusic;
#[cfg(feature = "audio")]
pub use input::{list_devices, push_block_drop_count, AudioInputConfig, MicCapture, CAPTURE_RATE};
#[cfg(feature = "audio")]
pub use output::{
    push_clip_blocking, AudioOutputConfig, OutputPump, Playback, PlaybackHandle, CLIP_SAMPLE_RATE,
};
pub use resample::{resample_offline, LinearResampler};

#[cfg(feature = "audio")]
use cpal::traits::DeviceTrait;

#[cfg(feature = "audio")]
use crate::error::{AudioError, Result};

/// Best-effort device name (cpal 0.18: `DeviceDescription::name()` with a
/// `Display` fallback). Shared by input/output device picking and
/// `--list-devices`.
#[cfg(feature = "audio")]
pub(crate) fn device_name(device: &cpal::Device) -> Option<String> {
    match device.description() {
        Ok(desc) => Some(desc.name().to_string()),
        Err(_) => {
            let name = device.to_string();
            (!name.is_empty()).then_some(name)
        }
    }
}

/// Picks the device named `name` from `devices`, or falls back to `default`
/// when no name is configured. Shared by capture and playback setup.
#[cfg(feature = "audio")]
pub(crate) fn pick_device<I>(
    name: Option<&str>,
    devices: std::result::Result<I, cpal::Error>,
    default: Option<cpal::Device>,
) -> Result<cpal::Device>
where
    I: Iterator<Item = cpal::Device>,
{
    match name {
        Some(name) => Ok(devices
            .map_err(|err| AudioError::StreamConfig(err.to_string()))?
            .find(|dev| device_name(dev).as_deref() == Some(name))
            .ok_or(AudioError::NoDevice)?),
        None => Ok(default.ok_or(AudioError::NoDevice)?),
    }
}

/// Collects a device's supported config ranges. A device that cannot even
/// report configurations is unusable (headless machines commonly list such a
/// phantom ALSA "default" PCM); report it as "no suitable device" (the cause
/// is logged).
#[cfg(feature = "audio")]
pub(crate) fn config_ranges<I>(
    kind: &str,
    configs: std::result::Result<I, cpal::Error>,
) -> Result<Vec<cpal::SupportedStreamConfigRange>>
where
    I: Iterator<Item = cpal::SupportedStreamConfigRange>,
{
    match configs {
        Ok(ranges) => Ok(ranges.collect()),
        Err(err) => {
            tracing::warn!(%err, "{kind} device config query failed; device unusable");
            Err(AudioError::NoDevice.into())
        }
    }
}

/// Picks a supported stream config: prefers f32, then i16, then u16; within
/// a format prefers `preferred_rate` when the range covers it, else the
/// range's maximum sample rate. Any channel count is accepted (the capture
/// callback mixes down to mono; playback duplicates mono across channels).
#[cfg(feature = "audio")]
pub(crate) fn negotiate_config(
    ranges: &[cpal::SupportedStreamConfigRange],
    preferred_rate: u32,
) -> Option<cpal::SupportedStreamConfig> {
    for format in [
        cpal::SampleFormat::F32,
        cpal::SampleFormat::I16,
        cpal::SampleFormat::U16,
    ] {
        let mut fallback = None;
        for range in ranges
            .iter()
            .copied()
            .filter(|r| r.sample_format() == format)
        {
            if let Some(cfg) = range.try_with_sample_rate(preferred_rate) {
                return Some(cfg);
            }
            fallback.get_or_insert_with(|| range.with_max_sample_rate());
        }
        if fallback.is_some() {
            return fallback;
        }
    }
    None
}
