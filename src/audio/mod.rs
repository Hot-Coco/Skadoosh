//! Audio I/O: cpal capture/playback plus a zero-dependency resampler.
//!
//! Capture converts device-rate input to 16 kHz mono and pushes it into a
//! lock-free ring buffer; playback pops 24 kHz [`crate::tts::TtsClip`]s and
//! resamples to the device rate. Real-time callbacks never allocate or lock.

pub mod input;
pub mod output;
pub mod resample;

pub use input::{list_devices, AudioInputConfig, MicCapture};
pub use output::{AudioOutputConfig, Playback, PlaybackHandle};
pub use resample::{resample_offline, LinearResampler};
