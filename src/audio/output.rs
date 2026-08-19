//! Speaker playback: clips mpsc → playback std thread → SPSC ring → cpal
//! output callback.
//!
//! The playback thread owns the cpal output stream and is the *sole* pusher
//! into its own 48 000-sample (2 s @ 24 kHz) `HeapRb<f32>`: it pops
//! [`TtsClip`]s from the clips mpsc and pushes them, retrying the remaining
//! slice every ~5 ms when full (natural backpressure up the clips mpsc).
//! Each retry re-checks the flush epoch, discarding the clip remainder after
//! a bump so a blocked push can never delay a barge-in flush. The RT callback
//! pulls from the ring, resamples to the device rate, and outputs silence
//! when empty.
//!
//! Flush is lock-free: [`PlaybackHandle::flush`] only bumps an epoch atomic;
//! the RT callback compares epochs each period and calls `Consumer::clear()`
//! itself — flush latency is bounded by one callback period (~5–10 ms) with
//! zero locks on the RT thread.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, SampleFormat, SizedSample, StreamConfig, SupportedBufferSize};
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use tokio::sync::mpsc;

use crate::error::{AudioError, Result};
use crate::tts::TtsClip;

use super::resample::{resample_offline, LinearResampler};
use super::{config_ranges, device_name, negotiate_config, pick_device};

/// Sample rate of [`TtsClip`] audio (24 kHz; Kokoro and MockTts both emit
/// it). Aliases [`crate::tts::TTS_SAMPLE_RATE`]: the TTS module owns the
/// single source of truth for the clip-rate contract.
pub const CLIP_SAMPLE_RATE: u32 = crate::tts::TTS_SAMPLE_RATE;

/// Playback ring capacity: 2 s of 24 kHz audio (plan §6/§7).
const PLAYBACK_RING_CAPACITY: usize = 48_000;

/// Clips mpsc capacity (plan §7).
const CLIPS_CHANNEL_CAPACITY: usize = 8;

/// Retry interval for a full playback ring and for an idle clips channel
/// (plan §6: ~5 ms).
const RETRY_INTERVAL: Duration = Duration::from_millis(5);

/// Output buffer target: ~2 typical device periods (~10 ms at 48 kHz).
const OUTPUT_BUFFER_FRAMES: u32 = 512;

/// Worst-case callback period assumed when the device doesn't advertise a
/// buffer range (`BufferSize::Default`): RT scratch is sized for this many
/// frames per period. A larger period still works — the scratch grows once —
/// but that growth would happen on the RT thread (debug-asserted against).
const FALLBACK_PERIOD_FRAMES: usize = 8192;

/// Configuration for speaker playback.
#[derive(Debug, Clone, Default)]
pub struct AudioOutputConfig {
    /// Output device name; `None` selects the default output device.
    pub device_name: Option<String>,
}

/// Owning end of the playback thread. [`Playback::stop`] ends playback.
pub struct Playback {
    worker: std::thread::JoinHandle<()>,
    stop: Arc<AtomicBool>,
}

impl Playback {
    /// Spawns the playback thread, opens the output stream, and returns the
    /// owning handle plus the cloneable [`PlaybackHandle`] used by the
    /// pipeline to queue clips and flush on barge-in.
    pub fn start(cfg: &AudioOutputConfig) -> Result<(Playback, PlaybackHandle)> {
        let host = cpal::default_host();
        let device = pick_device(
            cfg.device_name.as_deref(),
            host.output_devices(),
            host.default_output_device(),
        )?;
        tracing::info!(
            device = device_name(&device).as_deref().unwrap_or("<unknown>"),
            "playing to output device"
        );

        // Prefer the device's default rate (most compatible) in an f32
        // stream; fall back to i16/u16 arms when f32 is unavailable.
        let preferred_rate = device
            .default_output_config()
            .map(|cfg| cfg.sample_rate())
            .unwrap_or(48_000);
        let ranges = config_ranges("output", device.supported_output_configs())?;
        let negotiated = negotiate_config(&ranges, preferred_rate).ok_or_else(|| {
            AudioError::StreamConfig(
                "device offers no f32/i16/u16 output configuration".to_string(),
            )
        })?;
        let sample_format = negotiated.sample_format();
        let mut config = negotiated.config();
        // Small output buffer (~2 device periods) so a flush lands within one
        // callback period; leave the default when the device doesn't advertise
        // a range.
        if let SupportedBufferSize::Range { min, max } = *negotiated.buffer_size() {
            config.buffer_size = BufferSize::Fixed(OUTPUT_BUFFER_FRAMES.clamp(min, max));
        }
        let device_rate = config.sample_rate;
        // Worst-case frames per output callback, used to preallocate the RT
        // scratch so the callback never allocates: exact when we pinned a
        // fixed buffer, a documented assumption otherwise.
        let max_period_frames = match config.buffer_size {
            BufferSize::Fixed(n) => n as usize,
            BufferSize::Default => FALLBACK_PERIOD_FRAMES,
        };
        tracing::debug!(
            ?sample_format,
            channels = config.channels,
            device_rate,
            buffer_size = ?config.buffer_size,
            "negotiated output stream config"
        );

        let (prod, cons) = HeapRb::new(PLAYBACK_RING_CAPACITY).split();
        let (clips_tx, clips_rx) = mpsc::channel::<TtsClip>(CLIPS_CHANNEL_CAPACITY);
        let flush_epoch = Arc::new(AtomicU64::new(0));
        let is_playing = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));

        let pump = OutputPump::new(
            cons,
            device_rate,
            max_period_frames,
            Arc::clone(&flush_epoch),
            Arc::clone(&is_playing),
        );
        let stream = match sample_format {
            SampleFormat::F32 => {
                build_output::<f32>(&device, config, max_period_frames, pump, |s| s)
            }
            SampleFormat::I16 => {
                build_output::<i16>(&device, config, max_period_frames, pump, |s| {
                    (s.clamp(-1.0, 1.0) * 32767.0) as i16
                })
            }
            SampleFormat::U16 => {
                build_output::<u16>(&device, config, max_period_frames, pump, |s| {
                    ((s.clamp(-1.0, 1.0) + 1.0) * 32767.5) as u16
                })
            }
            other => {
                return Err(AudioError::StreamConfig(format!(
                    "unsupported output sample format {other:?}"
                ))
                .into());
            }
        }?;
        // cpal 0.18 does not auto-start streams.
        stream
            .play()
            .map_err(|err| AudioError::StreamBuild(err.to_string()))?;

        let worker = std::thread::Builder::new()
            .name("skadoosh-playback".to_string())
            .spawn({
                let flush_epoch = Arc::clone(&flush_epoch);
                let stop = Arc::clone(&stop);
                move || playback_loop(stream, clips_rx, prod, flush_epoch, stop)
            })
            .map_err(|err| {
                AudioError::StreamBuild(format!("failed to spawn playback thread: {err}"))
            })?;

        Ok((
            Playback { worker, stop },
            PlaybackHandle {
                clips_tx,
                flush_epoch,
                is_playing,
                sample_rate: CLIP_SAMPLE_RATE,
            },
        ))
    }

    /// Signals the playback thread to exit and joins it.
    pub fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.worker.join();
    }
}

/// Cloneable, `Send` handle to the running playback thread. Internally just
/// clones of the clips mpsc `Sender` plus the epoch/`is_playing` atomics — it
/// never touches the ring directly.
#[derive(Clone)]
pub struct PlaybackHandle {
    clips_tx: mpsc::Sender<TtsClip>,
    flush_epoch: Arc<AtomicU64>,
    is_playing: Arc<AtomicBool>,
    sample_rate: u32,
}

impl PlaybackHandle {
    /// Queues a clip for playback, awaiting mpsc capacity (backpressure).
    pub async fn queue_clip(&self, clip: TtsClip) -> Result<()> {
        self.clips_tx
            .send(clip)
            .await
            .map_err(|_| AudioError::StreamBuild("playback thread exited".to_string()).into())
    }

    /// Lock-free flush: bumps the flush epoch. The RT callback clears the
    /// ring itself within one output callback period.
    pub fn flush(&self) {
        self.flush_epoch.fetch_add(1, Ordering::Release);
    }

    /// Whether non-silent samples are currently being emitted (VAD/barge-in
    /// input). Set by the RT callback, cleared when the ring runs dry.
    pub fn is_playing(&self) -> bool {
        self.is_playing.load(Ordering::Acquire)
    }

    /// Sample rate of queued clips (24 000 Hz).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// RT-safe core of the output callback, extracted from the cpal closure so
/// the flush-epoch and silence logic can be driven headless by tests.
///
/// Each [`OutputPump::render`] call fills one output period of mono
/// device-rate samples: it compares the flush epoch once per period (clearing
/// the ring itself on a bump), pulls queued 24 kHz clip audio from the ring,
/// resamples to the device rate, and zero-fills whatever the ring could not
/// provide. `is_playing` is set whenever real samples were emitted and
/// cleared when the ring runs dry. All scratch is preallocated for
/// `max_period_frames` per render, so renders at or below that size perform
/// no heap allocation.
pub struct OutputPump {
    cons: HeapCons<f32>,
    resampler: LinearResampler,
    flush_epoch: Arc<AtomicU64>,
    seen_epoch: u64,
    is_playing: Arc<AtomicBool>,
    /// Worst-case frames per [`OutputPump::render`] call the scratch was
    /// sized for (from the negotiated buffer size when known).
    max_period_frames: usize,
    /// Resampled-but-unemitted samples carried into the next period (a
    /// resampled block rarely aligns with the period length).
    pending: Vec<f32>,
    /// Scratch: raw 24 kHz samples popped from the ring.
    ring_scratch: Vec<f32>,
    /// Scratch: one resampled block at device rate.
    block_scratch: Vec<f32>,
}

impl OutputPump {
    /// Creates the pump draining `cons`, resampling 24 kHz clip audio to
    /// `device_rate`. The atomics are shared with the [`PlaybackHandle`].
    ///
    /// `max_period_frames` is the worst-case frame count of one `render`
    /// call (the negotiated output buffer size when known); scratch is
    /// preallocated for it so renders up to that size never allocate. A
    /// larger render still works but grows the scratch on the RT thread
    /// (debug-asserted against).
    pub fn new(
        cons: HeapCons<f32>,
        device_rate: u32,
        max_period_frames: usize,
        flush_epoch: Arc<AtomicU64>,
        is_playing: Arc<AtomicBool>,
    ) -> Self {
        let seen_epoch = flush_epoch.load(Ordering::Acquire);
        let resampler = LinearResampler::new(CLIP_SAMPLE_RATE, device_rate);
        // Worst-case 24 kHz samples popped per period (mirrors render()'s
        // `want`) and worst-case device-rate samples resampled from them
        // (mirrors the resampler's reserve), each with headroom.
        let frames = max_period_frames as u64;
        let clip = u64::from(CLIP_SAMPLE_RATE);
        let device = u64::from(device_rate);
        let ring_cap = (frames * clip / device + 2) as usize + 8;
        let block_cap = ((frames * clip / device + 2) * device / clip + 2) as usize + 8;
        Self {
            cons,
            resampler,
            flush_epoch,
            seen_epoch,
            is_playing,
            max_period_frames,
            // One block's overshoot at most.
            pending: Vec::with_capacity(block_cap),
            ring_scratch: Vec::with_capacity(ring_cap),
            block_scratch: Vec::with_capacity(block_cap),
        }
    }

    /// Fills one output period (mono, device rate) from the ring, emitting
    /// silence beyond what the ring provides.
    pub fn render(&mut self, out: &mut [f32]) {
        // Lock-free flush: one epoch comparison per period; the consumer
        // clears the ring itself, so `flush()` never touches the RT thread.
        let epoch = self.flush_epoch.load(Ordering::Acquire);
        if epoch != self.seen_epoch {
            self.seen_epoch = epoch;
            self.cons.clear();
            self.pending.clear();
        }

        let frames = out.len();
        debug_assert!(
            frames <= self.max_period_frames,
            "output period of {frames} frames exceeds the preallocated {}-frame scratch; \
             this render allocates on the RT thread",
            self.max_period_frames
        );
        let mut written = 0usize;

        let take = self.pending.len().min(frames);
        out[..take].copy_from_slice(&self.pending[..take]);
        self.pending.drain(..take);
        written += take;

        while written < frames {
            let remaining = frames - written;
            // Enough 24 kHz source samples to produce `remaining` outputs
            // (see the resampler's phase math), capped by ring occupancy.
            let want = (remaining * CLIP_SAMPLE_RATE as usize / self.resampler.dst_rate() as usize
                + 2)
            .min(self.cons.occupied_len());
            if want == 0 {
                break; // ring dry
            }
            self.ring_scratch.clear();
            self.ring_scratch.resize(want, 0.0);
            let got = self.cons.pop_slice(&mut self.ring_scratch);
            self.ring_scratch.truncate(got);
            self.resampler
                .process(&self.ring_scratch, &mut self.block_scratch);
            if self.block_scratch.is_empty() {
                break; // need more source samples than the ring holds
            }
            let take = self.block_scratch.len().min(frames - written);
            out[written..written + take].copy_from_slice(&self.block_scratch[..take]);
            written += take;
            if take < self.block_scratch.len() {
                self.pending.extend_from_slice(&self.block_scratch[take..]);
            }
        }

        out[written..].fill(0.0);
        self.is_playing.store(written > 0, Ordering::Release);
    }
}

/// Pushes a clip's samples into the playback ring under the §6 full-policy:
/// when the ring is full, retries the remaining slice every ~5 ms (natural
/// backpressure up the clips mpsc), re-checking the flush epoch and the stop
/// flag on every retry. Returns `true` when the whole clip was pushed;
/// `false` when the remainder was discarded after a flush-epoch bump (so a
/// blocked push can never delay a barge-in flush) or `stop` was set.
pub fn push_clip_blocking(
    prod: &mut HeapProd<f32>,
    samples: &[f32],
    flush_epoch: &AtomicU64,
    seen_epoch: &mut u64,
    stop: &AtomicBool,
) -> bool {
    let mut rest = samples;
    while !rest.is_empty() {
        let epoch = flush_epoch.load(Ordering::Acquire);
        if epoch != *seen_epoch {
            *seen_epoch = epoch;
            return false;
        }
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        let pushed = prod.push_slice(rest);
        rest = &rest[pushed..];
        if !rest.is_empty() {
            std::thread::sleep(RETRY_INTERVAL);
        }
    }
    true
}

/// Playback thread main loop: owns the stream and is the sole ring pusher.
/// Pops clips, normalizes their rate to 24 kHz if a producer violates the
/// contract, pushes under the full-policy, and drains queued clips after a
/// flush bump so no stale audio can follow a barge-in flush.
fn playback_loop(
    stream: cpal::Stream,
    mut clips_rx: mpsc::Receiver<TtsClip>,
    mut prod: HeapProd<f32>,
    flush_epoch: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
) {
    // The stream plays until this binding is dropped at loop exit.
    let _stream = stream;
    let mut seen_epoch = flush_epoch.load(Ordering::Acquire);
    while !stop.load(Ordering::Relaxed) {
        match clips_rx.try_recv() {
            Ok(clip) => {
                let samples = if clip.sample_rate == CLIP_SAMPLE_RATE {
                    clip.samples
                } else {
                    tracing::warn!(
                        clip_rate = clip.sample_rate,
                        "clip sample rate violates the {CLIP_SAMPLE_RATE} Hz contract; resampling"
                    );
                    resample_offline(&clip.samples, clip.sample_rate, CLIP_SAMPLE_RATE)
                };
                if !push_clip_blocking(&mut prod, &samples, &flush_epoch, &mut seen_epoch, &stop) {
                    while clips_rx.try_recv().is_ok() {}
                }
            }
            Err(mpsc::error::TryRecvError::Empty) => std::thread::sleep(RETRY_INTERVAL),
            // Every sender dropped: the pipeline is shutting down.
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        }
    }
}

/// Builds the typed output stream for one sample format, wiring the RT
/// callback: pump a mono period from the ring, then convert + interleave.
/// `max_period_frames` preallocates the mono scratch (see [`OutputPump::new`]).
fn build_output<T>(
    device: &cpal::Device,
    config: StreamConfig,
    max_period_frames: usize,
    mut pump: OutputPump,
    convert: fn(f32) -> T,
) -> Result<cpal::Stream>
where
    T: SizedSample + 'static,
{
    let channels = (config.channels as usize).max(1);
    let mut mono: Vec<f32> = Vec::with_capacity(max_period_frames);
    device
        .build_output_stream(
            config,
            move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
                let frames = out.len() / channels;
                debug_assert!(
                    frames <= mono.capacity(),
                    "output period of {frames} frames exceeds the preallocated {}-frame scratch; \
                     this callback allocates on the RT thread",
                    mono.capacity()
                );
                mono.clear();
                mono.resize(frames, 0.0);
                pump.render(&mut mono);
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = convert(mono[i / channels]);
                }
            },
            move |err| {
                tracing::warn!(%err, "playback stream error");
            },
            None,
        )
        .map_err(|err| AudioError::StreamBuild(err.to_string()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pump(device_rate: u32, max_period_frames: usize) -> (HeapProd<f32>, OutputPump) {
        let (prod, cons) = HeapRb::<f32>::new(PLAYBACK_RING_CAPACITY).split();
        let pump = OutputPump::new(
            cons,
            device_rate,
            max_period_frames,
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        (prod, pump)
    }

    /// Renders at or below the negotiated period must never grow the
    /// preallocated scratch (steady-state no-alloc on the RT thread).
    #[test]
    fn render_within_period_hint_does_not_grow_scratch() {
        let (mut prod, mut pump) = test_pump(48_000, 512);
        let ring_cap = pump.ring_scratch.capacity();
        let block_cap = pump.block_scratch.capacity();
        let pending_cap = pump.pending.capacity();
        prod.push_slice(&vec![0.5f32; PLAYBACK_RING_CAPACITY]);
        let mut period = [0.0f32; 512];
        for _ in 0..16 {
            pump.render(&mut period);
            assert_eq!(pump.ring_scratch.capacity(), ring_cap, "ring scratch grew");
            assert_eq!(
                pump.block_scratch.capacity(),
                block_cap,
                "block scratch grew"
            );
            assert_eq!(pump.pending.capacity(), pending_cap, "pending grew");
            assert!(period.iter().all(|&s| s == 0.5));
        }
    }

    /// The review scenario: a heavy downsample ratio (24 kHz → 8 kHz) with a
    /// big period pops >8192 source samples per render. Sized from the
    /// negotiated buffer, the scratch absorbs it without growing.
    #[test]
    fn heavy_downsample_big_period_does_not_grow_scratch() {
        let (mut prod, mut pump) = test_pump(8_000, 4096);
        let ring_cap = pump.ring_scratch.capacity();
        assert!(ring_cap > 8192, "hint-based sizing must exceed 8192 here");
        let block_cap = pump.block_scratch.capacity();
        prod.push_slice(&vec![0.5f32; PLAYBACK_RING_CAPACITY]);
        let mut period = [1.0f32; 4096];
        pump.render(&mut period);
        pump.render(&mut period);
        assert_eq!(pump.ring_scratch.capacity(), ring_cap);
        assert_eq!(pump.block_scratch.capacity(), block_cap);
        assert!(period.iter().all(|&s| s == 0.5));
    }
}
