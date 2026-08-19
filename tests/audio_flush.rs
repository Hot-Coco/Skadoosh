//! Headless tests for playback flush-epoch semantics and the playback
//! thread's full-ring push policy (plan §10 task 1.3, §11). No cpal streams
//! are opened: the RT callback core ([`OutputPump`]) and the pusher policy
//! ([`push_clip_blocking`]) are driven directly with scripted rings.
//!
//! Tests that need a real device are gated behind `SKADOOSH_AUDIO_TESTS=1`
//! and skip with a printed reason otherwise.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;
use skadoosh::audio::output::{push_clip_blocking, OutputPump, CLIP_SAMPLE_RATE};

const DEVICE_RATE: u32 = 48_000;
const PERIOD: usize = 512;

#[test]
fn flush_epoch_discards_pending_samples_on_next_period() {
    let (mut prod, cons) = HeapRb::<f32>::new(48_000).split();
    let flush_epoch = Arc::new(AtomicU64::new(0));
    let is_playing = Arc::new(AtomicBool::new(false));
    let mut pump = OutputPump::new(
        cons,
        DEVICE_RATE,
        PERIOD,
        Arc::clone(&flush_epoch),
        Arc::clone(&is_playing),
    );
    let mut period = [1.0f32; PERIOD];

    // Dry ring → pure silence, not playing.
    pump.render(&mut period);
    assert!(period.iter().all(|&s| s == 0.0));
    assert!(!is_playing.load(Ordering::Acquire));

    // Queue 0.5 s of constant 24 kHz audio → next period is non-silent.
    prod.push_slice(&vec![0.25f32; 12_000]);
    pump.render(&mut period);
    assert!(
        period.iter().all(|&s| s == 0.25),
        "constant clip must render as a constant period"
    );
    assert!(is_playing.load(Ordering::Acquire));
    assert!(
        prod.occupied_len() > 11_000,
        "test precondition: ring still holds plenty of audio before flush"
    );

    // Barge-in: bump the epoch. The very next period must be silent even
    // though the ring was nearly full — the consumer clears it itself.
    flush_epoch.fetch_add(1, Ordering::Release);
    pump.render(&mut period);
    assert!(
        period.iter().all(|&s| s == 0.0),
        "pending samples must be discarded on the period after a flush"
    );
    assert!(
        !is_playing.load(Ordering::Acquire),
        "is_playing must clear once the flushed ring runs dry"
    );

    // Fresh audio queued after the flush plays normally.
    prod.push_slice(&vec![0.5f32; 2400]);
    pump.render(&mut period);
    assert!(period.contains(&0.5));
    assert!(is_playing.load(Ordering::Acquire));
}

#[test]
fn is_playing_clears_when_ring_runs_dry() {
    let (mut prod, cons) = HeapRb::<f32>::new(48_000).split();
    let flush_epoch = Arc::new(AtomicU64::new(0));
    let is_playing = Arc::new(AtomicBool::new(false));
    let mut pump = OutputPump::new(
        cons,
        DEVICE_RATE,
        PERIOD,
        flush_epoch,
        Arc::clone(&is_playing),
    );
    let mut period = [0.0f32; PERIOD];

    // Less than one period of source audio: partially audible, then dry.
    prod.push_slice(&vec![0.75f32; 100]);
    pump.render(&mut period);
    assert!(period[..100].contains(&0.75));
    assert!(is_playing.load(Ordering::Acquire));

    pump.render(&mut period);
    assert!(period.iter().all(|&s| s == 0.0));
    assert!(!is_playing.load(Ordering::Acquire));
}

#[test]
fn push_clip_blocking_discards_remainder_after_flush_bump() {
    let (mut prod, mut cons) = HeapRb::<f32>::new(1024).split();
    let flush_epoch = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let mut seen_epoch = 0;

    // Ring has room: the push completes synchronously.
    let clip = vec![0.1f32; 512];
    assert!(push_clip_blocking(
        &mut prod,
        &clip,
        &flush_epoch,
        &mut seen_epoch,
        &stop
    ));
    assert_eq!(prod.occupied_len(), 512);

    // Fill the ring; a blocked push must give up soon after the epoch bumps —
    // a blocked push can never delay a barge-in flush.
    prod.push_slice(&clip);
    assert_eq!(prod.vacant_len(), 0);
    let big = vec![0.2f32; 4096];
    std::thread::scope(|scope| {
        let pusher = scope
            .spawn(|| push_clip_blocking(&mut prod, &big, &flush_epoch, &mut seen_epoch, &stop));
        std::thread::sleep(Duration::from_millis(50));
        flush_epoch.fetch_add(1, Ordering::Release);
        let pushed = pusher.join().expect("pusher thread panicked");
        assert!(
            !pushed,
            "blocked push must discard the remainder after a flush bump"
        );
    });

    // The epoch bump was observed by the pusher; after draining, a fresh push
    // at the current epoch succeeds again.
    cons.clear();
    assert!(push_clip_blocking(
        &mut prod,
        &big[..512],
        &flush_epoch,
        &mut seen_epoch,
        &stop
    ));
    assert_eq!(prod.occupied_len(), 512);

    // Stop flag also aborts a push.
    stop.store(true, Ordering::Relaxed);
    assert!(!push_clip_blocking(
        &mut prod,
        &big[..8],
        &flush_epoch,
        &mut seen_epoch,
        &stop
    ));
}

#[test]
fn push_clip_blocking_waits_for_ring_space() {
    let (mut prod, mut cons) = HeapRb::<f32>::new(1024).split();
    prod.push_slice(&vec![0.3f32; 1024]); // full
    let flush_epoch = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let mut seen_epoch = 0;
    let clip = vec![0.4f32; 2048];

    let started = Instant::now();
    std::thread::scope(|scope| {
        let pusher = scope
            .spawn(|| push_clip_blocking(&mut prod, &clip, &flush_epoch, &mut seen_epoch, &stop));
        // Drain the ring in two stages, 20 ms apart: the pusher can only
        // finish after the second drain.
        for _ in 0..2 {
            std::thread::sleep(Duration::from_millis(20));
            let mut buf = [0.0f32; 1024];
            cons.pop_slice(&mut buf);
        }
        let pushed = pusher.join().expect("pusher thread panicked");
        assert!(pushed, "push must complete once the ring drains");
    });
    assert!(
        started.elapsed() >= Duration::from_millis(15),
        "push on a full ring must wait for space instead of dropping"
    );
    assert_eq!(prod.occupied_len(), 1024);
}

#[test]
fn list_devices_is_headless_safe() {
    let names =
        skadoosh::audio::list_devices().expect("list_devices must not fail on a headless machine");
    eprintln!("list_devices (empty is fine with no audio hardware): {names:?}");
}

fn audio_tests_enabled() -> bool {
    std::env::var("SKADOOSH_AUDIO_TESTS").is_ok_and(|v| v == "1")
}

/// Name of the first input device whose configs actually negotiate — or None.
/// Headless boxes often list a phantom ALSA "default" PCM that cannot open,
/// so the gated tests select a workable device by name (also exercising the
/// `device_name` config path).
fn workable_input_device() -> Option<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    host.input_devices()
        .ok()?
        .find(|dev| {
            dev.supported_input_configs()
                .is_ok_and(|mut configs| configs.next().is_some())
        })
        .and_then(|dev| dev.description().ok().map(|desc| desc.name().to_string()))
}

/// Name of the first output device whose configs actually negotiate.
fn workable_output_device() -> Option<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    host.output_devices()
        .ok()?
        .find(|dev| {
            dev.supported_output_configs()
                .is_ok_and(|mut configs| configs.next().is_some())
        })
        .and_then(|dev| dev.description().ok().map(|desc| desc.name().to_string()))
}

/// Gated (needs a speaker): plays a 100 ms sine clip and observes the
/// `is_playing` flag. Run with `SKADOOSH_AUDIO_TESTS=1`.
#[tokio::test]
async fn playback_plays_clip_on_real_device() {
    if !audio_tests_enabled() {
        eprintln!(
            "skipping playback_plays_clip_on_real_device: needs audio hardware; \
             set SKADOOSH_AUDIO_TESTS=1 to enable"
        );
        return;
    }
    use skadoosh::audio::output::{AudioOutputConfig, Playback};
    use skadoosh::tts::TtsClip;

    let Some(device_name) = workable_output_device() else {
        eprintln!("skipping playback_plays_clip_on_real_device: no usable output device found");
        return;
    };
    let cfg = AudioOutputConfig {
        device_name: Some(device_name),
    };
    let (playback, handle) = Playback::start(&cfg).expect("playback start");
    let n = (CLIP_SAMPLE_RATE / 10) as usize; // 100 ms
    let samples: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / CLIP_SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    // Keep the ring fed while polling: timing-free devices (e.g. the ALSA
    // null plugin) run far faster than realtime, so a single clip drains in
    // microseconds and a sleeping poll would miss the is_playing window.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut observed_playing = false;
    while Instant::now() < deadline && !observed_playing {
        handle
            .queue_clip(TtsClip {
                samples: samples.clone(),
                sample_rate: CLIP_SAMPLE_RATE,
            })
            .await
            .expect("queue clip");
        observed_playing = handle.is_playing();
    }
    assert!(
        observed_playing,
        "is_playing was never set while clips streamed"
    );
    playback.stop();
}

/// Gated (needs a microphone): captures ~300 ms from the default input
/// device. Run with `SKADOOSH_AUDIO_TESTS=1`.
#[test]
fn mic_capture_receives_samples_on_real_device() {
    if !audio_tests_enabled() {
        eprintln!(
            "skipping mic_capture_receives_samples_on_real_device: needs audio hardware; \
             set SKADOOSH_AUDIO_TESTS=1 to enable"
        );
        return;
    }
    use skadoosh::audio::input::{AudioInputConfig, MicCapture};

    let Some(device_name) = workable_input_device() else {
        eprintln!("skipping mic_capture_receives_samples_on_real_device: no usable input device");
        return;
    };
    let cfg = AudioInputConfig {
        device_name: Some(device_name),
    };
    let (capture, cons) = MicCapture::start(&cfg).expect("mic start");
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        cons.occupied_len() > 0,
        "no samples captured from the default input device"
    );
    capture.stop();
}
