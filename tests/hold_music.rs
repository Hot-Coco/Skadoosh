//! Integration checks for procedural hold music.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use skadoosh::audio::HoldMusic;

#[test]
fn hold_music_generates_non_zero_when_active() {
    let active = Arc::new(AtomicBool::new(true));
    let mut music = HoldMusic::new(active);
    let samples = music.generate(1024);
    assert!(
        !samples.is_empty(),
        "hold music should generate samples when active"
    );
    let sum_of_squares: f32 = samples.iter().map(|s| s * s).sum();
    assert!(
        sum_of_squares > 0.0,
        "active hold music should contain non-zero audio energy"
    );
}

#[test]
fn hold_music_generates_silence_when_inactive() {
    let active = Arc::new(AtomicBool::new(false));
    let mut music = HoldMusic::new(active);
    let samples = music.generate(1024);
    assert!(
        !samples.is_empty(),
        "hold music should return a buffer of the requested size even when inactive"
    );
    for (i, &sample) in samples.iter().enumerate() {
        assert_eq!(
            sample, 0.0,
            "sample {i} should be silence when the active flag is false"
        );
    }
}

#[test]
fn hold_music_toggles_active_mid_stream() {
    let active = Arc::new(AtomicBool::new(true));
    let mut music = HoldMusic::new(Arc::clone(&active));

    let active_samples = music.generate(512);
    let energy_before: f32 = active_samples.iter().map(|s| s * s).sum();
    assert!(energy_before > 0.0, "active samples should have energy");

    active.store(false, Ordering::Relaxed);
    let silent_samples = music.generate(512);
    for (i, &sample) in silent_samples.iter().enumerate() {
        assert_eq!(
            sample, 0.0,
            "sample {i} should be silent after the active flag is toggled off"
        );
    }

    active.store(true, Ordering::Relaxed);
    let resumed_samples = music.generate(512);
    let energy_after: f32 = resumed_samples.iter().map(|s| s * s).sum();
    assert!(
        energy_after > 0.0,
        "resumed samples should have energy again"
    );
}

#[test]
fn hold_music_advances_elapsed_counter() {
    let active = Arc::new(AtomicBool::new(true));
    let mut music = HoldMusic::new(active);
    let before = music.elapsed_samples();
    let samples = music.generate(1024);
    let after = music.elapsed_samples();
    assert_eq!(
        after - before,
        samples.len() as u64,
        "elapsed_samples must advance by the number of generated samples"
    );
}
