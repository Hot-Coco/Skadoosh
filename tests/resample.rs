//! Headless tests for the linear resampler and the mic-ring overflow policy
//! (plan §10 tasks 1.1/1.2, §11). No audio hardware required.

use std::f32::consts::TAU;
use std::sync::atomic::{AtomicU64, Ordering};

use ringbuf::traits::{Consumer, Observer, Split};
use ringbuf::HeapRb;
use skadoosh::audio::input::push_block_drop_count;
use skadoosh::audio::resample::{resample_offline, LinearResampler};

/// Deterministic swept sine (200 Hz → 2 kHz), peak 0.8.
fn swept_sine(n: usize, rate: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / rate;
            let freq = 200.0 + 1800.0 * (i as f32 / n as f32);
            (TAU * freq * t).sin() * 0.8
        })
        .collect()
}

#[test]
fn identity_ratio_is_bit_exact_passthrough() {
    let input = swept_sine(4096, 16_000.0);

    let offline = resample_offline(&input, 16_000, 16_000);
    assert_eq!(offline, input, "offline identity must be bit-exact");

    // Chunked identity is bit-exact too (every chunk round-trips unchanged).
    let mut resampler = LinearResampler::new(16_000, 16_000);
    let mut out = Vec::new();
    let mut chunked = Vec::new();
    for chunk in input.chunks(500) {
        resampler.process(chunk, &mut out);
        assert_eq!(&out[..], chunk, "identity chunk must be bit-exact");
        chunked.extend_from_slice(&out);
    }
    assert_eq!(chunked, input);
}

#[test]
fn downsample_48k_to_16k_length() {
    assert!(resample_offline(&[], 48_000, 16_000).is_empty());
    for n in [1usize, 2, 3, 4, 5, 100, 300, 301, 302, 4800, 48_000] {
        let input = vec![0.5f32; n];
        let out = resample_offline(&input, 48_000, 16_000);
        let expected = n / 3;
        assert!(
            out.len().abs_diff(expected) <= 1,
            "n={n}: got {} samples, expected {expected} ± 1",
            out.len()
        );
    }
}

#[test]
fn upsample_16k_to_48k_length_and_shape() {
    // A DC offset resamples to the same DC offset (within float tolerance).
    let input = vec![0.25f32; 1600];
    let out = resample_offline(&input, 16_000, 48_000);
    let expected = 1600 * 3;
    assert!(
        out.len().abs_diff(expected) <= 4,
        "got {} samples, expected ≈{expected}",
        out.len()
    );
    assert!(
        out.iter().all(|&s| (s - 0.25).abs() < 1e-6),
        "DC must survive resampling"
    );
}

#[test]
fn chunked_processing_matches_offline_reference() {
    let rate = 44_100u32;
    let input = swept_sine(30_000, rate as f32);
    let reference = resample_offline(&input, rate, 16_000);

    // Awkward chunk sizes (tiny, non-divisor, varying) stress the fractional
    // phase carry across boundaries.
    let pattern = [1usize, 7, 313, 1024, 3, 2000];
    let mut resampler = LinearResampler::new(rate, 16_000);
    let mut out = Vec::new();
    let mut chunked = Vec::new();
    let mut offset = 0;
    let mut i = 0;
    while offset < input.len() {
        let size = pattern[i % pattern.len()].min(input.len() - offset);
        resampler.process(&input[offset..offset + size], &mut out);
        chunked.extend_from_slice(&out);
        offset += size;
        i += 1;
    }

    assert_eq!(
        chunked.len(),
        reference.len(),
        "chunked output length diverged from offline"
    );
    let max_diff = chunked
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "phase discontinuity at a chunk boundary: max sample diff {max_diff}"
    );
}

#[test]
fn steady_state_process_does_not_reallocate_scratch() {
    let mut resampler = LinearResampler::new(48_000, 16_000);
    let block = swept_sine(4800, 48_000.0);
    let mut out = Vec::new();
    // Warm up: the first call grows the scratch to its steady-state capacity.
    resampler.process(&block, &mut out);
    resampler.process(&block, &mut out);
    let capacity = out.capacity();
    let len = out.len();
    assert!(len > 0);
    for _ in 0..64 {
        resampler.process(&block, &mut out);
        assert_eq!(
            out.capacity(),
            capacity,
            "scratch Vec reallocated in steady state"
        );
        assert_eq!(out.len(), len, "steady-state output length changed");
    }
}

#[test]
fn push_block_drop_count_drops_full_block_on_overflow() {
    let (mut prod, mut cons) = HeapRb::<f32>::new(64).split();
    let dropped = AtomicU64::new(0);

    let first = vec![1.0f32; 60];
    push_block_drop_count(&mut prod, &first, &dropped);
    assert_eq!(prod.occupied_len(), 60);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);

    // Only 4 slots free: an 8-sample block is dropped entirely and counted.
    let block = vec![2.0f32; 8];
    push_block_drop_count(&mut prod, &block, &dropped);
    assert_eq!(dropped.load(Ordering::Relaxed), 8);
    assert_eq!(
        prod.occupied_len(),
        60,
        "a partial block must never be pushed"
    );

    // An exactly-fitting block still fits without bumping the counter.
    let exact = vec![3.0f32; 4];
    push_block_drop_count(&mut prod, &exact, &dropped);
    assert_eq!(prod.occupied_len(), 64);
    assert_eq!(dropped.load(Ordering::Relaxed), 8);

    // Ring contents are intact (dropped block is nowhere to be seen).
    let mut buf = [0.0f32; 64];
    let got = cons.pop_slice(&mut buf);
    assert_eq!(got, 64);
    assert!(buf[..60].iter().all(|&s| s == 1.0));
    assert!(buf[60..].iter().all(|&s| s == 3.0));

    // After draining, pushes succeed again with no further drops.
    push_block_drop_count(&mut prod, &block, &dropped);
    assert_eq!(dropped.load(Ordering::Relaxed), 8);
    assert_eq!(prod.occupied_len(), 8);
}
