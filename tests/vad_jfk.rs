//! Real-model VAD test: `silero_vad.onnx` over `tests/data/jfk.wav`
//! (plan §10 task 2.1/2.2 acceptance). Skips with a printed reason if the
//! model or fixture is absent (both are fetched by `scripts/download_models.sh`).

use std::path::Path;

use skadoosh::error::{SkadooshError, VadError};
use skadoosh::vad::{SileroVad, VadEvent, VadSegmenter, FRAME_LEN};

const VAD_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/silero_vad.onnx");
const JFK_WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jfk.wav");

/// Loads jfk.wav, asserting the expected 16 kHz / 16-bit / mono PCM contract,
/// and converts i16 samples to f32.
fn load_jfk() -> Vec<f32> {
    let reader = hound::WavReader::open(JFK_WAV).expect("failed to open jfk.wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "jfk.wav must be 16 kHz");
    assert_eq!(spec.channels, 1, "jfk.wav must be mono");
    assert_eq!(spec.bits_per_sample, 16, "jfk.wav must be 16-bit PCM");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    reader
        .into_samples::<i16>()
        .map(|s| f32::from(s.expect("jfk.wav sample read failed")) / 32768.0)
        .collect()
}

/// Splits audio into whole 512-sample frames, then appends `zeros` extra
/// silent frames (real streams keep running after the wav ends — this forces
/// the trailing-silence endpoint close).
fn framed(samples: &[f32], zeros: usize) -> Vec<[f32; FRAME_LEN]> {
    let mut frames: Vec<[f32; FRAME_LEN]> = samples
        .chunks_exact(FRAME_LEN)
        .map(|c| <[f32; FRAME_LEN]>::try_from(c).expect("chunk is FRAME_LEN"))
        .collect();
    frames.extend(std::iter::repeat_n([0.0; FRAME_LEN], zeros));
    frames
}

#[test]
fn silero_jfk_detects_speech_and_segments() {
    if !Path::new(VAD_MODEL).is_file() || !Path::new(JFK_WAV).is_file() {
        eprintln!(
            "skipping silero_jfk_detects_speech_and_segments: {VAD_MODEL} or {JFK_WAV} missing \
             (run scripts/download_models.sh)"
        );
        return;
    }
    let mut vad = SileroVad::new(Path::new(VAD_MODEL)).expect("failed to load silero_vad.onnx");

    // 10 frames of zeros must stay below 0.1 speech probability.
    for i in 0..10 {
        let prob = vad
            .process(&[0.0; FRAME_LEN])
            .expect("zero-frame inference");
        assert!(
            (0.0..=1.0).contains(&prob),
            "prob {prob} out of [0, 1] on zero frame {i}"
        );
        assert!(prob < 0.1, "zero frame {i} prob {prob} >= 0.1");
    }

    // Non-512 frames are rejected.
    let err = vad
        .process(&[0.0; 256])
        .expect_err("must reject short frame");
    assert!(matches!(
        err,
        SkadooshError::Vad(VadError::BadFrameLen {
            expected: FRAME_LEN,
            actual: 256
        })
    ));

    // Real speech: probs must cross the 0.5 threshold somewhere and the
    // segmenter must emit at least one Segment (preceded by SpeechStart).
    vad.reset_state();
    let mut segmenter = VadSegmenter::new(0.5, 300);
    let mut max_prob: f32 = 0.0;
    let mut events = Vec::new();
    for frame in framed(&load_jfk(), 16) {
        let prob = vad.process(&frame).expect("jfk inference");
        max_prob = max_prob.max(prob);
        if let Some(event) = segmenter.push(&frame, prob) {
            events.push(event);
        }
    }
    eprintln!("max speech prob on jfk.wav: {max_prob:.3}");
    assert!(
        max_prob > 0.5,
        "speech prob never crossed 0.5 on jfk.wav (max {max_prob})"
    );
    let n_segments = events
        .iter()
        .filter(|e| matches!(e, VadEvent::Segment(_)))
        .count();
    assert!(
        n_segments >= 1,
        "expected >= 1 segment, got events {events:?}"
    );
    let first_segment = events
        .iter()
        .position(|e| matches!(e, VadEvent::Segment(_)))
        .expect("no segment");
    assert!(
        events[..first_segment]
            .iter()
            .any(|e| matches!(e, VadEvent::SpeechStart)),
        "Segment without a preceding SpeechStart: {events:?}"
    );
    for event in &events {
        if let VadEvent::Segment(audio) = event {
            assert!(!audio.is_empty() && audio.len() % FRAME_LEN == 0);
        }
    }

    // State reset after the burst: zeros stay quiet again.
    vad.reset_state();
    let prob = vad
        .process(&[0.0; FRAME_LEN])
        .expect("post-reset inference");
    assert!(prob < 0.1, "post-reset zero-frame prob {prob} >= 0.1");
}
