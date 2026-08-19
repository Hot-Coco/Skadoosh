//! Scripted-probability tests for the pure `VadSegmenter` state machine
//! (plan §10 task 2.2 acceptance: onset fires; close only after silence_ms;
//! preroll included; sub-min blip rejected; second burst → second Segment).

use skadoosh::vad::{VadEvent, VadSegmenter, FRAME_LEN};

/// One frame filled with a constant marker value.
fn frame(value: f32) -> [f32; FRAME_LEN] {
    [value; FRAME_LEN]
}

/// Pushes `n` copies of a frame/prob pair, collecting any events.
fn push_n(seg: &mut VadSegmenter, frame: &[f32; FRAME_LEN], prob: f32, n: usize) -> Vec<VadEvent> {
    (0..n).filter_map(|_| seg.push(frame, prob)).collect()
}

/// Asserts a 512-sample block of `audio` starting at frame `f` is constant.
fn assert_frame_eq(audio: &[f32], f: usize, value: f32) {
    assert!(
        audio[f * FRAME_LEN..(f + 1) * FRAME_LEN]
            .iter()
            .all(|&x| x == value),
        "frame {f} of segment is not all {value}"
    );
}

#[test]
fn onset_fires_speech_start() {
    let mut seg = VadSegmenter::new(0.5, 300);
    for _ in 0..5 {
        assert_eq!(seg.push(&frame(0.0), 0.0), None);
    }
    assert_eq!(seg.push(&frame(0.6), 0.9), Some(VadEvent::SpeechStart));
    // Threshold is inclusive (first frame ≥ threshold).
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(0.6), 0.5), Some(VadEvent::SpeechStart));
}

#[test]
fn closes_only_after_silence_ms() {
    // 300 ms at 32 ms/frame → closes on the 10th trailing silent frame
    // (9 × 32 = 288 ≤ 300; 10 × 32 = 320 > 300).
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 8).is_empty());
    for i in 1..=9 {
        assert!(
            seg.push(&frame(0.0), 0.0).is_none(),
            "segment closed early at silent frame {i}"
        );
    }
    match seg.push(&frame(0.0), 0.0) {
        Some(VadEvent::Segment(_)) => {}
        other => panic!("expected Segment on 10th silent frame, got {other:?}"),
    }
}

#[test]
fn silence_boundary_uses_exact_ms_math() {
    // silence_ms = 320 = exactly 10 frames: 10 × 32 = 320 is NOT > 320, so
    // the segment must still be open; it closes on the 11th silent frame.
    let mut seg = VadSegmenter::new(0.5, 320);
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 8).is_empty());
    for i in 1..=10 {
        assert!(
            seg.push(&frame(0.0), 0.0).is_none(),
            "closed at silent frame {i} with exactly 320 ms of silence"
        );
    }
    match seg.push(&frame(0.0), 0.0) {
        Some(VadEvent::Segment(_)) => {}
        other => panic!("expected Segment on 11th silent frame, got {other:?}"),
    }
}

#[test]
fn segment_includes_preroll() {
    let mut seg = VadSegmenter::new(0.5, 300);
    // 12 distinct pre-onset frames; the ring of 8 keeps values 4.0..=11.0.
    for i in 0..12 {
        assert_eq!(seg.push(&frame(i as f32), 0.0), None);
    }
    assert_eq!(seg.push(&frame(100.0), 0.9), Some(VadEvent::SpeechStart));
    // 8 more speech frames → 9 active frames total.
    assert!(push_n(&mut seg, &frame(50.0), 0.9, 8).is_empty());
    // 10 trailing silent frames → close.
    let events = push_n(&mut seg, &frame(0.0), 0.0, 10);
    let [VadEvent::Segment(audio)] = &events[..] else {
        panic!("expected exactly one Segment, got {events:?}");
    };
    assert_eq!(audio.len(), (8 + 9 + 10) * FRAME_LEN);
    for f in 0..8 {
        assert_frame_eq(audio, f, 4.0 + f as f32);
    }
    assert_frame_eq(audio, 8, 100.0);
    for f in 9..17 {
        assert_frame_eq(audio, f, 50.0);
    }
    for f in 17..27 {
        assert_frame_eq(audio, f, 0.0);
    }
}

#[test]
fn sub_min_length_blip_rejected() {
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    // 3 active frames total (96 ms < 250 ms min) then silence → rejected.
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 2).is_empty());
    for i in 0..12 {
        assert!(
            seg.push(&frame(0.0), 0.0).is_none(),
            "blip emitted an event at frame {i}"
        );
    }
}

#[test]
fn min_length_boundary() {
    // 7 active frames = 224 ms < 250 ms → rejected.
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 6).is_empty());
    assert!(push_n(&mut seg, &frame(0.0), 0.0, 10).is_empty());

    // 8 active frames = 256 ms ≥ 250 ms → emitted.
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 7).is_empty());
    let events = push_n(&mut seg, &frame(0.0), 0.0, 10);
    let [VadEvent::Segment(audio)] = &events[..] else {
        panic!("expected Segment for 256 ms burst, got {events:?}");
    };
    // Onset was the very first frame, so the preroll ring was empty:
    // 8 speech frames + 10 trailing silent frames.
    assert_eq!(audio.len(), (8 + 10) * FRAME_LEN);
}

#[test]
fn second_burst_yields_second_segment() {
    let mut seg = VadSegmenter::new(0.5, 300);
    // Burst 1.
    assert_eq!(seg.push(&frame(1.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(1.0), 0.9, 8).is_empty());
    let events = push_n(&mut seg, &frame(0.0), 0.0, 10);
    let [VadEvent::Segment(_)] = &events[..] else {
        panic!("expected first Segment, got {events:?}");
    };
    // Burst 2 right after: back in Listening, onset fires again, and the
    // preroll is the trailing silence of burst 1 (8 zero frames).
    assert_eq!(seg.push(&frame(2.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(2.0), 0.9, 8).is_empty());
    let events = push_n(&mut seg, &frame(0.0), 0.0, 10);
    let [VadEvent::Segment(audio)] = &events[..] else {
        panic!("expected second Segment, got {events:?}");
    };
    assert_eq!(audio.len(), (8 + 9 + 10) * FRAME_LEN);
    for f in 0..8 {
        assert_frame_eq(audio, f, 0.0);
    }
    assert_frame_eq(audio, 8, 2.0);
}

#[test]
fn preroll_partially_filled_at_stream_start() {
    // Onset after only 3 frames: preroll contains exactly those 3.
    let mut seg = VadSegmenter::new(0.5, 300);
    assert_eq!(seg.push(&frame(7.0), 0.0), None);
    assert_eq!(seg.push(&frame(8.0), 0.0), None);
    assert_eq!(seg.push(&frame(9.0), 0.0), None);
    assert_eq!(seg.push(&frame(100.0), 0.9), Some(VadEvent::SpeechStart));
    assert!(push_n(&mut seg, &frame(50.0), 0.9, 8).is_empty());
    let events = push_n(&mut seg, &frame(0.0), 0.0, 10);
    let [VadEvent::Segment(audio)] = &events[..] else {
        panic!("expected Segment, got {events:?}");
    };
    assert_eq!(audio.len(), (3 + 9 + 10) * FRAME_LEN);
    assert_frame_eq(audio, 0, 7.0);
    assert_frame_eq(audio, 1, 8.0);
    assert_frame_eq(audio, 2, 9.0);
    assert_frame_eq(audio, 3, 100.0);
}
