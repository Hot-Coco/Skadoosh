//! Cookbook 15 — Push-to-talk.
//!
//! Demonstrates the push-to-talk config ([`Config::push_to_talk`]) and the
//! record-gating state machine it enables: press-and-hold to record, release
//! to send the segment. This overrides VAD-based segmentation — instead of
//! detecting speech/silence, the user explicitly bounds each utterance.
//!
//! The simulation mirrors `src/pipeline.rs`'s `push_to_talk_task`: a rising
//! edge (press) clears the accumulator and starts a turn; a falling edge
//! (release) emits the accumulated samples as a segment. No server, no
//! models, no audio device.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 15_push_to_talk
//! ```

use skadoosh::Config;

/// One input event for the push-to-talk simulation.
enum Event {
    /// Key pressed (start recording) / released (stop recording).
    Press,
    Release,
    /// A chunk of captured audio samples arriving while recording.
    Samples(usize),
}

fn main() -> skadoosh::Result<()> {
    let config = Config {
        push_to_talk: true,
        ..Config::default()
    };

    println!("config.push_to_talk = {}", config.push_to_talk);
    println!("(press-and-hold to record, release to send)\n");

    // A scripted interaction: two held-and-released utterances.
    let events = [
        Event::Press,
        Event::Samples(16_000), // ~1 s of 16 kHz audio
        Event::Samples(16_000),
        Event::Release, // -> send segment
        Event::Press,
        Event::Samples(8_000), // ~0.5 s
        Event::Release,        // -> send segment
        Event::Press,
        Event::Release, // tap with no audio -> nothing sent
    ];

    // The state machine (same edges as push_to_talk_task).
    let mut recording = false;
    let mut was_recording = false;
    let mut accumulating: usize = 0;
    let mut segments: Vec<usize> = Vec::new();

    for event in events {
        match event {
            Event::Press => recording = true,
            Event::Release => recording = false,
            Event::Samples(n) => {
                if recording {
                    accumulating += n;
                }
            }
        }

        // Rising edge: a new turn starts; clear the accumulator.
        if recording && !was_recording {
            println!("[rec] recording started (buffer cleared)");
            accumulating = 0;
        }

        // Falling edge: release sends whatever was accumulated.
        if !recording && was_recording {
            if accumulating > 0 {
                println!("[send] released -> segment of {accumulating} samples");
                segments.push(accumulating);
            } else {
                println!("[skip] released with no audio -> nothing sent");
            }
        }

        was_recording = recording;
    }

    println!("\nsent {} segment(s): {:?}", segments.len(), segments);

    assert_eq!(
        segments,
        vec![32_000, 8_000],
        "two utterances sent on release"
    );
    assert!(config.push_to_talk, "push_to_talk enabled");

    println!("\n15_push_to_talk: OK");
    Ok(())
}
