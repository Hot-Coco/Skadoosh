//! Real-model STT test: `ggml-tiny.en.bin` over `tests/data/jfk.wav`
//! (plan §10 task 3.1 acceptance). Skips with a printed reason if the model
//! or fixture is absent (both are fetched by `scripts/download_models.sh`).

use std::path::Path;
use std::time::{Duration, Instant};

use skadoosh::stt::{SttConfig, WhisperStt};

const WHISPER_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/ggml-tiny.en.bin");
const JFK_WAV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/jfk.wav");

const TIMEOUT: Duration = Duration::from_secs(30);

fn fixtures_present() -> bool {
    let present = Path::new(WHISPER_MODEL).is_file() && Path::new(JFK_WAV).is_file();
    if !present {
        eprintln!(
            "skipping STT test: {WHISPER_MODEL} or {JFK_WAV} missing \
             (run scripts/download_models.sh)"
        );
    }
    present
}

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

/// Lowercases and strips punctuation so transcript assertions don't depend on
/// whisper's exact casing/comma choices.
fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_punctuation() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[tokio::test]
async fn whisper_jfk_transcribes_ask_not() {
    if !fixtures_present() {
        return;
    }
    let stt = WhisperStt::start(Path::new(WHISPER_MODEL), &SttConfig::default())
        .expect("failed to load ggml-tiny.en.bin");
    let started = Instant::now();
    let text = tokio::time::timeout(TIMEOUT, stt.transcribe(load_jfk()))
        .await
        .expect("transcription exceeded 30 s")
        .expect("worker reply channel closed")
        .expect("transcription failed");
    let elapsed = started.elapsed();
    eprintln!("transcript in {elapsed:?}: {text}");
    assert!(elapsed < TIMEOUT, "transcription took {elapsed:?} (> 30 s)");
    assert!(
        normalize(&text).contains("ask not what your country"),
        "transcript missing the JFK phrase: {text:?}"
    );
    stt.stop();
}

#[tokio::test]
async fn stop_joins_worker_thread_promptly() {
    if !fixtures_present() {
        return;
    }
    let stt = WhisperStt::start(Path::new(WHISPER_MODEL), &SttConfig::default())
        .expect("failed to load ggml-tiny.en.bin");
    // start() only returns once the model is loaded and the worker is parked
    // in recv(), so stop() must join essentially immediately.
    let started = Instant::now();
    stt.stop();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "stop() join took {elapsed:?} (> 2 s)"
    );
}

#[tokio::test]
async fn full_queue_drops_oldest_and_counts() {
    if !fixtures_present() {
        return;
    }
    let stt = WhisperStt::start(Path::new(WHISPER_MODEL), &SttConfig::default())
        .expect("failed to load ggml-tiny.en.bin");
    let samples = load_jfk();
    // jfk takes ~seconds per decode on tiny.en while these submits take
    // microseconds, so the bounded queue (capacity 2, worker busy with job 1)
    // is necessarily full by the time later jobs arrive.
    let mut replies = Vec::new();
    for _ in 0..4 {
        replies.push(stt.transcribe(samples.clone()));
    }
    let mut answered = 0usize;
    for reply in replies {
        match tokio::time::timeout(TIMEOUT, reply).await {
            Ok(Ok(Ok(text))) => {
                assert!(
                    normalize(&text).contains("ask not what your country"),
                    "transcript missing the JFK phrase: {text:?}"
                );
                answered += 1;
            }
            Ok(Err(_)) => {} // evicted before the worker saw it: expected
            other => panic!("unexpected job outcome: {other:?}"),
        }
    }
    let dropped = stt.dropped_jobs();
    eprintln!("dropped {dropped} of 4 queued jfk jobs, answered {answered}");
    assert!(dropped >= 1, "expected at least one dropped job");
    assert_eq!(answered + dropped as usize, 4);
    stt.stop();
}
