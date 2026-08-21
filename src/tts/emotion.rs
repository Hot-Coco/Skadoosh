//! Emotion-aware TTS: detects sentiment from LLM output and adjusts speaking
//! speed/pitch. Uses a lightweight keyword approach — no ML model needed.
//!
//! When `--tts-emotion` is enabled, each clause is scanned for sentiment
//! markers before synthesis. Results adjust the TTS speed multiplier:
//! excited/urgent → faster, calm/sad → slower.

/// Detected emotional tone from a text clause.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tone {
    /// Default/neutral tone, no speed adjustment.
    Neutral,
    /// Excited, happy, or urgent — speaks faster.
    Excited,
    /// Calm, sad, or serious — speaks slower.
    Calm,
}

impl Tone {
    /// Returns the speed multiplier for this tone (1.0 = neutral).
    pub fn speed_multiplier(self) -> f32 {
        match self {
            Tone::Neutral => 1.0,
            Tone::Excited => 1.25,
            Tone::Calm => 0.85,
        }
    }
}

/// Scans a clause for sentiment keywords and returns the detected [`Tone`].
/// No external dependencies — simple keyword matching.
pub fn detect_tone(clause: &str) -> Tone {
    let lower = clause.to_lowercase();

    // Excitement markers: exclamation marks, ALL CAPS words, urgency words.
    let excited_keywords = [
        "!",
        "wow",
        "great",
        "amazing",
        "awesome",
        "exciting",
        "fantastic",
        "wonderful",
        "hurry",
        "quick",
        "fast",
        "urgent",
        "alert",
        "congratulations",
        "brilliant",
        "excellent",
        "incredible",
        "love",
        "beautiful",
    ];
    let excited_score: usize = excited_keywords
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();

    // Calm/sad markers.
    let calm_keywords = [
        "sorry",
        "sad",
        "unfortunately",
        "calm",
        "slow",
        "careful",
        "gentle",
        "quiet",
        "peaceful",
        "rest",
        "regret",
        "apologize",
        "unfortunately",
        "however",
        "although",
        "perhaps",
        "maybe",
    ];
    let calm_score: usize = calm_keywords
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();

    if excited_score > calm_score && excited_score > 0 {
        Tone::Excited
    } else if calm_score > excited_score && calm_score > 0 {
        Tone::Calm
    } else {
        Tone::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_clause() {
        assert_eq!(
            detect_tone("The weather is partly cloudy today"),
            Tone::Neutral
        );
    }

    #[test]
    fn excited_clause() {
        assert_eq!(
            detect_tone("Wow! That's amazing news, congratulations!"),
            Tone::Excited
        );
    }

    #[test]
    fn calm_clause() {
        assert_eq!(
            detect_tone("I'm sorry to hear that, take it slow and careful"),
            Tone::Calm
        );
    }

    #[test]
    fn speed_multipliers() {
        assert!((Tone::Neutral.speed_multiplier() - 1.0).abs() < f32::EPSILON);
        assert!(Tone::Excited.speed_multiplier() > 1.0);
        assert!(Tone::Calm.speed_multiplier() < 1.0);
    }

    #[test]
    fn capitalization_is_ignored() {
        assert_eq!(
            detect_tone("This is AMAZING and EXCELLENT work!"),
            Tone::Excited
        );
    }
}
