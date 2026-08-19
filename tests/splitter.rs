//! Clause splitter boundary/UTF-8 tests (plan task 4.1 acceptance).

use skadoosh::llm::ClauseSplitter;

/// Default bounds used by the LLM client (min 4, max 160); tests shrink
/// `max_len` where the hard flush is under test.
fn splitter() -> ClauseSplitter {
    ClauseSplitter::new(4, 160)
}

#[test]
fn emits_on_each_boundary_char() {
    for (text, boundary) in [
        ("Hi there.", '.'),
        ("Hi there?", '?'),
        ("Hi there!", '!'),
        ("Hi there, friend", ','),
    ] {
        let mut s = splitter();
        let out = s.push(text);
        assert!(
            !out.is_empty(),
            "no clause emitted for boundary {boundary:?} in {text:?}"
        );
        assert!(
            out[0].ends_with(boundary),
            "clause {:?} should end with {boundary:?}",
            out[0]
        );
    }
}

#[test]
fn boundary_requires_min_len() {
    let mut s = splitter();
    // The only boundary (".", 1 char) is below min_len (4): nothing emitted.
    assert!(s.push(".ab").is_empty());
    // More text lets the *second* boundary satisfy min_len; the splitter
    // scans for the first boundary whose clause is long enough.
    let out = s.push("cd.");
    assert_eq!(out, vec![".abcd.".to_string()]);
}

#[test]
fn fragmented_tokens_merge_into_one_clause() {
    let mut s = splitter();
    assert!(s.push("hel").is_empty());
    let out = s.push("lo.");
    assert_eq!(out, vec!["hello.".to_string()]);
}

#[test]
fn multiple_boundaries_emit_in_order_in_one_push() {
    let mut s = splitter();
    let out = s.push("One. Two? Three!");
    assert_eq!(
        out,
        vec![
            "One.".to_string(),
            " Two?".to_string(),
            " Three!".to_string()
        ]
    );
    assert!(s.flush().is_none());
}

#[test]
fn max_len_flush_breaks_at_last_whitespace() {
    let mut s = ClauseSplitter::new(4, 20);
    // 22 chars, no boundary chars; last whitespace within the first 20 is
    // at char index 17.
    let out = s.push("aa bb cc dd ee ff gg h");
    assert_eq!(out, vec!["aa bb cc dd ee ff".to_string()]);
    // The whitespace the clause broke on is consumed; remainder stays.
    assert_eq!(s.flush(), Some("gg h".to_string()));
}

#[test]
fn max_len_hard_cut_without_whitespace() {
    let mut s = ClauseSplitter::new(4, 20);
    let out = s.push("aaaaaaaaaaaaaaaaaaaaaa"); // 22 a's
    assert_eq!(out, vec!["a".repeat(20)]);
    assert_eq!(s.flush(), Some("aa".to_string()));
}

#[test]
fn max_len_never_splits_multibyte_chars() {
    // 7 emoji (4 bytes each), max_len 4 chars: hard cut after 4 emoji.
    let mut s = ClauseSplitter::new(4, 4);
    let out = s.push("😀😀😀😀😀😀😀");
    assert_eq!(out, vec!["😀😀😀😀".to_string()]);
    assert_eq!(s.flush(), Some("😀😀😀".to_string()));

    // CJK: 9 chars, max_len 4 → "你好世界" + "测试字符" + flush("串").
    let mut s = ClauseSplitter::new(4, 4);
    let text = "你好世界测试字符串";
    let mut all = s.push(text);
    if let Some(rest) = s.flush() {
        all.push(rest);
    }
    assert_eq!(all.concat(), text, "reassembly must be exact");
    assert!(all.iter().all(|c| c.chars().count() <= 4));
}

#[test]
fn flush_drains_remainder_exactly_once() {
    let mut s = splitter();
    assert!(s.push("hello").is_empty());
    assert_eq!(s.flush(), Some("hello".to_string()));
    assert_eq!(s.flush(), None, "second flush must be empty");
    // Splitter stays usable after a flush.
    let out = s.push(" world.");
    assert_eq!(out, vec![" world.".to_string()]);
    assert_eq!(s.flush(), None);
}

#[test]
fn flush_drops_whitespace_only_remainder() {
    let mut s = splitter();
    let out = s.push("Hiya. ");
    assert_eq!(out, vec!["Hiya.".to_string()]);
    assert_eq!(s.flush(), None, "whitespace-only remainder is dropped");
}

#[test]
fn clauses_reassemble_to_input_for_punctuated_text() {
    let mut s = splitter();
    let mut all = Vec::new();
    for chunk in ["Hello, ", "world. ", "How are ", "you?"] {
        all.extend(s.push(chunk));
    }
    if let Some(rest) = s.flush() {
        all.push(rest);
    }
    assert_eq!(all.concat(), "Hello, world. How are you?");
}
