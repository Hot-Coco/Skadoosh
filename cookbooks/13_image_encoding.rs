//! Cookbook 13 — Image encoding to a base64 data URI.
//!
//! Encodes an image file to a `data:<mime>;base64,…` URI with
//! [`image_to_data_uri`] — the format multimodal (vision) models accept in a
//! [`ContentBlock::Image`]. The MIME type is auto-detected from the file
//! extension. Uses a real 1×1 PNG; no server, no models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 13_image_encoding
//! ```

use std::path::PathBuf;

use base64::Engine;
use skadoosh::llm::{image_to_data_uri, ContentBlock, ImageUrl};

/// Wraps any `Display` error (io / base64) into the crate's umbrella error.
fn wrap<E: std::fmt::Display>(e: E) -> skadoosh::SkadooshError {
    anyhow::anyhow!("{e}").into()
}

/// A minimal valid 1×1 RGB PNG (generated offline), as base64.
const ONE_PX_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

fn main() -> skadoosh::Result<()> {
    // Materialize the tiny PNG on disk so image_to_data_uri can read it.
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(ONE_PX_PNG_B64)
        .map_err(wrap)?;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cookbook_13_image.png");
    std::fs::write(&path, &png_bytes).map_err(wrap)?;
    println!("wrote {} ({} bytes)", path.display(), png_bytes.len());

    // Encode it to a base64 data URI (MIME detected from the .png extension).
    let data_uri = image_to_data_uri(&path).map_err(wrap)?;
    let prefix = "data:image/png;base64,";
    assert!(
        data_uri.starts_with(prefix),
        "expected a png data URI, got: {data_uri}"
    );

    let encoded = &data_uri[prefix.len()..];
    println!("data URI: {prefix}…");
    println!("  mime   : image/png");
    println!("  encoded: {} base64 chars", encoded.len());

    // The encoded payload must round-trip back to the original bytes.
    let roundtrip = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(wrap)?;
    assert_eq!(roundtrip, png_bytes, "base64 round-trips losslessly");

    // Show it embedded in a ContentBlock::Image, as an LLM request would carry it.
    let block = ContentBlock::Image {
        image_url: ImageUrl {
            url: data_uri,
            detail: Some("auto".to_string()),
        },
    };
    let json = serde_json::to_string(&block).map_err(wrap)?;
    println!("\nas ContentBlock::Image:\n  {json}");

    // Cleanup.
    let _ = std::fs::remove_file(&path);

    println!("\n13_image_encoding: OK");
    Ok(())
}
