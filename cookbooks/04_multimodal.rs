//! Cookbook 04 — Multimodal message construction.
//!
//! Builds an OpenAI-compatible multimodal user message combining a text
//! block and an image block ([`ContentBlock::Text`] + [`ContentBlock::Image`]).
//! The image is a real 1×1 PNG, encoded to a `data:image/png;base64,…` URI
//! with [`image_to_data_uri`], exactly as [`LlmClient`] would send it to a
//! vision model. The assembled [`Message`] is serialized to JSON to show the
//! wire shape. No server, no models.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example 04_multimodal
//! ```

use std::path::PathBuf;

use base64::Engine;
use skadoosh::llm::{image_to_data_uri, ContentBlock, ImageUrl, Message, MessageContent};

/// Wraps any `Display` error into the crate's umbrella error for `?`.
fn wrap<E: std::fmt::Display>(e: E) -> skadoosh::SkadooshError {
    anyhow::anyhow!("{e}").into()
}

/// A minimal valid 1×1 RGB PNG (generated offline), as base64.
const ONE_PX_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

fn main() -> skadoosh::Result<()> {
    // 1. Materialize the tiny PNG on disk so image_to_data_uri can read it.
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(ONE_PX_PNG_B64)
        .map_err(wrap)?;
    let img_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/cookbook_04_pixel.png");
    std::fs::write(&img_path, &png_bytes).map_err(wrap)?;

    // 2. Encode it to a base64 data URI (auto-detects MIME from the extension).
    let data_uri = image_to_data_uri(&img_path).map_err(wrap)?;
    assert!(
        data_uri.starts_with("data:image/png;base64,"),
        "expected a png data URI, got: {data_uri}"
    );
    println!("image data URI ({} bytes):", data_uri.len());
    println!("  {}…", &data_uri[..data_uri.len().min(60)]);

    // 3. Assemble the multimodal content: a text prompt followed by the image.
    let content = MessageContent::Blocks(vec![
        ContentBlock::Text {
            text: "What is in this image?".to_string(),
        },
        ContentBlock::Image {
            image_url: ImageUrl {
                url: data_uri,
                detail: Some("auto".to_string()),
            },
        },
    ]);

    let message = Message {
        role: "user".to_string(),
        content,
        tool_call_id: None,
        tool_calls: None,
    };

    // 4. Serialize to JSON to inspect the OpenAI-compatible shape: content is
    //    a JSON array of typed blocks (`{"type":"text",...}` / `{"type":"image_url",...}`).
    let json = serde_json::to_string_pretty(&message).map_err(wrap)?;
    println!("multimodal message JSON:\n{json}");

    assert!(json.contains("\"type\": \"text\""), "text block present");
    assert!(
        json.contains("\"type\": \"image_url\""),
        "image block present"
    );
    assert!(
        json.contains("data:image/png;base64,"),
        "data URI embedded in the image block"
    );

    // Cleanup the temp image.
    let _ = std::fs::remove_file(&img_path);

    println!("04_multimodal: OK");
    Ok(())
}
