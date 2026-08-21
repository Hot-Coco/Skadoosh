#!/usr/bin/env bash
# Downloads the model and fixture files for skadoosh.
#
#   models/silero_vad.onnx      Silero VAD v5 (~2 MB)     — snakers4/silero-vad, tag v5.1.2
#   models/ggml-tiny.en.bin     Whisper tiny.en (~74 MB)  — ggerganov/whisper.cpp on Hugging Face
#   tests/data/jfk.wav          STT/VAD fixture (~344 KB) — whisper.cpp samples
#
# With --with-kokoro (opt-in, ~320 MB):
#
#   models/kokoro-v0_19.onnx    Kokoro-82M fp32 (~310 MB) — thewh1teagle/kokoro-onnx, release "model-files"
#   models/voices.bin           Kokoro voice style bank (~5.5 MB, 2D [n,256] f32 npy) — same release
#
# With --with-rag (opt-in, ~90 MB) — enables retrieval-augmented generation
# (`--rag-dir`):
#
#   models/all-MiniLM-L6-v2.onnx          all-MiniLM-L6-v2 sentence embedder (~90 MB)
#       — sentence-transformers/all-MiniLM-L6-v2, onnx/model.onnx
#   models/all-MiniLM-L6-v2-vocab.txt     BERT uncased WordPiece vocab (~231 KB)
#       — same repo, vocab.txt
#
# The script is idempotent: existing files at or above their expected minimum
# size are skipped; undersized files are re-downloaded; downloads that end up
# too small fail loudly. RAG URLs verified 2026-08-21; others 2026-08-18.

set -euo pipefail

WITH_KOKORO=0
WITH_RAG=0
for arg in "$@"; do
    case "$arg" in
        --with-kokoro) WITH_KOKORO=1 ;;
        --with-rag) WITH_RAG=1 ;;
        -h | --help)
            echo "usage: $0 [--with-kokoro] [--with-rag]" >&2
            exit 0
            ;;
        *)
            echo "unknown argument: $arg (usage: $0 [--with-kokoro] [--with-rag])" >&2
            exit 2
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models"
DATA_DIR="$ROOT_DIR/tests/data"
mkdir -p "$MODELS_DIR" "$DATA_DIR"

file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# download <url> <dest> <min_bytes>
download() {
    local url="$1" dest="$2" min_bytes="$3" size

    if [[ -f "$dest" ]]; then
        size="$(file_size "$dest")"
        if ((size >= min_bytes)); then
            echo "ok:      $dest ($size bytes) — already present, skipping"
            return 0
        fi
        echo "warn:    $dest exists but is too small ($size < $min_bytes) — re-downloading"
        rm -f "$dest"
    fi

    echo "fetch:   $url"
    echo "     ->  $dest"
    curl -fL --retry 3 --connect-timeout 15 -o "$dest.tmp" "$url"
    mv "$dest.tmp" "$dest"

    size="$(file_size "$dest")"
    if ((size < min_bytes)); then
        echo "error:   $dest is suspiciously small ($size < $min_bytes bytes);" >&2
        echo "         the upstream URL may have moved: $url" >&2
        rm -f "$dest"
        exit 1
    fi
    echo "ok:      $dest ($size bytes)"
}

# Note: at tag v5.1.2 the model lives under src/silero_vad/data/ (the files/
# directory only ships the logo). Fallback mirror: the same file is attached
# to the snakers4/silero-vad v5.1.2 release page.
download \
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx" \
    "$MODELS_DIR/silero_vad.onnx" \
    1000000

download \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" \
    "$MODELS_DIR/ggml-tiny.en.bin" \
    70000000

# whisper.cpp now lives in the ggml-org GitHub org (ggerganov/whisper.cpp redirects).
download \
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav" \
    "$DATA_DIR/jfk.wav" \
    100000

if ((WITH_KOKORO)); then
    # The "model-files" release pairs kokoro-v0_19.onnx with voices.bin.
    # (voices-v1.0.bin only exists in the separate model-files-v1.0 release,
    # paired with kokoro-v1.0.onnx.)
    download \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" \
        "$MODELS_DIR/kokoro-v0_19.onnx" \
        300000000

    download \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" \
        "$MODELS_DIR/voices.bin" \
        1000000
fi

if ((WITH_RAG)); then
    # all-MiniLM-L6-v2 sentence embedder for --rag-dir. The ONNX export
    # outputs last_hidden_state [B,S,384] (mean-pooled in Rust); the vocab is
    # the standard BERT uncased WordPiece list (30522 tokens).
    download \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" \
        "$MODELS_DIR/all-MiniLM-L6-v2.onnx" \
        90000000

    download \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" \
        "$MODELS_DIR/all-MiniLM-L6-v2-vocab.txt" \
        200000
fi

echo "done."
