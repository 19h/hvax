#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${HVAX_MODELS_DIR:-$ROOT/models}"
mkdir -p "$DEST"
det="$DEST/det_10g.onnx"
rec="$DEST/w600k_r50.onnx"
DET_SHA="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
REC_SHA="4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"
base="https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l"
temporary=""
trap 'if [[ -n "$temporary" ]]; then rm -f "$temporary"; fi' EXIT
sha256_file() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  else
    shasum -a 256 "$file" | awk '{print $1}'
  fi
}
fetch_verified() {
  local expected="$1"
  local file="$2"
  local url="$3"
  local actual=""
  if [[ -f "$file" ]]; then
    actual="$(sha256_file "$file")"
    if [[ "$actual" == "$expected" ]]; then
      echo "$file: OK"
      return
    fi
    echo "SHA-256 mismatch for $file; downloading a verified replacement" >&2
  fi
  temporary="$(mktemp "$DEST/.$(basename "$file").XXXXXX")"
  if ! curl --fail --location --retry 3 --output "$temporary" "$url"; then
    rm -f "$temporary"
    temporary=""
    return 1
  fi
  actual="$(sha256_file "$temporary")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$temporary"
    temporary=""
    echo "SHA-256 mismatch for downloaded $file: expected $expected, got $actual" >&2
    return 1
  fi
  mv -f "$temporary" "$file"
  temporary=""
  echo "$file: OK"
}
fetch_verified "$DET_SHA" "$det" "$base/det_10g.onnx"
fetch_verified "$REC_SHA" "$rec" "$base/w600k_r50.onnx"
echo "models ok in $DEST"
