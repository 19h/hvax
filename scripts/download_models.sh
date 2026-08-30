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
if [[ ! -f "$det" ]]; then
  curl -L --fail -o "$det" "$base/det_10g.onnx"
fi
if [[ ! -f "$rec" ]]; then
  curl -L --fail -o "$rec" "$base/w600k_r50.onnx"
fi
verify_sha256() {
  local expected="$1"
  local file="$2"
  local actual
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$file" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$file" | awk '{print $1}')"
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $file: expected $expected, got $actual" >&2
    exit 1
  fi
  echo "$file: OK"
}
verify_sha256 "$DET_SHA" "$det"
verify_sha256 "$REC_SHA" "$rec"
echo "models ok in $DEST"
