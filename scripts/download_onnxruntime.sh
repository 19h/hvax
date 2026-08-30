#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
version="${HVAX_ORT_VERSION:-1.29.0}"
arch="$(uname -m)"
os="$(uname -s)"
if [[ "$os/$arch" != "Darwin/arm64" ]]; then
  echo "this bootstrap downloads the CoreML-enabled macOS arm64 package; got $os/$arch" >&2
  exit 1
fi
package="onnxruntime-osx-arm64-$version"

if [[ "$version" == "1.29.0" ]]; then
  expected_sha256="d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92"
elif [[ -n "${HVAX_ORT_SHA256:-}" ]]; then
  expected_sha256="$HVAX_ORT_SHA256"
else
  echo "set HVAX_ORT_SHA256 when overriding HVAX_ORT_VERSION" >&2
  exit 1
fi

destination="$root/third_party/$package"
if [[ -f "$destination/include/onnxruntime_cxx_api.h" && -f "$destination/lib/libonnxruntime.dylib" ]]; then
  echo "ONNX Runtime already present in $destination"
  exit 0
fi

archive="$(mktemp "${TMPDIR:-/tmp}/$package.XXXXXX.tgz")"
trap 'rm -f "$archive"' EXIT
url="https://github.com/microsoft/onnxruntime/releases/download/v$version/$package.tgz"
curl --fail --location --retry 3 --output "$archive" "$url"
actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "ONNX Runtime SHA-256 mismatch: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi
tar -xzf "$archive" -C "$root/third_party"
test -f "$destination/include/onnxruntime_cxx_api.h"
echo "ONNX Runtime $version installed in $destination"
