#!/usr/bin/env bash
# Download the trained weights the submission (v38) was built with from this repository's GitHub release and check
# them against SHA256SUMS. Our nets go to models/, the teammate's run outputs to third_party/mali/outputs/.
#   bash scripts/fetch_weights.sh <owner/repo> [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?owner/repo}; TAG=${2:-v38}
URL=https://github.com/$REPO/releases/download/$TAG
TMP=$(mktemp -d)
for a in SHA256SUMS models_team.tar models_video.tar mali_outputs.tar; do
  curl -fL --retry 3 -o "$TMP/$a" "$URL/$a"
done
tar -xf "$TMP/models_team.tar" -C "$TMP" && tar -xf "$TMP/models_video.tar" -C "$TMP" && tar -xf "$TMP/mali_outputs.tar" -C "$TMP"
(cd "$TMP" && sha256sum -c --quiet SHA256SUMS)
mkdir -p models third_party/mali/outputs
cp -r "$TMP/models/." models/
cp -r "$TMP/mali/." third_party/mali/outputs/
rm -rf "$TMP"
echo "weights in models/ and third_party/mali/outputs/ (sha256 checked)"
