#!/usr/bin/env bash
# Download the trained weights the submission (v28) was built with from this repository's GitHub release and check
# them against SHA256SUMS. Our nets go to models/, the teammate's delivered probabilities to cache/team_in/mehmet_ali_topkara/barath/.
#   bash scripts/fetch_weights.sh <owner/repo> [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?owner/repo}; TAG=${2:-v28}
URL=https://github.com/$REPO/releases/download/$TAG
TMP=$(mktemp -d)
for a in SHA256SUMS models_team.tar models_video.tar mali_delivered.tar; do
  curl -fL --retry 3 -o "$TMP/$a" "$URL/$a"
done
tar -xf "$TMP/models_team.tar" -C "$TMP" && tar -xf "$TMP/models_video.tar" -C "$TMP" && tar -xf "$TMP/mali_delivered.tar" -C "$TMP"
(cd "$TMP" && sha256sum -c --quiet SHA256SUMS)
mkdir -p models cache/team_in/mehmet_ali_topkara/barath
cp -r "$TMP/models/." models/
cp "$TMP/mali_delivered/components.npz" cache/team_in/mehmet_ali_topkara/barath/
rm -rf "$TMP"
echo "weights in models/ and cache/team_in/mehmet_ali_topkara/barath/ (sha256 checked)"
