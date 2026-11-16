#!/usr/bin/env bash
# Fetch Metrica Sports' public sample data (3 matches, tracking + events, ~175 MB) at a pinned
# commit, into data/external/metrica-sample-data.
#
# Same provider as the challenge data, and Sample Game 3 uses the same FIFA/EPTS export as our
# ATD. The repository states no licence; fine for development and comparison, but ask the
# organisers before training a submitted model on it. See literature/README.md.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MTAH_DATA_DIR:-$ROOT/data}/external/metrica-sample-data"
COMMIT=e706dd506b360d69d9d123d5b8026e7294b13996

if [ ! -d "$DEST/.git" ]; then
  git clone --quiet https://github.com/metrica-sports/sample-data "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin "$COMMIT"
git -C "$DEST" checkout --quiet "$COMMIT"
echo "metrica sample data at $(git -C "$DEST" rev-parse --short HEAD) -> $DEST"
