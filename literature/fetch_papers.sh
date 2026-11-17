#!/usr/bin/env bash
# Rebuild literature/papers/ and literature/text/ from sources.tsv.
# The PDFs and their extracted text are gitignored (not ours to redistribute), so a fresh clone
# runs this once. Needs curl and pdftotext (poppler-utils). Existing files are skipped.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/papers" "$HERE/text"
grep -v '^#' "$HERE/sources.tsv" | while IFS=$'\t' read -r name url; do
  [ -z "$name" ] && continue
  out="$HERE/papers/$name"
  if [ ! -s "$out" ]; then
    case "$url" in
      MANUAL*) echo "manual  $name  <- ${url#MANUAL }"; continue ;;
    esac
    curl -sSL -A "Mozilla/5.0" --max-time 90 -o "$out" "$url"
    if ! file -b "$out" | grep -q PDF; then
      echo "FAILED  $name  (not a PDF; try a browser: $url)"; rm -f "$out"; continue
    fi
    echo "fetched $name"
  fi
  txt="$HERE/text/${name%.pdf}.txt"
  [ -s "$txt" ] || pdftotext "$out" "$txt"
done
