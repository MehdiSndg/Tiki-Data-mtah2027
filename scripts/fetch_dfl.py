"""Download the Sportec Open DFL dataset (Bassek et al. 2025) and verify every file by MD5.

    uv run python scripts/fetch_dfl.py                   # into data/external/dfl
    uv run python scripts/fetch_dfl.py --only matchinformation events   # skip the 2.5 GB positions

Seven Bundesliga / 2. Bundesliga matches, 25 Hz TRACAB tracking with a per-frame BallPossession
(1 home, 2 away) and BallStatus (0 dead, 1 alive) on the ball -- the same structure as this
challenge's ground truth, on other fixtures. CC BY 4.0: https://doi.org/10.6084/m9.figshare.28196177
See literature/README.md for why it is here and what to watch out for.

Files that already exist with the right MD5 are skipped, and an interrupted download resumes from
where it stopped, so re-running after a network failure is always safe. data/ is gitignored, but
the licence would allow redistribution anyway -- this one is not the gated match bundle.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

# Params
ARTICLE_API = "https://api.figshare.com/v2/articles/28196177"
OUT_DIR = os.path.join(DATA_DIR, "external", "dfl")
KINDS = ("matchinformation", "events", "positions")
CHUNK = 1 << 20
RETRIES = 5
# figshare rejects the default urllib user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (mtah2027 fetch_dfl.py)"}


def get(url, offset=0):
    headers = dict(HEADERS)
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, path):
    """Stream one file to path, resuming a partial .part file, with retries."""
    partial = path + ".part"
    for attempt in range(1, RETRIES + 1):
        offset = os.path.getsize(partial) if os.path.exists(partial) else 0
        if offset == entry["size"]:
            # Fully written but interrupted before the rename; a Range request would get a 416.
            os.replace(partial, path)
            return
        try:
            with get(entry["download_url"], offset) as response:
                # A server that ignores Range sends the whole file again: start over.
                mode = "ab" if offset and response.status == 206 else "wb"
                with open(partial, mode) as handle:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        handle.write(block)
            os.replace(partial, path)
            return
        except OSError as error:
            print("    attempt {}/{} failed: {}".format(attempt, RETRIES, error))
            time.sleep(5 * attempt)
    raise IOError("giving up on {}".format(entry["name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=KINDS, default=list(KINDS),
                        help="which file kinds to fetch")
    parser.add_argument("--out", default=OUT_DIR, help="where to put the files")
    args = parser.parse_args()

    with get(ARTICLE_API) as response:
        article = json.load(response)
    files = sorted((f for f in article["files"] if any(k in f["name"] for k in args.only)),
                   key=lambda f: f["name"])
    os.makedirs(args.out, exist_ok=True)
    print("{} files, {:.1f} MB -> {}".format(
        len(files), sum(f["size"] for f in files) / 2 ** 20, args.out))

    bad = []
    for entry in files:
        path = os.path.join(args.out, entry["name"])
        if os.path.exists(path) and md5(path) == entry["computed_md5"]:
            print("  have  {}".format(entry["name"]))
            continue
        print("  fetch {}  ({:.1f} MB)".format(entry["name"], entry["size"] / 2 ** 20))
        download(entry, path)
        if md5(path) != entry["computed_md5"]:
            bad.append(entry["name"])
            print("    MD5 MISMATCH -- delete it and re-run")

    print("\n{}".format("all files verified" if not bad else "bad: " + ", ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
