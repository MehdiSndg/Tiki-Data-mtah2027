"""IDSSE / DFL open data (Bassek et al., Sci Data 2025; CC BY 4.0) -> TrackingData with labels.

Seven Bundesliga 1/2 matches (2022/23) with per-frame ball status and ball possession in the raw
Sportec position XML. kloppy maps them to ball_state / ball_owning_team. Two traps handled here:
  * kloppy's default only_alive=True silently drops every dead-ball frame -> we pass False.
  * a missing BallPossession attribute becomes "away" in kloppy -> irrelevant when dead; when
    alive we keep it (it is the provider's own value), and count how often it happens.

Raw files are downloaded once to <data>/external/idsse/<match_id>/ and parsed from there.

Source: the Hugging Face mirror pysport/idsse-data (same files, CC BY 4.0). The original Figshare
host sits behind an AWS WAF browser challenge that blocks scripted downloads; if the mirror is
ever unavailable, download meta.xml / positions.xml by hand from Figshare into the folder above.
Each positions file is ~420 MB.
"""

import json
import urllib.request
from pathlib import Path

import numpy as np

from ..constants import A, D, H, X
from ..data import TrackingData
from ..paths import EXTERNAL, ensure

HF_API = "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main"
HF_FILE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
MATCHES = ("J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")


def _remote_files():
    with urllib.request.urlopen(HF_API, timeout=60) as resp:
        return {item["path"]: item.get("size", 0) for item in json.load(resp)}


def _remote_names(match_id, files=None):
    files = files or _remote_files()
    meta = [p for p in files if "matchinformation" in p and match_id in p]
    pos = [p for p in files if "positions_raw" in p and match_id in p]
    if len(meta) != 1 or len(pos) != 1:
        raise FileNotFoundError(f"{match_id}: expected one meta and one positions file on the mirror")
    return meta[0], pos[0]


def _download(url, dest):
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url} -> {dest.name}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                print(f"    {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    tmp.replace(dest)
    return dest


def download(match_id):
    folder = ensure(EXTERNAL / "idsse" / match_id)
    meta, track = folder / "meta.xml", folder / "positions.xml"
    if meta.is_file() and track.is_file():
        return meta, track
    meta_name, pos_name = _remote_names(match_id)
    _download(HF_FILE.format(meta_name), meta)
    _download(HF_FILE.format(pos_name), track)
    return meta, track


def load_idsse(match_id, limit=None):
    """Download (if needed) and convert one match. Labels: D if dead else H/A from possession."""
    from kloppy import sportec
    from kloppy.domain import BallState

    meta_path, track_path = download(match_id)
    ds = sportec.load_tracking(meta_data=str(meta_path), raw_data=str(track_path),
                               only_alive=False, limit=limit)
    # kloppy converts to its own normalised system (0-1) by default; the real pitch size is kept
    # on the dimensions object. Scale every coordinate back to metres from a corner origin.
    dims = ds.metadata.pitch_dimensions
    x0, x1 = dims.x_dim.min, dims.x_dim.max
    y0, y1 = dims.y_dim.min, dims.y_dim.max
    length = float(getattr(dims, "pitch_length", None) or (x1 - x0))
    width = float(getattr(dims, "pitch_width", None) or (y1 - y0))
    sx, sy = length / float(x1 - x0), width / float(y1 - y0)

    home, away = ds.metadata.teams
    players = list(home.players) + list(away.players)
    index = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), np.int8)

    n = len(ds.records)
    pos = np.full((n, len(players), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    frame = np.zeros(n, np.int64)
    period = np.zeros(n, np.int8)
    labels = np.full(n, X, np.int8)
    for t, fr in enumerate(ds.records):
        frame[t] = fr.frame_id
        period[t] = fr.period.id
        if fr.ball_coordinates is not None and fr.ball_coordinates.x is not None:
            ball[t] = ((fr.ball_coordinates.x - x0) * sx, (fr.ball_coordinates.y - y0) * sy)
        for player, pdata in fr.players_data.items():
            c = pdata.coordinates
            if c is not None and c.x is not None:
                pos[t, index[player.player_id]] = ((c.x - x0) * sx, (c.y - y0) * sy)
        if fr.ball_state == BallState.DEAD:
            labels[t] = D
        elif fr.ball_owning_team is not None:
            labels[t] = H if fr.ball_owning_team == home else A

    return TrackingData(
        name=f"idsse_{match_id}", fps=int(round(ds.metadata.frame_rate)), pitch=(length, width),
        frame=frame, period=period, players=pos, ball=ball, team=team, labels=labels,
        meta={"source": "idsse_dfl", "match_id": match_id, "home": home.name, "away": away.name},
    )


def cached_idsse(match_id, rebuild=False):
    if match_id == "WCSF":                      # 5 Nov: the Groningen match joins the DFL-like matches
        from .wcsf import cached_game
        return cached_game(rebuild)
    name = f"idsse_{match_id}"
    if not rebuild and TrackingData.exists(name):
        return TrackingData.load(name)
    data = load_idsse(match_id)
    data.save()
    return data


def file_sizes():
    """{match_id: (meta bytes, positions bytes)} from the mirror's file listing."""
    files = _remote_files()
    out = {}
    for mid in MATCHES:
        meta_name, pos_name = _remote_names(mid, files)
        out[mid] = (files[meta_name], files[pos_name])
    return out


def local_path(match_id):
    return Path(EXTERNAL / "idsse" / match_id)
