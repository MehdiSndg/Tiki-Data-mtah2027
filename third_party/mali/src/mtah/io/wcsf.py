"""WCSF 2023 Solution Room match (University of Groningen; SciSports-processed) -> TrackingData at 25 fps.

Source: github.com/MatKempeGroningen/WCSF_SolutionRoom (README: data collected for research; please cite
the data source). One anonymised professional European league match; optical tracking 25 Hz, provided
downsampled to 10 Hz and smoothed; 22 players + ball on every frame (no ball gaps: clean optical data, so
it is corrupted like DFL). Files under EXTERNAL/wcsf/: positions_FIRST_HALF.csv.zip,
positions_SECOND_HALF.csv.zip (read inside the zip).

Coordinates are MIRRORED per team (own goal always at x = -52.5): the away team's frame is the home
frame rotated by 180 degrees (checked: away-row ball = - home-row ball on every frame). We keep the home
frame (away players and nothing else negated), centre origin -> corner origin, pitch 105 x 68 m.

Labels: provider channels, like the target's ground truth: AliveFlag 0 -> D; otherwise PossessionFlag
1 -> H (group 1 = home), 2 -> A. Per frame the flags are taken from the home rows.
"""

import zipfile

import numpy as np
import pandas as pd

from ..constants import A, D, H
from ..data import TrackingData
from ..paths import EXTERNAL

PITCH = (105.0, 68.0)
FILES = ("positions_FIRST_HALF.csv.zip", "positions_SECOND_HALF.csv.zip")
NAME = "WCSF"


def root():
    return EXTERNAL / "wcsf"


def _read(path):
    with zipfile.ZipFile(path) as z:
        member = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")]
        if len(member) != 1:
            raise ValueError(f"{path}: expected one csv, found {member}")
        with z.open(member[0]) as fh:
            return pd.read_csv(fh, index_col=0)


def half_arrays(df, slot_of):
    """10 Hz arrays of one half: players (S, N, 2) home frame, ball (S, 2), labels (S,)."""
    ts = np.sort(df.timestamp_ms.unique())
    if np.any(np.diff(ts) != 100):
        raise ValueError("WCSF frames are not a contiguous 10 Hz grid")
    t_idx = {t: i for i, t in enumerate(ts)}
    rows = df.timestamp_ms.map(t_idx).to_numpy()
    sign = np.where(df.group_id.to_numpy() == 2, -1.0, 1.0)
    players = np.full((len(ts), len(slot_of), 2), np.nan, np.float32)
    slots = df.player_id.map(slot_of).to_numpy()
    players[rows, slots, 0] = sign * df.x.to_numpy()
    players[rows, slots, 1] = sign * df.y.to_numpy()
    home = df[df.group_id == 1].groupby("timestamp_ms").first()
    home = home.reindex(ts)
    ball = home[["x_ball", "y_ball"]].to_numpy(np.float32)
    alive, poss = home.AliveFlag.to_numpy(), home.PossessionFlag.to_numpy()
    labels = np.where(alive == 0, D, np.where(poss == 1, H, A)).astype(np.int8)
    return players, ball, labels


def load_game(base=None):
    from .skillcorner import upsample
    base = base or root()
    halves = [_read(base / f) for f in FILES]
    ids = pd.concat([h[["player_id", "group_id"]] for h in halves]).drop_duplicates("player_id")
    ids = ids.sort_values(["group_id", "player_id"])
    slot_of = {p: i for i, p in enumerate(ids.player_id)}
    team = (ids.group_id.to_numpy() == 2).astype(np.int8)
    parts = []
    for k, df in enumerate(halves, start=1):
        pl, b, lab = half_arrays(df, slot_of)
        n = len(lab)
        _, per, pl, b, lab = upsample(np.arange(n), np.full(n, k, np.int8), pl, b, lab)   # 10 -> 25 fps
        parts.append((per, pl, b, lab))
    period = np.concatenate([p[0] for p in parts])
    shift = np.array([PITCH[0] / 2, PITCH[1] / 2], np.float32)
    players = np.concatenate([p[1] for p in parts]) + shift
    ball = np.concatenate([p[2] for p in parts]) + shift
    labels = np.concatenate([p[3] for p in parts])
    return TrackingData(name="wcsf_1", fps=25, pitch=PITCH, frame=np.arange(len(period), dtype=np.int64),
                        period=period, players=players, ball=ball, team=team, labels=labels,
                        meta={"source": "wcsf_groningen_2023", "match_id": NAME})


def cached_game(rebuild=False):
    if not rebuild and TrackingData.exists("wcsf_1"):
        return TrackingData.load("wcsf_1")
    data = load_game()
    data.save()
    return data
