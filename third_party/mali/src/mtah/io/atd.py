"""Metrica FIFA-EPTS automatic tracking (ATD) -> TrackingData.

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>"  (46 player slots + ball, NaN = missing).
Turning ':' and ';' into ',' makes every line a flat CSV row of 1 + 3*47 numbers, which pandas'
C parser reads in a few seconds instead of a minute of Python string handling.
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .. import constants as C
from ..data import TrackingData
from ..paths import tactical


def read_metadata(xml_path):
    """(slot_team (N,) int8 0=home/1=away, team names, slot order) from the EPTS XML.

    Slot order follows the DataFormatSpecification (playerK_x channels), the order the raw
    file actually uses; the first <Team> is the home side (FIFA EPTS convention).
    """
    root = ET.parse(xml_path).getroot()
    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter("Team")]
    home_id = teams[0][0]
    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    order = []
    for ref in root.iter("PlayerChannelRef"):
        cid = ref.get("playerChannelId")
        if cid.endswith("_x"):
            order.append(cid[len("player"):-2])
    slot_team = np.array([0 if team_of_player[pid] == home_id else 1 for pid in order], np.int8)
    return slot_team, [name for _, name in teams], order


def read_raw(txt_path, n_slots):
    """(frame_id (T,), players (T,N,3), ball (T,3)) float32, NaN where missing."""
    raw = Path(txt_path).read_bytes().translate(bytes.maketrans(b":;", b",,"))
    table = pd.read_csv(io.BytesIO(raw), header=None, dtype=np.float32, engine="c")
    values = table.to_numpy()
    expected = 1 + 3 * (n_slots + 1)
    if values.shape[1] != expected:
        raise ValueError(f"{txt_path}: {values.shape[1]} columns, expected {expected}")
    frame_id = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:]
    return frame_id, players, ball


def load_barath():
    """The tactical recording of BAR-ATH as TrackingData (metres, video-frame indexed)."""
    slot_team, names, _ = read_metadata(tactical("atd_meta"))
    frame_id, players, ball = read_raw(tactical("atd_raw"), len(slot_team))
    if len(frame_id) != C.N_FRAMES or frame_id[0] != 1 or not np.all(np.diff(frame_id) == 1):
        raise ValueError("ATD is not one record per video frame numbered from 1")
    length, width = C.PITCH
    scale = np.array([length, width], np.float32)
    data = TrackingData(
        name="barath",
        fps=C.FPS,
        pitch=C.PITCH,
        frame=frame_id - 1,                             # video frame = frame_id - 1
        period=C.period_array(),
        players=players[:, :, :2] * scale,
        ball=ball[:, :2] * scale,
        team=slot_team,
        labels=None,
        meta={"source": "metrica_atd_tactical", "home": names[0], "away": names[1],
              "raw_player_speed_ms": None},
    )
    return data, players[:, :, 2]


def cached_barath(rebuild=False):
    """Load from cache/barath (building it on first use)."""
    if not rebuild and TrackingData.exists("barath"):
        return TrackingData.load("barath")
    data, _ = load_barath()
    data.save()
    return data
