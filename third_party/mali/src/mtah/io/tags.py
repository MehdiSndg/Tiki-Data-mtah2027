"""Smart Tagging (SportsCode XML) -> DataFrame of video-frame intervals.

Codes are upper case, tag values title case; everything is matched case-insensitively.
Defending codes are exact mirrors of attacking ones (same interval, other team) and are flagged
so they are never counted twice.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..paths import tactical

ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
MIRRORS = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
           "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
TEAM_CODE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def load_tags(path=None, fps=25):
    rows = []
    for ins in ET.parse(path or tactical("smart_tagging")).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in ins.findall("label")}
        code = (ins.findtext("code") or "").strip().upper()
        team = TEAM_CODE.get((tags.get("Team") or "").strip().lower(), -1)
        start_s, end_s = float(ins.findtext("start")), float(ins.findtext("end"))
        rows.append(dict(code=code, team=team, type=(tags.get("Type") or "").strip(),
                         half=(tags.get("Half") or "").strip(),
                         start=int(round(start_s * fps)), end=int(round(end_s * fps)),
                         is_attacking=code in ATTACKING, is_mirror=code in MIRRORS))
    return pd.DataFrame(rows)


def interval_signal(tags, n, mask, value=None):
    """Sum of `value` (or 1) over tag intervals selected by boolean `mask`, as a length-n array."""
    out = np.zeros(n + 1, np.float32)
    sel = tags[mask]
    vals = np.ones(len(sel), np.float32) if value is None else np.asarray(value, np.float32)
    starts = np.clip(sel["start"].to_numpy(), 0, n)
    ends = np.clip(sel["end"].to_numpy(), 0, n)
    np.add.at(out, starts, vals)
    np.add.at(out, ends, -vals)
    return np.cumsum(out)[:n]
