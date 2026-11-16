"""Standard in-memory format shared by every data source (BAR-ATH ATD, IDSSE/DFL, ...).

Coordinates are metres with the origin at a pitch corner: x in [0, length], y in [0, width].
Every array is indexed by a contiguous frame axis of length T. Slots are anonymous positions in
the player array; only their team is known (identities are never used).

On disk a dataset is two files in cache/<name>/:
    arrays.npz      players (T,N,2) float32, ball (T,2) float32, team (N,) int8, frame, period, labels
    frames.parquet  one row per frame: frame, period, ball_x, ball_y, ball_visible, label  (inspection / joins)
plus meta.json (fps, pitch, source, home attack direction per period).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .constants import LABELS, X
from .paths import CACHE, ensure


@dataclass
class TrackingData:
    name: str
    fps: int
    pitch: tuple
    frame: np.ndarray                     # (T,) int64 source frame index (video frame for BAR-ATH)
    period: np.ndarray                    # (T,) int8; 0 = outside any scored/played period
    players: np.ndarray                   # (T,N,2) float32 metres, NaN = not tracked
    ball: np.ndarray                      # (T,2) float32 metres, NaN = not tracked
    team: np.ndarray                      # (N,) int8: 0 home, 1 away
    labels: Optional[np.ndarray] = None   # (T,) int8 codes H/A/D/X (X = unknown)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.frame)

    def runs(self):
        """[(start, stop)] maximal blocks of equal non-zero period: the decoding units."""
        out, p = [], self.period
        edges = np.flatnonzero(np.diff(p.astype(np.int16)) != 0) + 1
        bounds = np.concatenate(([0], edges, [len(p)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if p[a] > 0:
                out.append((int(a), int(b)))
        return out

    def copy(self):
        return TrackingData(self.name, self.fps, tuple(self.pitch), self.frame.copy(),
                            self.period.copy(), self.players.copy(), self.ball.copy(),
                            self.team.copy(), None if self.labels is None else self.labels.copy(),
                            dict(self.meta))

    # -- persistence -----------------------------------------------------------------------

    def save(self, root=CACHE):
        folder = ensure(Path(root) / self.name)
        arrays = dict(players=self.players, ball=self.ball, team=self.team,
                      frame=self.frame, period=self.period)
        if self.labels is not None:
            arrays["labels"] = self.labels
        np.savez(folder / "arrays.npz", **arrays)
        meta = dict(self.meta, name=self.name, fps=self.fps, pitch=list(self.pitch))
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.frames_table().to_parquet(folder / "frames.parquet", index=False)
        return folder

    @classmethod
    def load(cls, name, root=CACHE):
        folder = Path(root) / name
        with np.load(folder / "arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(name=name, fps=int(meta["fps"]), pitch=tuple(meta["pitch"]),
                   frame=arrays["frame"], period=arrays["period"], players=arrays["players"],
                   ball=arrays["ball"], team=arrays["team"], labels=arrays.get("labels"),
                   meta=meta)

    @staticmethod
    def exists(name, root=CACHE):
        return (Path(root) / name / "arrays.npz").is_file()

    def frames_table(self):
        lab = (np.full(self.n_frames, X, np.int8) if self.labels is None else self.labels)
        return pd.DataFrame({
            "frame": self.frame, "period": self.period,
            "ball_x": self.ball[:, 0], "ball_y": self.ball[:, 1],
            "ball_visible": ~np.isnan(self.ball[:, 0]),
            "label": np.array(LABELS)[lab],
        })
