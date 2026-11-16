"""Tactical-camera homography (BAR-ATH only): where the camera looks on the pitch.

The CSV gives one 3x3 matrix per video frame mapping normalised pitch coordinates (0-1, the ATD
convention) to image pixels. The image centre back-projected through the inverse is the camera's
aim point. Measured on ball-visible frames (Gün 3, Ek L.12): aim x vs ATD ball x correlation 0.83,
median |dx| 12 m; aim y carries no information (the camera always looks at the far half), so only
x is used. Cached as cache/homography_aim.npy.
"""

import numpy as np
import pandas as pd

from ..paths import CACHE, tactical

IMAGE_CENTRE = (960.0, 540.0)


def camera_aim_x(n_frames, length_m, cache=True):
    """(n_frames,) float64 camera aim x in metres (pitch frame of TrackingData), NaN where the
    homography is missing or invalid."""
    path = CACHE / "homography_aim.npy"
    if cache and path.is_file():
        x = np.load(path)
        if len(x) == n_frames:
            return x * length_m
    h = pd.read_csv(tactical("homography"))
    frame = h["frame"].to_numpy()
    ok = (frame >= 0) & (frame < n_frames) & (h["valid"].to_numpy() > 0)
    H = h[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].to_numpy()[ok].reshape(-1, 3, 3)
    v = np.linalg.inv(H) @ np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
    xn = np.full(n_frames, np.nan)
    xn[frame[ok]] = v[:, 0] / v[:, 2]
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, xn)
    return xn * length_m
