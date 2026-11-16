"""Shared, lazily computed quantities every feature group draws from.

Everything is in metres / seconds and expressed from the HOME team's point of view (canonical x:
home attacks towards +x in every period), so a group written once works for BAR-ATH, IDSSE and
any future TrackingData source with identical column names and units.
"""

import warnings
from functools import cached_property

import numpy as np

from ..preprocess import filled_period, goalkeepers_and_direction, velocity


def nan_call(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


class FeatureContext:
    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        self.length, self.width = data.pitch
        self.fps = data.fps
        self.n = data.n_frames
        self.P, self.B = data.players, data.ball
        self.home, self.away = data.team == 0, data.team == 1

    def frames(self, seconds):
        """A duration in seconds as a frame count for this source."""
        return max(1, int(round(seconds * self.fps)))

    # -- direction, goalkeepers --------------------------------------------------------------
    @cached_property
    def gk_dir(self):
        return goalkeepers_and_direction(self.data)

    @cached_property
    def home_pos(self):
        phase = filled_period(self.data.period)
        out = np.zeros(self.n, bool)
        for per, (_, _, pos_dir) in self.gk_dir.items():
            out[phase == per] = pos_dir
        return out

    @cached_property
    def outfield(self):
        phase = filled_period(self.data.period)
        out = np.ones(self.P.shape[:2], bool)
        for per, (gk_h, gk_a, _) in self.gk_dir.items():
            out[np.ix_(phase == per, [gk_h, gk_a])] = False
        return out

    @cached_property
    def sign(self):
        return np.where(self.home_pos, 1.0, -1.0).astype(np.float32)

    # -- canonical coordinates (home attacks +x) -----------------------------------------------
    @cached_property
    def xc_p(self):
        return np.where(self.home_pos[:, None], self.P[:, :, 0], self.length - self.P[:, :, 0])

    @cached_property
    def xc_b(self):
        return np.where(self.home_pos, self.B[:, 0], self.length - self.B[:, 0])

    @cached_property
    def y_p(self):
        return self.P[:, :, 1]

    # -- kinematics ----------------------------------------------------------------------------
    @cached_property
    def vp(self):
        return velocity(self.P, self.fps, int(self.cfg["preprocess"]["velocity_half_window"]))

    @cached_property
    def sp(self):
        return np.hypot(self.vp[..., 0], self.vp[..., 1])

    @cached_property
    def vx_c(self):
        """Player x-velocity in canonical (home-attack) direction."""
        return self.vp[..., 0] * self.sign[:, None]

    @cached_property
    def vb(self):
        return velocity(self.B, self.fps, int(self.cfg["preprocess"]["velocity_half_window"]))

    @cached_property
    def sb(self):
        s = np.hypot(self.vb[:, 0], self.vb[:, 1])
        s[s > self.cfg["preprocess"]["ball_max_speed_ms"]] = np.nan
        return s

    @cached_property
    def vis(self):
        return ~np.isnan(self.B[:, 0])

    # -- ball-player geometry ------------------------------------------------------------------
    @cached_property
    def dist(self):
        return np.hypot(self.P[:, :, 0] - self.B[:, 0:1], self.P[:, :, 1] - self.B[:, 1:2])

    @cached_property
    def d_home(self):
        return nan_call(np.nanmin, np.where(self.home, self.dist, np.nan), axis=1)

    @cached_property
    def d_away(self):
        return nan_call(np.nanmin, np.where(self.away, self.dist, np.nan), axis=1)

    @cached_property
    def team_sign_near(self):
        """tanh((d_away - d_home)/2): +1 home clearly closer, -1 away clearly closer."""
        return np.tanh((self.d_away - self.d_home) / 2.0)

    @cached_property
    def outfield_home(self):
        return self.outfield & self.home

    @cached_property
    def outfield_away(self):
        return self.outfield & self.away
