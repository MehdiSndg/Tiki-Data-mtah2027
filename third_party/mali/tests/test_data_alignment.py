"""Data-dependent checks; skipped when the match data is not present."""

import numpy as np
import pytest

from mtah import constants as C
from mtah.paths import tactical

pytestmark = pytest.mark.skipif(not tactical("atd_raw").is_file(), reason="match data not present")


def test_barath_cache_shapes_and_alignment():
    from mtah.io.atd import cached_barath
    data = cached_barath()
    assert data.players.shape == (C.N_FRAMES, C.N_SLOTS, 2)
    assert data.frame[0] == 0 and data.frame[-1] == C.N_FRAMES - 1
    assert data.runs() == [C.HALVES[1], C.HALVES[2]]
    assert list(np.flatnonzero(np.diff(data.team))) == [23]


def test_idsse_cache_is_metric_and_labelled():
    from mtah.data import TrackingData
    if not TrackingData.exists("idsse_J03WPY"):
        pytest.skip("IDSSE J03WPY not converted")
    d = TrackingData.load("idsse_J03WPY")
    assert 100 < d.pitch[0] < 110 and 60 < d.pitch[1] < 75          # metres, not normalised
    assert np.nanmax(d.players[..., 0]) > 90
    played = d.period > 0
    assert np.all(d.labels[played] != C.X)
    assert set(np.unique(d.labels[played])) <= {C.H, C.A, C.D}


def test_goalkeepers_and_direction():
    from mtah.io.atd import cached_barath
    from mtah.preprocess import goalkeepers_and_direction
    gk = goalkeepers_and_direction(cached_barath())
    assert gk[1][:2] == C.GK_SLOTS and gk[2][:2] == C.GK_SLOTS
    assert gk[1][2] is True and gk[2][2] is False      # Barcelona attacks +x in the first half
