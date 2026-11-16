import glob
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import DFL, from_dfl, merge_short  # noqa: E402
from mtah.match import spell_stats  # noqa: E402

DFL_IDS = sorted(p.rsplit("_", 1)[-1][:-4] for p in glob.glob(os.path.join(DFL, "*positions*.xml")))
needs_dfl = pytest.mark.skipif(len(DFL_IDS) != 7, reason="DFL open data not downloaded")


@needs_dfl
def test_dfl_goalkeepers_at_the_ends():
    m = from_dfl("DFL-MAT-J03WOH")
    for t in (0, 1):
        x = np.nanmean(m["px"][m["half"] == 1, m["gk"][t]])
        assert x < 0.2 or x > 0.8


@needs_dfl
def test_dfl_merge_matches_overnight_sweep():
    s = spell_stats(np.concatenate([from_dfl(i)["labels"] for i in DFL_IDS]))
    assert s["spells_per_5min"] == pytest.approx(18.5, abs=0.2)     # analysis/g3_dfl_conversion.py


@needs_dfl
def test_dfl_halves_are_separated_and_normalised():
    m = from_dfl("DFL-MAT-J03WOH")
    gap = slice(np.flatnonzero(m["half"] == 1)[-1] + 1, np.flatnonzero(m["half"] == 2)[0])
    assert gap.stop > gap.start and np.all(m["labels"][gap] == pm.X)
    assert np.nanmin(m["px"]) > -0.1 and np.nanmax(m["px"]) < 1.1 and m["pitch"] == (105.0, 68.0)


def test_merge_short_absorbs_a_short_sandwiched_spell():
    H, A, D = pm.H, pm.A, pm.D
    c = np.array([H] * 50 + [A] * 30 + [H] * 50 + [D] * 10 + [A] * 30 + [D] * 5, np.int8)
    out = merge_short(c, 1.6)
    assert np.all(out[50:80] == H)                 # 1.2 s of A between H and H
    assert np.all(out[140:170] == A)               # not sandwiched by the same team: kept
    assert np.array_equal(merge_short(c, 0), c)
