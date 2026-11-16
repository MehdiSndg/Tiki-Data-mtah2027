"""Gün 5 (Ek N.10): consensus team anchors (labels.weak.team_consensus)."""
import numpy as np
import pandas as pd
import pytest

from mtah.constants import A, D, H, X
from mtah.labels.weak import CONSENSUS_COLUMNS, team_consensus


def _feats(n=200):
    f = pd.DataFrame({c: np.ones(n) for c in CONSENSUS_COLUMNS})
    f.loc[100:, list(CONSENSUS_COLUMNS)] = -1.0                 # all agree: home before 100, away after
    f.loc[50:59, "prox_ddiff_m25"] = -1.0                       # one dissenting signal
    return f


def test_unanimity_adds_only_free_allowed_frames():
    f = _feats()
    codes = np.full(200, X, np.int8)
    codes[10:20] = D                                            # existing anchors are never overwritten
    allowed = np.ones(200, bool)
    allowed[150:160] = False                                    # e.g. slow tempo
    out, n_add, n_mask = team_consensus(f, codes, allowed, {"need": 4, "mask_switch_frames": 0})
    assert (out[10:20] == D).all()
    assert (out[50:60] == X).all()                              # 3 of 4 is not unanimity
    assert (out[150:160] == X).all()
    assert (out[0:10] == H).all() and (out[100:150] == A).all()
    assert n_add == int(np.isin(out, (H, A)).sum()) and n_mask == 0


def test_three_of_four_and_switch_mask():
    f = _feats()
    f["tc_last_team"] = np.where(np.arange(200) < 100, 1.0, -1.0)
    out, _, n_mask = team_consensus(f, np.full(200, X, np.int8), np.ones(200, bool), {"need": 3, "mask_switch_frames": 35})
    assert (out[50:60] == H).all()                              # 3 of 4 agree -> anchor with need 3
    assert (out[65:136] == X).all() and n_mask > 0             # +-35 around the switch at 100
    assert (out[:65] == H).all() and (out[136:] == A).all()


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        team_consensus(pd.DataFrame({"x": [1.0]}), np.array([X], np.int8), np.array([True]), {})
