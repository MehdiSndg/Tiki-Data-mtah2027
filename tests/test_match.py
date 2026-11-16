import os
import sys

import numpy as np
import pytest

from conftest import needs_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from mtah.external import METRICA, from_metrica_sample  # noqa: E402
from mtah.match import from_tactical, spell_stats  # noqa: E402

needs_metrica = pytest.mark.skipif(not os.path.isdir(METRICA), reason="Metrica sample data not downloaded")


@needs_metrica
def test_metrica_labels_reproduce_overnight_stats():
    m = from_metrica_sample(1)
    s = spell_stats(m["labels"])
    assert s["spells_per_5min"] == pytest.approx(18.58, abs=0.01)   # analysis/g2_metrica_events.py, 22 Oct
    assert s["d_share"] == pytest.approx(0.38, abs=0.005)


@needs_metrica
def test_metrica_halves_are_separated():
    m = from_metrica_sample(1)
    assert np.any(m["labels"][np.flatnonzero(m["half"] == 1)[-1] + 1:np.flatnonzero(m["half"] == 2)[0]] == pm.X) \
        or np.flatnonzero(m["half"] == 1)[-1] + 1 == np.flatnonzero(m["half"] == 2)[0]


@needs_metrica
def test_metrica_goalkeepers_at_the_ends():
    m = from_metrica_sample(1)
    for t in (0, 1):
        x = np.nanmean(m["px"][m["half"] == 1, m["gk"][t]])
        assert x < 0.2 or x > 0.8
    home_gk_x = np.nanmean(m["px"][m["half"] == 1, m["gk"][0]])
    assert m["sign"][1] == (1 if home_gk_x < 0.5 else -1) and m["sign"][2] == -m["sign"][1]


@needs_data
def test_tactical_match_has_atd_shape():
    m = from_tactical()
    assert m["px"].shape == (199319, 46) and m["gk"] == {0: 23, 1: 45}
    assert m["pitch"] == (105.3, 68.0)


def test_spell_stats_on_a_toy_sequence():
    c = np.array([pm.X] + [pm.H] * 50 + [pm.D] * 25 + [pm.A] * 75 + [pm.X], np.int8)
    s = spell_stats(c)
    assert s["d_share"] == pytest.approx(25 / 150)
    assert s["spells_per_5min"] == pytest.approx(2 / (150 / 25 / 300))


@needs_metrica
def test_metrica_events_share_the_tracking_clock():
    """Review focus 3: at a RECOVERY the recovering player is at the ball (median < 1.5 m)."""
    import pandas as pd
    from mtah.external import METRICA_PITCH
    m = from_metrica_sample(1)
    ev = pd.read_csv(os.path.join(METRICA, "Sample_Game_1", "Sample_Game_1_RawEventsData.csv"))
    rec = ev[ev["Type"] == "RECOVERY"]
    L, W = METRICA_PITCH
    d = []
    for _, r in rec.iterrows():
        f, t = int(r["Start Frame"]), 0 if r["Team"] == "Home" else 1
        if np.isnan(m["bx"][f]):
            continue
        cols = m["slot_team"] == t
        d.append(np.nanmin(np.hypot((m["px"][f, cols] - m["bx"][f]) * L, (m["py"][f, cols] - m["by"][f]) * W)))
    assert len(d) > 50 and np.median(d) < 1.5


@needs_metrica
def test_metrica_game3_loads_with_halves_and_keepers():
    from mtah.external import from_metrica_sample
    m = from_metrica_sample(3)
    assert m["name"] == "metrica3" and m["px"].shape[0] == len(m["bx"]) == len(m["labels"])
    assert set(np.unique(m["half"])) == {0, 1, 2}
    for t in (0, 1):
        x = np.nanmean(m["px"][m["half"] == 1, m["gk"][t]])
        assert x < 0.2 or x > 0.8
    s = spell_stats(m["labels"])
    assert 12 <= s["spells_per_5min"] <= 25 and 0.15 <= s["d_share"] <= 0.5     # games 1-2: 17-18.6, 0.37-0.38


@needs_metrica
def test_metrica_game3_events_share_the_tracking_clock():
    import json
    from mtah.external import METRICA_PITCH, from_metrica_sample
    m = from_metrica_sample(3)
    ev = json.load(open(os.path.join(METRICA, "Sample_Game_3", "Sample_Game_3_events.json")))["data"]
    L, W = METRICA_PITCH
    d = []
    for e in ev:
        if e["type"]["name"] != "RECOVERY":
            continue
        f, t = e["start"]["frame"], 0 if e["team"]["name"] == "Team A" else 1
        if f >= len(m["bx"]) or np.isnan(m["bx"][f]):
            continue
        sl = m["slot_team"] == t
        d.append(np.nanmin(np.hypot((m["px"][f, sl] - m["bx"][f]) * L, (m["py"][f, sl] - m["by"][f]) * W)))
    assert len(d) > 100 and np.median(d) < 1.5


def test_goalkeeper_slots_come_from_the_metadata(tmp_path):
    from mtah.data import goalkeeper_slots
    xml = """<main><Teams><Team id="h"/><Team id="a"/></Teams><Players>
    <Player teamId="h"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Unknown</Value></ProviderParameter></ProviderPlayerParameters></Player>
    <Player teamId="a"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>
    <Player teamId="h"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>
    </Players></main>"""
    p = tmp_path / "m.xml"
    p.write_text(xml)
    assert goalkeeper_slots(str(p)) == {0: 2, 1: 1}


def test_keepers_and_sign_when_both_teams_sit_past_halfway():
    from mtah.match import keepers_and_sign
    # first half: home spread over 0.10..0.90 (mean 0.52), away 0.40..0.95 (mean 0.64)
    home = np.array([0.10, 0.45, 0.55, 0.60, 0.90])
    away = np.array([0.40, 0.55, 0.65, 0.70, 0.95])
    px = np.tile(np.r_[home, away], (10, 1)).astype(np.float32)
    half = np.ones(10, np.int8)
    slot_team = np.r_[np.zeros(5), np.ones(5)].astype(np.int8)
    gk, sign = keepers_and_sign(px, half, slot_team)
    assert gk == {0: 0, 1: 9} and sign == {1: 1, 2: -1}


def test_swap_teams_mirrors_labels_slots_keepers_and_sign():
    import possession_metrics as pm
    from mtah.match import swap_teams
    m = {"labels": np.array([pm.H, pm.A, pm.D, pm.X], np.int8), "slot_team": np.array([0, 0, 1], np.int8),
         "gk": {0: 1, 1: 2}, "sign": {1: 1, 2: -1}, "name": "x", "px": np.zeros((4, 3))}
    s = swap_teams(m)
    assert s["labels"].tolist() == [pm.A, pm.H, pm.D, pm.X]
    assert s["slot_team"].tolist() == [1, 1, 0]
    assert s["gk"] == {0: 2, 1: 1} and s["sign"] == {1: -1, 2: 1}
    assert m["labels"].tolist() == [pm.H, pm.A, pm.D, pm.X]          # the original is untouched
    assert s["px"] is m["px"]
