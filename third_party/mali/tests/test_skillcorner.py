"""SkillCorner open-data reader: synthetic files only (no download)."""

import json

import numpy as np
import pandas as pd

from mtah.constants import A, D, H, X
from mtah.io import skillcorner as sc


def write_game(base, game="9", n=40):
    g = base / game
    g.mkdir(parents=True)
    players = [{"id": 1, "team_id": 100}, {"id": 2, "team_id": 200}]
    (g / f"{game}_match.json").write_text(json.dumps(
        {"home_team": {"id": 100}, "away_team": {"id": 200}, "players": players,
         "pitch_length": 104, "pitch_width": 68}), encoding="utf-8")
    lines = []
    for f in range(n):
        period = None if f < 2 else 1
        lines.append(json.dumps({
            "frame": f, "period": period,
            "ball_data": {"x": float(f), "y": 0.0, "z": 0.0, "is_detected": f % 10 != 5},
            "player_data": [] if period is None else [
                {"player_id": 1, "x": float(f), "y": 1.0, "is_detected": True},
                {"player_id": 2, "x": -float(f), "y": -1.0, "is_detected": False}]}))
    text = "\n".join(lines) + "\n"
    (g / f"{game}_tracking_extrapolated.jsonl").write_bytes(text.encode())
    pd.DataFrame({"frame_start": [5, 21], "frame_end": [14, 30], "team_in_possession_id": [100, 200]}) \
        .to_csv(g / f"{game}_phases_of_play.csv", index=False)
    (base / "expected_sizes.json").write_text(json.dumps({game: len(text.encode())}), encoding="utf-8")
    return game


def test_labels_are_dead_outside_phases_and_phase_team_inside(tmp_path):
    game = write_game(tmp_path)
    frame, period, *_ = sc.read_source(game, tmp_path)
    phases = pd.read_csv(tmp_path / game / f"{game}_phases_of_play.csv")
    lab = sc.phase_labels(phases, frame, period, 100)
    assert np.all(lab[:2] == X)                       # outside a period
    assert np.all(lab[2:5] == D) and np.all(lab[15:21] == D) and np.all(lab[31:] == D)
    assert np.all(lab[5:15] == H) and np.all(lab[21:31] == A)


def test_upsampled_track_is_25fps_metres_with_corner_origin_and_undetected_ball_dropped(tmp_path):
    game = write_game(tmp_path)
    data = sc.load_game(game, tmp_path)
    assert data.fps == 25 and data.pitch == (104.0, 68.0)
    assert data.n_frames == (40 - 1) * 25 // 10 + 1
    t = 10 * 25 // 10                                  # 25 fps frame of source frame 10
    np.testing.assert_allclose(data.players[t, 0], [10 + 52, 1 + 34])
    np.testing.assert_allclose(data.players[t + 1, 0], [10.4 + 52, 1 + 34], rtol=1e-6)
    np.testing.assert_allclose(data.players[t, 1], [-10 + 52, -1 + 34])   # extrapolated players kept
    # source frame 15 has an undetected ball: every 25 fps frame touching it is NaN
    near15 = [k for k in range(data.n_frames) if 14 < k * 0.4 < 16]
    assert np.all(np.isnan(data.ball[near15, 0]))
    assert np.isfinite(data.ball[t, 0])
    assert set(np.unique(data.labels[data.period > 0])) <= {H, A, D}
    assert list(data.team) == [0, 1]


def test_partial_download_is_not_listed(tmp_path):
    game = write_game(tmp_path)
    assert sc.games(tmp_path) == [game]
    f = tmp_path / game / f"{game}_tracking_extrapolated.jsonl"
    f.write_bytes(f.read_bytes()[:-10])
    assert sc.games(tmp_path) == []


def test_screen_config_differs_from_reference_only_by_skillcorner_games():
    from mtah import config
    from mtah.pipeline import adapt
    assert adapt.DEFAULTS["skillcorner_games"] == ()
    for name in ("main_ext_met", "main_ext_met_dt"):
        assert not adapt.params(config.load(name))[0]["skillcorner_games"]
    sc_cfg, ref = config.load("ev_ext_met_sc"), config.load("ev_ext_met")
    assert len(sc_cfg["adapt"].pop("skillcorner_games")) == 8
    sc_cfg.pop("name"), ref.pop("name")
    assert sc_cfg == ref


def test_full_screen_config_uses_all_games_thinned_and_same_reference():
    from mtah import config
    sc20, ref = config.load("ev_ext_met_sc20"), config.load("ev_ext_met")
    assert len(set(sc20["adapt"].pop("skillcorner_games"))) == 20 and sc20["adapt"].pop("skillcorner_stride") == 2
    sc20.pop("name"), ref.pop("name")
    assert sc20 == ref


def test_dead_only_configs_differ_only_by_team_weight():
    from mtah import config
    from mtah.pipeline import adapt
    assert adapt.DEFAULTS["skillcorner_team"] == 1.0
    for d, base in (("ev_ext_met_scd", "ev_ext_met_sc"), ("ev_ext_met_sc20d", "ev_ext_met_sc20")):
        a, b = config.load(d), config.load(base)
        assert a["adapt"].pop("skillcorner_team") == 0.0
        a.pop("name"), b.pop("name")
        assert a == b


def test_team_weight_zero_window_ignores_which_team_is_labelled():
    import torch
    from mtah.models import seqnet
    torch.manual_seed(0)
    B, T = 2, 30
    logits, bl = torch.randn(B, T, 3), torch.randn(B, T)
    y = torch.randint(0, 3, (B, T))
    known = torch.ones(B, T, dtype=torch.bool)
    p = dict(tmse_tau=4.0, lambda_tmse=0.15, lambda_iou=0.5, lambda_bnd=0.5, bnd_pos_weight=5.0)
    w, bon, ton = torch.ones(B), torch.zeros(B, dtype=torch.bool), torch.zeros(B)
    swapped = torch.where(y < 2, 1 - y, y)
    a, _ = seqnet.loss_fn(logits, bl, y, known, torch.zeros(B, T), p, w, bon, ton)
    b, _ = seqnet.loss_fn(logits, bl, swapped, known, torch.zeros(B, T), p, w, bon, ton)
    assert torch.allclose(a, b)


def test_draw_and_big_net_configs_differ_only_where_intended():
    from mtah import config
    from mtah.pipeline import adapt
    ref = config.load("ev_ext_met")
    assert adapt.DEFAULTS["extra_draws"] == ()
    dr = config.load("ev_ext_met_dr3")
    assert dr["adapt"].pop("extra_draws") == [11, 13]
    big = config.load("ev_ext_met_big")
    assert big.pop("seqnet") == {**ref["seqnet"], "hidden": 192, "gru_hidden": 192, "gru_layers": 3}
    for c in (dr, big, ref):
        c.pop("name")
    assert dr == ref and big == {k: v for k, v in ref.items() if k != "seqnet"}


def test_wcsf_config_adds_one_unseen_match_and_keeps_the_dfl3_referees_apart():
    import sys
    from pathlib import Path
    from mtah import config
    from mtah.pipeline import adapt
    ac, _ = adapt.params(config.load("ev_ext_met_wcsf"))
    ref, _ = adapt.params(config.load("ev_ext_met"))
    assert ac["extra_matches"] == ref["extra_matches"] + ["WCSF"]
    assert "WCSF" not in config.load("ev_ext_met_wcsf")["joint"]["matches"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import run_day_screen as screen
    row = {"net": "adapted", "w_boost": 0.25, "spell_gt35": 0.9,
           "per_match_gt35": {"A": 0.7, "B": 0.8, "WCSF": 0.5, "metrica_1": 0.6}}
    s = {"dfl_lomo": [row], "dfl_lomo_eval": [{"w_boost": 0.25, "spell_gt35": 0.9, "per_match_gt35": {"A": 0.6, "B": 0.7, "WCSF": 0.4}}],
         "bar": [{"net": "adapted", "w_boost": 0.25, "n_possessions": 250, "dead_share": 0.45, "home_share": 0.6}]}
    out = screen.met_scores(s)
    assert out["wcsf_v1"] == 0.5 and out["wcsf_v2b"] == 0.4
    assert abs(out["dfl_v1"] - 0.75) < 1e-9 and abs(out["dfl_v2b"] - 0.65) < 1e-9 and "WCSF" not in out["dfl_v2b_per_match"]
