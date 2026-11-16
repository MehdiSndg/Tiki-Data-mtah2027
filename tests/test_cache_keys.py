import os

from mtah import frames, grumodel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_model_hash_covers_every_file_the_models_depend_on():
    covered = {os.path.relpath(p, ROOT) for p in grumodel._hash_files()}
    for f in ("mtah/setgru.py", "mtah/deadball.py", "analysis/m1_dead_classifier.py", "evaluation/possession_metrics.py",
              "mtah/grumodel.py", "mtah/teammodel.py", "mtah/degrade.py", "mtah/external.py", "mtah/match.py"):
        assert f in covered, f


def test_frames_cache_watches_games_py():
    assert os.path.join(ROOT, "games.py") in frames._sources()


def test_close_short_gaps_fills_slivers_between_dead_frames_only():
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from p4_common import close_short_gaps
    dead = np.zeros(300, bool)
    dead[:100] = True
    dead[110:200] = True          # 10-frame sliver between two dead blocks -> closed
    g = np.ones(300, bool)
    out = close_short_gaps(dead, g, 50)
    assert out[100:110].all() and not out[200:].any()
    g2 = g.copy()
    g2[105] = False               # a half boundary inside the sliver: left alone
    assert not close_short_gaps(dead, g2, 50)[100:105].any()


def test_model_tag_names_non_default_seeds():
    assert grumodel.model_tag(swap=True, metrica=True, short_k=2.0, arch="set", multitask=True) == "_swap_met_k2_set_mt"
    assert grumodel.model_tag(swap=True, metrica=True, short_k=2.0, arch="set", multitask=True, seeds=(0, 1, 2, 3, 4, 5)) == "_swap_met_k2_set_mt_s012345"
    assert grumodel.model_tag(swap=True, seeds=grumodel.SEEDS) == "_swap"
    assert grumodel.model_tag(swap=True, metrica=True, short_k=2.0, arch="big", seeds=(0, 1, 2, 3, 4, 5), tail=True) == "_swap_met_k2_big_s012345_tail"


def test_model_tag_names_extra_degradation_draws():
    assert grumodel.model_tag(swap=True, metrica=True, short_k=2.0, arch="big", seeds=(0, 1, 2, 3, 4, 5), draws=3) == "_swap_met_k2_big_s012345_d3"
    assert grumodel.model_tag(swap=True, metrica=True, short_k=2.0, arch="big", seeds=(0, 1, 2, 3, 4, 5), draws=1) == "_swap_met_k2_big_s012345"


def test_extra_draws_add_seeded_copies_of_every_training_match(monkeypatch):
    import mtah.degrade as D
    import mtah.external as E
    calls = []
    monkeypatch.setattr(E, "from_metrica_sample", lambda g: {"game": g})
    monkeypatch.setattr(D, "degrade", lambda m, rates, seed: calls.append((m["game"], seed)) or {"game": m["game"], "seed": seed})
    ms = grumodel.metrica_training_matches(draws=3)
    assert sorted(calls) == [(g, s) for g in (1, 2, 3) for s in (0, 1, 2)]
    assert len(ms) == 9


def test_feature_stats_match_nanmean_nanstd_of_the_concatenation():
    import numpy as np
    rng = np.random.default_rng(0)
    Fs = [(rng.normal(size=(500, 4)) * 10 + 3).astype(np.float32) for _ in range(3)]
    Fs[1][::5, 2] = np.nan
    mu, sd = grumodel.feature_stats(Fs)
    allF = np.concatenate(Fs)
    assert np.allclose(mu, np.nanmean(allF, axis=0), atol=1e-4)
    assert np.allclose(sd, np.nanstd(allF, axis=0) + 1e-6, atol=1e-4)


def test_dead_training_mask_drops_x_frames_only_when_asked():
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import p4_common
    from mtah.degrade import pm
    m = {"half": np.array([0, 1, 1, 1, 2], np.int8), "labels": np.array([pm.X, pm.X, pm.D, pm.H, pm.A], np.int8)}
    assert p4_common.dead_train_mask(m).tolist() == [False, True, True, True, True]
    assert p4_common.dead_train_mask(m, xmask=True).tolist() == [False, False, True, True, True]


def test_dead_gru_file_names_the_v8_recipe():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    import p4_common
    assert p4_common.dead_recipe_tag() == ""
    assert p4_common.dead_recipe_tag(metrica=(1, 2, 3), xmask=True) == "_m123x"
    assert p4_common.dead_recipe_tag(metrica=(1, 2, 3), xmask=True, nseeds=6) == "_m123x_s6"
