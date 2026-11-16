"""One evaluation for every probability-level candidate (autonomous loop, 22 Oct).

evaluate(cfg, bar_probs, dfl_probs, post_fn=None) ->
  dfl_spell_gt35 / dfl_spell_gt11 / dfl_macro : DFL leave-one-match-out probabilities decoded with
      the config's HSMM (+ optional post-processing), scored on the 35-aligned (Metrica-like) and
      the training (<= 11) GT, mean over matches
  bar_*  : BAR-ATH full-match statistics of the decoded submission + inside the reference bands
  pub_*  : public window (DIAGNOSTIC; selection never uses it -- CLAUDE.md rule 3)
post_fn(labels, runs, probs, ctx) -> labels, where ctx has ball / pitch / accel / near_home /
near_away / kick of that source.
"""

import json

import numpy as np

from .. import config as config_mod  # noqa: F401
from .. import constants as C
from ..features import build_features
from ..io.broadcast import bar_data
from ..io.tags import load_tags
from ..labels.align import suppress_short_returns
from ..labels.masking import gap_lengths
from ..paths import OUTPUTS
from ..validation import sanity
from ..validation.official import pm
from ..validation.scoring import score_blocks
from ..validation.splits import labelled_blocks, public_blocks
from . import dfl_referee, engine, m0
from .ablation import _in_bands, official_reference, reference_bands

_CTX = {}


def contexts(cfg):
    """Cached decoding contexts: DFL preps + GTs + kinematic arrays, BAR arrays, blocks, bands."""
    key = cfg["name"]
    if key in _CTX:
        return _CTX[key]
    lengths = gap_lengths(bar_data(cfg))
    dfl = {}
    for m in cfg["joint"]["matches"]:
        prep = dfl_referee.prepare(m, cfg, lengths)
        f = prep["feats"]
        dfl[m] = dict(prep=prep,
                      gt35=np.where(prep["data"].period > 0,
                                    suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                                    C.X).astype(np.int8),
                      ball=prep["corrupted"].ball, pitch=prep["data"].pitch,
                      accel=f["kin_acc"].to_numpy(), kick_thr=(f["kin_acc"] / f["kin_acc_rel"]).to_numpy(),
                      near_home=f["prox_dH"].to_numpy(), near_away=f["prox_dA"].to_numpy(),
                      kick=f["kin_kick"].to_numpy() > 0, vis=~np.isnan(prep["corrupted"].ball[:, 0]))
    bar = bar_data(cfg)
    bf = build_features(bar, cfg, load_tags(), cache_key=bar.name)
    barc = dict(ball=bar.ball, pitch=bar.pitch, accel=bf["kin_acc"].to_numpy(),
                kick_thr=(bf["kin_acc"] / bf["kin_acc_rel"]).to_numpy(),
                near_home=bf["prox_dH"].to_numpy(), near_away=bf["prox_dA"].to_numpy(),
                kick=bf["kin_kick"].to_numpy() > 0, vis=~np.isnan(bar.ball[:, 0]))
    _, ref_stats, _ = official_reference(cfg)
    bands = reference_bands(cfg["acceptance"], None, ref_stats)
    _CTX[key] = dict(dfl=dfl, bar=barc, blocks=public_blocks(), bands=bands)
    return _CTX[key]


def evaluate(cfg, bar_probs, dfl_probs, post_fn=None, decoder_cfg=None):
    ctx = contexts(cfg)
    dcfg = decoder_cfg or cfg
    sp35, sp11, mac = [], [], []
    for m, d in ctx["dfl"].items():
        if m not in dfl_probs:
            continue
        runs = d["prep"]["runs"]
        pred = engine.decode_runs(dfl_probs[m], runs, dcfg)
        if post_fn is not None:
            pred = post_fn(pred, runs, dfl_probs[m], d)
        s = dfl_referee.score(pred, d["prep"], None)
        sp11.append(s["spell_f1"])
        mac.append(s["macro_f1"])
        sp35.append(pm.evaluate(d["gt35"], pred)["spell_f1_mean"])
    lab = m0.decode_match(bar_probs, dcfg)
    if post_fn is not None:
        lab = post_fn(lab, list(C.HALVES.values()), bar_probs, ctx["bar"])
    pub = score_blocks(lab, ctx["blocks"], bootstrap=0)["ALL"]
    st = sanity.stats(lab)
    ok, _ = _in_bands(st, ctx["bands"])
    return {"dfl_spell_gt35": float(np.mean(sp35)), "dfl_spell_gt11": float(np.mean(sp11)),
            "dfl_macro": float(np.mean(mac)), "dfl_per_match_gt35": [round(x, 4) for x in sp35],
            "bar_poss": st["n_possessions"], "bar_dead": round(st["dead_share"], 4),
            "bar_home": round(st["home_share"], 4), "bar_in_bands": bool(ok),
            "pub_spell": pub["spell_f1_mean"], "pub_macro": pub["macro_f1"], "labels": lab}


def load_run_probs(run):
    """(bar_probs, dfl_probs) of a finished run: 'main_lgbm' (the booster stage) or an
    outputs/<run> folder written by adapt / hybrid (e.g. 'main', the official main model)."""
    if run == "main_lgbm":
        from .. import config as config_mod
        from ..validation.splits import cache_tag
        tag = cache_tag(config_mod.load("main"))
        bar = np.load(OUTPUTS / "main_lgbm" / "probs.npy")
        dfl = {m: np.load(OUTPUTS / "decoder" / "main" / f"lomo_{m}{tag}.npy")
               for m in ("J03WPY", "J03WN1", "J03WMX", "J03WOH")}
        return bar, dfl
    folder = OUTPUTS / run
    z = np.load(folder / "dfl_lomo_blend.npz")
    return np.load(folder / "probs.npy"), {m: z[m] for m in z.files}
