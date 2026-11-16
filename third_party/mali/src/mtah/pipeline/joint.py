"""Joint training (Adım 3.3): BAR-ATH weak labels + IDSSE/DFL true labels, and its DFL check.

Training set = BAR-ATH source (weak anchors + masked copies, exactly as m0.run) + one source per
DFL match: the ATD-like corrupted track (dfl_referee.corrupt), its TRUE labels (aligned), and
state-aware masked copies. Sample weights give BAR-ATH `joint.bar_share` of the total weight and
split the rest equally over the DFL matches, whatever their row counts.

DFL validation = leave-one-match-out: for each available DFL match m the model is trained on
BAR-ATH + every OTHER DFL match and scored on m (second half, comparable with the referee, and
the whole match). With a single DFL match the fold falls back to "train on m's first half".
Source-specific columns (Smart Tagging) are NaN on DFL rows unless joint.drop_source_specific.
"""

from dataclasses import dataclass

import numpy as np

from .. import constants as C
from ..data import TrackingData
from . import dfl_referee, engine

JOINT_DEFAULTS = dict(bar_share=0.5, dfl_stride=4, drop_source_specific=False, external_only=False)
# external_only (organisers' clarification, Nov 2026: "If you need to train (or retrain), you MUST use only
# external data"): the booster never trains on the provided game -- no BAR weak-anchor source; the DFL
# matches carry the whole weight and only the common (non source-specific) feature columns are used.


def training_sources(cfg, bar_source, matches, bar_lengths, train_period=None, exclude_metrica=()):
    """[BAR source (unless external_only)] + DFL sources; DFL shares the whole weight when BAR is absent.
    joint.metrica_booster (6 Nov, external_only only): the Metrica open games (minus `exclude_metrica`, the
    games being predicted) join as booster sources; every match (DFL or Metrica) gets an equal weight share."""
    jc = joint_cfg(cfg)
    met = [g for g in (jc.get("metrica_booster") or []) if g not in set(exclude_metrica)]
    if met:
        assert jc["external_only"], "metrica_booster needs external_only"
        n = len(matches) + len(met)
        return (dfl_sources(cfg, matches, bar_lengths, train_period, share_total=len(matches) / n)
                + metrica_sources(cfg, met, 1.0 / n))
    if jc["external_only"] or bar_source is None:
        return dfl_sources(cfg, matches, bar_lengths, train_period, share_total=1.0)
    bar_source.share = jc["bar_share"]
    return [bar_source] + dfl_sources(cfg, matches, bar_lengths, train_period)


def joint_cfg(cfg):
    return dict(JOINT_DEFAULTS, **cfg.get("joint", {}))


def available(matches):
    """The given DFL matches, with cache/idsse_<id> built from the raw files when missing.
    25 Oct (clean-room run): this used to drop uncached matches SILENTLY, so a fresh checkout
    trained the booster without DFL and still produced a submission. Now a missing match raises."""
    from ..io.idsse import cached_idsse
    for m in matches:
        if not TrackingData.exists(f"idsse_{m}"):
            cached_idsse(m)
    return list(matches)


def dfl_sources(cfg, matches, bar_lengths, train_period=None, share_total=None):
    jc = joint_cfg(cfg)
    matches = available(matches)
    share_total = (1.0 - jc["bar_share"]) if share_total is None else share_total
    # Gün 7 (Ek Q.4): dfl_mix_eval trains on both the training and the evaluation corruption of every
    # DFL match (dfl_referee + dfl_referee_eval), sharing the DFL weight equally.
    variants = [dfl_referee.train_cfg(cfg)]     # 5 Nov: dfl_referee_train (v3_dr) for training copies
    ecfg = dfl_referee.eval_cfg(cfg)
    if cfg.get("dfl_mix_eval") and ecfg is not None:
        variants.append(ecfg)
    out = []
    for mid in matches:
        for vi, vcfg in enumerate(variants):
            prep = dfl_referee.prepare(mid, vcfg, bar_lengths)
            data = prep["data"]
            train_ok = data.period > 0 if train_period is None else data.period == train_period
            out.append(engine.TrainSource(
                f"dfl_{mid}" + (f"_v{vi}" if vi else ""), prep["corrupted"], prep["feats"], prep["aligned"],
                np.ones(data.n_frames, bool), train_ok, prep["key"], prep.get("lengths", bar_lengths), None,
                int(jc["dfl_stride"]), share=share_total / max(len(matches) * len(variants), 1),
                state_labels=prep["aligned"], feat_cfg=prep["cfg"]))
    return out


@dataclass
class FixedSource(engine.TrainSource):
    """A source trained as it is: no masked copies (Metrica's own ball gaps are the masking)."""

    def matrix(self, cfg, i):
        return engine.training_matrix(self.feats, [], self.labels, self.team_ok, self.train_ok, self.stride)


def metrica_sources(cfg, games, share_per_game):
    """Metrica open games as booster sources: raw track + a player-degraded copy (the training corruption,
    ball untouched), half the game's share each (tools/booster_met.py, 5 Nov)."""
    from . import metrica_src
    stride = int(joint_cfg(cfg)["dfl_stride"])
    rc = dfl_referee.referee_cfg(cfg)
    out = []
    for g in games:
        for kind, prep in (("raw", metrica_src.prepare(g, cfg)), ("deg", metrica_src.prepare_corrupted(g, cfg, rc))):
            d = prep["data"]
            out.append(FixedSource(f"metrica_{g}_{kind}", prep["corrupted"], prep["feats"], prep["aligned"],
                                   np.ones(d.n_frames, bool), d.period > 0, prep["key"], None, None, stride,
                                   share=share_per_game / 2))
    return out


def metrica_games_in(names):
    """Metrica game numbers among target names (metrica_2, metrica_2_v2b, metrica_2_dr_s11, ...)."""
    return sorted({int(n.split("_")[1]) for n in names if n.startswith("metrica_")})


def evaluate_lomo(cfg, matches, bar_source, columns, bar_lengths, return_probs=False, ecfg=None):
    """[{match, h2: score dict, full: score dict}] -- one fold per available DFL match. With `ecfg`
    (dfl_referee.eval_cfg) the fold's models also predict the held-out match under the evaluation
    corruption: row['probs_eval'] (needs return_probs)."""
    jc = joint_cfg(cfg)
    matches = available(matches)
    rows = []
    for m in matches:
        others = [o for o in matches if o != m]
        if others:
            sources = training_sources(cfg, bar_source, others, bar_lengths)
            fold = "lomo"
        else:
            sources = training_sources(cfg, bar_source, [m], bar_lengths, train_period=1)
            fold = "h1->h2"
        prep = dfl_referee.prepare(m, cfg, bar_lengths)
        targets = [prep["feats"]]
        if ecfg is not None:
            targets.append(dfl_referee.prepare(m, ecfg, bar_lengths)["feats"])
        per_target, _, counts = engine.fit_predict_sources(sources, targets, cfg, columns)
        per_seed = per_target[0]
        probs = engine.ensemble(per_seed)
        data = prep["data"]
        runs = prep["runs"]
        pred_full = engine.decode_runs(probs, runs, cfg)
        h2_runs = [(a, b) for a, b in runs if data.period[a] == 2]
        pred_h2 = engine.decode_runs(probs, h2_runs, cfg)
        rows.append({"match": m, "fold": fold, "train_rows": counts,
                     "h2": dfl_referee.score(pred_h2, prep, 2),
                     "full": dfl_referee.score(pred_full, prep, None)})
        if return_probs:
            rows[-1]["probs"] = probs
            if ecfg is not None:
                rows[-1]["probs_eval"] = engine.ensemble(per_target[1])
    return rows
