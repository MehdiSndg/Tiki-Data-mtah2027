"""Hybrid pipeline (Gün 2): gradient-boosting probabilities + temporal net (models/seqnet.py),
log-pool blend, then the same HSMM decoder.

Training data of the net = the DFL matches of the config with REAL (aligned) labels; its input
probabilities there are the config's leave-one-match-out probabilities (out-of-fold, so the net
never learns to trust in-sample confidence). BAR-ATH is only predicted (its boosting
probabilities are in-sample on weak anchors; using them for training would leak).
Caveat: the LOMO probability of DFL match j was produced by a booster that also saw match m, so
the net's LOMO score on m has a second-order dependence on m's labels (no nested CV).

Evaluation:
  DFL   the net itself is leave-one-match-out: trained on the other DFL matches, scored on m
        (whole match; GT aligned <= 11 frames as in training, and <= 35 + control).
  BAR   net trained on all DFL matches -> public window + full-match statistics + the acceptance
        rule against the config's own seeds; writes outputs/<name>/{probs.npy, submission.csv}.
Blend  log p = w * log p_boost + (1 - w) * log p_net / T, renormalised; w (and T) picked on the
        DFL LOMO (35-aligned GT), never on the public window.
"""

import copy
import datetime as dt
import json
import time
import warnings

import numpy as np

from .. import config as config_mod
from .. import constants as C
from ..features import build_features, common_groups
from ..io import submission
from ..io.broadcast import bar_data
from ..io.tags import load_tags
from ..labels.align import suppress_short_returns
from ..labels.masking import gap_lengths
from ..paths import OUTPUTS, ensure
from ..validation import sanity
from ..validation.compare import paired_spell_bootstrap
from ..validation.scoring import score_blocks
from ..validation.splits import labelled_blocks, merged_gt, public_blocks
from . import decoder_tune, dfl_referee, engine, m0
from .ablation import _in_bands, official_reference, reference_bands


def robust_scale(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # all-NaN columns
        med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
    iqr = np.where(np.isfinite(q75 - q25) & ((q75 - q25) > 1e-6), q75 - q25, 1.0)
    Z = (X - np.where(np.isfinite(med), med, 0.0)) / iqr
    return np.clip(np.nan_to_num(Z, nan=0.0), -5, 5).astype(np.float32)


def step_view(feats, columns, probs, labels, runs, stride):
    """[(x (S,F+3), y (S,), known (S,), (a, b))] per run on the stride grid (block means)."""
    X = robust_scale(feats[columns].to_numpy(np.float64))
    lp = np.log(np.clip(probs, 1e-4, 1.0)).astype(np.float32)
    Z = np.concatenate([X, lp], axis=1)
    out = []
    for a, b in runs:
        n = (b - a) // stride
        if n < 8:
            continue
        e = a + n * stride
        x = Z[a:e].reshape(n, stride, -1).mean(1)
        if labels is None:
            y, known = np.zeros(n, np.int64), np.zeros(n, bool)
        else:
            lab = labels[a:e].reshape(n, stride)[:, stride // 2]
            known = lab != C.X
            y = np.where(known, lab, 0).astype(np.int64)
        out.append((x.astype(np.float32), y, known, (a, b)))
    return out


def to_frames(step_probs, a, b, stride):
    """Step probabilities -> per-frame (b-a, 3) by linear interpolation between step centres."""
    n = len(step_probs)
    centres = np.arange(n) * stride + stride / 2.0
    t = np.arange(b - a) + 0.5
    out = np.stack([np.interp(t, centres, step_probs[:, k]) for k in range(3)], axis=1)
    return out / out.sum(1, keepdims=True)


def log_pool(p_boost, p_net, w, T=1.0):
    lp = w * np.log(np.clip(p_boost, 1e-6, 1)) + (1 - w) * np.log(np.clip(p_net, 1e-6, 1)) / T
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return np.round(p / p.sum(1, keepdims=True), 6)


def compose_dead_team(p_dead, p_team):
    """Two-model composition (12 Nov): P(D) of `p_dead`; the alive mass 1 - P(D) is split between H and A in
    the proportion of `p_team`. (T, 3) float64, rows sum to 1."""
    out = np.asarray(p_team, np.float64).copy()
    alive = np.maximum(out[:, 0] + out[:, 1], 1e-9)
    pd_ = np.asarray(p_dead, np.float64)[:, 2]
    out[:, 2] = pd_
    out[:, :2] *= ((1.0 - pd_) / alive)[:, None]
    return out


FINE_COLS = ("ball_vis", "ball_speed", "ball_oob_dist", "ball_d_touchline", "prox_dH", "prox_dA",
             "prox_ddiff", "prox_dmin", "prox_contact", "prox_nH_r3", "prox_nA_r3", "kin_acc_rel",
             "kin_acc_z", "kin_dtheta", "kin_dspeed", "kin_kick")
FINE_DIM = len(FINE_COLS) + 3                      # + booster log-probabilities


def fine_view(feats, probs, labels, steps, stride, cols=FINE_COLS):
    """Per step item (x, y, known, (a, b)): (xf (n, stride, F) robust-scaled frame features +
    booster log-probs, yf (n, stride) frame labels, -1 unknown)."""
    Z = np.concatenate([robust_scale(feats[list(cols)].to_numpy(np.float64)),
                        np.log(np.clip(probs, 1e-4, 1.0)).astype(np.float32)], axis=1)
    out = []
    for x, _, _, (a, b) in steps:
        n = len(x)
        e = a + n * stride
        xf = Z[a:e].reshape(n, stride, -1).astype(np.float32)
        if labels is None:
            yf = np.full((n, stride), -1, np.int64)
        else:
            lab = np.asarray(labels[a:e]).reshape(n, stride)
            yf = np.where(lab == C.X, -1, lab).astype(np.int64)
        out.append((xf, yf))
    return out


def net_probs(models, steps, n_frames, base_probs, stride, fine=None):
    """Per-frame net probabilities: 5 Hz steps interpolated to frames, or (fine branch) the
    25 fps output directly when `fine` (fine_view of the same steps) is given."""
    from ..models import seqnet
    out = base_probs.copy()
    for i, (x, _, _, (a, b)) in enumerate(steps):
        if fine is not None:
            pf = np.mean([seqnet.predict_fine(m, x, fine[i][0]) for m in models], axis=0)
            out[a:a + len(pf)] = pf / pf.sum(1, keepdims=True)
            continue
        ps = np.mean([seqnet.predict(m, x) for m in models], axis=0)
        e = a + len(ps) * stride
        out[a:e] = to_frames(ps, a, e, stride)
    return out


def net_boundary(models, steps, n_frames, stride):
    """Per-frame boundary-head probability (seed mean), each step's value repeated over its
    `stride` frames so frame a + t*stride holds step t of the run starting at a (the decoder's
    step grid, decoding.hsmm.boundary_bonus). 0 outside the runs."""
    from ..models import seqnet
    out = np.zeros(n_frames)
    for x, _, _, (a, b) in steps:
        bs = np.mean([seqnet.predict_boundary(m, x) for m in models], axis=0)
        fr = np.repeat(bs, stride)[:b - a]
        out[a:a + len(fr)] = fr
    return np.round(out, 6)


def run(config="main", weights=(0.0, 0.25, 0.5, 0.75, 1.0), temps=(1.0,), name=None, p_over=None):
    from ..models import seqnet
    t0 = time.time()
    cfg = config_mod.load(config)
    p = {**seqnet.DEFAULTS, **cfg.get("seqnet", {}), **(p_over or {})}
    stride = int(p["stride"])
    name = name or f"hybrid_{config}"
    out_dir = ensure(OUTPUTS / name)
    lengths = gap_lengths(bar_data(cfg))
    lomo = decoder_tune.lomo_probs(cfg, ensure(OUTPUTS / "decoder" / config))
    matches = list(lomo)
    groups = common_groups(cfg["features"]["groups"])
    preps, steps_by = {}, {}
    for m in matches:
        prep = dfl_referee.prepare(m, cfg, lengths)
        preps[m] = prep
        columns = list(prep["feats"].columns)
        steps_by[m] = step_view(prep["feats"], columns, lomo[m], prep["aligned"], prep["runs"], stride)
    print(f"[{time.time() - t0:5.0f}s] DFL step data: {[(m, sum(len(s[0]) for s in steps_by[m])) for m in matches]} "
          f"features {len(columns)} + 3 log-probs, device {seqnet.device()}", flush=True)

    # ---- DFL leave-one-match-out for the net ---------------------------------------------
    gt35 = {m: suppress_short_returns(preps[m]["data"].labels, preps[m]["runs"], 35, preps[m]["data"], True)
            for m in matches}
    net_lomo, logs = {}, {}
    for m in matches:
        train_seqs = [(x, y, k) for o in matches if o != m for x, y, k, _ in steps_by[o]]
        models = []
        for i in range(int(p["n_seeds"])):
            lg = []
            models.append(seqnet.train(train_seqs, p, int(cfg["seed"]) + 1000 * i, lg))
            logs[f"{m}_s{i}"] = lg[-1]
        net_lomo[m] = net_probs(models, steps_by[m], len(lomo[m]), lomo[m], stride)
        print(f"[{time.time() - t0:5.0f}s] net LOMO fold {m} done (last epoch {logs[f'{m}_s0']})", flush=True)
    table = []
    for w in weights:
        for T in temps:
            sp11, sp35, mac = [], [], []
            for m in matches:
                prep = preps[m]
                pred = engine.decode_runs(log_pool(lomo[m], net_lomo[m], w, T), prep["runs"], cfg)
                s = dfl_referee.score(pred, prep, None)
                sp11.append(s["spell_f1"])
                mac.append(s["macro_f1"])
                from ..validation.official import pm
                g = np.where(prep["data"].period > 0, gt35[m], C.X).astype(np.int8)
                sp35.append(pm.evaluate(g, pred)["spell_f1_mean"])
            table.append({"w_boost": w, "T": T, "dfl_spell_gt11": float(np.mean(sp11)),
                          "dfl_spell_gt35": float(np.mean(sp35)), "dfl_macro": float(np.mean(mac)),
                          "per_match_gt35": [round(x, 4) for x in sp35]})
            print(f"  w_boost {w:.2f} T {T}: DFL LOMO spell {np.mean(sp11):.4f} (gt35 {np.mean(sp35):.4f}) "
                  f"macro {np.mean(mac):.4f}", flush=True)
    best = max(table, key=lambda r: (r["dfl_spell_gt35"], r["dfl_macro"]))
    print(f"chosen on DFL LOMO (gt35): {best}", flush=True)

    # ---- BAR-ATH: net trained on all DFL matches --------------------------------------------
    all_seqs = [(x, y, k) for o in matches for x, y, k, _ in steps_by[o]]
    models = [seqnet.train(all_seqs, p, int(cfg["seed"]) + 1000 * i) for i in range(int(p["n_seeds"]))]
    bar = bar_data(cfg)
    feats = build_features(bar, cfg, load_tags(), cache_key=bar.name)
    bcols = [c for g in groups for c in feats.attrs["group_columns"][g]]
    assert bcols == columns, "BAR and DFL common columns differ"
    p_boost = np.load(OUTPUTS / f"{cfg['name']}_lgbm" / "probs.npy")
    bsteps = step_view(feats, bcols, p_boost, None, list(C.HALVES.values()), stride)
    p_net = net_probs(models, bsteps, len(p_boost), p_boost, stride)
    blocks = public_blocks()               # 'public' score = organisers' window only
    gt = merged_gt(blocks)
    ref_labels = m0.decode_match(p_boost, cfg)
    ref_seeds = json.loads((OUTPUTS / f"{cfg['name']}_lgbm" / "seeds.json").read_text())
    bands = reference_bands(cfg["acceptance"], ref_seeds, official_reference(cfg)[1])
    rows = []
    for w in sorted({best["w_boost"], 0.0, 1.0}):
        probs = log_pool(p_boost, p_net, w, best["T"])
        lab = m0.decode_match(probs, cfg)
        r = score_blocks(lab, blocks, bootstrap=1000)["ALL"]
        st = sanity.stats(lab)
        ok, detail = _in_bands(st, bands)
        pb = paired_spell_bootstrap(gt, ref_labels, lab)["p_b_better"]
        rows.append({"w_boost": w, "spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
                     "boot": [r.get("spell_bootstrap_low"), r.get("spell_bootstrap_high")],
                     "dead_share": st["dead_share"], "n_possessions": st["n_possessions"],
                     "home_share": st["home_share"], "in_ref_bands": ok, "p_better_vs_ref": pb})
        print(f"  BAR w_boost {w:.2f}: public spell {r['spell_f1_mean']:.4f} macro {r['macro_f1']:.4f} "
              f"D {st['dead_share']:.3f} poss {st['n_possessions']} home {st['home_share']:.3f} "
              f"bands {ok} P(better) {pb:.2f}", flush=True)
        if w == best["w_boost"]:
            np.save(out_dir / "probs.npy", probs)
            sub = submission.write(out_dir / "submission.csv", lab)
            reason = submission.validate(sub)
            print(f"  -> {sub} official format check: {'OK' if reason is None else reason}", flush=True)
    res = {"config": config, "seqnet": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
           "dfl_lomo": table, "chosen": best, "bar": rows, "train_logs": logs,
           "time": dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out_dir}")
    return res


TOKEN_SHAPE = (48, 9)


def player_tokens(data, cfg, runs, stride, n_max=TOKEN_SHAPE[0]):
    """[(S, n_max * 9) float32] per run, aligned with step_view (centre frame of each block).
    Token = [x, y (centred, canonical home-attack direction, /half pitch), vx, vy (/10 m/s),
    team sign, goalkeeper flag, dx, dy to the ball (/20 m; 0 when the ball is missing),
    present]. Missing slots / padding are all-zero (present = 0)."""
    from ..features.context import FeatureContext
    from ..preprocess import filled_period
    ctx = FeatureContext(data, cfg)
    L_, W_ = data.pitch
    xc, yy = ctx.xc_p, ctx.y_p                              # (T, N) canonical x, y
    vx, vy = ctx.vx_c, ctx.vp[..., 1]
    team_sign = np.where(data.team == 0, 1.0, -1.0)
    phase = filled_period(data.period)
    gk = np.zeros(xc.shape, bool)
    for per, (gh, ga, _) in ctx.gk_dir.items():
        gk[np.ix_(phase == per, [gh, ga])] = True
    bx, by = ctx.xc_b, data.ball[:, 1]
    N = min(xc.shape[1], n_max)
    out = []
    for a, b in runs:
        n = (b - a) // stride
        if n < 8:
            continue
        c = a + np.arange(n) * stride + stride // 2          # centre frames
        tok = np.zeros((n, n_max, 9), np.float32)
        present = np.isfinite(xc[c, :N])
        ball_ok = np.isfinite(bx[c])[:, None]
        f = [(xc[c, :N] - L_ / 2) / (L_ / 2), (yy[c, :N] - W_ / 2) / (W_ / 2),
             vx[c, :N] / 10.0, vy[c, :N] / 10.0,
             np.broadcast_to(team_sign[:N], (n, N)), gk[c, :N].astype(np.float32),
             np.where(ball_ok, (xc[c, :N] - bx[c, None]) / 20.0, 0.0),
             np.where(ball_ok, (yy[c, :N] - by[c, None]) / 20.0, 0.0)]
        for k, arr in enumerate(f):
            tok[:, :N, k] = np.where(present, np.nan_to_num(np.clip(arr, -5, 5)), 0.0)
        tok[:, :N, 8] = present
        out.append(tok.reshape(n, -1))
    return out


def with_tokens(steps, tokens):
    """Append token columns to step_view items (same runs, same order)."""
    assert len(steps) == len(tokens)
    return [(np.concatenate([x, t], axis=1), y, k, ab) for (x, y, k, ab), t in zip(steps, tokens)]
