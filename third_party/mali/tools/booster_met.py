r"""Booster that also learns from the Metrica open games (Rapor 10, item 5; CPU only, the net is unchanged).

The run's booster trained on 4 DFL matches and never saw Metrica; in the blend (w_boost 0.25) it costs Metrica
LOGO ~0.03 while it helps DFL v2b. Here a second booster is trained on the same 4 DFL sources + Metrica games
(raw track: its ball already has ATD-like gaps, so no extra ball masking; plus a player-degraded copy with the
run's training corruption, ball untouched) and swapped into the blend in place of the old one:

  Metrica game g   booster on 4 DFL + the OTHER two Metrica games (nested leave-one-game-out)
  DFL-3 (v1, v2b)  booster on 4 DFL + 3 Metrica (it never saw these matches)
  BAR              same, for the label-free health gate only

The net probabilities are the run's own (saved / recovered from the saved blend), so this measures only the
booster's share of the blend. The net still received the OLD booster's probabilities as input; if this helps,
the full test retrains the net with the new booster input (GPU).

The reference (old booster) is recomputed with the same code and must match the run's summary.

  .venv\Scripts\python.exe tools/booster_met.py --run ev_ext_met --out outputs/booster_met_ev_ext_met.json
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah import cloud, config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.labels.align import suppress_short_returns  # noqa: E402
from mtah.paths import OUTPUTS, ensure  # noqa: E402
from mtah.pipeline import adapt, decoder_tune, dfl_referee, engine, joint, m0, metrica_src  # noqa: E402
from mtah.pipeline.hybrid import log_pool  # noqa: E402
from mtah.validation import sanity  # noqa: E402
from mtah.validation.official import pm  # noqa: E402

GAMES = (1, 2, 3)


@dataclass
class FixedSource(engine.TrainSource):
    """A source trained as it is: no masked copies (Metrica's own ball gaps are the masking)."""

    def matrix(self, cfg, i):
        return engine.training_matrix(self.feats, [], self.labels, self.team_ok, self.train_ok, self.stride)


def metrica_sources(cfg, games, share_per_game, rc, lengths=()):
    out = []
    stride = int(joint.joint_cfg(cfg)["dfl_stride"])
    for g in games:
        for kind, prep in (("raw", metrica_src.prepare(g, cfg)), ("deg", metrica_src.prepare_corrupted(g, cfg, rc, 0.0, lengths))):
            d = prep["data"]
            out.append(FixedSource(f"metrica_{g}_{kind}", prep["corrupted"], prep["feats"], prep["aligned"],
                                   np.ones(d.n_frames, bool), d.period > 0, prep["key"], None, None, stride,
                                   share=share_per_game / 2))
    return out


V2 = {}               # --v2 (12 Nov): view config + held-out PFF games; the new booster also predicts the BAR-like views


def v2_metrica(g, lengths, boost=False):
    """BAR-like view of a Metrica game: dead ball filled in, wandering, masked like a DFL track; ball-status labels.
    boost: with the new booster's feature groups (V2['bcfg'], --groups) instead of the run's."""
    c = V2.get("bcfg", V2["cfg"]) if boost else V2["cfg"]
    return metrica_src.prepare_corrupted(g, c, dfl_referee.referee_cfg(dfl_referee.eval_cfg(c)), 0.0, lengths)


def v2_pff(g, boost=False):
    from mtah.pipeline import skillcorner_src
    c = V2.get("bcfg", V2["cfg"]) if boost else V2["cfg"]
    return skillcorner_src.prepare_corrupted(f"pff_{g}", c, dfl_referee.referee_cfg(dfl_referee.eval_cfg(c)))


PFF_GAMES = []        # --pff N: PFF training games of the new booster (11 Nov), each weighted as a quarter match


def pff_sources(cfg, games, share_per_game):
    from mtah.pipeline import skillcorner_src
    out = []
    stride = int(joint.joint_cfg(cfg)["dfl_stride"])
    rc = dfl_referee.referee_cfg(dfl_referee.eval_cfg(cfg))
    for g in games:
        for kind, prep in (("v1", skillcorner_src.prepare(f"pff_{g}", cfg)),
                           ("v2b", skillcorner_src.prepare_corrupted(f"pff_{g}", cfg, rc))):
            d = prep["data"]
            out.append(FixedSource(f"pff_{g}_{kind}", prep["corrupted"], prep["feats"], prep["aligned"],
                                   np.ones(d.n_frames, bool), d.period > 0, prep["key"], None, None, stride,
                                   share=share_per_game / 2))
    return out


def fit(cfg, met_games, targets, columns, lengths):
    n = len(cfg["joint"]["matches"]) + len(met_games) + 0.25 * len(PFF_GAMES)
    sources = joint.dfl_sources(cfg, cfg["joint"]["matches"], lengths, share_total=len(cfg["joint"]["matches"]) / n)
    sources += metrica_sources(cfg, met_games, 1.0 / n, dfl_referee.referee_cfg(cfg), lengths)
    sources += pff_sources(cfg, PFF_GAMES, 0.25 / n)
    per_target, imp, counts = engine.fit_predict_sources(sources, targets, cfg, columns)
    print(f"    rows per source: {counts}", flush=True)
    return [engine.ensemble(p) for p in per_target]


def new_booster(cfg, columns, lengths, extra, ecfg, cache):
    if cache.is_file():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    out = {}
    for g in GAMES:
        others = [o for o in GAMES if o != g]
        print(f"[{time.time() - t0:5.0f}s] Metrica fold {g}: booster on 4 DFL + Metrica {others}", flush=True)
        tg = [metrica_src.prepare(g, cfg)["feats"]] + ([v2_metrica(g, lengths, True)["feats"]] if V2 else [])
        pr = fit(cfg, others, tg, columns, lengths)
        out[f"metrica_{g}"] = pr[0]
        if V2:
            out[f"metrica_{g}_bar"] = pr[1]
    print(f"[{time.time() - t0:5.0f}s] full: booster on 4 DFL + Metrica {list(GAMES)} -> DFL-3 v1/v2b + BAR", flush=True)
    ctx = adapt.bar_context(cfg)
    targets = ([dfl_referee.prepare(m, cfg, lengths)["feats"] for m in extra]
               + [dfl_referee.prepare(m, ecfg, lengths)["feats"] for m in extra] + [ctx["feats"]])
    targets += [v2_pff(g, True)["feats"] for g in V2.get("pff", [])]
    probs = fit(cfg, list(GAMES), targets, columns, lengths)
    for i, g in enumerate(V2.get("pff", [])):
        out[f"pff_{g}"] = probs[2 * len(extra) + 1 + i]
    for i, m in enumerate(extra):
        out[m] = probs[i]
        out[m + "_v2b"] = probs[len(extra) + i]
    out["BAR"] = probs[2 * len(extra)]
    np.savez_compressed(cache, **out)
    print(f"[{time.time() - t0:5.0f}s] booster done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="ev_ext_met")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="0,0.1,0.25,0.4")
    ap.add_argument("--decoder-config", help="score with this config's tuned decoder (e.g. main_ext_met_dt)")
    ap.add_argument("--groups", nargs="+", help="10 Nov: feature groups of the NEW booster only (screen of groups in "
                    "the blend; the nets and the old booster are unchanged); needs --tag")
    ap.add_argument("--tag", help="cache / variant name suffix for --groups")
    ap.add_argument("--gt-labels", choices=["ball"], help="11 Nov: score Metrica against the ball-status labels "
                    "(metrica_labels: ball) whatever the run trained on")
    ap.add_argument("--booster-labels", choices=["ball"], help="11 Nov: train the NEW booster's Metrica games on the "
                    "ball-status labels (one cache for all runs: booster_met/labels_ball.npz)")
    ap.add_argument("--dead-fill", action="store_true", help="11 Nov: new booster's degraded Metrica copy shows the "
                    "dead ball like a DFL track (metrica_dead_fill); cache booster_met/labels_ball_fill.npz")
    ap.add_argument("--dead-drift", type=float, help="11 Nov: with --dead-fill, the filled dead ball wanders at the "
                    "target's out-of-bounds ball speeds times this scale (metrica_dead_drift)")
    ap.add_argument("--dfl-drift", type=float, help="11 Nov: the new booster's DFL evaluation-corruption training copy "
                    "gets the wandering dead ball too (dfl_referee_eval.dead_drift); referee views unchanged")
    ap.add_argument("--pff", type=int, default=0, help="11 Nov: add the first N PFF training games (group games g[::3]) "
                    "to the new booster")
    ap.add_argument("--v2", action="store_true", help="12 Nov: also score the BAR-like Metrica view (fold nets) and 12 "
                    "held-out PFF games (final nets); needs the run's saved nets")
    ap.add_argument("--booster-event", action="store_true", help="12 Nov: the new booster is the OFFICIAL one (event-derived "
                    "Metrica labels, no fill / drift) whatever the run config says; one shared cache (event_v2.npz)")
    ap.add_argument("--team-avg-run", help="12 Nov: average --run's net probabilities with this run's (same seed set and "
                    "booster source) before anything else: a two-recipe team ensemble")
    ap.add_argument("--dead-run", help="12 Nov: two-run composition -- P(D) from this run's nets (same seed set, same "
                    "booster source), the team split of the alive mass from --run's nets")
    ap.add_argument("--dead-mix", type=float, default=1.0, help="12 Nov: with --dead-run, P(D) = mix * dead run + "
                    "(1 - mix) * --run (1 = dead run only; 0.5 = plain average of the two recipes)")
    ap.add_argument("--save-bar", help="write the decoded BAR labels of the new-booster blend (w = last weight) here (.npy)")
    args = ap.parse_args()
    if args.pff:
        from mtah.io import pff
        PFF_GAMES.extend([x for x in pff.games() if x.startswith("38")][::3][:args.pff])
    run = OUTPUTS / args.run
    summ = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summ["config"])
    if not cfg.get("joint", {}).get("external_only"):
        raise SystemExit("external_only runs only")
    ac, _ = adapt.params(cfg)
    w0 = float(ac["w_boost"])
    dcfg = adapt.blend_cfg(cfg, ac)
    if args.decoder_config:
        dt = config_mod.load(args.decoder_config)
        dcfg = dict(dcfg, decoder=adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"])
    folder = OUTPUTS / "decoder" / (ac.get("booster_source") or cfg["name"])
    lengths = cloud.bar_lengths(cfg)
    ecfg = dfl_referee.eval_cfg(cfg)
    extra = [m for m in ac["eval_matches"] if not m.startswith("metrica_")]
    assert not set(extra) & set(cfg["joint"]["matches"])
    columns = list(dfl_referee.prepare(extra[0], cfg, lengths)["feats"].columns)
    run_labels = cfg.get("metrica_labels")
    gcfg = dict(cfg, metrica_labels=args.gt_labels or run_labels)          # referee labels
    bcfg = dict(cfg, metrica_labels=args.booster_labels or run_labels)     # new booster's Metrica labels
    if args.v2:
        from mtah.io import pff
        V2["cfg"] = dict(cfg, metrica_labels="ball", metrica_dead_fill=True, metrica_dead_drift=1.5)   # one view for all runs
        V2["pff"] = [x for x in pff.games() if x.startswith("38")][1::3][:12]
        assert not {f"pff_{g}" for g in V2["pff"]} & {str(g) for g in ac.get("skillcorner_games") or ()}
        assert not set(V2["pff"]) & set(PFF_GAMES)
    if args.booster_event:
        bcfg = {k: v for k, v in cfg.items() if k not in ("metrica_labels", "metrica_dead_fill", "metrica_dead_drift",
                                                           "metrica_min_dead")}
        bcfg["metrica_labels"] = None
    if not args.booster_event and (args.dead_fill or cfg.get("metrica_dead_fill")):
        bcfg["metrica_dead_fill"] = True
    if args.dead_drift:
        bcfg["metrica_dead_drift"] = args.dead_drift
    if args.dfl_drift:
        assert bcfg.get("dfl_mix_eval") and bcfg.get("dfl_referee_eval"), "--dfl-drift needs the mixed training copies"
        speeds = [round(float(x), 3) for x in metrica_src.bar_dead_speeds(cfg)]
        bcfg["dfl_referee_eval"] = dict(bcfg["dfl_referee_eval"], dead_drift=args.dfl_drift, dead_drift_speeds=speeds)
    print(f"Metrica labels: referee {gcfg['metrica_labels'] or 'event'}, new booster {bcfg['metrica_labels'] or 'event'}",
          flush=True)

    old_v1 = decoder_tune.unseen_probs(cfg, extra, folder)
    for g in GAMES:
        m = f"metrica_{g}"
        old_v1[m] = decoder_tune.unseen_probs(cfg, [m], folder, preps=[metrica_src.prepare(g, cfg)])[m]
    old_v2b = decoder_tune.eval_probs(cfg, folder, tuple(extra))
    cache = ensure(OUTPUTS / "booster_met") / (f"labels_{bcfg['metrica_labels']}{'_fill' if bcfg.get('metrica_dead_fill') else ''}{'_dd%g' % bcfg['metrica_dead_drift'] if bcfg.get('metrica_dead_drift') else ''}{'_dfl%g' % args.dfl_drift if args.dfl_drift else ''}{'_pff%d' % args.pff if args.pff else ''}{'_v2' if args.v2 else ''}.npz" if bcfg["metrica_labels"]
                                                else ("event" if args.booster_event else args.run) + ('_v2' if args.v2 else '') + ".npz")
    if args.groups:
        assert args.tag, "--groups needs --tag"
        ncfg = config_mod.with_groups(bcfg, args.groups)
        if args.v2:
            V2["bcfg"] = config_mod.with_groups(V2["cfg"], args.groups)
        ncols = list(dfl_referee.prepare(extra[0], ncfg, lengths)["feats"].columns)
        print(f"new booster groups {args.groups}: {len(ncols)} columns (run: {len(columns)})", flush=True)
        # the booster does not depend on the net seed set: one cache per tag
        new = new_booster(ncfg, ncols, lengths, extra, dfl_referee.eval_cfg(ncfg),
                          cache.with_name(f"groups_{args.tag}.npz" if not bcfg["metrica_labels"]
                                          else cache.stem + f"_g{args.tag}.npz"))
    else:
        new = new_booster(bcfg, columns, lengths, extra, ecfg, cache)      # targets: cfg-independent views (v1, ecfg)

    from mtah.pipeline.hybrid import compose_dead_team as compose
    if args.dead_mix < 1.0:
        _compose, _mx = compose, float(args.dead_mix)
        compose = lambda pd_, pt: _compose(_mx * np.asarray(pd_, np.float64) + (1 - _mx) * np.asarray(pt, np.float64), pt)  # noqa: E731

    drun = OUTPUTS / args.dead_run if args.dead_run else None
    xrun = OUTPUTS / args.team_avg_run if args.team_avg_run else None
    avg = lambda a, b: (np.asarray(a, np.float64) + np.asarray(b, np.float64)) / 2.0      # noqa: E731
    net_v1 = np.load(run / "dfl_lomo_net_adapted.npz")
    bnd_v1 = np.load(run / "dfl_lomo_bnd.npz")
    blend_v2b = np.load(run / "dfl_lomo_eval_blend.npz")
    bnd_v2b = np.load(run / "dfl_lomo_eval_bnd.npz")

    def net_from_blend(pr, pb):     # exact inverse of the log-pool up to the 6-digit rounding
        lp = (np.log(np.clip(pr, 1e-6, 1)) - w0 * np.log(np.clip(pb, 1e-6, 1))) / (1 - w0)
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    refs = []          # (view, match, prep, gt, net, old booster, new booster, boundary)
    for m in net_v1.files:
        if m.startswith("metrica_"):
            prep = metrica_src.prepare(int(m.split("_")[1]), gcfg)
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
        else:
            prep = dfl_referee.prepare(m, cfg, lengths)
            gt = np.where(prep["data"].period > 0,
                          suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                          C.X).astype(np.int8)
        pn = net_v1[m] if xrun is None else avg(net_v1[m], np.load(xrun / "dfl_lomo_net_adapted.npz")[m])
        pn = pn if drun is None else compose(np.load(drun / "dfl_lomo_net_adapted.npz")[m], pn)
        refs.append(("metrica" if m.startswith("metrica_") else "v1", m, prep, gt, pn, old_v1[m], new[m], bnd_v1[m]))
    for m in blend_v2b.files:
        prep = dfl_referee.prepare(m, ecfg, lengths)
        gt = np.where(prep["data"].period > 0,
                      suppress_short_returns(prep["data"].labels, prep["runs"], 35, prep["data"], True),
                      C.X).astype(np.int8)
        pn = net_from_blend(blend_v2b[m], old_v2b[m])
        if xrun is not None:
            pn = avg(pn, net_from_blend(np.load(xrun / "dfl_lomo_eval_blend.npz")[m], old_v2b[m]))
        if drun is not None:
            pn = compose(net_from_blend(np.load(drun / "dfl_lomo_eval_blend.npz")[m], old_v2b[m]), pn)
        refs.append(("v2b", m, prep, gt, pn, old_v2b[m], new[m + "_v2b"], bnd_v2b[m]))

    if args.v2:
        import bar_from_nets
        from mtah.pipeline import skillcorner_src
        from mtah.pipeline.hybrid import net_boundary, net_probs, step_view
        stride = int(adapt.params(cfg)[1]["stride"])

        def net_view(folder_nets, prep, pin):
            nets, ncols = bar_from_nets.load_nets(folder_nets)
            assert list(prep["feats"].columns) == ncols
            models = [n for _, n in nets]
            steps = step_view(prep["feats"], ncols, pin, None, prep["runs"], stride)
            return net_probs(models, steps, len(pin), pin, stride, None), net_boundary(models, steps, len(pin), stride)

        for g in GAMES:
            prep = v2_metrica(g, lengths)
            nm = f"metrica_{g}_v2bfd1.5"
            pin = decoder_tune.unseen_probs(cfg, [nm], folder, preps=[prep])[nm]
            pn, b = net_view(run / "fold_nets" / f"metrica_{g}", prep, pin)
            if xrun is not None:
                pn = avg(pn, net_view(xrun / "fold_nets" / f"metrica_{g}", prep, pin)[0])
            if drun is not None:
                pn = compose(net_view(drun / "fold_nets" / f"metrica_{g}", prep, pin)[0], pn)
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            refs.append(("metbar", f"metrica_{g}", prep, gt, pn, pin, new[f"metrica_{g}_bar"], b))
        for g in V2["pff"]:
            prep = v2_pff(g)
            nm = skillcorner_src.name(f"pff_{g}", V2["cfg"]) + "_v2b"
            pin = decoder_tune.unseen_probs(cfg, [nm], folder, preps=[prep])[nm]
            pn, b = net_view(run / "nets", prep, pin)
            if xrun is not None:
                pn = avg(pn, net_view(xrun / "nets", prep, pin)[0])
            if drun is not None:
                pn = compose(net_view(drun / "nets", prep, pin)[0], pn)
            gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
            refs.append(("pff", g, prep, gt, pn, pin, new[f"pff_{g}"], b))

    def score(which, w):
        per = {}
        for view, m, prep, gt, pn, pb_old, pb_new, b in refs:
            pb = pb_old if which == "old" else pb_new
            pred = engine.decode_runs(log_pool(pb, pn, w), prep["runs"], dcfg, boundary=b,
                                      regime=contact.regime_for(prep["corrupted"], dcfg))
            per.setdefault(view, {})[m] = round(pm.evaluate(gt, pred)["spell_f1_mean"], 4)
        return {v: {"mean": round(float(np.mean(list(d.values()))), 4), "per_match": d} for v, d in per.items()}

    bar_net, bar_bnd = np.load(run / "bar_net_adapted.npy"), np.load(run / "bar_bnd_adapted.npy")
    if xrun is not None:
        bar_net = avg(bar_net, np.load(xrun / "bar_net_adapted.npy"))
    if drun is not None:
        bar_net = compose(np.load(drun / "bar_net_adapted.npy"), bar_net)
    bar_old = np.load(OUTPUTS / f"{ac.get('booster_source') or cfg['name']}_lgbm" / "probs.npy")
    bar_data = adapt.bar_context(cfg)["data"]

    def bar_stats(pb, w):
        lab = m0.decode_match(log_pool(pb, bar_net, w), dcfg, bar_bnd, regime=contact.regime_for(bar_data, dcfg))
        if args.save_bar and pb is not bar_old:
            np.save(args.save_bar, np.asarray(lab, np.int8))
        st = sanity.stats(lab)
        return {k: st[k] for k in ("n_possessions", "dead_share", "home_share")}

    res = {"run": args.run, "w_boost_run": w0, "reference_summary": None, "rows": [],
           "metrica_labels": {"referee": gcfg["metrica_labels"] or "event", "booster": bcfg["metrica_labels"] or "event"}}
    try:
        import run_day_screen as screen
        res["reference_summary"] = {k: screen.met_scores(summ)[k] for k in ("dfl_v1", "dfl_v2b", "metrica")}
    except Exception as e:      # noqa: BLE001
        print("summary read failed:", e)
    for w in [float(x) for x in args.weights.split(",")]:
        for which in ("old", "new"):
            if w == 0.0 and which == "new":
                continue
            s = score(which, w)
            row = {"booster": which if w else "none (net only)", "w": w, **{k: v["mean"] for k, v in s.items()},
                   "per_match": {k: v["per_match"] for k, v in s.items()},
                   "bar": bar_stats(bar_old if which == "old" else new["BAR"], w)}
            res["rows"].append(row)
            print(f"  w {w:.2f} {row['booster']:<16} v1 {row['v1']:.4f} v2b {row['v2b']:.4f} Metrica {row['metrica']:.4f} "
                  + (f"MetBAR {row['metbar']:.4f} PFF {row['pff']:.4f} " if args.v2 else "")
                  + f"| BAR poss {row['bar']['n_possessions']} D {row['bar']['dead_share']:.3f} "
                  f"home {row['bar']['home_share']:.3f}", flush=True)
    print("reference from summary:", res["reference_summary"])
    Path(args.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
