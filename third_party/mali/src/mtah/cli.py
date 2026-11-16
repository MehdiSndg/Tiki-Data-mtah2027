"""Command-line dispatcher used by run.py."""

import argparse
import time


def _num(v):
    """'3' -> 3, '0.5' / '1e-4' -> float, anything else (e.g. 'split') -> str."""
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run.py", description="MTAH 2027 pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cache", help="parse BAR-ATH tracking into cache/barath")

    p_m0 = sub.add_parser("m0", help="end-to-end M0 baseline")
    p_m0.add_argument("--emission", choices=("lgbm", "rules"), default="lgbm")
    p_m0.add_argument("--config", default="m0")
    p_m0.add_argument("--bootstrap", type=int, default=1000)

    p_abl = sub.add_parser("ablate", help="group-wise forward ablation with the acceptance rule")
    p_abl.add_argument("--base", default="m0")
    p_abl.add_argument("--groups", nargs="+", default=["kin", "geo", "press"])
    p_abl.add_argument("--bootstrap", type=int, default=1000)
    p_abl.add_argument("--verbose", action="store_true")
    p_abl.add_argument("--seeds", type=int, default=None,
                       help="seed-ensemble size (default acceptance.n_seeds)")
    p_abl.add_argument("--dfl", nargs="*", default=["J03WPY"], help="IDSSE matches for the referee")

    p_a3 = sub.add_parser("adim3", help="compare configs vs a reference: BAR rule + DFL referee/transfer")
    p_a3.add_argument("--reference", default="main")
    p_a3.add_argument("--configs", nargs="+", required=True)
    p_a3.add_argument("--bootstrap", type=int, default=1000)
    p_a3.add_argument("--verbose", action="store_true")

    p_hy = sub.add_parser("hybrid", help="boosting + temporal net (TCN+BiGRU) blend, DFL LOMO + BAR submission")
    p_hy.add_argument("--config", default="main")
    p_hy.add_argument("--name", default=None)

    p_ad = sub.add_parser("adapt", help="fine-tune the temporal net on BAR weak anchors + re-blend")
    p_ad.add_argument("--config", default="main")
    p_ad.add_argument("--name", default="hybrid_adapted")
    p_ad.add_argument("--d-share", type=float, default=None, help="D weight share of BAR anchors (e.g. 0.35)")
    p_ad.add_argument("--bar-dead-only", action="store_true", help="BAR windows teach dead vs alive only")
    p_ad.add_argument("--bar-team-weight", type=float, default=None, help="lambda of the BAR team loss (0..1)")
    p_ad.add_argument("--dfl-target", choices=("gt11", "gt15", "gt20", "gt25", "gt35", "ctrl"), default=None,
                      help="DFL labels the net trains on (ctrl = Metrica-like controlled possession)")
    p_ad.add_argument("--extra-matches", nargs="*", default=None, help="unseen DFL matches added to the net's data")
    p_ad.add_argument("--net-seeds", type=int, default=None, help="temporal-net ensemble size (default 3)")
    p_ad.add_argument("--metrica", nargs="*", type=int, default=None, help="Metrica open games to add (1 2 3)")
    p_ad.add_argument("--metrica-eval", nargs="*", type=int, default=None,
                      help="Metrica open games used only as a referee (never trained on)")
    p_ad.add_argument("--net-set", nargs="*", default=None, help="seqnet overrides key=value (e.g. lambda_tmse=0.3)")
    p_ad.add_argument("--adapt-set", nargs="*", default=None, help="adapt overrides key=value (bar_share, ft_epochs, ft_lr, w_boost)")

    p_ds = sub.add_parser("dsweep", help="joint decoder sweep (beta x turnover x gamma) on a finished run; DFL LOMO + Metrica + full-match stats")
    p_ds.add_argument("--config", default="main")
    p_ds.add_argument("--run", default="main")
    p_ds.add_argument("--betas", nargs="*", type=float, default=[0.8, 1.0, 1.25, 1.5])
    p_ds.add_argument("--turnovers", nargs="*", type=float, default=[8.0, 10.0, 11.0, 12.0, 14.0])
    p_ds.add_argument("--gammas", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    p_main = sub.add_parser("main", help="official recipe (30 Oct: main_ha = mixed v1+v2b + H<->A) -> outputs/<config>; "
                                         "promotion to outputs/main is a separate sealed step")
    p_main.add_argument("--config", default="main_ha", help="main = previous official cc1a recipe")

    p_score = sub.add_parser("score", help="score a submission on the public window (format / sanity diagnostic only)")
    p_score.add_argument("csv")
    p_score.add_argument("--bootstrap", type=int, default=1000)

    p_idsse = sub.add_parser("idsse", help="download + convert IDSSE/DFL open matches")
    p_idsse.add_argument("matches", nargs="*", help="match ids (default: all seven)")
    p_idsse.add_argument("--sizes", action="store_true", help="only print remote file sizes")

    args = parser.parse_args(argv)

    if args.cmd == "cache":
        from .io.atd import cached_barath
        t0 = time.time()
        data = cached_barath(rebuild=True)
        print(f"cache/barath: players {data.players.shape}, ball {data.ball.shape} "
              f"in {time.time() - t0:.1f}s")
    elif args.cmd == "m0":
        from .pipeline.m0 import run
        run(emission=args.emission, cfg_name=args.config, bootstrap=args.bootstrap)
    elif args.cmd == "ablate":
        from .pipeline.ablation import run_ablation
        run_ablation(base=args.base, candidates=args.groups, bootstrap=args.bootstrap,
                     verbose=args.verbose, n_seeds=args.seeds, dfl_matches=tuple(args.dfl))
    elif args.cmd == "adim3":
        from .pipeline.adim3 import run as run_adim3
        run_adim3(reference=args.reference, candidates=args.configs, bootstrap=args.bootstrap,
                  verbose=args.verbose)
    elif args.cmd == "hybrid":
        from .pipeline.hybrid import run as run_hybrid
        run_hybrid(config=args.config, name=args.name)
    elif args.cmd == "dsweep":
        from .pipeline.decoder_tune import joint_sweep
        joint_sweep(args.config, args.run, tuple(args.betas), tuple(args.turnovers), tuple(args.gammas))
    elif args.cmd == "adapt":
        from .pipeline.adapt import run as run_adapt
        over = {} if args.d_share is None else {"d_share": args.d_share}
        if args.bar_dead_only:
            over["bar_team"] = False
        if args.bar_team_weight is not None:
            over["bar_team"] = args.bar_team_weight
        if args.dfl_target is not None:
            over["dfl_target"] = args.dfl_target
        if args.extra_matches:
            over["extra_matches"] = tuple(args.extra_matches)
        net = {}
        if args.net_seeds:
            net["n_seeds"] = args.net_seeds
        for kv in args.net_set or ():
            k, v = kv.split("=", 1)
            net[k] = _num(v)
        if net:
            over["seqnet"] = net
        if args.metrica:
            over["metrica_games"] = tuple(args.metrica)
        if args.metrica_eval:
            over["metrica_eval"] = tuple(args.metrica_eval)
        for kv in args.adapt_set or ():
            k, v = kv.split("=", 1)
            over[k] = float(v) if any(c in v for c in ".e") else int(v)
        run_adapt(config=args.config, name=args.name, over=over or None)
    elif args.cmd == "main":
        from .pipeline.mainrun import run as run_main
        run_main(config=args.config)
    elif args.cmd == "score":
        from .validation.official import pm
        from .validation.scoring import report, score_blocks
        table = pm.read_label_csv(args.csv)
        reason = pm.validate_submission(table, len(table))
        if reason:
            print(f"FAIL {reason}")
            return 1
        print(report(score_blocks(pm.label_codes(table), bootstrap=args.bootstrap), args.csv))
    elif args.cmd == "idsse":
        from .io import idsse
        if args.sizes:
            for mid, (meta, track) in idsse.file_sizes().items():
                print(f"{mid}: meta {meta / 1e6:.1f} MB, positions {track / 1e6:.1f} MB")
            return 0
        for mid in args.matches or list(idsse.MATCHES):
            t0 = time.time()
            data = idsse.cached_idsse(mid)
            lab = data.labels
            share = {k: float((lab == i).mean()) for i, k in enumerate("HADX")}
            print(f"{mid}: {data.players.shape} periods {sorted(set(data.period.tolist()))} "
                  f"labels {share} ({time.time() - t0:.0f}s)")
    return 0
