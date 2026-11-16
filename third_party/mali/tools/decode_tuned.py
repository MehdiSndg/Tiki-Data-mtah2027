r"""Decode the provided game with a decoder tuned on external data only (tools/tune_decoder_ext.py output).

Fixed probabilities of a finished `adapt` run; nothing is retrained and no ground truth is read.
Writes outputs/<out>/submission.csv (official format check) and a label-free summary.

  .venv\Scripts\python.exe tools/decode_tuned.py --run main_ext_met --config main_ext_met_dt --out main_ext_met_dt
  (or --tuned <json> instead of --config: the decoder straight from the tuning output)
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import config as config_mod  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.io import submission  # noqa: E402
from mtah.io.broadcast import bar_data  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, m0  # noqa: E402
from mtah.validation import sanity  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tuned", help="JSON written by tools/tune_decoder_ext.py")
    src.add_argument("--config", help="config holding the tuned decoder (e.g. main_ext_met_dt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--booster", help="booster stage whose BAR probabilities replace the run's in the blend "
                                      "(outputs/<name>_lgbm, same features), e.g. ev_ext_mb")
    ap.add_argument("--dead-run", help="12 Nov: two-model composition -- P(D) from this run's BAR net probabilities, "
                                       "the team split of the alive mass from --run's (pipeline.hybrid.compose_dead_team)")
    ap.add_argument("--dead-mix", type=float, default=1.0, help="12 Nov: with --dead-run, P(D) = mix * dead run + "
                    "(1 - mix) * --run (1 = dead run only; 0.5 = plain average of the two recipes)")
    args = ap.parse_args()
    run = OUTPUTS / args.run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summary["config"])
    ac, _ = adapt.params(cfg)
    c = copy.deepcopy(adapt.blend_cfg(cfg, ac))
    if args.tuned:
        tuned = json.loads(Path(args.tuned).read_text(encoding="utf-8"))
        if tuned["run"] != args.run:
            raise SystemExit(f"tuned JSON belongs to run {tuned['run']}, not {args.run}")
        c["decoder"] = tuned["decoder_after"]
    else:
        dt = config_mod.load(args.config)
        c["decoder"] = adapt.blend_cfg(dt, adapt.params(dt)[0])["decoder"]
    probs, bnd = np.load(run / "probs.npy"), np.load(run / "bar_bnd_adapted.npy")
    if args.dead_run:
        if not args.booster:
            raise SystemExit("--dead-run needs --booster (the blend is rebuilt from the nets)")
    if args.booster:
        # 6 Nov: swap the blend's booster for another booster stage on the SAME features (e.g. ev_ext_mb:
        # the run's booster + Metrica). The run's own blend must be reproduced from its saved net first.
        from mtah.pipeline.hybrid import log_pool
        net = np.load(run / "bar_net_adapted.npy")
        old = np.load(OUTPUTS / f"{ac.get('booster_source') or cfg['name']}_lgbm" / "probs.npy")
        w = float(ac["w_boost"])
        err = float(np.nanmax(np.abs(log_pool(old, net, w) - probs)))
        if err > 1e-5:
            raise SystemExit(f"cannot reproduce the run's blend from its net (max diff {err})")
        if args.dead_run:
            from mtah.pipeline.hybrid import compose_dead_team
            dead = np.load(OUTPUTS / args.dead_run / "bar_net_adapted.npy").astype(np.float64)
            net = compose_dead_team(args.dead_mix * dead + (1.0 - args.dead_mix) * np.asarray(net, np.float64), net)
        probs = log_pool(np.load(OUTPUTS / f"{args.booster}_lgbm" / "probs.npy"), net, w)
    labels = m0.decode_match(probs, c, bnd, regime=contact.regime_for(bar_data(cfg), c))
    out = OUTPUTS / args.out
    path = submission.write(out / "submission.csv", labels)
    problem = submission.validate(path)
    if problem:
        raise SystemExit(f"official format check failed: {problem}")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stats = {k: v for k, v in sanity.stats(labels).items() if k in ("n_possessions", "dead_share", "home_share")}
    base = np.asarray(m0.decode_match(probs, adapt.blend_cfg(cfg, ac), bnd,
                                      regime=contact.regime_for(bar_data(cfg), adapt.blend_cfg(cfg, ac))))
    changed = float(np.mean(base != np.asarray(labels)))
    res = {"run": args.run, "tuned": args.tuned, "config": args.config, "booster": args.booster, "dead_run": args.dead_run, "dead_mix": args.dead_mix if args.dead_run else None, "sha16": sha, "bar_label_free": stats,
           "frames_changed_vs_untuned": changed, "decoder": c["decoder"]}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("sha16", "bar_label_free", "frames_changed_vs_untuned")}, default=float))


if __name__ == "__main__":
    main()
