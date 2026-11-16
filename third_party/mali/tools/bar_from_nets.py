r"""Predict the provided game LOCALLY from nets trained elsewhere (cloud mode, src/mtah/cloud.py).

Reads outputs/<run>/nets/net_<seed>.pt and the local booster probabilities of the run's booster source
(outputs/<source>_lgbm/probs.npy; build them first with `run.py m0 --config <source>` if missing), then
does exactly what `adapt` does for the provided game in an external_only run: step view, net
probabilities and boundary, log-pool blend, decode, label-free statistics, submission. Writes
probs.npy, bar_net_adapted.npy, bar_bnd_adapted.npy, submission.csv and the summary's "bar" rows
(no public-window score: that is a format check only and is not needed for the gates).

  .venv\Scripts\python.exe tools/bar_from_nets.py --run ev_ext_met_cloudref
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import config as config_mod  # noqa: E402
from mtah import constants as C  # noqa: E402
from mtah.decoding import contact  # noqa: E402
from mtah.io import submission  # noqa: E402
from mtah.paths import OUTPUTS  # noqa: E402
from mtah.pipeline import adapt, m0  # noqa: E402
from mtah.pipeline.hybrid import log_pool, net_boundary, net_probs, step_view  # noqa: E402
from mtah.validation import sanity  # noqa: E402


def load_nets(folder):
    import torch
    from mtah.models import seqnet
    nets, columns = [], None
    for f in sorted(folder.glob("net_*.pt"), key=lambda f: int(f.stem.split("_")[1])):
        d = torch.load(f, map_location="cpu", weights_only=False)
        mdl = seqnet.SeqNet(d["n_in"], d["p"])
        mdl.load_state_dict(d["state"])
        nets.append((d["seed"], mdl.to(seqnet.device()).eval()))
        columns = d["columns"] if columns is None else columns
        assert d["columns"] == columns
    if not nets:
        raise SystemExit(f"no nets in {folder}")
    return nets, columns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    out = OUTPUTS / args.run
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cfg = config_mod.load(summary["config"])
    if not cfg.get("joint", {}).get("external_only"):
        raise SystemExit("only external_only runs can be predicted from saved nets")
    ac, p = adapt.params(cfg)
    dcfg = adapt.blend_cfg(cfg, ac)
    stride = int(p["stride"])
    seeds_run = [int(s) for s in summary["net_seeds"]]
    nets, columns = load_nets(out / "nets")
    if [s for s, _ in nets] != seeds_run:
        raise SystemExit(f"nets {[s for s, _ in nets]} do not match the run's seeds {seeds_run}")
    source = ac.get("booster_source") or cfg["name"]
    p_boost = np.load(OUTPUTS / f"{source}_lgbm" / "probs.npy")
    ctx = adapt.bar_context(cfg)
    local_cols = [c for g in adapt.common_groups(cfg["features"]["groups"]) for c in ctx["feats"].attrs["group_columns"][g]]
    if local_cols != columns:
        raise SystemExit("feature columns of the provided game differ from the nets' training columns")
    inf = step_view(ctx["feats"], columns, p_boost, None, list(C.HALVES.values()), stride)
    models = [m for _, m in nets]
    p_net = net_probs(models, inf, len(p_boost), p_boost, stride, None)
    bnd = net_boundary(models, inf, len(p_boost), stride)
    np.save(out / "bar_net_adapted.npy", p_net)
    np.save(out / "bar_net_base.npy", p_net)            # external_only: base = adapted
    np.save(out / "bar_bnd_adapted.npy", bnd)
    rows = []
    for label in ("base", "adapted"):
        for w in (0.0, ac["w_boost"]):
            probs = log_pool(p_boost, p_net, w)
            lab = m0.decode_match(probs, dcfg, bnd, regime=contact.regime_for(ctx["data"], dcfg))
            st = sanity.stats(lab)
            rows.append({"net": label, "w_boost": w, "dead_share": st["dead_share"], "n_possessions": st["n_possessions"],
                         "home_share": st["home_share"], "predicted_locally": True})
            if label == "adapted" and w == ac["w_boost"]:
                np.save(out / "probs.npy", probs)
                sub = submission.write(out / "submission.csv", lab)
                reason = submission.validate(sub)
                print(f"  -> {sub} official format check: {'OK' if reason is None else reason}")
            print(f"  BAR {label:<8} w_boost {w:.2f}: D {st['dead_share']:.3f} poss {st['n_possessions']} "
                  f"home {st['home_share']:.3f}")
    summary["bar"] = rows
    summary["bar_predicted_locally"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("BAR_FROM_NETS_COMPLETE")


if __name__ == "__main__":
    main()
