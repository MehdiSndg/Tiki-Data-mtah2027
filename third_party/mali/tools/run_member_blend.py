"""CPU evaluation of two predeclared equal-weight posterior/boundary mixtures; no training."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

import run_mix_promotion as common

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "ev_set_base_rf0"
MEMBERS = {"base_plus_gru": "ev_set_gru_rf0", "base_plus_set": "ev_set_pool_rf0"}


def mix_probabilities(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Incompatible probability outputs")
    mixed = (a + b) * .5
    return mixed / mixed.sum(1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    job = Path(parser.parse_args().job_dir).resolve()
    if os.environ.get("MTAH_OFFLINE") != "1" or socket.socket.connect.__module__ != "sitecustomize":
        raise RuntimeError("Offline guard is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("This evaluation must use CPU only")
    sys.path.insert(0, str(ROOT / "src"))
    from mtah import config, constants as C
    from mtah.decoding import contact
    from mtah.io.broadcast import bar_data
    from mtah.labels.align import suppress_short_returns
    from mtah.labels.masking import gap_lengths
    from mtah.paths import OUTPUTS
    from mtah.pipeline import adapt, dfl_referee, engine, metrica_src
    from mtah.validation import sanity
    from mtah.validation.official import pm
    from mtah.validation.possession import diagnostics
    from run_short_team_screen import scores
    cfg = config.load(REFERENCE)
    dc = adapt.blend_cfg(cfg, adapt.params(cfg)[0])
    expected = scores(json.loads((OUTPUTS / REFERENCE / "summary.json").read_text(encoding="utf-8")))
    data = bar_data(cfg)
    lengths = gap_lengths(data)
    names = [REFERENCE, *MEMBERS.values()]
    rows = {m: {g: {} for g in ("dfl_v1", "dfl_v2b", "metrica")} for m in ("reference", *MEMBERS)}
    for group, pf, bf in (("dfl_v1", "dfl_lomo_blend.npz", "dfl_lomo_bnd.npz"),
                          ("dfl_v2b", "dfl_lomo_eval_blend.npz", "dfl_lomo_eval_bnd.npz"),
                          ("metrica", "metrica_eval_blend.npz", "metrica_eval_bnd.npz")):
        with ExitStack() as stack:
            ps = {n: stack.enter_context(np.load(OUTPUTS / n / pf)) for n in names}
            bs = {n: stack.enter_context(np.load(OUTPUTS / n / bf)) for n in names}
            if any(set(ps[n].files) != set(ps[REFERENCE].files) for n in names):
                raise ValueError("Members have different held-out matches")
            for match in ps[REFERENCE].files:
                if group == "metrica":
                    prep = metrica_src.prepare(int(match.split("_")[-1]), cfg)
                    gt = np.where(prep["data"].period > 0, prep["aligned"], C.X).astype(np.int8)
                else:
                    prep = dfl_referee.prepare(match, dfl_referee.eval_cfg(cfg) if group == "dfl_v2b" else cfg, lengths)
                    clean = prep["data"]
                    gt = np.where(clean.period > 0, suppress_short_returns(clean.labels, clean.runs(), 35,
                                  clean, True), C.X).astype(np.int8)
                regime = contact.regime_for(prep.get("corrupted", prep["data"]), dc)
                for method in rows:
                    p, b = ps[REFERENCE][match], bs[REFERENCE][match]
                    if method != "reference":
                        p = mix_probabilities(p, ps[MEMBERS[method]][match])
                        b = (b + bs[MEMBERS[method]][match]) * .5
                    pred = engine.decode_runs(p, prep["runs"], dc, boundary=b, regime=regime)
                    r = pm.evaluate(gt, pred)
                    rows[method][group][match] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                        "segments": diagnostics(gt, pred, prep["runs"], prep["data"].fps)}
    for method, row in rows.items():
        row["mean_scores"] = {g: float(np.mean([r["spell"] for r in row[g].values()]))
                              for g in ("dfl_v1", "dfl_v2b", "metrica")}
        p = np.load(OUTPUTS / REFERENCE / "probs.npy")
        b = np.load(OUTPUTS / REFERENCE / "bar_bnd_adapted.npy")
        if method != "reference":
            p = mix_probabilities(p, np.load(OUTPUTS / MEMBERS[method] / "probs.npy"))
            b = (b + np.load(OUTPUTS / MEMBERS[method] / "bar_bnd_adapted.npy")) * .5
        row["bar_health"] = sanity.stats(engine.decode_runs(p, list(C.HALVES.values()), dc, C.FORCED_DEAD,
                                             boundary=b, regime=contact.regime_for(data, dc)))
    for group in ("dfl_v1", "dfl_v2b", "metrica"):
        if abs(rows["reference"]["mean_scores"][group] - expected[group]) > 1e-8:
            raise RuntimeError("Reference reproduction mismatch")
    if rows["reference"]["bar_health"]["n_possessions"] != expected["bar"]["n_possessions"]:
        raise RuntimeError("Reference full-match health mismatch")
    result = {"weight": .5, "methods": rows, "note": "Fixed posterior and boundary averages; no weight selection, public scoring or promotion"}
    common.write_json(job / "fixed_blends.json", result)
    print("FIXED_BLENDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
