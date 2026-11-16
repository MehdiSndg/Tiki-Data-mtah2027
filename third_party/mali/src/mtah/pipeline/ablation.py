"""Group-wise forward ablation: n-seed ensembles, user-approved acceptance rule, DFL referee.

Starting from the reference config's groups, each candidate group is added on its own on top of
the current accepted set. Reference and candidate are both cfg['acceptance']['n_seeds']-seed
ensembles (feature matrices cached; only LightGBM refitted per seed). A candidate is KEPT only if:
    1. DFL referee spellF1 ('weak' mode: the BAR weak-label pipeline on DFL, real labels) improves,
    2. DFL referee macroF1 drops by at most acceptance.macro_tolerance,
    3. its full-match statistics lie inside mean +- band_sigmas * std of the REFERENCE's
       single-seed decodes (bands re-derived whenever the reference changes).
25 Oct (upstream rules): the public window is not a tuning set; its score is reported only. Every BAR-ATH run is logged to outputs/experiments.csv.
"""

import datetime as dt

import numpy as np

from .. import config as config_mod
from ..io.broadcast import bar_data
from ..labels.masking import gap_lengths
from ..paths import CONFIGS, OUTPUTS, ensure
from ..validation.compare import paired_spell_bootstrap
from . import dfl_referee, m0


def derive_bands(seed_rows, ac):
    """mean +- max(band_sigmas * sd, band_floor) of the reference's single-seed decodes.
    The floor (user rule, 22 Oct) keeps an artificially tight seed spread from rejecting
    models whose full-match statistics are physically plausible."""
    out = {}
    floor = ac.get("band_floor", {})
    for key in ac["band_stats"]:
        vals = np.array([s[key] for s in seed_rows], float)
        mu = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        half = max(ac["band_sigmas"] * sd, float(floor.get(key, 0.0)))
        out[key] = (float(mu - half), float(mu + half))
    return out


def ensemble_bands(stats, ac):
    """Reference ENSEMBLE full-match statistic +- band_floor (user rule, 22 Oct evening): the
    submission is the seed ensemble, and for the temporal net single-seed decodes split far more
    (275-305 possessions vs 239 for the 5-seed average), so seed-spread bands do not describe it."""
    floor = ac.get("band_floor", {})
    return {k: (float(stats[k] - floor.get(k, 0.0)), float(stats[k] + floor.get(k, 0.0)))
            for k in ac["band_stats"]}


def reference_bands(ac, seed_rows=None, ens_stats=None):
    """acceptance.band_mode 'ensemble' (default since 22 Oct evening) -> ensemble_bands;
    'seeds' -> derive_bands (seed mean +- max(band_sigmas * sd, floor))."""
    if ac.get("band_mode", "ensemble") == "ensemble" and ens_stats is not None:
        return ensemble_bands(ens_stats, ac)
    return derive_bands(seed_rows, ac)


def official_reference(cfg, exclude=None):
    """(labels, full-match stats, path) of the official main submission (outputs/main), or of
    the config's booster stage (outputs/<name>_lgbm) when outputs/main is missing or is the run
    being built (exclude='main')."""
    import numpy as np
    import pandas as pd
    from ..validation import sanity
    main = OUTPUTS / "main" / "submission.csv"
    path = main if main.is_file() and exclude != "main" else OUTPUTS / f"{cfg['name']}_lgbm" / "submission.csv"
    labels = pd.read_csv(path)["label"].map({"H": 0, "A": 1, "D": 2, "X": 3}).to_numpy(np.int8)
    return labels, sanity.stats(labels), path


def _in_bands(stats, bands):
    detail = {k: (stats[k], lo <= stats[k] <= hi) for k, (lo, hi) in bands.items()}
    return all(ok for _, ok in detail.values()), detail


def _seed_txt(res):
    sp = [s["spell_f1"] for s in res["seeds"]]
    return f"{np.mean(sp):.3f}±{np.std(sp, ddof=1):.3f}" if len(sp) > 1 else "-"


def _dfl_txt(d):
    if not d:
        return "-"
    return " · ".join(f"{m} {v['spell_f1']:.3f}/{v['macro_f1']:.3f}" for m, v in d.items())


def run_ablation(base="m0", candidates=("kin", "geo", "press"), bootstrap=1000, verbose=False,
                 n_seeds=None, dfl_matches=("J03WPY",)):
    cfg0 = config_mod.load(base)
    ac = cfg0["acceptance"]
    cfg0["lgbm"]["n_seeds"] = int(n_seeds or ac["n_seeds"])
    groups = list(cfg0["features"]["groups"])
    bar_lengths = gap_lengths(bar_data(cfg0))
    assert dfl_matches, "the decision needs a DFL referee match (--dfl J03WPY)"
    print(f"ablation base={base} groups={groups} candidates={list(candidates)} "
          f"n_seeds={cfg0['lgbm']['n_seeds']} DFL referee={list(dfl_matches)}")
    print(f"rule: DFL referee spellF1 up AND macroF1 >= ref - {ac['macro_tolerance']} AND stats within "
          f"ref seed mean +- {ac['band_sigmas']} sd\n")

    def evaluate(groups_):
        name = f"abl5_{'+'.join(groups_)}"
        res = m0.run(cfg=config_mod.with_groups(cfg0, groups_, name), bootstrap=bootstrap,
                     verbose=verbose, compare_baselines=False)
        dfl = {}
        for mid in dfl_matches:
            dfl = dfl_referee.evaluate(mid, groups_, cfg0, bar_lengths)   # one match: report it
        return res, dfl

    ref, ref_dfl = evaluate(groups)
    bands = reference_bands(ac, ref["seeds"], ref["stats"])
    ok_ref, _ = _in_bands(ref["stats"], bands)
    s = ref["summary"]
    print(f"REFERENCE {'+'.join(groups):<24} spellF1 {s['spell_f1']:.4f} macroF1 {s['macro_f1']:.4f} "
          f"seeds {_seed_txt(ref)}  D {s['sanity_dead_share']:.3f} poss {s['sanity_n_possessions']} "
          f"home {s['sanity_home_share']:.3f}  (ensemble in own bands: {ok_ref})  DFL {_dfl_txt(ref_dfl)}")
    print("  bands: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in bands.items()))
    rows = [dict(step="reference", groups="+".join(groups), res=ref, dfl=ref_dfl, paired=None,
                 bands=bands, verdict="-")]

    for g in candidates:
        cand_groups = groups + [g]
        res, dfl = evaluate(cand_groups)
        s = res["summary"]
        spell_up = dfl["weak"]["spell_f1"] > ref_dfl["weak"]["spell_f1"]
        macro_ok = dfl["weak"]["macro_f1"] >= ref_dfl["weak"]["macro_f1"] - ac["macro_tolerance"]
        in_bands, detail = _in_bands(res["stats"], bands)
        paired = paired_spell_bootstrap(ref["gt"], ref["labels"], res["labels"])
        accept = spell_up and macro_ok and in_bands
        why = ([] if spell_up else ["spellF1 not up"]) + ([] if macro_ok else ["macroF1 drop > tol"])
        why += [f"{k} {v:.3f} outside band" for k, (v, ok) in detail.items() if not ok]
        verdict = "KEEP" if accept else "DROP (" + "; ".join(why) + ")"
        print(f"+{g:<5} {'+'.join(cand_groups):<24} spellF1 {s['spell_f1']:.4f} macroF1 {s['macro_f1']:.4f} "
              f"seeds {_seed_txt(res)}  D {s['sanity_dead_share']:.3f} poss {s['sanity_n_possessions']} "
              f"home {s['sanity_home_share']:.3f}  P(better) {paired['p_b_better']:.2f}  "
              f"DFL {_dfl_txt(dfl)}  -> {verdict}")
        rows.append(dict(step=f"+{g}", groups="+".join(cand_groups), res=res, dfl=dfl,
                         paired=paired, bands=bands, verdict=verdict))
        if accept:
            groups, ref, ref_dfl = cand_groups, res, dfl
            bands = reference_bands(ac, ref["seeds"], ref["stats"])
            print("  new bands: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in bands.items()))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ensure(OUTPUTS / "ablation")
    lines = [f"# Ablation {stamp} (base={base}, {cfg0['lgbm']['n_seeds']}-seed ensembles)", "",
             f"Rule: DFL referee spellF1 up AND macroF1 >= ref - {ac['macro_tolerance']} AND full-match stats inside "
             f"ref single-seed mean ± {ac['band_sigmas']} sd. DFL referee ({', '.join(dfl_matches)}; "
             "1st half -> 2nd half, real labels) decides; BAR public reported only.", "",
             "| step | groups | BAR spellF1 [boot] | BAR macroF1 | seed spellF1 mean±sd | dead | poss | home | "
             "P(better) | DFL weak spell/macro | DFL sup spell/macro | decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        s, d = row["res"]["summary"], row["dfl"]
        p = "" if row["paired"] is None else "{:.2f}".format(row["paired"]["p_b_better"])
        dw = "{:.3f} / {:.3f}".format(d["weak"]["spell_f1"], d["weak"]["macro_f1"]) if "weak" in d else ""
        ds = "{:.3f} / {:.3f}".format(d["sup"]["spell_f1"], d["sup"]["macro_f1"]) if "sup" in d else ""
        lines.append(f"| {row['step']} | {row['groups']} | {s['spell_f1']:.4f} [{s['boot_low']:.3f}–{s['boot_high']:.3f}] | "
                     f"{s['macro_f1']:.4f} | {_seed_txt(row['res'])} | {s['sanity_dead_share']:.3f} | "
                     f"{s['sanity_n_possessions']} | {s['sanity_home_share']:.3f} | {p} | {dw} | {ds} | {row['verdict']} |")
    lines += ["", f"Selected groups: {groups}", "",
              "Reference bands: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in rows[0]["bands"].items())]
    report_path = folder / f"ablation_{stamp}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = CONFIGS / "t1_selected.yaml"
    selected.write_text(f"# Written by run.py ablate on {stamp}. Groups kept by the acceptance rule.\n"
                        f"inherit: {base}\nname: t1_selected\nfeatures:\n  groups: {groups}\n"
                        f"lgbm:\n  n_seeds: {cfg0['lgbm']['n_seeds']}\n", encoding="utf-8")
    print(f"\nselected groups: {groups}\n -> {report_path}\n -> {selected}")
    return groups, rows
