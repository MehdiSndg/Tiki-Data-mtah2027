"""Adım 3 driver: compare configurations against a reference with the acceptance rule and with
every DFL view.

For each config (reference first):
  BAR-ATH   m0.run: public spellF1 / macroF1 (n-seed ensemble), seed spread, full-match stats
  DFL weak  referee (J03WPY 1st -> 2nd half): the BAR weak-label pipeline on DFL, incl. D recall
            split by ball visibility and the dead-anchor precision / coverage on the 1st half
  DFL sup   referee with the true labels (information content)
  transfer  the BAR-ATH model (+ the other DFL matches when the config is joint) scored on every
            downloaded DFL match it did not train on (leave-one-match-out); J03WPY 2nd half for
            comparability with the referee, and the mean over matches on whole matches
A candidate is accepted against the reference on DFL LOMO spellF1 / macroF1 (transfer mean) and
the full-match bands; the public window is reported only (upstream rules: not a tuning set).
"""

import datetime as dt
import json

import numpy as np

from .. import config as config_mod
from ..io.broadcast import bar_data
from ..io.idsse import MATCHES
from ..labels.masking import gap_lengths
from ..paths import OUTPUTS, ensure
from ..validation.compare import paired_spell_bootstrap
from . import dfl_referee, joint, m0
from .ablation import _in_bands, reference_bands

REFEREE_MATCH = "J03WPY"
CORE4 = ("J03WPY", "J03WN1", "J03WMX", "J03WOH")   # the matches of main's LOMO (comparable means)


def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def evaluate_config(name, bootstrap, dfl_matches, verbose=False, referee=True):
    cfg = config_mod.load(name)
    res = m0.run(cfg=cfg, bootstrap=bootstrap, verbose=verbose, compare_baselines=False)
    lengths = gap_lengths(bar_data(cfg))
    out = {"name": name, "res": res, "cfg": cfg}
    if referee:
        modes = ("weak", "sup")
        out["referee"] = dfl_referee.evaluate(REFEREE_MATCH, cfg["features"]["groups"], cfg, lengths, modes)
    is_joint = bool(cfg.get("joint", {}).get("matches"))
    matches = joint.available(dfl_matches if not is_joint else cfg["joint"]["matches"])
    bar = res["bar_source"]
    if is_joint:
        rows = joint.evaluate_lomo(cfg, matches, bar, res["columns"], lengths)
    else:
        rows = transfer(cfg, matches, bar, res["columns"], lengths)
    out["transfer"] = rows
    return out


def transfer(cfg, matches, bar_source, columns, lengths):
    """BAR-ATH-only model scored on DFL matches (zero-shot transfer)."""
    from . import engine
    bar_source.share = 1.0
    preps = [dfl_referee.prepare(m, cfg, lengths) for m in matches]
    per_target, _, _ = engine.fit_predict_sources([bar_source], [p["feats"] for p in preps], cfg, columns)
    rows = []
    for m, prep, per_seed in zip(matches, preps, per_target):
        probs = engine.ensemble(per_seed)
        data, runs = prep["data"], prep["runs"]
        pred_full = engine.decode_runs(probs, runs, cfg)
        pred_h2 = engine.decode_runs(probs, [(a, b) for a, b in runs if data.period[a] == 2], cfg)
        rows.append({"match": m, "fold": "bar-only", "h2": dfl_referee.score(pred_h2, prep, 2),
                     "full": dfl_referee.score(pred_full, prep, None)})
    return rows


def _transfer_summary(rows):
    j = [r for r in rows if r["match"] == REFEREE_MATCH]
    full = [r["full"] for r in rows]
    mean = lambda k: float(np.mean([f[k] for f in full])) if full else float("nan")  # noqa: E731
    core = [r["full"] for r in rows if r["match"] in CORE4]
    return {"j_h2": j[0]["h2"] if j else None, "n": len(rows), "spell": mean("spell_f1"),
            "macro": mean("macro_f1"), "d_recall": mean("d_recall"), "team_acc": mean("team_acc"),
            "spell4": float(np.mean([f["spell_f1"] for f in core])) if core else float("nan"),
            "macro4": float(np.mean([f["macro_f1"] for f in core])) if core else float("nan"),
            "d_rec_hidden": mean("d_recall_ball_hidden")}


def run(reference="main", candidates=(), bootstrap=1000,
        dfl_matches=MATCHES, verbose=False):
    ref = evaluate_config(reference, bootstrap, dfl_matches, verbose)
    ac = ref["cfg"]["acceptance"]
    bands = reference_bands(ac, ref["res"]["seeds"], ref["res"]["stats"])
    print(f"reference {reference}: bands " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in bands.items()))
    rows = [dict(item=ref, verdict="-", paired=None)]
    for name in candidates:
        item = evaluate_config(name, bootstrap, dfl_matches, verbose)
        s = item["res"]["summary"]
        # 25 Oct (upstream rules): the public window is not a tuning set -> decide on DFL LOMO
        t, rt = _transfer_summary(item["transfer"]), _transfer_summary(ref["transfer"])
        spell_up = t["spell"] > rt["spell"]
        macro_ok = t["macro"] >= rt["macro"] - ac["macro_tolerance"]
        in_bands, detail = _in_bands(item["res"]["stats"], bands)
        paired = paired_spell_bootstrap(ref["res"]["gt"], ref["res"]["labels"], item["res"]["labels"])
        why = ([] if spell_up else ["spellF1 not up"]) + ([] if macro_ok else ["macroF1 drop > tol"])
        why += [f"{k} {v:.3f} outside band" for k, (v, ok) in detail.items() if not ok]
        verdict = "KEEP" if (spell_up and macro_ok and in_bands) else "DROP (" + "; ".join(why) + ")"
        rows.append(dict(item=item, verdict=verdict, paired=paired))
        print(f"{name}: spellF1 {s['spell_f1']:.4f} macroF1 {s['macro_f1']:.4f} -> {verdict}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ensure(OUTPUTS / "adim3")
    lines = [f"# Adım 3 karşılaştırması {stamp}", "",
             f"Referans: `{reference}`. Kural: DFL LOMO spellF1 ↑, DFL macroF1 ≥ ref − {ac['macro_tolerance']}, "
             f"bantlar = ref tek-seed ort. ± max({ac['band_sigmas']} sd, taban {ac.get('band_floor')}).", "",
             "Bantlar: " + ", ".join(f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in bands.items()), "",
             "| config | BAR spellF1 [boot] | seed ort.±sd | BAR macroF1 | D | poss | ev | P(daha iyi) | "
             "DFL weak spell/macro | weak D rec (görünür/gizli) | weak D çapası P/kapsama | weak satır | "
             "DFL sup spell/macro | transfer J03WPY-H2 spell/macro | transfer tüm maç ort. spell/macro/D rec (n) | "
             "ortak 4 maç spell/macro | gizli top D rec | karar |",
             "|" + "---|" * 18]
    dump = []
    for row in rows:
        it = row["item"]
        s = it["res"]["summary"]
        sp = [x["spell_f1"] for x in it["res"]["seeds"]]
        seed_txt = f"{np.mean(sp):.3f}±{np.std(sp, ddof=1):.3f}" if len(sp) > 1 else "–"
        w, su = it["referee"]["weak"], it["referee"]["sup"]
        t = _transfer_summary(it["transfer"])
        jh2 = t["j_h2"]
        p = "" if row["paired"] is None else f"{row['paired']['p_b_better']:.2f}"
        lines.append(
            f"| {it['name']} | {s['spell_f1']:.4f} [{_f(s['boot_low'], 2)}–{_f(s['boot_high'], 2)}] | {seed_txt} | "
            f"{s['macro_f1']:.4f} | {s['sanity_dead_share']:.3f} | {s['sanity_n_possessions']} | "
            f"{s['sanity_home_share']:.3f} | {p} | {_f(w['spell_f1'])} / {_f(w['macro_f1'])} | "
            f"{_f(w['d_recall'], 2)} ({_f(w['d_recall_ball_visible'], 2)}/{_f(w['d_recall_ball_hidden'], 2)}) | "
            f"{_f(w.get('d_anchor_precision_h1'), 2)} / {_f(w.get('d_anchor_cover_h1'), 2)} | {w['train_rows']} | "
            f"{_f(su['spell_f1'])} / {_f(su['macro_f1'])} | "
            f"{_f(jh2['spell_f1']) if jh2 else '–'} / {_f(jh2['macro_f1']) if jh2 else '–'} | "
            f"{_f(t['spell'])} / {_f(t['macro'])} / {_f(t['d_recall'], 2)} ({t['n']}) | "
            f"{_f(t['spell4'])} / {_f(t['macro4'])} | {_f(t['d_rec_hidden'], 2)} | {row['verdict']} |")
        dump.append({"config": it["name"], "summary": s, "seeds": it["res"]["seeds"],
                     "anchor_info": it["res"]["anchor_info"], "referee": it["referee"],
                     "transfer": it["transfer"], "verdict": row["verdict"],
                     "p_better": None if row["paired"] is None else row["paired"]["p_b_better"]})
    path = folder / f"adim3_{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / f"adim3_{stamp}.json").write_text(json.dumps(dump, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n -> {path}")
    return rows
