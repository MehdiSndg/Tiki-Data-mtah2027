r"""Checks of team/PROTOCOL.md section 6 on a hand-over folder: tools/team_verify.py <team folder>
(run from the project root with the project's venv; the score uses the official metric via mtah.validation.official)."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtah.validation.official import pm  # noqa: E402

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
out = Path(sys.argv[1])
summ = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))

for folder in (out / "barath", out / "barath" / "cand_mix_pff"):
    lab = np.array([CODES[ln.strip().split(",")[1]] for ln in
                    (folder / "submission.csv").read_text(encoding="utf-8").splitlines()[1:] if ln.strip()], np.int8)
    z = np.load(folder / "probs.npz")
    p = np.stack([z["p_home"], z["p_away"], z["p_dead"]], 1).astype(np.float64)
    v = z["valid"]
    c = np.load(folder / "components.npz")
    print(f"{folder.relative_to(out).as_posix()}: N {len(p)} (submission rows {len(lab)}), NaN {int(np.isnan(p).sum())}, "
          f"|sum-1| max {np.abs(p.sum(1) - 1).max():.4f}, valid {int(v.sum())}, X in submission {int((lab == 3).sum())}, "
          f"argmax == submission on valid frames {100 * np.mean(p.argmax(1)[v] == lab[v]):.2f} %, "
          f"components keys {sorted(c.files)} len {len(c['p_dead'])}")

for sub, key in (("external", "external"), ("external/cand_mix_pff", "external_candB")):
    scores = []
    for f in sorted((out / sub).glob("*.npz")):
        z = np.load(f)
        n = {k: len(z[k]) for k in ("p_home", "p_away", "p_dead", "valid", "labels", "half", "frame", "pred")}
        lab = z["labels"]
        dist = " ".join(f"{k} {100 * np.mean(lab == i):.1f}" for k, i in CODES.items())
        p = np.stack([z["p_home"], z["p_away"], z["p_dead"]], 1).astype(np.float64)
        s = pm.evaluate(lab, z["pred"])["spell_f1_mean"]
        scores.append(s)
        ok = abs(s - summ[key][f.stem]["spell_f1"]) < 5e-5
        print(f"{sub}/{f.name}: equal lengths {len(set(n.values())) == 1} (M {n['labels']}), fps {float(z['fps']):g}, "
              f"halves {sorted(set(z['half'][z['valid']].tolist()))}, frame {int(z['frame'][0])}..{int(z['frame'][-1])}, "
              f"NaN {int(np.isnan(p).sum())}, labels % {dist}, spellF1 recomputed {s:.4f} table match {ok}")
    print(f"{sub}: mean spellF1 {np.mean(scores):.4f} over {len(scores)} files")
