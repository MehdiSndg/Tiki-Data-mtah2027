r"""Bring a cloud run home: unpack <NAME>.tar (downloaded from Drive MyDrive/mtah_cloud/runs/) into
outputs/<NAME>, predict the provided game locally from its nets, and (with --reference) apply the 4 gates
against another run trained in the SAME place (cloud vs cloud; GPU differences make cross-machine
comparisons unfair).

  .venv\Scripts\python.exe tools/cloud_fetch.py C:\Users\ASUS\Downloads\cl_ev_ext_met_dr3.tar --reference cl_ev_ext_met
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mtah.paths import OUTPUTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tar")
    ap.add_argument("--reference", help="another cloud run under outputs/ to compare with (4 gates)")
    args = ap.parse_args()
    tar = Path(args.tar)
    name = tar.stem
    dest = OUTPUTS / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; never overwrite a run")
    with tarfile.open(tar) as t:
        members = t.getmembers()
        if any(not m.name.startswith(name + "/") and m.name != name for m in members):
            raise SystemExit("unexpected paths in the archive")
        t.extractall(OUTPUTS, filter="data")
    print(f"unpacked -> {dest}")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "bar_from_nets.py"), "--run", name])
    if r.returncode != 0 or "bar" not in json.loads((dest / "summary.json").read_text(encoding="utf-8")):
        raise SystemExit("local prediction of the provided game failed")
    if args.reference:
        import run_day_screen as screen
        ref = screen.met_scores(json.loads((OUTPUTS / args.reference / "summary.json").read_text(encoding="utf-8")))
        cand = screen.met_scores(json.loads((dest / "summary.json").read_text(encoding="utf-8")))
        g = screen.four_gates({name: {"reference": ref, "candidate": cand}})
        res = {"reference": args.reference, "candidate": name, "reference_scores": ref, "candidate_scores": cand,
               "four_gates": g}
        (dest / "comparison.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(f"v1 {ref['dfl_v1']:.4f} -> {cand['dfl_v1']:.4f} | v2b {ref['dfl_v2b']:.4f} -> {cand['dfl_v2b']:.4f} | "
              f"Metrica {ref['metrica']:.4f} -> {cand['metrica']:.4f} | BAR {ref['bar']['n_possessions']} -> "
              f"{cand['bar']['n_possessions']}")
        print(f"composite {g['delta_composite']:+.4f}, positive {g['positive_matches']}/6, gates {g['gates']}, pass {g['pass']}")


if __name__ == "__main__":
    main()
