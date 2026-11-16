r"""Write the only provided-game-derived input the cloud runs need: the ball-gap length list that the
DFL ball masking samples from (aggregate statistic, a few KB) -> cache/cloud/bar_gap_lengths.npy.

  .venv\Scripts\python.exe tools/cloud_export.py --config ev_ext_met
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah import cloud, config as config_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ev_ext_met")
    args = ap.parse_args()
    path, lengths = cloud.export_bar_lengths(config_mod.load(args.config))
    print(f"{path}: {len(lengths)} gaps, {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
