"""Filesystem layout. Nothing else in the package spells a path out."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "more-than-a-hack-2027"
DATA = Path(os.environ.get("MTAH_DATA_DIR") or OFFICIAL / "data")
# A pinned code snapshot may use production storage; clean-clone checks leave these unset.
CACHE = Path(os.environ.get("MTAH_CACHE_DIR") or ROOT / "cache")
OUTPUTS = Path(os.environ.get("MTAH_OUTPUTS_DIR") or ROOT / "outputs")
CONFIGS = ROOT / "configs"
EXTERNAL = DATA / "external"

PUBLIC_GT = OFFICIAL / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
BASELINE_RESULTS = OFFICIAL / "evaluation" / "results"


def tactical(asset):
    """Path of a tactical asset: atd_meta, atd_raw, homography, smart_tagging, video."""
    suffix = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt", "homography": "_homography.csv",
              "smart_tagging": "_smart_tagging.xml", "video": ".mp4"}[asset]
    return DATA / "tactical" / f"BAR-ATH_tactical{suffix}"


def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
