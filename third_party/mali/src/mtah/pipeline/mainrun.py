"""The official main model end to end (configs/main.yaml): booster stage, then the temporal
net with its BAR-ATH adaptation and the log-pool blend -> outputs/main.

    run.py main            (~10-15 min on the RTX 5070 Ti; caches make reruns faster)
"""

import hashlib
import time

from .. import config as config_mod
from ..paths import OUTPUTS
from . import adapt, m0


def run(config="main", bootstrap=1000):
    t0 = time.time()
    cfg = config_mod.load(config)
    v = cfg.get("validation") or {}
    if v.get("region_folds") and not v.get("region_cv"):
        from . import regionrun                 # Gün 5: region-fold out-of-fold submission
        return regionrun.run(config, bootstrap=0)
    res_b = m0.run(cfg=cfg, bootstrap=bootstrap, verbose=False, compare_baselines=False)
    s = res_b["summary"]
    print(f"[main] booster -> outputs/{cfg['name']}_lgbm: public {s['spell_f1']:.4f} / {s['macro_f1']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    res = adapt.run(config=config, name=cfg["name"])
    sub = OUTPUTS / cfg["name"] / "submission.csv"
    h = hashlib.sha256(sub.read_bytes()).hexdigest()
    print(f"[main] {sub}  sha256 {h[:16]}  ({time.time() - t0:.0f}s)", flush=True)
    return res, h
