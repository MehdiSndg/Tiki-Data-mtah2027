"""PathCRF (Kim et al. 2026) on its own held-out DFL match: preprocessing, as its README does it.

    cd data/external/pathcrf && uv run --project ../../.. --with rdp python ../../../analysis/p5d_pathcrf_prep.py J03WR9

The repository's datatools/preprocess.py loops over all seven matches; this runs the same steps for
the named ones only (the held-out test match of checkpoint 140 is the last one, J03WR9). Run from
the PathCRF repository root: its modules read relative paths. Nothing of ours is changed.
"""

import os
import sys

sys.path.insert(0, os.getcwd())
import pandas as pd  # noqa: E402
from datatools import utils  # noqa: E402
from datatools.preprocess import Preprocessor  # noqa: E402
from datatools.sportec_data import SportecData  # noqa: E402

for match_id in sys.argv[1:]:
    os.makedirs("data/sportec/event_processed", exist_ok=True)
    os.makedirs("data/sportec/tracking_processed", exist_ok=True)
    events = pd.read_parquet(f"data/sportec_synced/{match_id}.parquet")
    tracking = SportecData(match_id, load_tracking=True).tracking.reset_index(drop=True)
    tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)
    proc = Preprocessor(events, tracking)
    touches = proc.detect_touches()
    touches.to_parquet(f"data/sportec/event_processed/{match_id}.parquet")
    tracking = proc.merge_tracking_poss(touches)
    utils.calculate_running_features(tracking).to_parquet(f"data/sportec/tracking_processed/{match_id}.parquet")
    print(match_id, "done", tracking.shape, flush=True)
