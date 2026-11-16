"""PathCRF (Kim et al. 2026) on its held-out DFL match: reproduce the paper, then score it our way.

    cd data/external/pathcrf && uv run --project ../../.. --with rdp python ../../../analysis/p5d_pathcrf_eval.py J03WR9 140

Steps of the repository's tutorial.ipynb (load the trial, inference, detect_events,
compute_true_pred_poss), then the frame-level predicted team is scored with the challenge's metric
against DFL's labels (raw, and merged at 1.6 s as our proxy), next to our control rule on the same
match; the dead ball is taken from the labels for both, so only the team call is compared.
Writes eda_out/pathcrf_<match>_<trial>.npz (frame, pred_team) for later use. Run from the PathCRF
repository root; nothing of theirs is modified.
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # this repository (run from PathCRF's)
sys.path.insert(0, os.getcwd())
import pandas as pd  # noqa: E402
from applications.team_poss import compute_true_pred_poss  # noqa: E402
from datatools.postprocess import detect_events, simplify_events  # noqa: E402
from inference import inference  # noqa: E402
from models.utils import build_model, load_trial_args, resolve_model_path  # noqa: E402

match_id, trial = sys.argv[1], int(sys.argv[2])
tracking = pd.read_parquet(f"data/sportec/tracking_processed/{match_id}.parquet")
events = pd.read_parquet(f"data/sportec/event_processed/{match_id}.parquet")
save_path = f"saved/{trial:03d}"
args = load_trial_args(save_path)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = build_model(args, device=device)
name = "state_dict_best_pe.pt" if args.get("micro_type", "edge") == "ball" else "state_dict_best_acc.pt"
model.load_state_dict(torch.load(resolve_model_path(save_path, name), map_location=device, weights_only=False))
macro_prev, macro_next, micro_pred, stats = inference(model, tracking, use_crf=args.get("crf_weight", 0) > 0, decode="indep",
                                                      window_seconds=args.get("window_seconds", None), correct_episode_lasts=True)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items() if not str(k).startswith("ep_")}, flush=True)
pred_events = detect_events(tracking, micro_pred)
frames = compute_true_pred_poss(simplify_events(events), pred_events, tracking)
print("per-frame team possession accuracy (paper: 0.9240): {:.4f}".format((frames["true_team"] == frames["pred_team"]).mean()))
print(frames.head(3).to_string(), flush=True)
out = os.path.join(ROOT, "eda_out", f"pathcrf_{match_id}_{trial}.npz")
cols = {c: frames[c].values for c in frames.columns if frames[c].dtype != object}
cols.update({c: frames[c].astype(str).values for c in frames.columns if frames[c].dtype == object})
np.savez_compressed(out, **cols)
print("saved", out, list(frames.columns))
