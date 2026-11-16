# Tiki-Data — More Than A Hack 2027, Phase 1

Frame-level team possession (`H` / `A` / `D` / `X`) for the BAR-ATH match, from the provided automatic tracking,
tactical video and Smart Tagging. Every model is trained on **external** labelled data only (DFL / Sportec open data,
Metrica sample data, SoccerTrack v2); nothing is trained, fitted, tuned or calibrated on the provided game.

- Technical document: [`docs/Tiki-Data.md`](docs/Tiki-Data.md) (approach, assets, training data, possession rules,
  limitations, results).
- The submission file is produced by `scripts/make_submission.py v30` (see Reproduction).
- The organisers' challenge documents, scorer (`evaluation/`) and starter code (`starter/`) are kept unchanged; their
  README is `docs/CHALLENGE_README.md`.

## Layout

| path | what |
|---|---|
| `mtah/` | the library: data loading, degradation model, features, sequence models, dead ball, semi-Markov decoder |
| `analysis/` | one script per experiment (P-numbers in the technical document), including the judges |
| `scripts/make_submission.py` | builds the CSV from the inputs and the trained weights; `scripts/reproduce.sh` lists every step |
| `scripts/kaggle/`, `analysis/*_colab.py` | the training jobs (Kaggle T4 / Colab) |
| `third_party/mali/` | our teammate Mehmet Ali Topkara's pipeline (LightGBM + TCN-BiGRU, external data only), one member of the ensemble |
| `tests/` | `pytest tests/` |

## Installation

Python 3.11, CUDA GPU for the video stage.

```bash
pip install uv
uv sync --frozen                       # pinned environment (uv.lock): torch 2.7.1 + cu128, numpy, pandas, scikit-learn, ...
git clone https://github.com/SoccerNet/sn-teamspotting third_party/sn-teamspotting    # T-DEED code (video model)
```

The teammate's member runs in its own environment: `third_party/mali/kurulum/requirements-gpu.txt`
(`third_party/mali/KURULUM.md`).

## Data

- **The match bundle is not in this repository** (not redistributable). Unpack it as the challenge describes:
  `data/tactical/`, `data/broadcast/`.
- External data: `python scripts/fetch_dfl.py` (DFL / IDSSE open data, CC BY 4.0), `bash scripts/fetch_metrica.sh`
  (Metrica sample data). SoccerTrack v2 (CC BY 4.0) is only needed to retrain the video heads.

## Trained weights

Training on GPUs is not bit-reproducible across machines, so the weights the submission was built with are released
as assets of this repository's GitHub release `v30` and go into `models/` (and `third_party/mali/outputs/` for the
teammate's nets). The release lists each file with its sha256 and the command that trained it. The dead-ball
classifier (HistGradientBoosting, fixed seed) is refitted during the build.

```bash
bash scripts/fetch_weights.sh <owner/repo>     # downloads the release, checks SHA256SUMS, fills models/ and third_party/mali/outputs/
```

## Reproduction

```bash
bash scripts/reproduce.sh
```

Steps: (1) caches of the provided match (tracking and frame tables, broadcast → tactical frame map; no model, no
label), (2) the video channel (virtual-camera frames of the tactical video, inference with the seven video models),
(3) the teammate's component, (4) `python scripts/make_submission.py v30 evaluation/results/Tiki-Data.csv` (about
7.4 GB RAM; run it alone), (5) `python evaluation/validate_submission.py evaluation/results/Tiki-Data.csv` and the
sha256 recorded in `scripts/reproduce.sh`. With the released weights the build gives the same file byte for byte on our
machine (checked twice).

## Licences and credits

Our code: see `LICENSE` (the organisers' licence covers their files). Data and models used: DFL / Sportec open data
(CC BY 4.0, Bassek et al.), Metrica Sports sample data, SoccerTrack v2 (CC BY 4.0, Scott et al.), SoccerNet T-DEED
(code and checkpoint from `SoccerNet/sn-teamspotting`), PFF FC World Cup 2022 data (evaluation only). Ultralytics is
AGPL-3.0 (used only for label-free checks).
