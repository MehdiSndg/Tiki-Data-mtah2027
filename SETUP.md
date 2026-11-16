# Setting up the environment

One locked environment for everything we run: the scorer, the baselines, data loading, video and
model training. `pyproject.toml` declares it, `uv.lock` pins every package to an exact version,
and `.venv/` holds it (gitignored). The organisers' `requirements*.txt` are left untouched.

The bar this exists for: running the repository against the data must reproduce the submission
**byte for byte** (RULES.md). An unpinned environment is the easiest way to miss that.

---

## Once per machine

| | |
|---|---|
| OS | Linux or WSL2. Keep the repository and `data/` on the Linux filesystem, not under `/mnt/c`: reads through `/mnt/c` are many times slower |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| NVIDIA driver | any driver supporting CUDA ≥ 12.4. `nvidia-smi` should list the GPU. CPU-only works too, slowly |
| poppler-utils | only for `literature/fetch_papers.sh` (`pdftotext`) |

Python itself is not a prerequisite: uv fetches 3.11.

## Every clone

```bash
UV_HTTP_TIMEOUT=900 uv sync                    # ~3.5 GB, mostly CUDA wheels
uv run python scripts/check_env.py             # interpreter, packages, GPU, data, video, audio
uv run python scripts/check_baselines.py       # rebuilds the 4 baselines, must be byte-identical
uv run python evaluation/run_evaluation.py     # the scorer, on the public 5 minutes
```

`UV_HTTP_TIMEOUT` is there because the CUDA wheels are several hundred MB each and uv's default
30 s timeout fails on a slow connection. A failed `uv sync` is safe to re-run.

Optional extras:

```bash
uv run python scripts/fetch_dfl.py             # Sportec Open DFL, 2.5 GB -> data/external/dfl
./scripts/fetch_metrica.sh                     # Metrica sample data, 175 MB, pinned commit
./literature/fetch_papers.sh                   # the papers behind literature/README.md
```

Both datasets land under `data/external/`, which `data/*` in `.gitignore` already keeps out of git.
`fetch_dfl.py` verifies every file against figshare's MD5 and resumes an interrupted download.

---

## What is pinned, and why

| | version | why this one |
|---|---|---|
| Python | 3.11 | one of the two versions the official scorer is tested under |
| numpy / pandas | 2.3 / 2.3 | `floodlight` requires pandas < 3 |
| kloppy | 3.18.0 | the version `requirements-starter.txt` pins for the Metrica EPTS export |
| floodlight | 1.2.0 | loader for the Sportec Open DFL dataset; requires lxml < 6 |
| torch / torchvision | 2.7.1 / 0.22.1, CUDA 12.8 | from the PyTorch cu128 index, configured in `pyproject.toml`; 12.8 is the first build that runs on Blackwell (Colab G4), and it runs on our RTX 4060 and on A100 too (`notes/video/P5_gpu_bench.md`) |
| PyAV | 18.1.0 | video **and audio** decoding with ffmpeg bundled; the last release with cp311 wheels |
| OpenCV | headless | frame access; no GUI dependency |

To add a package: `uv add <name>`, then commit `pyproject.toml` and `uv.lock` together.
