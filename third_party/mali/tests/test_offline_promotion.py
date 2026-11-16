"""No real match content or GPU training is used by these promotion checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("mix_promotion", ROOT / "tools" / "run_mix_promotion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_promotes_exact_mixed_recipe_with_eight_folds():
    from mtah import config
    from mtah.pipeline.regionrun import fold_configs
    from mtah.validation.splits import cache_tag
    cfg = config.load("main")
    assert cfg["dfl_mix_eval"] is True
    assert cfg["adapt"]["dfl_target"] == "gt20" and cfg["adapt"]["metrica_eval"] == [1, 2, 3]
    assert not cfg["adapt"].get("short_team") and not cfg["adapt"].get("booster_source")
    assert not cfg["adapt"].get("eval_matches")
    rows = fold_configs("main")
    assert len(rows) == 2 and all(len(row) == 4 for row in rows)
    assert {fc["seed"] for row in rows for _, fc in row} == {2027, 7027}
    for k in (0, 1):
        main, measured = config.load(f"main_rf{k}"), config.load(f"ev_mix_rf{k}")
        main.pop("name")
        measured.pop("name")
        assert main == measured
    assert cache_tag(config.load("main_rf0")) == cache_tag(config.load("ev_mix_rf0"))
    assert not config.load("ev_ref2b_rf0").get("dfl_mix_eval", False)
    assert not config.load("main_v1").get("dfl_mix_eval", False)


def test_clean_clone_environment_does_not_inherit_production_storage(monkeypatch, tmp_path):
    runner = _runner()
    monkeypatch.setenv("MTAH_CACHE_DIR", "wrong-cache")
    monkeypatch.setenv("MTAH_OUTPUTS_DIR", "wrong-outputs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    clean = runner.offline_env(tmp_path / "code", tmp_path / "data")
    assert "MTAH_CACHE_DIR" not in clean and "MTAH_OUTPUTS_DIR" not in clean
    assert "CUDA_VISIBLE_DEVICES" not in clean
    assert clean["MTAH_OFFLINE"] == clean["HF_HUB_OFFLINE"] == "1"
    prod = runner.offline_env(tmp_path / "code", tmp_path / "data", tmp_path / "production")
    assert Path(prod["MTAH_CACHE_DIR"]) == tmp_path / "production" / "cache"
    assert Path(prod["MTAH_OUTPUTS_DIR"]) == tmp_path / "production" / "outputs"


def test_outbound_connect_and_dns_are_blocked_without_using_the_network():
    script = """
import socket
for fn in (lambda: socket.getaddrinfo('example.invalid', 443),
           lambda: socket.socket().connect(('198.51.100.1', 443)),
           lambda: socket.socket().connect_ex(('198.51.100.1', 443))):
    try:
        fn()
    except OSError as exc:
        assert 'MTAH offline mode' in str(exc)
    else:
        raise AssertionError('network guard missing')
assert socket.getaddrinfo('127.0.0.1', 1)
print('offline-guard-ok')
"""
    env = dict(os.environ, MTAH_OFFLINE="1", PYTHONPATH=str(ROOT / "tools" / "offline_site"))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert "offline-guard-ok" in result.stdout


def test_missing_raw_data_fails_before_any_training(tmp_path):
    runner = _runner()
    with pytest.raises(FileNotFoundError, match="Offline raw files missing"):
        runner.preflight(tmp_path, tmp_path / "empty")


def test_archive_preserves_main_reference_and_moves_only_named_folders(tmp_path):
    runner = _runner()
    outputs = tmp_path / "outputs"
    for name in ("main", "main_rf0", "main_rf0_lgbm", "unrelated"):
        (outputs / name).mkdir(parents=True)
        (outputs / name / "toy.txt").write_text(name)
    job = outputs / "mix_promotion" / "test"
    job.mkdir(parents=True)
    runner.archive_previous(tmp_path, job)
    assert (outputs / "main" / "toy.txt").read_text() == "main"
    assert (outputs / "unrelated" / "toy.txt").exists()
    assert not (outputs / "main_rf0").exists()
    assert (job / "previous_official" / "main_rf0" / "toy.txt").exists()


def test_completion_marker_alone_does_not_accept_incomplete_outputs(tmp_path):
    runner = _runner()
    with pytest.raises(RuntimeError, match="Incomplete output"):
        runner.output_summary(tmp_path, "toy")
    folder = tmp_path / "toy"
    folder.mkdir()
    for name in ("summary.json", "submission.csv", "probs.npy", "bar_bnd_adapted.npy"):
        (folder / name).write_text('{}')
    assert runner.output_summary(tmp_path, "toy") == {}


def test_promoted_short_screen_is_paired_and_reuses_only_matching_booster():
    from mtah import config
    from mtah.validation.splits import cache_tag
    base = config.load("ev_short_main_base_rf0")
    trial = config.load("ev_short_main_team_rf0")
    assert base["dfl_mix_eval"] and base["seqnet"]["n_seeds"] == 2
    assert base["adapt"]["booster_source"] == "main_rf0"
    assert base["adapt"]["eval_matches"] == ["J03WQQ", "J03WOY", "J03WR9"]
    assert base["adapt"]["metrica_eval"] == [1, 2, 3]
    assert cache_tag(base) == cache_tag(trial) == cache_tag(config.load("main_rf0"))
    assert trial["adapt"].pop("short_team") == {"max_seconds": 5., "pad_frames": 12, "factor": 2.}
    base.pop("name")
    trial.pop("name")
    assert base == trial


def test_deferred_reproduction_never_claims_verified_and_keeps_production_sha(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import resume_mix_production as resume
    state = {"revision": "pinned", "completed_folds": {"production": {"fold": {}}}}
    sealed = {"revision": "pinned", "sha256": "synthetic-hash"}
    job = tmp_path / "job"
    resume.defer_reproduction(job, tmp_path, state, sealed)
    seal = json.loads((tmp_path / "outputs/main/official_seal.json").read_text())
    assert seal["sha256"] == "synthetic-hash" and seal["reproduction"] == "deferred_by_user"
    assert state["phase"] == state["status"] == "production_complete"
    assert "not been verified" in (job / "RESULT.md").read_text()
    assert not (job / "reproduction_seal.json").exists()


def test_adoption_refuses_unrelated_worker_before_waiting(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import resume_mix_production as resume
    monkeypatch.setattr(resume.subprocess, "check_output", lambda *a, **k: '{"CommandLine":"python unrelated.py"}')
    with pytest.raises(RuntimeError, match="not a fold worker"):
        resume.validate_adopted_worker(1234, tmp_path)
