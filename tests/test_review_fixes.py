"""Tests for the 13 Nov code-review findings (stale caches, the memory guard's cascade, deadsim clean-up)."""

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import possession_metrics as pm  # noqa: E402


# --- recipe C's model file name follows the realism sources and seeds -------------------------------------------------

def test_members_c_cache_path_follows_the_realism_sources(tmp_path, monkeypatch):
    from mtah import members_c
    src = tmp_path / "deadsim.py"
    src.write_text("A = 1\n")
    monkeypatch.setattr(members_c, "RECIPE_FILES", (str(src),))
    before = members_c.cache_path("team", "gru", False, (0,), 1)
    src.write_text("A = 2\n")
    assert members_c.cache_path("team", "gru", False, (0,), 1) != before


def test_members_c_cache_path_follows_the_realistic_dead_seed(monkeypatch):
    from mtah import members_c
    before = members_c.cache_path("team", "gru", False, (0,), 1)
    monkeypatch.setattr(members_c, "REAL_SEED", 2)
    assert members_c.cache_path("team", "gru", False, (0,), 1) != before


# --- the mix dead-ball GRU's file name follows the realistic copies' recipe ----------------------------------------------

def test_dead_gru_mix_path_follows_the_realistic_rates(monkeypatch):
    import p4_common
    before = p4_common.dead_gru_path(mix=True)
    monkeypatch.setattr(p4_common, "REALISTIC_RATES", {"smooth_frames": 13})
    assert p4_common.dead_gru_path(mix=True) != before


def test_dead_gru_production_path_ignores_the_realistic_rates(monkeypatch):
    import p4_common
    before = p4_common.dead_gru_path(mix=False)
    monkeypatch.setattr(p4_common, "REALISTIC_RATES", {"smooth_frames": 13})
    assert p4_common.dead_gru_path(mix=False) == before


# --- the PFF cache is rebuilt when the labelling or the source changes ---------------------------------------------------

def fake_pff(root, gid="g1", n=200):
    """Two periods of 100 frames, 2 x 2 players, one home pass at frame 10 of each period."""
    frame = np.arange(n)
    period = np.r_[np.ones(n // 2), np.full(n // 2, 2)].astype(np.int8)
    ev = [{"gameEventType": "OTB", "start_frame": f, "end_frame": f + 5, "home_ball": True, "possessionEventType": "PA",
           "setpieceType": "O"} for f in (10, 110)]
    rng = np.random.default_rng(0)
    np.savez(os.path.join(root, gid + ".npz"), events=np.array(json.dumps(ev)), frame=frame, period=period,
             t_ms=frame * 40.0, xy=rng.integers(-3000, 3000, (n, 2, 2, 2)).astype(np.int16),
             vis=np.ones((n, 2, 2), np.uint8), ball=rng.integers(-3000, 3000, (n, 3)).astype(np.int16),
             ball_vis=np.ones(n, np.uint8))
    return gid


def test_pff_cache_is_rebuilt_when_the_labelling_changes(tmp_path, monkeypatch):
    from mtah import pff
    monkeypatch.setattr(pff, "CACHE", str(tmp_path / "cache"))
    gid = fake_pff(str(tmp_path))
    assert (pff._pff_raw(gid, root=str(tmp_path))["labels"] == pm.H).any()

    def all_away(ev, frame, period, controlled=False):
        return np.full(len(frame), pm.A, np.int8)
    monkeypatch.setattr(pff, "pff_labels", all_away)
    d = pff._pff_raw(gid, root=str(tmp_path))
    assert not (d["labels"][d["half"] > 0] == pm.H).any()


def test_pff_cache_tells_export_roots_apart(tmp_path, monkeypatch):
    from mtah import pff
    monkeypatch.setattr(pff, "CACHE", str(tmp_path / "cache"))
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    r1.mkdir()
    r2.mkdir()
    fake_pff(str(r1))
    fake_pff(str(r2), n=300)
    pff._pff_raw("g1", root=str(r1))
    assert len(pff._pff_raw("g1", root=str(r2))["bx"]) > 250


# --- the memory guard stops one job, then waits for Windows to see the memory come back ----------------------------------

def _dummy_jobs(tmp_path, k):
    venv = tmp_path / "guardtest" / ".venv" / "bin"
    venv.mkdir(parents=True)
    os.symlink(sys.executable, venv / "python")
    return [subprocess.Popen([str(venv / "python"), "-c", "import time; time.sleep(60)"]) for _ in range(k)]


def _run_guard(tmp_path, seconds, **env):
    e = dict(os.environ, FREE_MB_CMD="echo 100", MATCH=str(tmp_path / "guardtest"), EVICT_CMD="true", EVICT_WAIT_S="0")
    e.update(env)
    g = subprocess.Popen(["bash", os.path.join(ROOT, "scripts", "win_guard.sh"), str(tmp_path / "guard.log")], env=e)
    time.sleep(seconds)
    g.kill()
    g.wait()


def test_win_guard_polls_at_the_given_interval(tmp_path):
    jobs = _dummy_jobs(tmp_path, 2)
    time.sleep(0.5)
    try:
        _run_guard(tmp_path, 5, POLL_S="1", COOLDOWN_S="0")
        assert all(j.poll() is not None for j in jobs)
    finally:
        for j in jobs:
            j.kill()


def test_win_guard_waits_after_stopping_a_job(tmp_path):
    jobs = _dummy_jobs(tmp_path, 2)
    time.sleep(0.5)
    try:
        _run_guard(tmp_path, 5, POLL_S="1", COOLDOWN_S="60")
        assert sum(j.poll() is not None for j in jobs) == 1
        assert jobs[0].poll() is None                    # the newest one went first
    finally:
        for j in jobs:
            j.kill()


def test_win_guard_first_evicts_the_page_cache_and_spares_the_job_when_that_is_enough(tmp_path):
    jobs = _dummy_jobs(tmp_path, 1)
    free = tmp_path / "free"
    free.write_text("100\n")
    time.sleep(0.5)
    try:
        _run_guard(tmp_path, 4, POLL_S="1", COOLDOWN_S="0", EVICT_WAIT_S="1",
                   FREE_MB_CMD="cat {}".format(free), EVICT_CMD="echo 9000 > {}".format(free))
        assert jobs[0].poll() is None
    finally:
        for j in jobs:
            j.kill()


def test_win_guard_stops_the_job_when_eviction_is_not_enough(tmp_path):
    jobs = _dummy_jobs(tmp_path, 1)
    time.sleep(0.5)
    try:
        _run_guard(tmp_path, 4, POLL_S="1", COOLDOWN_S="0", EVICT_WAIT_S="1", EVICT_CMD="true")
        assert jobs[0].poll() is not None
    finally:
        for j in jobs:
            j.kill()


def test_evict_cache_drops_a_files_pages(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(os.urandom(4 << 20))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "evict_cache.py"), str(tmp_path)], capture_output=True, text=True)
    assert out.returncode == 0 and "evicted 1 files" in out.stdout


def test_win_guard_leaves_jobs_alone_when_memory_is_free(tmp_path):
    jobs = _dummy_jobs(tmp_path, 1)
    time.sleep(0.5)
    try:
        _run_guard(tmp_path, 3, POLL_S="1", COOLDOWN_S="0", FREE_MB_CMD="echo 9000")
        assert jobs[0].poll() is None
    finally:
        for j in jobs:
            j.kill()


# --- deadsim's clean-up keeps its output byte for byte ---------------------------------------------------------------------

def test_realistic_dead_output_is_unchanged_by_the_clean_up():
    from test_realism_modules import dead_stretch_match
    from mtah.deadsim import realistic_dead
    m, a, b = dead_stretch_match()
    h = hashlib.sha256()
    for s in range(5):
        o = realistic_dead(dict(m), m, seed=s)
        h.update(np.ascontiguousarray(o["bx"]).tobytes())
        h.update(np.ascontiguousarray(o["by"]).tobytes())
    assert h.hexdigest() == "b7b58c5a2980f33da33eeae61bafa4187c7059ac865be725afa7261bf77d85cb"   # recorded before it


def test_deadsim_carry_offset_spread_matches_its_docstring():
    """The carry offset's Ornstein-Uhlenbeck step (0.998 decay, 0.04 m noise) has a stationary sd of ~0.63 m."""
    from mtah import deadsim
    assert abs(deadsim.OU_SD_M - 0.04 / np.sqrt(1 - 0.998 ** 2)) < 1e-9
    assert "0.63" in deadsim.__doc__
