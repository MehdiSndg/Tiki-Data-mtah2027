import numpy as np
import pandas as pd
import pytest

from conftest import needs_data

EXPECTED_COLUMNS = [
    "period", "half",
    "ball_X", "ball_Y", "ball_vx", "ball_vy", "ball_speed", "ball_acc", "ball_status", "ball_off_pitch",
    "ball_stationary", "ball_gap_age", "ball_gap_len",
    "H_n", "A_n", "H_near_d", "A_near_d", "H_gk_d", "A_gk_d", "H_cx", "A_cx", "med_speed", "H_gk_X", "A_gk_X",
    "view_xmin", "view_xmax", "tag_H", "tag_A", "setpiece", "setpiece_type", "whistle",
]


@pytest.fixture(scope="module")
def frames():
    from mtah.frames import build_frames
    return build_frames()


@needs_data
def test_frames_shape_and_columns(frames):
    assert len(frames) == 199319 and frames.index[0] == 0 and frames.index[-1] == 199318
    assert set(EXPECTED_COLUMNS) <= set(frames.columns)


@needs_data
def test_frames_deterministic(frames):
    from mtah.frames import build_frames
    pd.testing.assert_frame_equal(frames, build_frames())


@needs_data
def test_frames_do_not_depend_on_ground_truth(frames, monkeypatch):
    import mtah.frames as fr
    from mtah.data import load_tactical
    real = load_tactical()
    rng = np.random.RandomState(0)

    def scrambled():
        m = dict(real)
        m["gt_public"] = rng.randint(0, 4, len(m["gt_public"])).astype(np.int8)
        return m
    monkeypatch.setattr(fr, "load_tactical", scrambled)
    pd.testing.assert_frame_equal(frames, fr.build_frames())


@needs_data
def test_nearest_distance_only_where_the_ball_is_usable(frames):
    usable = frames["ball_status"].isin([0, 1])
    assert frames.loc[~usable, "H_near_d"].isna().all()
    assert frames.loc[usable & (frames["H_n"] > 0), "H_near_d"].notna().all()


def test_load_frames_never_shows_a_half_written_cache(tmp_path, monkeypatch):
    """Parallel runs read the cache while one rebuilds it: the file is swapped in whole, never written in place."""
    import mtah.frames as fr
    target = tmp_path / "frames_v1.parquet"
    seen = []

    def fake_build(version):
        return pd.DataFrame({"period": [1, 2, 3]})

    real_to_parquet = pd.DataFrame.to_parquet

    def spy(self, path, *a, **k):
        seen.append(str(path))
        return real_to_parquet(self, path, *a, **k)

    monkeypatch.setattr(fr, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fr, "build_frames", fake_build)
    monkeypatch.setattr(fr, "_sources", lambda: [])
    monkeypatch.setattr(pd.DataFrame, "to_parquet", spy)
    df = fr.load_frames()
    assert list(df["period"]) == [1, 2, 3]
    assert seen and all(p != str(target) for p in seen)      # written elsewhere, then moved into place
    assert sorted(x.name for x in tmp_path.iterdir()) == ["frames_v1.parquet"]
