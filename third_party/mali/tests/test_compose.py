import numpy as np

from mtah.pipeline.hybrid import compose_dead_team


def test_compose_keeps_dead_and_team_ratio():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([1, 1, 1], 200)
    b = rng.dirichlet([1, 1, 1], 200)
    c = compose_dead_team(a, b)
    assert np.allclose(c.sum(1), 1.0) and np.allclose(c[:, 2], a[:, 2])
    assert np.allclose(c[:, 0] / (c[:, 0] + c[:, 1]), b[:, 0] / (b[:, 0] + b[:, 1]))
    assert np.allclose(compose_dead_team(a, a), a)
