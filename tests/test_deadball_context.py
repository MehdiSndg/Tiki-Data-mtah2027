import numpy as np
import pandas as pd

from mtah import deadball


def test_context_features_summarise_past_and_future_at_each_scale():
    n = 200 * 25
    sig = pd.DataFrame({"s": np.r_[np.zeros(n // 2), np.ones(n // 2)]})
    c = deadball.context(sig, scales_s=(4, 20))
    assert list(c.columns) == ["s_past4", "s_next4", "s_past20", "s_next20"]
    i = n // 2 - 1                                  # last frame of the zeros
    assert c["s_past4"].iloc[i] == 0 and c["s_next4"].iloc[i] == 1
    assert not c.isna().any().any()
