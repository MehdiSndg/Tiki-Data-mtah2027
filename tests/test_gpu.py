import numpy as np


def test_strict_fp32_turns_tf32_off_and_cudnn_deterministic():
    import torch
    from mtah.gpu import strict_fp32
    torch.backends.cudnn.allow_tf32 = True             # torch's default for convolutions
    info = strict_fp32()
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert info["tf32_cudnn"] is False and info["tf32_matmul"] is False


def test_quantize_boxes_rounds_to_the_published_grid():
    from mtah.gpu import CONF_STEP, XY_STEP, quantize_boxes
    conf = np.array([0.123456, 0.9], np.float32)
    xywh = np.array([[100.004, 200.006, 10.0049, 8.0], [5.0, 6.0, 1.0, 1.0]], np.float32)
    c, b = quantize_boxes(conf, xywh)
    assert np.allclose(c, [0.1235, 0.9]) and np.allclose(b[0], [100.0, 200.01, 10.0, 8.0])
    assert XY_STEP == 0.01 and CONF_STEP == 1e-4


def test_quantize_boxes_absorbs_last_bit_differences():
    from mtah.gpu import quantize_boxes
    rng = np.random.default_rng(0)
    xywh = rng.uniform(0, 1920, (1000, 4)).astype(np.float32)
    nudged = np.nextafter(xywh, np.float32(np.inf))           # one float32 ULP up, as P5 saw
    conf = rng.uniform(0, 1, 1000).astype(np.float32)
    _, a = quantize_boxes(conf, xywh)
    _, b = quantize_boxes(conf, nudged)
    assert np.mean(a == b) > 0.98                              # all but values sitting on a rounding edge
