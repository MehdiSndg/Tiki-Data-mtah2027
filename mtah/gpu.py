"""Deterministic detector inference: strict fp32 and a fixed output grid. (P5)

    from mtah.gpu import strict_fp32, quantize_boxes
    info = strict_fp32()                       # before the first CUDA call
    conf, xywh = quantize_boxes(conf, xywh)    # before anything downstream reads the boxes

torch runs cuDNN convolutions in TF32 by default (`cudnn.allow_tf32` is True); our earlier "fp32"
passes did, and so differed across devices. With TF32 off, cuDNN deterministic and no autotuning,
an RTX 4060 (torch 2.6/cu124) and an RTX PRO 6000 Blackwell (torch 2.7.1/cu128) gave the same
boxes on 200 frames, up to the last float32 bit (notes/BULGULAR.md). Rounding the boxes
to a fixed grid removes those last-bit differences, except for a value that sits on a rounding
edge -- so the submission chain also keeps the detector output it used, with its checksum.
"""

import os

import numpy as np

# Params
XY_STEP = 0.01          # px: box centre and size
CONF_STEP = 1e-4        # detector confidence


def strict_fp32():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def quantize_boxes(conf, xywh):
    c = (np.round(np.asarray(conf, np.float64) / CONF_STEP) * CONF_STEP).astype(np.float32)
    b = (np.round(np.asarray(xywh, np.float64) / XY_STEP) * XY_STEP).astype(np.float32)
    return c, b
