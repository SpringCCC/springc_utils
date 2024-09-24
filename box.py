import numpy as np
import torch


def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xywhxyxy_reverse_normalize(x, h, w):
    # x:(cx cy w h) or (x1 y1 x2 y2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] * w  # top left x
    y[..., 1] = x[..., 1] * h  # top left y
    y[..., 2] = x[..., 0] * w  # bottom right x
    y[..., 3] = x[..., 1] * h  # bottom right y
    return y


def xywhxyxy_normalize(x, h, w):
    # x:(cx cy w h) or (x1 y1 x2 y2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] / w  # top left x
    y[..., 1] = x[..., 1] / h  # top left y
    y[..., 2] = x[..., 0] / w  # bottom right x
    y[..., 3] = x[..., 1] / h  # bottom right y
    return y