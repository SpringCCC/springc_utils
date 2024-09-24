import torch
import numpy as np
from PIL import Image

def toTensor(a, cuda=True):
    if isinstance(a, torch.Tensor):
        pass
    else:
        a = torch.from_numpy(toNumpy(a))
    return a.cuda() if cuda else a

def toNumpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        return np.asarray(a)

def toPil(a):
    if isinstance(a, np.ndarray):
        a = Image.fromarray(a)
    elif isinstance(a, torch.Tensor):
        a = Image.fromarray(toNumpy(a))
    return a

