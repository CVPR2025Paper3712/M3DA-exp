import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


PathLike = Union[Path, str]


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def flush(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)


def path_mkdir(path):
    path = Path(path)
    path.mkdir()
    return path


def idx2train(idx, n_folds: int = 6):
    return idx // (n_folds - 1)


def idx2test(idx, n_folds: int = 6):
    train = idx2train(idx, n_folds=n_folds)
    test = idx % (n_folds - 1)
    return test + (test >= train)
