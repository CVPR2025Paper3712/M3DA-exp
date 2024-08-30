import numpy as np
from connectome import Transform
from deli import load_json

from m3da_exp.paths import REPO_PATH


class NormalizeLIDC(Transform):
    __inherit__ = True

    def mask(cancer):
        return np.int8(cancer)


class AddFold(Transform):
    __inherit__ = True
    _lidc_split: tuple = tuple((k, tuple(v)) for k, v in load_json(REPO_PATH / 'dataset' / 'lidc_split.json').items())

    def fold(id, _lidc_split):
        f = None
        for _fold, ids in _lidc_split:
            if id in ids:
                f = _fold
        return f
