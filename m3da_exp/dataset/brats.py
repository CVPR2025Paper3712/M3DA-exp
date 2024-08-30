import numpy as np
from connectome import Transform
from deli import load_json

from ..paths import REPO_PATH


class BuildMask(Transform):
    __inherit__ = True

    def mask(mask):
        mask_new = np.copy(mask)
        mask_new[mask_new == 4] = 3
        return mask_new


class AddFold(Transform):
    __inherit__ = True
    _brats_subject_split: tuple = tuple(map(tuple, load_json(REPO_PATH / 'dataset' / 'brats_subject_split.json')))

    def fold(subject_id, modality, _brats_subject_split):
        domain = 'src' if subject_id in _brats_subject_split[0] else 'trg'
        split = 'tr' if (domain == 'src') else ('tr' if subject_id in _brats_subject_split[1] else 'ts')
        return f'{modality}|{domain}|{split}'
