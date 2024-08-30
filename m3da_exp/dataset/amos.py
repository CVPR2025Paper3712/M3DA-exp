from connectome import Transform
from deli import load_json
import nibabel as nib
import numpy as np

from ..const import LDCT_NOISE_INTENSITY, RANDOM_STATE, CT_SOFT_A_RANGE, CT_SOFT_B_RANGE, CT_SHARP_A_RANGE, \
    CT_SHARP_B_RANGE, CT_SOFT_A, CT_SOFT_B, CT_SHARP_A, CT_SHARP_B
from ..paths import REPO_PATH, AMOS_PATH_LDCT, AMOS_PATH_SOFT, AMOS_PATH_SHARP


class AddFold(Transform):
    __inherit__ = True
    _amos_ct_split: tuple = tuple(map(tuple, load_json(REPO_PATH / 'dataset' / 'amos_ct_split.json')))
    # _random_state: int = RANDOM_STATE
    # _check_split: bool = False

    def fold(id, mask, _amos_ct_split):
        m = 'ct' if int(id) <= 500 else 'mri'
        l = 'u' if mask is None else 'l'
        f = None if (m == 'mri' or l == 'u') else (0 if id in _amos_ct_split[0] else 1)
        # if _check_split:
        #     ids = _amos_ct_split[0][:]
        #     ids.extend(_amos_ct_split[1])
        #     check_split = train_test_split(ids, test_size=0.5, random_state=_random_state)
        #     if (sorted(check_split[0]) != _amos_ct_split[0]) or (sorted(check_split[1]) != _amos_ct_split[1]):
        #         raise IndexError('Split check failed.')
        return f'{m}|{l}|{f}'


class CTNoise(Transform):
    __inherit__ = True
    _mode: str = 'ldct'
    _random_state: int = RANDOM_STATE
    _noise_param: float = LDCT_NOISE_INTENSITY
    _apply_folds: tuple = tuple()
    _hu_min: float = -1000
    _theta: int = 900

    def image(id, image, affine, fold, _apply_folds, _mode, _random_state, _hu_min, _theta, _noise_param):
        if (_mode is None) or (fold not in _apply_folds):
            return image
        assert _mode == 'ldct', f"{_mode}"

        data_path = AMOS_PATH_LDCT
        image_path = data_path / f"{id}.nii.gz"

        if image_path.exists():
            return nib.load(str(image_path)).get_fdata()

        else:
            from ct_augmentation import simulate_ct_dose

            img = simulate_ct_dose(image, _noise_param, axes=(0, 1), random_state=_random_state, theta=_theta)
            img = np.clip(np.round(img, 0), image.min(), image.max())
            img[image <= _hu_min] = image[image <= _hu_min]

            img = img.astype(np.int16)
            nib.save(nib.Nifti1Image(img, affine=affine), image_path)
            return img


class CTKernel(Transform):
    __inherit__ = True
    _random_state: np.random.RandomState = np.random.RandomState(RANDOM_STATE)
    _deterministic: bool = True

    _soft_folds: tuple = tuple()
    _sharp_folds: tuple = tuple()

    _hu_min: float = -1000
    _theta: int = 900

    def image(id, image, affine, fold, _soft_folds, _sharp_folds, _random_state, _deterministic, _hu_min, _theta):
        for f in _soft_folds:
            if f in _sharp_folds:
                raise ValueError(f"Fold {f} is set to soft and sharp simultaneously. This is not a desired behavior.")

        if (fold not in _soft_folds) and (fold not in _sharp_folds):
            return image

        is_soft = fold in _soft_folds
        data_path = AMOS_PATH_SOFT if is_soft else AMOS_PATH_SHARP
        image_path = data_path / f"{id}.nii.gz"

        if image_path.exists():
            return nib.load(str(image_path)).get_fdata()

        else:
            from ct_augmentation import apply_conv_filter

            if _deterministic:
                a, b = (CT_SOFT_A, CT_SOFT_B) if is_soft else (CT_SHARP_A, CT_SHARP_B)
            else:
                a, b = (CT_SOFT_A_RANGE, CT_SOFT_B_RANGE) if is_soft else (CT_SHARP_A_RANGE, CT_SHARP_B_RANGE)
                a = _random_state.uniform(*a)
                b = _random_state.uniform(*b)

            img = apply_conv_filter(image, a, b, axes=(0, 1), theta=_theta)
            img = np.clip(np.round(img, 0), image.min(), image.max())
            img[image <= _hu_min] = image[image <= _hu_min]

            img = img.astype(np.int16)
            nib.save(nib.Nifti1Image(img, affine=affine), image_path)
            return img
