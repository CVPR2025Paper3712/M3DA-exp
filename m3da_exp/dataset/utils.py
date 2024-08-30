from typing import Union

import numpy as np
from amid.utils import propagate_none
from connectome import Transform
from imops import zoom, label


class Identity(Transform):
    __inherit__ = True


class MultiClassZoom(Transform):
    __inherit__ = True
    _new_spacing: Union[tuple, float, int] = (None, None, None)
    _labels_list: tuple

    def _spacing(spacing, _new_spacing) -> tuple:
        _new_spacing = np.broadcast_to(_new_spacing, len(spacing)).copy()
        _new_spacing[np.isnan(_new_spacing)] = np.array(spacing)[np.isnan(_new_spacing)]
        return tuple(_new_spacing.tolist())

    def _scale_factor(spacing, _spacing):
        return np.nan_to_num(np.float32(spacing) / np.float32(_spacing), nan=1)

    def spacing(_spacing):
        return _spacing

    def image(image, _scale_factor) -> np.ndarray:
        return zoom(image.astype(np.float32), _scale_factor)

    @propagate_none
    def mask(mask, _scale_factor, _labels_list) -> np.ndarray:
        onehot = (np.asarray(_labels_list) == mask[..., None]).transpose(-1, 0, 1, 2)
        out = np.array(zoom(onehot.astype(np.float32), _scale_factor, axis=(1, 2, 3)) > 0.5, dtype=mask.dtype)
        labels = out.argmax(axis=0)  # 3d-image
        return labels


def scale_q(x, min_q=0.5, max_q=99.5):
    assert max_q > min_q
    x = np.clip(np.float32(x), *np.percentile(np.float32(x), [min_q, max_q]))
    x -= np.min(x)
    x /= np.max(x)
    return x


class ScaleQ(Transform):
    __inherit__ = True
    _min_q: float = 0.5
    _max_q: float = 99.5

    def image(image, _min_q, _max_q):
        return scale_q(image, _min_q, _max_q)


class Shape(Transform):
    __inherit__ = True

    def shape(image):
        return image.shape


class Orig(Transform):
    __inherit__ = True

    def orig_image(image):
        return image

    def orig_shape(image):
        return image.shape

    @propagate_none
    def orig_mask(mask):
        return mask


class None2ZeroMask(Transform):
    __inherit__ = True

    def mask(image, mask):
        return np.zeros_like(image) if mask is None else mask


class TumorCenters(Transform):
    __inherit__ = True

    def _labels_n_labels(mask):
        return label(mask > 0.5, return_num=True, connectivity=3)

    def n_tumors(_labels_n_labels):
        return _labels_n_labels[1]

    def tumor_centers(_labels_n_labels):
        labels, n_labels = _labels_n_labels
        return np.int16([np.round(np.mean(np.argwhere(labels == i), axis=0)) for i in range(1, n_labels + 1)])
