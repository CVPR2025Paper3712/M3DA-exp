from functools import partial
from typing import Callable

import numpy as np

from m3da_exp.predict.utils import zoom_pred2true


def sample_fwd_and_inv_augm(random_state: np.random.RandomState, flip_dims: tuple, rot_dims: tuple):
    axis = random_state.choice(flip_dims)
    flip_fn_inv = flip_fn_fwd = partial(np.flip, axis=axis)

    axes = tuple(random_state.permutation(rot_dims)[:2])
    k = random_state.randint(1, 4)
    rot_fn_fwd = partial(np.rot90, k=k, axes=axes)
    rot_fn_inv = partial(np.rot90, k=-k, axes=axes)

    augm_fn_fwd = lambda x: rot_fn_fwd(flip_fn_fwd(x))
    augm_fn_inv = lambda x: flip_fn_inv(rot_fn_inv(x))

    return augm_fn_fwd, augm_fn_inv


def predict_tta(img: np.ndarray, true: np.ndarray, predict: Callable, pred_init: np.ndarray, n_tta_iter: int,
                zoom_dims: tuple, order: int, random_state: np.random.RandomState, flip_dims: tuple, rot_dims: tuple):
    pred_sum = pred_init.copy()
    for _ in range(n_tta_iter):
        augm_fn_fwd, augm_fn_inv = sample_fwd_and_inv_augm(random_state, flip_dims, rot_dims)
        pred_sum += zoom_pred2true(augm_fn_inv(predict(augm_fn_fwd(img))), true, dims=zoom_dims, order=order)
    return pred_sum / float(n_tta_iter)
