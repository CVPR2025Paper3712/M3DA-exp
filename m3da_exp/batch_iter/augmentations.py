import warnings

import numpy as np
from imops import zoom, zoom_to_shape
from scipy.ndimage import gaussian_filter
from skimage.exposure import adjust_gamma

from .pipeline import SPATIAL_DIMS


def d48_augm(inputs, random_state: np.random.RandomState, flip_dims: tuple = None, rot_dims: tuple = None):
    if flip_dims is None:
        flip_dims = SPATIAL_DIMS
    if rot_dims is None:
        rot_dims = SPATIAL_DIMS

    outputs = inputs

    if random_state.rand() < 0.5:
        axis = random_state.choice(flip_dims)
        outputs = [np.flip(x, axis=axis) for x in outputs]

    if random_state.rand() < (23 / 24):
        axes = tuple(random_state.permutation(rot_dims)[:2])
        k = random_state.randint(1, 4)
        outputs = [np.rot90(x, k=k, axes=axes) for x in inputs]

    return outputs


def gamma_augm(x, random_state: np.random.RandomState, max_gamma: float = 2.0):
    gamma = random_state.choice((random_state.uniform(1 / max_gamma, 1), random_state.uniform(1, max_gamma)))
    return adjust_gamma(x, gamma=gamma)


def nnunet_augm(x, random_state: np.random.RandomState):
    x = np.copy(x)
    # 1. scale (0.7, 1.4), p=0.2
    # skipped here, since it needs y transform

    # 2. rot90
    # already applied

    # 3. Gaussian noise
    if random_state.rand() < 0.1:
        x += random_state.normal(0, random_state.uniform(0, 0.1), size=x.shape)
        x = np.clip(x, a_min=0, a_max=1)

    # 4. Gaussian blur
    if random_state.rand() < 0.1:
        x = gaussian_filter(x, random_state.uniform(0.5, 1))

    # 5. Brightness
    if random_state.rand() < 0.15:
        x *= random_state.uniform(0.75, 1.25)

    # 6. Contrast
    if random_state.rand() < 0.15:
        a_min, mean, a_max = x.min(), x.mean(), x.max()
        x = (x - mean) * random_state.uniform(0.75, 1.25) + mean
        x = np.clip(x, a_min, a_max)

    # 7. Lowres
    if random_state.rand() < 0.125:
        a_min, a_max = x.min(), x.max()
        x_low = zoom(x, random_state.uniform(0.5, 1.0), order=0)
        # with warnings.catch_warnings(action="ignore"):  # available in python>=3.11
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # UserWarning: ... Falling back to scipy's implementation.
            x = zoom_to_shape(x_low, x.shape, order=3, )
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min) * (a_max - a_min) + a_min

    # 8. Gamma & "gamma+inverse"
    if random_state.rand() < 0.37:
        x = adjust_gamma(x, gamma=random_state.uniform(0.7, 1.5))

    return x
