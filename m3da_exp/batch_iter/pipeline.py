import numpy as np
from dpipe.im.box import get_centered_box
from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.shape_ops import crop_to_box


SPATIAL_DIMS = (-3, -2, -1)


def sample_center_uniformly(shape, patch_size, spatial_dims, random_state):
    return np.array([sample_box_center_uniformly((ss, ), np.array([ps, ]), random_state).item()
                     if ps <= ss else (ss // 2)
                     for ps, ss in zip(patch_size, np.asarray(shape)[list(spatial_dims)])])


def center_choice_random(inputs, y_patch_size, random_state: np.random.RandomState):
    x, y = inputs
    return x, y, sample_center_uniformly(y.shape, y_patch_size, spatial_dims=SPATIAL_DIMS, random_state=random_state)


def _sync_centers(center, shape_from, shape_to):
    center_rel = np.array(center) / np.array(shape_from)
    center_sync = np.int16(np.round(center_rel * np.array(shape_to)))
    return center_sync


def center_choice_random_uda(inputs, y_patch_size, random_state: np.random.RandomState, sync_sampling: bool = False):
    xs, xt, ys = inputs
    cs = center_choice_random((xs, ys), y_patch_size, random_state)[-1]
    if sync_sampling:
        ct = _sync_centers(cs, xs.shape, xt.shape)
    else:
        ct = center_choice_random((xt, xt), y_patch_size, random_state)[-1]
    return xs, xt, ys, cs, ct


def center_choice_ts(inputs, y_patch_size, random_state: np.random.RandomState, nonzero_fraction: float = 0.5,
                     center_random_shift=(19, 19, 19)):
    x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    center_random_shift = np.array(center_random_shift)

    if len(centers) > 0 and random_state.uniform() < nonzero_fraction:
        center = centers[random_state.choice(np.arange(len(centers)))]
        # shift augm:
        max_shift = y_patch_size // 2
        low = np.maximum(max_shift, center - center_random_shift)
        high = np.minimum(np.array(y.shape) - max_shift, center + center_random_shift + 1)
        center = center if np.any(low >= high) else random_state.randint(low=low, high=high, size=len(SPATIAL_DIMS))
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size,
                                         spatial_dims=SPATIAL_DIMS, random_state=random_state)

    return x, y, center


def center_choice_uda_ts(inputs, y_patch_size, random_state: np.random.RandomState, nonzero_fraction: float = 0.5,
                         center_random_shift=(19, 19, 19), sync_sampling: bool = False):
    xs, xt, ys, cs = inputs

    y_patch_size = np.array(y_patch_size)
    center_random_shift = np.array(center_random_shift)

    if len(cs) > 0 and random_state.uniform() < nonzero_fraction:
        center = cs[random_state.choice(np.arange(len(cs)))]
        # shift augm:
        max_shift = y_patch_size // 2
        low = np.maximum(max_shift, center - center_random_shift)
        high = np.minimum(np.array(ys.shape) - max_shift, center + center_random_shift + 1)
        center = center if np.any(low >= high) else random_state.randint(low=low, high=high, size=len(SPATIAL_DIMS))
    else:
        center = sample_center_uniformly(ys.shape, patch_size=y_patch_size,
                                         spatial_dims=SPATIAL_DIMS, random_state=random_state)

    if sync_sampling:
        center_t = _sync_centers(center, xs.shape, xt.shape)
    else:
        center_t = sample_center_uniformly(ys.shape, patch_size=y_patch_size,
                                           spatial_dims=SPATIAL_DIMS, random_state=random_state)

    return xs, xt, ys, center, center_t


def extract_patch(inputs, patch_sizes, padding_values, spatial_dims=SPATIAL_DIMS):
    *inputs, center = inputs
    return [crop_to_box(inp, box=get_centered_box(center, patch_size), padding_values=padding_value, axis=spatial_dims)
            for inp, patch_size, padding_value in zip(inputs, patch_sizes, padding_values)]
    
    
def extract_patch_uda(inputs, *args, **kwargs):
    xs, xt, ys, cs, ct = inputs
    xs, y = extract_patch((xs, ys, cs), *args, **kwargs)
    xt = extract_patch((xt, ct), *args, **kwargs)[0]
    return xs, xt, y


def sample_patch(inputs, patch_size, random_state: np.random.RandomState):
    x1, x2, y = inputs
    center = sample_center_uniformly(y.shape, patch_size=patch_size, spatial_dims=SPATIAL_DIMS,
                                     random_state=random_state)
    return x1, x2, y, center
