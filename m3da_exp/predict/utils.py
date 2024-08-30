import numpy as np
from imops import zoom_to_shape

from m3da_exp.im.histogram_matching import histogram_matching


def load_x_hm(_id, load_x, train_ids, random_state: np.random.RandomState):
    image = load_x(_id)
    reference = load_x(random_state.choice(train_ids))
    matched = histogram_matching(image=image, reference=reference)
    return matched


def zoom_pred2true(pred, true, dims: tuple, order: int = 1):
    if true.shape[-len(dims):] != pred.shape[-len(dims):]:
        pred = zoom_to_shape(pred.astype(np.float32), true.shape[-len(dims):], axis=dims, order=order)
    return pred
