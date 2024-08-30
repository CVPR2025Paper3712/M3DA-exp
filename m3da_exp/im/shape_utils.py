import numpy as np
from dpipe.im import crop_to_box
from dpipe.im.box import mask2bounding_box
from dpipe.im.shape_utils import prepend_dims
from skimage.filters import threshold_otsu


def prepend_dims_pw4d(array: np.ndarray):
    return prepend_dims(array) if (array.ndim == 3) else array


def crop_to_body(img, return_box=False):
    box = mask2bounding_box(img > threshold_otsu(img))
    return (crop_to_box(img, box), box) if return_box else crop_to_box(img, box)
