import numpy as np
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu


def histogram_matching(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    th_i = threshold_otsu(image)
    th_r = threshold_otsu(reference)

    source_mask = image > th_i
    matched_part = match_histograms(image[source_mask], reference[reference > th_r])

    matched = image.copy()
    matched[source_mask] = matched_part

    return matched
