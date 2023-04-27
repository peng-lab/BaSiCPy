from typing import Optional

import numpy as np


def entropy(
    image_float: np.ndarray,
    vmin: float,
    vmax: float,
    ignore_zeros: bool = True,
    bins: int = 256,
    weights: Optional[np.ndarray] = None,
):
    """Calculate the entropy of an image.
    Parameters
    ----------
    image_float : array
        The image.
    vmin : float
        The minimum value of the histogram.
    vmax : float
        The maximum value of the histogram.
    ignore_zeros: bool
        If True, ignore the bin corresponding to zero.
    bins : int
        The number of bins to use for the histogram.
    weights:
        The relative weights for the histogram.

    Returns
    -------
    entropy : float
        The entropy of the image.
    """
    image = (image_float - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)
    hist = np.histogram(image, bins=bins, range=(0, 1), weights=weights)[0]
    if ignore_zeros:
        hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist)) / bins
    return entropy
