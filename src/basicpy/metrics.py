from typing import Optional

import numpy as np


def entropy(
    image: np.ndarray,
    vmin: float,
    vmax: float,
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
    prob_density, edges = np.histogram(
        image, bins=bins, range=(vmin, vmax), weights=weights, density=True
    )
    # density : p(x) ... normalized such that the integral over the range is 1
    dx = edges[1] - edges[0]
    assert np.allclose(dx, edges[1:] - edges[:-1])
    assert np.isclose(np.sum(prob_density) * dx, 1)
    prob_density = prob_density[prob_density > 0]
    entropy = -np.sum(prob_density * np.log(prob_density)) * dx
    return entropy
