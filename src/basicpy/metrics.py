from typing import Optional

import numpy as np
from scipy.fft import dctn


def entropy(
    image: np.ndarray,
    vmin: float,
    vmax: float,
    bins: int = 256,
    weights: Optional[np.ndarray] = None,
    clip: bool = True,
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
    clip: bool
        If True, clip the image to the range [vmin, vmax].

    Returns
    -------
    entropy : float
        The entropy of the image.
    """
    if clip:
        image = image[np.logical_and(image >= vmin, image <= vmax)]
    prob_density, edges = np.histogram(
        image, bins=bins, range=(vmin, vmax), weights=weights, density=True
    )
    # density : p(x) ... normalized such that the integral over the range is 1
    dx = edges[1] - edges[0]
    assert np.allclose(dx, edges[1:] - edges[:-1], atol=np.max(edges) * 0.01, rtol=0.01)
    #    assert np.isclose(
    #        np.sum(prob_density) * dx, 1
    #    ), f"{np.sum(prob_density) * dx} is not close to 1"
    prob_density = prob_density[prob_density > 0]
    entropy = -np.sum(prob_density * np.log(prob_density)) * dx
    return entropy


def fourier_L0_norm(
    image: np.ndarray,
    threshold: float = 1.0,
    fourier_radius=10,
):
    SF = dctn(image)
    xy = np.meshgrid(*[range(x) for x in image.shape])
    outside_radius = np.sum(np.array(xy) ** 2, axis=0) > fourier_radius**2
    L0_norm = np.sum(SF[outside_radius] > threshold) / np.sum(outside_radius)
    return L0_norm
