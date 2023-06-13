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
    fourier_radius: float = 10,
):
    SF = dctn(image)
    xy = np.meshgrid(*[range(x) for x in image.shape], indexing="ij")
    outside_radius = np.sum(np.array(xy) ** 2, axis=0) > fourier_radius**2
    L0_norm = np.sum(SF[outside_radius] > threshold) / np.sum(outside_radius)
    return L0_norm


def autotune_cost(
    transformed_image: np.ndarray,
    flatfield: np.ndarray,
    entropy_vmin: float,
    entropy_vmax: float,
    histogram_bins: int = 1000,
    fourier_l0_norm_image_threshold: float = 1.0,
    fourier_l0_norm_fourier_radius: float = 10,
    fourier_l0_norm_threshold: float = 1e-3,
    fourier_l0_norm_cost_coef: float = 1e4,
    weights: Optional[np.ndarray] = None,
):
    """Calculate the cost function for autotuning.

    Parameters
    ----------
    transformed_image : np.ndarray
        The transformed image.
    flatfield : np.ndarray
        The flatfield image.
    entropy_vmin : float
        The minimum value of the histogram for the entropy calculation.
    entropy_vmax : float
        The maximum value of the histogram for the entropy calculation.
    histogram_bins : int
        The number of bins to use for the histogram for the entropy calculation.
    fourier_l0_norm_image_threshold : float
        The threshold for image values for the fourier L0 norm calculation.
    fourier_l0_norm_fourier_radius : float
        The Fourier radius for the fourier L0 norm calculation.
    fourier_l0_norm_threshold : float
        The maximum preferred value for the fourier L0 norm.
    fourier_l0_norm_cost_coef : float
        The cost coefficient for the fourier L0 norm.

    """
    entropy_value = entropy(
        transformed_image,
        vmin=entropy_vmin,
        vmax=entropy_vmax,
        bins=histogram_bins,
        weights=weights,
        clip=True,
    )

    n = fourier_L0_norm(
        flatfield,
        fourier_l0_norm_image_threshold,
        fourier_l0_norm_fourier_radius,
    )

    if n < fourier_l0_norm_threshold:
        fourier_L0_norm_cost = 0
    else:
        fourier_L0_norm_cost = (
            n - fourier_l0_norm_threshold
        ) * fourier_l0_norm_cost_coef
    return entropy_value + fourier_L0_norm_cost
