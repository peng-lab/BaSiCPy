from typing import Optional

import numpy as np
import torch
import torch_dct as dct


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
        ind = (image >= vmin) * (image <= vmax)
        image = image[ind]
        if weights is not None:
            weights = weights[ind]
    prob_density, edges = torch.histogram(
        image.cpu(),
        bins,
        range=(vmin.item(), vmax.item()),
        weight=weights.cpu() if weights != None else None,
        density=True,
    )

    # density : p(x) ... normalized such that the integral over the range is 1
    dx = edges[1] - edges[0]

    # assert torch.allclose(
    #     dx, edges[1:] - edges[:-1], atol=torch.max(edges) * 0.01, rtol=0.01
    # )
    #    assert np.isclose(
    #        np.sum(prob_density) * dx, 1
    #    ), f"{np.sum(prob_density) * dx} is not close to 1"
    prob_density = prob_density[prob_density > 0]
    ent = -torch.sum(prob_density * torch.log(prob_density)) * dx
    return ent


def fourier_L0_norm(
    image: np.ndarray,
    threshold: float = 0.1,
    fourier_radius: float = 10,
    exclude_edges: bool = True,
):
    SF = dct.dct_2d(image).abs()
    xy = torch.meshgrid(
        *[torch.arange(x).to(image.device) for x in image.shape], indexing="ij"
    )
    outside_radius = sum([x**2 for x in xy]) > fourier_radius**2
    if exclude_edges:
        outside_radius = outside_radius & (xy[0] > 0) & (xy[1] > 0)
    L0_norm = torch.sum(SF[outside_radius] > threshold) / torch.sum(outside_radius)
    return L0_norm


def autotune_cost(
    transformed_image: np.ndarray,
    flatfield: np.ndarray,
    entropy_vmin: float,
    entropy_vmax: float,
    histogram_bins: int = 1000,
    fourier_l0_norm_image_threshold: float = 0.1,
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
        exclude_edges=True,
    )

    if n < fourier_l0_norm_threshold:
        fourier_L0_norm_cost = 0
    else:
        fourier_L0_norm_cost = (
            n - fourier_l0_norm_threshold
        ) * fourier_l0_norm_cost_coef
    return entropy_value + fourier_L0_norm_cost
