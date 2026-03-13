from typing import Optional, Union

import numpy as np
import torch
import torch_dct as dct


def entropy(
    image: Union[np.ndarray, torch.Tensor],
    vmin: Union[float, torch.Tensor],
    vmax: Union[float, torch.Tensor],
    bins: int = 256,
    weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
    # Convert to tensors if needed
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image)
    else:
        image_tensor = image.cpu() if image.is_cuda else image

    if isinstance(vmin, torch.Tensor):
        vmin_val = vmin.item()
    else:
        vmin_val = float(vmin)

    if isinstance(vmax, torch.Tensor):
        vmax_val = vmax.item()
    else:
        vmax_val = float(vmax)

    if weights is not None:
        if isinstance(weights, np.ndarray):
            weights_tensor = torch.from_numpy(weights)
        else:
            weights_tensor = weights.cpu() if weights.is_cuda else weights
    else:
        weights_tensor = None

    if clip:
        ind = (image_tensor >= vmin_val) * (image_tensor <= vmax_val)
        image_tensor = image_tensor[ind]
        if weights_tensor is not None:
            weights_tensor = weights_tensor[ind]

    prob_density, edges = torch.histogram(
        image_tensor,
        bins,
        range=(vmin_val, vmax_val),
        weight=weights_tensor,
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
    image: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.1,
    fourier_radius: float = 10,
    exclude_edges: bool = True,
):
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image)
    else:
        image_tensor = image
    SF = dct.dct_2d(image_tensor).abs()
    xy = torch.meshgrid(
        *[torch.arange(x).to(image_tensor.device) for x in image_tensor.shape],
        indexing="ij",
    )
    outside_radius = (
        torch.sum(torch.stack([x**2 for x in xy]), dim=0) > fourier_radius**2
    )
    if exclude_edges:
        outside_radius = outside_radius & (xy[0] > 0) & (xy[1] > 0)
    L0_norm = torch.sum(SF[outside_radius] > threshold) / torch.sum(outside_radius)
    return L0_norm


def autotune_cost(
    transformed_image: Union[np.ndarray, torch.Tensor],
    flatfield: Union[np.ndarray, torch.Tensor],
    entropy_vmin: Union[float, torch.Tensor],
    entropy_vmax: Union[float, torch.Tensor],
    histogram_bins: int = 1000,
    fourier_l0_norm_image_threshold: float = 0.1,
    fourier_l0_norm_fourier_radius: float = 10,
    fourier_l0_norm_threshold: float = 1e-3,
    fourier_l0_norm_cost_coef: float = 1e4,
    weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
