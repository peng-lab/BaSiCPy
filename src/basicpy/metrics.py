import numpy as np


def entropy(
    image_float: np.ndarray,
    qmin: float = 0.01,
    qmax: float = 0.99,
    ignore_zeros: bool = True,
    bins: int = 256,
):
    """Calculate the entropy of an image.
    Parameters
    ----------
    image_float : array
        The image.
    qmin : float
        The minimum quantile of the image.
    qmax : float
        The maximum quantile of the image.
    bins : int
        The number of bins to use for the histogram.
    Returns
    -------
    entropy : float
        The entropy of the image.
    """
    vmin, vmax = np.quantile(image_float, [qmin, qmax])
    image = (image_float - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)
    hist = np.histogram(image, bins=bins, range=(0, 1))[0]
    if ignore_zeros:
        hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist)) / bins
    return entropy
