import numpy as np


def entropy(image_float, qmin=0.01, qmax=0.99, ignore_zeros=True):
    """Calculate the entropy of an image.
    Parameters
    ----------
    image_float : array
        The image.
    vmin : float
        The minimum value of the image.
    vmax : float
        The maximum value of the image.
    Returns
    -------
    entropy : float
        The entropy of the image.
    """
    vmin, vmax = np.quantile(image_float, [qmin, qmax])
    image = (image_float - vmin) / (vmax - vmin)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    if ignore_zeros:
        hist = hist[hist > 0]
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist))
    return entropy
