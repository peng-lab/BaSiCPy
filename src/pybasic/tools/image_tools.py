"""Image manipulation tools."""


from typing import Iterable, List, Tuple

import numpy as np


def resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image.

    Args:
        image: input image
        size: desired image output size

    Returns:
        resized image
    """
    ...
    return


def load_image(fname: str) -> np.ndarray:
    """Loads an image.

    Args:
        fname: path to image file

    Returns:
        ndarray of image
    """
    ...
    return


def load_images(fnames: List[str], lazy: bool = False) -> Iterable[np.ndarray]:
    """Load images from files.

    Args:
        fnames: list of paths to images
        lazy: return a generator rather than a list

    Returns:
        an iterable of images (a `generator` if ``lazy = True``, otherwise a `list`)
    """
    if lazy:
        return (load_image(f) for f in fnames)
    else:
        return [load_image(f) for f in fnames]
