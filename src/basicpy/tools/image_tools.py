"""Image manipulation tools."""

import logging
from typing import Iterable, List, Tuple

import jax
import jax.numpy as jnp

from basicpy.types import ArrayLike, PathLike

# initialize logger with the package name
logger = logging.getLogger(__name__)


def resize(image: ArrayLike, shape: Tuple[int, int]) -> jnp.ndarray:
    """Resize an image.

    Args:
        image: input image
        shape: desired image output size

    Returns:
        resized image
    """
    return jax.image.resize(image, shape)


def load_image(fname: PathLike) -> jnp.ndarray:  # NOTE return type is a filler
    """Loads an image.

    Args:
        fname: path to image file

    Returns:
        ndarray of image
    """
    ...
    return


def load_images(
    fnames: List[PathLike], lazy: bool = False
) -> Iterable[jnp.ndarray]:  # NOTE return type is a filler
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
