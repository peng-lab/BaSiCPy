"""2D discrete consine transform tools."""

import jax.numpy as jnp

from ..types import ArrayLike


def dct2d(im_stack: ArrayLike) -> jnp.ndarray:
    """Calculates 2D discrete cosine transform.

    Args:
        im_stack: input image stack

    Returns:
        transformed image stack
    """
    ...
    return


def idct2d(im_stack: ArrayLike) -> jnp.ndarray:
    """Calculates 2D inverse discrete cosine transform.

    Args:
        im_stack: input image stack

    Returns:
        transformed image stack
    """
    ...
    return
