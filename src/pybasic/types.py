"""Types."""

from pathlib import Path
from typing import Union

# import jax
import jax.numpy as jnp
import numpy as np

ArrayLike = Union[np.ndarray, jnp.ndarray]  # dask.array.Array, zarr.Array
PathLike = Union[str, Path]
