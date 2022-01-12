"""Main BaSiC class."""

import os
from typing import Iterable, List, Union

import jax.numpy as jnp
import numpy as np

from .profile import Profile
from .settings import Settings
from .tools.dct2d_tools import dct2d, idct2d
from .tools.inexact_alm import inexact_alm_rspca_l1

PathLike = Union[str, bytes, os.PathLike]


mm = jnp.matmul


# NOTE how should multiple channels be handled?
# NOTE if settings need to be updated, should user reinitialize class?
class BaSiC:
    """A class for fitting and applying BaSiC correction models."""

    def __init__(
        self,
        settings: Settings,
        device: str = "cpu",
    ) -> None:
        """Inits the BaSiC class with the provided settings.

        Args:
            settings: :meth:`pybasic.Settings` object
            device: device to use, options are `"cpu"`, `"gpu"`, `"tpu"`
        """
        self._settings = settings
        self._device = device
        ...

    def fit(self, images: Iterable[np.ndarray]) -> List[Profile]:
        """Generate illumination correction profiles.

        Args:
            images: input images to predict illumination model

        Returns:
            correction profile model :meth:`pybasic.Model` objects

        Example:
            >>> from pybasic import BaSiC, Settings
            >>> from pybasic.tools import load_images
            >>> images = load_images('./images', lazy=True)
            >>> settings = Settings()
            >>> basic = BaSiC(settings)
            >>> profiles = basic.fit(images)

        Todo:
            * Encourage use of generator to provide images, reducing memory usage
        """
        settings = self._initialize_settings(images)
        return self._run(images, settings)

    def predict(
        self, images: Iterable[np.ndarray], profiles: List[np.ndarray]
    ) -> np.ndarray:
        """Apply profile to images.

        Args:
            images: input images to correct
            profiles: illumination correction profiles

        Returns:
            generator to apply illumination correction

        Example:
            >>> profiles = basic.fit(images)
            >>> corrected = basic.predict(images, profiles)
            >>> for i, im in enumerate(corrected):
            ...     imsave(f"image_{i}.tif")
        """

        def apply_profiles(im):
            for prof in profiles:
                im = prof.apply(im)

        return (apply_profiles(im) for im in images)

    def _run(self):
        """Run BaSiC."""
        ...

    def _initialize_settings(self, im_stack: np.ndarray) -> Settings:
        """Get initial settings.

        Args:
            im_stack: input images as a stack

        Returns:
            initialized settings
        """
        ...
        return

    def __repr__(self):
        return self._settings.__repr__()

    def __enter__(self):
        # for context management
        ...

    def __exit__(self):
        # for context management
        ...
