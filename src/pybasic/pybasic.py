"""Main BaSiC class.

Todo:
    Keep examples up to date with changing API.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple

# import jax
import jax.numpy as jnp
import numpy as np

# REFACTOR relative imports or package level?
from .profile import Profile
from .settings import Settings
from .types import ArrayLike

# from pybasic.tools.dct2d_tools import dct2d, idct2d
# from pybasic.tools.inexact_alm import inexact_alm_rspca_l1


# Shorthand for common operations
mm = jnp.matmul


# multiple channels should be handled by creating a `basic` object for each chan
class BaSiC:
    """A class for fitting and applying BaSiC illumination correction profiles."""

    # NOTE Docs could be consolidated. may be better design to pass *args, **kwargs
    # to Settings, but this alone will not copy function signature which is nice to
    # have for building documentation and for working in IDE
    def __init__(
        self,
        # REFACTOR defaults are defined here and settings class... will there ever be
        # a case where `basic` settings will be updated by the user?
        darkfield: bool = False,  # alphabetized
        epsilon: float = 0.1,
        estimation_mode: str = "l0",
        lambda_darkfield: float = 0,
        lambda_flatfield: float = 0,
        max_iterations: int = 500,
        max_reweight_iterations: int = 10,
        optimization_tol: float = 1e-6,
        reweighting_tol: float = 1e-3,
        timelapse: bool = False,
        varying_coeff: bool = True,
        working_size: int = 128,
        device: str = "cpu",  # device last
    ) -> None:
        """Initialize BaSiC with the provided settings.

        Args:
            darkfield: whether to estimate a darkfield correction
            epsilon:
            estimation_mode:
            lambda_darkfield:
            lambda_flatfield:
            max_iterations: maximum number of iterations allowed in the optimization
            max_reweight_iterations:
            optimization_tol: error tolerance in the optimization
            reweighting_tol:
            timelapse: whether to estimate photobleaching effect
            varying_coeff:
            working_size:
            device: device to use, options are `"cpu"`, `"gpu"`, `"tpu"`

        See Also:
            :meth:`pybasic.Settings`

        Todo:
            * Fill in parameter descriptions
        """
        self.settings = Settings(
            darkfield=darkfield,
            epsilon=epsilon,
            estimation_mode=estimation_mode,
            lambda_darkfield=lambda_darkfield,
            lambda_flatfield=lambda_flatfield,
            max_iterations=max_iterations,
            max_reweight_iterations=max_reweight_iterations,
            optimization_tol=optimization_tol,
            reweighting_tol=reweighting_tol,
            timelapse=timelapse,
            varying_coeff=varying_coeff,
            working_size=working_size,
        )

        # TODO run a check on device
        self._device = device
        ...

    def fit(self, images: Iterable[np.ndarray]) -> List[Profile]:
        """Generate illumination correction profiles.

        Args:
            images: input images to predict illumination model

        Returns:
            self after optimization

        Example:
            >>> from pybasic import BaSiC
            >>> from pybasic.tools import load_images
            >>> images = load_images('./images')
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        Tip:
            * Use a generator to provide images, reducing memory usage
        """
        # initial parameters from settings
        init_params = self._initialize_params(images)
        return self._run(images, init_params)

    def predict(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Apply profile to images.

        Args:
            images: input images to correct

        Returns:
            generator to apply illumination correction

        Example:

            .. code-block:: python

                ...
                >>> basic.fit(images)
                >>> corrected = basic.predict(images)
                >>> for i, im in enumerate(corrected):
                ...     imsave(f"image_{i}.tif")
        """
        if self.settings.timelapse:
            # calculate timelapse from input series
            ...

        def apply_profiles(im):
            for prof in self.profiles:
                im = prof.apply(im)

        return (apply_profiles(im) for im in images)

    # REFACTOR large datasets will probably prefer saving corrected images to
    # files directly, a generator may be handy
    def fit_predict(
        self, images: Iterable[ArrayLike]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Fit and predict on data.

        Args:
            images: input images to fit and correct

        Returns:
            profiles and corrected images

        Example:
            >>> profiles, corrected = basic.fit_predict(images)
        """
        self.fit(images)
        corrected = self.predict(images)
        profiles = self.profiles
        # NOTE or only return corrected images and user can get profiles separately
        return (profiles, corrected)

    @property
    def profiles(self) -> List[Profile]:
        """Estimated illumination correction profiles."""
        return self._profiles

    @profiles.setter
    def profiles(self, profiles: List[Profile]):
        """Alias for `load`."""
        self._profiles = profiles

    def load(self, profiles: List[Profile]):
        """Load `profiles`.

        Args:
            profiles: list of precalculated illumination correction profiles

        Example:
            >>> flatfield_prof = Profile(np.load("flatfield.npy"), type="flatfield")
            >>> darkfield_prof = Profile(np.load("darkfield.npy"), type="darkfield")
            >>> basic = BaSiC()
            >>> basic.load([flatfield_prof, darfield_prof])
        """
        self._profiles = profiles

    def _run(self, images, settings):
        """Run BaSiC.

        Returns:
            self after optimization
        """
        ...
        return self  # return like sklearn, but state still held in object

    # NOTE not sure yet about return type and state change
    def _initialize_params(self, im_stack: np.ndarray):
        """Get initial model parameters.

        Args:
            im_stack: input images as a stack

        Returns:
            initial parameters
        """
        ...
        return

    def __repr__(self):
        return self._settings.__repr__()
