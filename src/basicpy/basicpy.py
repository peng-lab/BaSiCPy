"""Main BaSiC class.

Todo:
    Keep examples up to date with changing API.
"""

# Core modules
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from multiprocessing import cpu_count
from typing import Dict, Iterable, Tuple, Union, Optional

# 3rd party modules
import numpy as np
from jax import device_put
import jax.numpy as jnp
from jax.image import resize, ResizeMethod
from pydantic import BaseModel, Field, PrivateAttr
from skimage.transform import resize as _resize

# Package modules
from basicpy.types import ArrayLike
from basicpy.tools.dct2d_tools import JaxDCT
from basicpy._jax_routines import LadmapFit, ApproximateFit

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d
newax = jnp.newaxis

# from basicpy.tools.dct2d_tools import dct2d, idct2d
# from basicpy.tools.inexact_alm import inexact_alm_rspca_l1

# Get number of available threads to limit CPU thrashing
# From preadator: https://pypi.org/project/preadator/
if hasattr(os, "sched_getaffinity"):
    # On Linux, we can detect how many cores are assigned to this process.
    # This is especially useful when running in a Docker container, when the
    # number of cores is intentionally limited.
    NUM_THREADS = len(os.sched_getaffinity(0))  # type: ignore
else:
    # Default back to multiprocessing cpu_count, which is always going to count
    # the total number of cpus
    NUM_THREADS = cpu_count()


class Device(Enum):

    cpu: str = "cpu"
    gpu: str = "gpu"
    tpu: str = "tpu"


class FittingMode(Enum):

    ladmap: str = "ladmap"
    approximate: str = "approximate"


# multiple channels should be handled by creating a `basic` object for each chan
class BaSiC(BaseModel):
    """A class for fitting and applying BaSiC illumination correction profiles."""

    darkfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the darkfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    device: Device = Field(
        Device.cpu,
        description="Must be one of ['cpu','gpu','tpu'].",
        exclude=True,  # Don't dump to output json/yaml
    )
    fitting_mode: FittingMode = Field(
        FittingMode.ladmap, description="Must be one of ['ladmap', 'approximate']"
    )

    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    flatfield: np.ndarray = Field(
        default_factory=lambda: np.zeros((128, 128), dtype=np.float64),
        description="Holds the flatfield component for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    lambda_flatfield_coef: float = Field(
        1.0 / 400 * 0.5, description="Weight of the flatfield term in the Lagrangian."
    )
    lambda_darkfield_coef: float = Field(
        0.2, description="Relative weight of the darkfield term in the Lagrangian."
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )
    max_reweight_iterations: int = Field(
        10,
        description="Maximum number of reweighting iterations.",
    )
    max_workers: int = Field(
        NUM_THREADS,
        description="Maximum number of threads used for processing.",
        exclude=True,  # Don't dump to output json/yaml
    )
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    resize_method: ResizeMethod = Field(
        ResizeMethod.CUBIC,
        description="Resize method to use when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance.",
    )
    residual_weighting: bool = Field(
        False,
        description="Weighting by the residuals.",
    )
    working_size: Optional[Union[int, Iterable[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _flatfield: np.ndarray = PrivateAttr(None)
    _darkfield: np.ndarray = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)

    class Config:

        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """Initialize BaSiC with the provided settings."""

        super().__init__(**kwargs)

        if self.working_size != 128:
            self.darkfield = np.zeros((self.working_size,) * 2, dtype=np.float64)
            self.flatfield = np.zeros((self.working_size,) * 2, dtype=np.float64)

        # Initialize the internal cache
        self._darkfield = np.zeros((self.working_size,) * 2, dtype=np.float64)
        self._flatfield = np.zeros((self.working_size,) * 2, dtype=np.float64)

        if self.device is not Device.cpu:
            # TODO: sanity checks on device selection
            pass

    def __call__(
        self, images: np.ndarray, timelapse: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for BaSiC.transform"""

        return self.transform(images, timelapse)

    def _resize(self, Im):
        """
        Resize the images to the working size.
        """
        if self.working_size is not None:
            if np.isscalar(self.working_size):
                working_shape = [self.working_size] * (Im.ndim - 1)
            else:
                if not Im.ndim - 1 == len(self.working_size):
                    raise ValueError(
                        "working_size must be a scalar or match the image dimensions"
                    )
                else:
                    working_shape = self.working_size
            Im = resize(Im, [Im.shape[0], *working_shape], self.resize_method)
        return Im

    def fit(self, images: np.ndarray) -> None:
        """
        Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional array with dimension of (T,Y,X).

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy.tools import load_images
            >>> images = load_images('./images')
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        Im = device_put(images).astype(jnp.float32)
        Im = self._resize(Im)

        mean_image = jnp.mean(Im, axis=2)
        mean_image = mean_image / jnp.mean(Im)
        mean_image_dct = dct2d(mean_image.T)
        lambda_flatfield = jnp.sum(jnp.abs(mean_image_dct)) * self.lambda_flatfield_coef

        spectral_norm = jnp.linalg.norm(Im.reshape((Im.shape[0], -1)), ord=2)
        init_mu = self.mu_coef / spectral_norm
        fit_params = self.dict()
        fit_params.update(
            dict(
                lambda_flatfield=lambda_flatfield,
                lambda_darkfield=lambda_flatfield * self.lambda_darkfield_coef,
                # matrix 2-norm (largest sing. value)
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=jnp.min(Im),
                image_norm=jnp.linalg.norm(Im.flatten(), ord=2),
            )
        )

        # Initialize variables
        W = jnp.ones_like(Im, dtype=jnp.float32)
        last_S = None
        last_D = None
        D = None

        if self.fitting_mode == FittingMode.ladmap:
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        for i in range(self.max_reweight_iterations):
            # TODO: loop jit?
            # TODO: reusing last values?
            S = jnp.zeros(Im.shape[1:], dtype=jnp.float32)
            D_R = jnp.zeros(Im.shape[1:], dtype=jnp.float32)
            D_Z = 0.0
            B = jnp.ones(Im.shape[0], dtype=jnp.float32)
            I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
            S, D_R, D_Z, I_R, B, norm_ratio, converged = fitting_step(
                Im,
                W,
                S,
                D_R,
                D_Z,
                B,
                I_R,
            )
            self._score = norm_ratio
            # TODO: warn if not converged
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfields
            B = B * mean_S  # baseline
            D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
            I_B = B[:, newax, newax] * S[newax, ...] + D_R[newax, ...] + D_Z
            W = jnp.ones_like(Im, dtype=np.float32) / (
                jnp.abs(I_R / I_B) + self.epsilon
            )
            self._weight = W
            self._residual = I_R

            if last_S is not None:
                mad_flatfield = jnp.sum(jnp.abs(S - last_S)) / jnp.sum(np.abs(last_S))
                if self.get_darkfield:
                    mad_darkfield = jnp.sum(jnp.abs(D - last_D)) / max(
                        jnp.sum(jnp.abs(last_D)), 1
                    )  # assumes the amplitude of darkfield is more than 1
                    self._reweight_score = max(mad_flatfield, mad_darkfield)
                else:
                    self._reweight_score = mad_flatfield
                if self._reweight_score <= self.reweighting_tol:
                    break
            last_S = S
            last_D = D

        self.flatfield = S
        self.darkfield = D

    def transform(
        self, images: np.ndarray, timelapse: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Apply profile to images.

        Todo:
            Add in baseline/timelapse correction.

        Args:
            images: input images to correct
            timelapse: calculate timelapse/photobleaching offsets. Currently
                does nothing.

        Returns:
            An array of the same size as images. If timelapse is True, returns
                a flat array of baseline corrections used in the calculations.

        Example:
            >>> basic.fit(images)
            >>> corrected = basic.transform(images)
            >>> for i, im in enumerate(corrected):
            ...     imsave(f"image_{i}.tif")
        """

        # Convert to the correct format
        im_float = images.astype(np.float64)

        # Check the image size
        if not all(i == d for i, d in zip(self._flatfield.shape, images.shape)):
            self._flatfield = _resize(self.flatfield, images.shape[:2])
            self._darkfield = _resize(self.darkfield, images.shape[:2])

        # Initialize the output
        output = np.zeros(images.shape, dtype=images.dtype)

        if timelapse:
            # calculate timelapse from input series
            ...

        def unshade(ins, outs, i, dark, flat):
            outs[..., i] = (ins[..., i] - dark) / flat

        # If one or fewer workers, don't user ThreadPool. Useful for debugging.
        if self.max_workers <= 1:
            for i in range(images.shape[-1]):
                unshade(im_float, output, i, self._darkfield, self._flatfield)

        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                threads = executor.map(
                    lambda x: unshade(
                        im_float, output, x, self._darkfield, self._flatfield
                    ),
                    range(images.shape[-1]),
                )

                # Get the result of each thread, this should catch thread errors
                for thread in threads:
                    assert thread is None

        return output.astype(images.dtype)

    # REFACTOR large datasets will probably prefer saving corrected images to
    # files directly, a generator may be handy
    def fit_transform(
        self, images: ArrayLike, timelapse: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit and transform on data.

        Args:
            images: input images to fit and correct

        Returns:
            profiles and corrected images

        Example:
            >>> profiles, corrected = basic.fit_transform(images)
        """
        self.fit(images)
        corrected = self.transform(images, timelapse)

        # NOTE or only return corrected images and user can get profiles separately
        return corrected

    @property
    def score(self):
        """The BaSiC fit final score"""
        return self._score

    @property
    def reweight_score(self):
        """The BaSiC fit final reweighting score"""
        return self._reweight_score

    @property
    def settings(self) -> Dict:
        """Current settings.

        Returns:
            current settings
        """
        return self.dict()
