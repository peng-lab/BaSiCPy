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
from typing import Dict, Tuple, Union
from functools import partial

# 3rd party modules
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel, Field, PrivateAttr
from scipy.fftpack import dct
from skimage.transform import resize

# Package modules
from basicpy.types import ArrayLike
from basicpy.tools.dct2d_tools import SciPyDCT
from jax import device_put
from basicpy.tools.dct2d_tools import JaxDCT

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d
newax = jnp.newaxis


def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


def _check_and_init_variables(images, weight, S, D_R, D_Z, B, I_R):
    if weight is None:
        weight = jnp.ones(images.shape, dtype=jnp.float32)
    if S is None:
        S = jnp.zeros(images.shape[1:], dtype=jnp.float32)
    if D_R is None:
        D_R = jnp.zeros(images.shape[1:], dtype=jnp.float32)
    if D_Z is None:
        D_Z = 0.0
    if B is None:
        B = jnp.ones(images.shape[0], dtype=jnp.float32)
    if I_R is None:
        I_R = jnp.zeros(images.shape, dtype=jnp.float32)

    if S.shape != images.shape[1:]:
        raise ValueError("S must have the same shape as images.shape[1:]")
    if D_R.shape != images.shape[1:]:
        raise ValueError("D_R must have the same shape as images.shape[1:]")
    if not jnp.isscalar(D_Z):
        raise ValueError("D_Z must be a scalar.")
    if B.shape != images.shape[:1]:
        raise ValueError("B must have the same shape as images.shape[:1]")
    if I_R.shape != images.shape:
        raise ValueError("I_R must have the same shape as images.shape")
    if weight.shape != images.shape:
        raise ValueError("weight must have the same shape as images.shape")
    return weight, S, D_R, D_Z, B, I_R


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
    lambda_darkfield: float = Field(
        0.0,
        description="Darkfield offset for weight updates.",
    )
    lambda_flatfield: float = Field(
        0.0,
        description="Flatfield offset for weight updates.",
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
    init_mu: float = Field(0, description="Initial value for mu.")
    D_Z_max: float = Field(0, description="Maximum value for D_Z.")
    image_norm: float = Field(0, description="The 2nd order norm for the images.")
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    max_mu: float = Field(0, description="The maximum value of mu.")
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance.",
    )
    varying_coeff: bool = Field(
        True,
        description="This description will need to be filled in.",
    )
    working_size: int = Field(
        128,
        description="Size for running computations. Should be a power of 2 (2^n).",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _flatfield: np.ndarray = PrivateAttr(None)
    _darkfield: np.ndarray = PrivateAttr(None)

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

    def _ladmap_step(self, Im, W, vals):
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        eta = jnp.sum(B**2) * 1.02
        S = S + jnp.sum(B[:, newax, newax] * (Im - I_B - I_R + Y / mu), axis=0) / eta
        S = idct2d(_jshrinkage(dct2d(S), self.lambda_flatfield / (eta * mu)))

        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        I_R = _jshrinkage(Im - I_B + Y / mu, W / mu)

        R = Im - I_R
        B = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2)) / jnp.sum(S**2)
        B = jnp.maximum(B, 0)

        BS = S[newax, ...] * B[:, newax, newax]
        if self.get_darkfield:
            D_Z = jnp.mean(Im - BS - D_R[newax, ...] - I_R + Y / 2.0 / mu)
            D_Z = jnp.clip(D_Z, 0, self.D_Z_max)
            eta_D = Im.shape[0] * 1.02
            D_R = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[newax, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R = idct2d(_jshrinkage(dct2d(D_R), self.lambda_darkfield / eta_D / mu))
            D_R = _jshrinkage(D_R, self.lambda_darkfield / eta_D / mu)

        I_B = BS + D_R[newax, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)

    def _ladmap_cond(self, vals):
        k, _, _, _, _, _, _, _, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return jnp.all(
            jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
        )

    def fit_ladmap_single(
        self,
        Im,
        W=None,
        S=None,
        D_R=None,
        D_Z=None,
        B=None,
        I_R=None,
    ):
        # initialize values
        W, S, D_R, D_Z, B, I_R = _check_and_init_variables(Im, W, S, D_R, D_Z, B, I_R)

        Y = jnp.ones_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf
        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)
        ladmap_step = partial(self._ladmap_step, Im, W)
        while self._ladmap_cond(vals):
            vals = ladmap_step(vals)
        # vals = lax.while_loop(self._ladmap_cond, ladmap_step, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations

    def fit_approximate_single(
        self, images, weight, S=None, D_R=None, D_Z=None, B=None, I_R=None
    ):
        pass

    def fit_ladmap(self, images: np.ndarray) -> None:
        """
        Fit the LADMAP algorithm to the images.
        """
        mean_image = np.mean(images, axis=2)
        mean_image = mean_image / np.mean(mean_image)
        mean_image_dct = SciPyDCT.dct2d(mean_image.T)
        self.lambda_flatfield = np.sum(np.abs(mean_image_dct)) / 400 * 0.5
        self.lambda_darkfield = self.lambda_flatfield * 0.2
        # matrix 2-norm (largest sing. value)
        spectral_norm = np.linalg.norm(images.reshape((images.shape[0], -1)), ord=2)
        self.init_mu = self.mu_coef / spectral_norm
        self.max_mu = self.init_mu * self.max_mu_coef
        self.D_Z_max = jnp.min(images)
        self.image_norm = np.linalg.norm(images.flatten(), ord=2)

        Im = device_put(images).astype(jnp.float32)
        W = jnp.ones_like(Im, dtype=jnp.float32)
        last_S = None
        last_D = None
        S = None
        D = None
        for i in range(self.max_reweight_iterations):
            # TODO: reuse the flatfield and darkfields?
            # TODO: loop jit?
            S, D_R, D_Z, I_R, B, norm_ratio, converged = self.fit_ladmap_single(
                Im,
                W,
            )
            # TODO: warn if not converged
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfield
            B = B * mean_S  # baseline
            D = D_R + D_Z  # darkfield
            W = jnp.ones_like(Im, dtype=np.float32) / (
                jnp.abs(I_R / S[newax, ...]) + self.epsilon
            )
            # weight = weight / jnp.mean(weight)

            if last_S is not None:
                mad_flatfield = jnp.sum(np.abs(S - last_S)) / jnp.sum(np.abs(last_S))
                if self.get_darkfield:
                    mad_darkfield = jnp.sum(jnp.abs(D - last_D)) / max(
                        jnp.sum(np.abs(last_D)), 1
                    )  # assumes the amplitude of darkfield is more than 1
                    self._reweight_score = max(mad_flatfield, mad_darkfield)
                else:
                    self._reweight_score = mad_flatfield
                if self._reweight_score <= self.reweighting_tol:
                    break
            last_S = S
            last_D = D
            print(i)
        self.flatfield = S
        self.darkfield = D

    def fit(self, images: np.ndarray) -> None:
        """Generate illumination correction profiles.

        Args:
            images: Input images to fit shading model. Images should be stacked
                along the z-dimension.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy.tools import load_images
            >>> images = load_images('./images')
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        assert images.ndim == 3

        # Resize the images
        images = images.astype(np.float64)
        working_shape = (self.working_size, self.working_size, images.shape[2])
        D = resize(
            images, (working_shape), order=1, mode="symmetric", preserve_range=True
        )

        # Get initial frequencies
        D_mean = D.mean(axis=2)
        D_mean /= D_mean.mean()
        W_D_mean = dct(dct(D_mean, norm="ortho").T, norm="ortho")

        # Set lambdas if null
        if self.lambda_flatfield == 0:
            self.lambda_flatfield = np.sum(np.abs(W_D_mean)) / 400 * 0.5
        if self.lambda_darkfield == 0:
            self.lambda_darkfield = self.lambda_flatfield * 0.2

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
            self._flatfield = resize(self.flatfield, images.shape[:2])
            self._darkfield = resize(self.darkfield, images.shape[:2])

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
