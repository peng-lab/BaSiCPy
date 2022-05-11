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

# 3rd party modules
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel, Field, PrivateAttr
from scipy.fftpack import dct
from skimage.transform import resize

# Package modules
from basicpy.types import ArrayLike
from basicpy.tools import fit
from basicpy.tools.dct2d_tools import SciPyDCT
from jax import jit, lax, device_put
from basicpy.tools.dct2d_tools import JaxDCT

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d


def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


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
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
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

    def _fit_ladmap_single(
        self,
        images,
        weight,
    ):
        # image dimension ... (time, Z, Y, X)
        assert np.array_equal(images.shape, weight.shape)

        # matrix 2-norm (largest sing. value)
        spectral_norm = np.linalg.norm(images.reshape((images.shape[0], -1)), ord=2)
        mu = self.mu_coef / spectral_norm
        max_mu = mu * self.max_mu_coef

        init_image_norm = np.linalg.norm(images.flatten(), ord=2)

        Im = device_put(images).astype(jnp.float32)
        D_Z_max = jnp.min(Im)

        # initialize values
        S = jnp.zeros(images.shape[1:], dtype=jnp.float32)
        D_R = jnp.zeros(images.shape[1:], dtype=jnp.float32)
        D_Z = 0
        B = jnp.ones(images.shape[0], dtype=jnp.float32)
        I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
        Y = jnp.ones_like(Im, dtype=jnp.float32)
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

        @jit
        def basic_step_ladmap(vals):
            k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
            I_B = (
                S[jnp.newaxis, ...] * B[:, jnp.newaxis, jnp.newaxis]
                + D_R[jnp.newaxis, ...]
                + D_Z
            )
            eta = jnp.sum(B**2) * 1.02
            S = (
                S
                + jnp.sum(
                    B[:, jnp.newaxis, jnp.newaxis] * (Im - I_B - I_R + Y / mu), axis=0
                )
                / eta
            )
            S = idct2d(_jshrinkage(dct2d(S), self.lambda_flatfield / (eta * mu)))

            I_B = (
                S[jnp.newaxis, ...] * B[:, jnp.newaxis, jnp.newaxis]
                + D_R[jnp.newaxis, ...]
                + D_Z
            )
            I_R = _jshrinkage(Im - I_B + Y / mu, weight / mu)

            R = Im - I_R
            B = jnp.sum(S[jnp.newaxis, ...] * (R + Y / mu), axis=(1, 2)) / jnp.sum(
                S**2
            )
            B = jnp.maximum(B, 0)

            BS = S[jnp.newaxis, ...] * B[:, jnp.newaxis, jnp.newaxis]
            if self.get_darkfield:
                D_Z = jnp.mean(Im - BS - D_R[jnp.newaxis, ...] - I_R + Y / 2.0 / mu)
                D_Z = jnp.clip(D_Z, 0, D_Z_max)
                eta_D = Im.shape[0] * 1.02
                D_R = D_R + 1.0 / eta_D * jnp.sum(
                    Im - BS - D_R[jnp.newaxis, ...] - D_Z - I_R + Y / mu, axis=0
                )
                D_R = idct2d(
                    _jshrinkage(dct2d(D_R), self.lambda_darkfield / eta_D / mu)
                )
                D_R = _jshrinkage(D_R, self.lambda_darkfield / eta_D / mu)

            I_B = BS + D_R[jnp.newaxis, ...] + D_Z
            fit_residual = R - I_B
            Y = Y + mu * fit_residual
            mu = jnp.minimum(mu * self.rho, max_mu)

            return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)

        @jit
        def continuing_cond(vals):
            k, _, _, _, _, _, _, _, fit_residual = vals
            norm_ratio = (
                jnp.linalg.norm(fit_residual.flatten(), ord=2) / init_image_norm
            )
            #            print(i,norm_ratio,D_Z)
            return jnp.all(
                jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
            )

        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)
        vals = lax.while_loop(continuing_cond, basic_step_ladmap, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / init_image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations

    def fit_ladmap(self, images: np.ndarray) -> None:
        """
        Fit the LADMAP algorithm to the images.
        """
        meanD = np.mean(images, axis=2)
        meanD = meanD / np.mean(meanD)
        W_meanD = SciPyDCT.dct2d(meanD.T)
        self.lambda_flatfield = np.sum(np.abs(W_meanD)) / 400 * 0.5
        self.lambda_darkfield = self.lambda_flatfield * 0.2

        weight = np.ones_like(images, dtype=np.float32)
        last_S = None
        last_D = None
        S = None
        D = None
        for i in range(self.max_reweight_iterations):
            # TODO: reuse the flatfield and darkfields?
            # TODO: loop jit?
            S, D_R, D_Z, I_R, B, norm_ratio, converged = fit._fit_ladmap_single(
                images,
                weight,
                self.lambda_darkfield,
                self.lambda_flatfield,
                self.get_darkfield,
                self.optimization_tol,
                self.max_iterations,
                self.rho,
                self.mu_coef,
                self.max_mu_coef,
            )
            # TODO: warn if not converged
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfield
            B = B * mean_S  # baseline
            D = D_R + D_Z  # darkfield
            weight = jnp.ones_like(images, dtype=np.float32) / (
                jnp.abs(I_R / S[jnp.newaxis, ...]) + self.epsilon
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
