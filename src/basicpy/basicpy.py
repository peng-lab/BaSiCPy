"""Main BaSiC class.

Todo:
    Keep examples up to date with changing API.
"""

# Core modules
from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from multiprocessing import cpu_count
from typing import Dict, Iterable, Optional, Tuple, Union

import jax.numpy as jnp

# 3rd party modules
import numpy as np
from jax import device_put
from jax.image import ResizeMethod, resize
from pydantic import BaseModel, Field, PrivateAttr
from skimage.transform import resize as _resize

from basicpy._jax_routines import ApproximateFit, LadmapFit
from basicpy.tools.dct2d_tools import JaxDCT

# Package modules
from basicpy.types import ArrayLike, PathLike

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

# initialize logger with the package name
logger = logging.getLogger(__name__)


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

    baseline: Optional[np.ndarray] = Field(
        None,
        description="Holds the baseline for the shading model.",
        exclude=True,  # Don't dump to output json/yaml
    )
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
    max_reweight_iterations_baseline: int = Field(
        5,
        description="Maximum number of reweighting iterations for baseline.",
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
    optimization_tol_diff: float = Field(
        1e-3,
        description="Optimization tolerance for update diff.",
    )
    resize_method: ResizeMethod = Field(
        ResizeMethod.CUBIC,
        description="Resize method to use when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Wheather or not to sort the intensities of the image.",
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
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)

    _settings_fname = "settings.json"
    _profiles_fname = "profiles.npy"

    class Config:

        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """Initialize BaSiC with the provided settings."""

        log_str = f"Initializing BaSiC {id(self)} with parameters: \n"
        for k, v in kwargs.items():
            log_str += f"{k}: {v}\n"
        logger.info(log_str)

        super().__init__(**kwargs)

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

    def fit(
        self, images: np.ndarray, fitting_weight: Optional[np.ndarray] = None
    ) -> None:
        """
        Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional array with dimension of (T,Y,X).
            fitting_weight: relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy.tools import load_images
            >>> images = load_images('./images')
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()
        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        Im = device_put(images).astype(jnp.float32)
        Im = self._resize(Im)

        if fitting_weight is not None:
            Ws = device_put(fitting_weight).astype(jnp.float32)
            Ws = self._resize(Ws)
            # normalize relative weight to 0 to 1
            Ws_min = jnp.min(Ws)
            Ws_max = jnp.max(Ws)
            Ws = (Ws - Ws_min) / (Ws_max - Ws_min)
        else:
            Ws = jnp.ones_like(Im)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = jnp.argsort(Im, axis=0)
            Im2 = jnp.take_along_axis(Im, inds, axis=0)
            Ws2 = jnp.take_along_axis(Ws, inds, axis=0)
        else:
            Im2 = Im
            Ws2 = Ws

        mean_image = jnp.mean(Im2, axis=2)
        mean_image = mean_image / jnp.mean(Im2)
        mean_image_dct = dct2d(mean_image.T)
        lambda_flatfield = jnp.sum(jnp.abs(mean_image_dct)) * self.lambda_flatfield_coef

        # spectral_norm = jnp.linalg.norm(Im.reshape((Im.shape[0], -1)), ord=2)
        if self.fitting_mode == FittingMode.ladmap:
            spectral_norm = jnp.linalg.norm(Im2.reshape((Im2.shape[0], -1)), ord=2)
        else:
            _temp = jnp.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
            spectral_norm = _temp[1][0]

        init_mu = self.mu_coef / spectral_norm
        fit_params = self.dict()
        fit_params.update(
            dict(
                lambda_flatfield=lambda_flatfield,
                lambda_darkfield=lambda_flatfield * self.lambda_darkfield_coef,
                # matrix 2-norm (largest sing. value)
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=jnp.min(Im2),
                image_norm=jnp.linalg.norm(Im2.flatten(), ord=2),
            )
        )

        # Initialize variables
        W = jnp.ones_like(Im2, dtype=jnp.float32)
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == FittingMode.ladmap:
            fitting_step = LadmapFit(**fit_params)
        else:
            fitting_step = ApproximateFit(**fit_params)

        for i in range(self.max_reweight_iterations):
            logger.info(f"reweighting iteration {i}")
            S = jnp.ones(Im2.shape[1:], dtype=jnp.float32)
            D_R = jnp.zeros(Im2.shape[1:], dtype=jnp.float32)
            D_Z = 0.0
            B = jnp.mean(Im2, axis=(1, 2))
            I_R = jnp.zeros(Im2.shape, dtype=jnp.float32)
            S, D_R, D_Z, I_R, B, norm_ratio, converged = fitting_step.fit(
                Im2,
                W,
                S,
                D_R,
                D_Z,
                B,
                I_R,
            )
            logger.info(f"single-step optimization score: {norm_ratio}.")
            logger.info(f"mean of S: {float(jnp.mean(S))}.")
            self._score = norm_ratio
            if not converged:
                logger.warning("single-step optimization did not converge.")
            self._S = S
            self._D_R = D_R
            self._B = B
            self._D_Z = D_Z
            D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
            mean_S = jnp.mean(S)
            S = S / mean_S  # flatfields
            B = B * mean_S  # baseline
            I_B = B[:, newax, newax] * S[newax, ...] + D[newax, ...]
            W = fitting_step.calc_weights(I_B, I_R) * Ws2

            self._weight = W
            self._residual = I_R

            logger.info(f"Iteration {i} finished.")
            if last_S is not None:
                mad_flatfield = jnp.sum(jnp.abs(S - last_S)) / jnp.sum(np.abs(last_S))
                if self.get_darkfield:
                    mad_darkfield = jnp.sum(jnp.abs(D - last_D)) / max(
                        jnp.sum(jnp.abs(last_D)), 1
                    )  # assumes the amplitude of darkfield is more than 1
                    self._reweight_score = max(mad_flatfield, mad_darkfield)
                else:
                    self._reweight_score = mad_flatfield
                logger.info(f"reweighting score: {self._reweight_score}")
                logger.info(f"elapsed time: {time.monotonic() - start_time} seconds")

                if self._reweight_score <= self.reweighting_tol:
                    logger.info("Reweighting converged.")
                    break
            if i == self.max_reweight_iterations - 1:
                logger.warning("Reweighting did not converge.")
            last_S = S
            last_D = D

        assert S is not None
        assert D is not None
        assert B is not None

        if self.sort_intensity:
            for i in range(self.max_reweight_iterations_baseline):
                B = jnp.ones(Im.shape[0], dtype=jnp.float32)
                if self.fitting_mode == FittingMode.approximate:
                    B = jnp.mean(Im, axis=(1, 2))
                I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
                logger.info(f"reweighting iteration for baseline {i}")
                I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
                    Im,
                    W,
                    S,
                    D,
                    B,
                    I_R,
                )
                I_B = B[:, newax, newax] * S[newax, ...] + D[newax, ...]
                W = fitting_step.calc_weights_baseline(I_B, I_R) * Ws
                self._weight = W
                self._residual = I_R
                logger.info(f"Iteration {i} finished.")

        self.flatfield = S
        self.darkfield = D
        self.baseline = B
        logger.info(
            f"=== BaSiC fit finished in {time.monotonic()-start_time} seconds ==="
        )

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

        logger.info("=== BaSiC transform started ===")
        start_time = time.monotonic()

        # Convert to the correct format
        im_float = images.astype(np.float32)

        # Rescale the flatfield and darkfield
        if not np.array_equal(self.flatfield.shape, im_float.shape[1:]):
            self._flatfield = _resize(self.flatfield, images.shape[1:])
            self._darkfield = _resize(self.darkfield, images.shape[1:])
        else:
            self._flatfield = self.flatfield
            self._darkfield = self.darkfield

        # Initialize the output
        output = np.empty(images.shape, dtype=images.dtype)

        if timelapse:
            # calculate timelapse from input series
            ...

        def unshade(ins, outs, i, dark, flat):
            outs[i] = (ins[i] - dark) / flat

        logger.info(f"unshading in {self.max_workers} threads")
        # If one or fewer workers, don't user ThreadPool. Useful for debugging.
        if self.max_workers <= 1:
            for i in range(images.shape[0]):
                unshade(im_float, output, i, self._darkfield, self._flatfield)

        else:
            with ThreadPoolExecutor(self.max_workers) as executor:
                threads = executor.map(
                    lambda x: unshade(
                        im_float, output, x, self._darkfield, self._flatfield
                    ),
                    range(images.shape[0]),
                )

                # Get the result of each thread, this should catch thread errors
                for thread in threads:
                    assert thread is None

        logger.info(
            f"=== BaSiC transform finished in {time.monotonic()-start_time} seconds ==="
        )
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

    def save_model(self, model_dir: PathLike, overwrite: bool = False) -> None:
        """Save current model to folder.

        Args:
            model_dir: path to model directory

        Raises:
            FileExistsError: if model directory already exists"""
        path = Path(model_dir)

        try:
            path.mkdir()
        except FileExistsError:
            if not overwrite:
                raise FileExistsError("Model folder already exists.")

        # save settings
        with open(path / self._settings_fname, "w") as fp:
            # see pydantic docs for output options
            fp.write(self.json())

        # NOTE emit warning if profiles are all zeros? fit probably not run
        # save profiles
        profiles = np.dstack((self.flatfield, self.darkfield))
        np.save(path / self._profiles_fname, profiles)

    @classmethod
    def load_model(cls, model_dir: PathLike) -> BaSiC:
        """Create a new instance from a model folder."""
        path = Path(model_dir)

        if not path.exists():
            raise FileNotFoundError("Model directory not found.")

        with open(path / cls._settings_fname) as fp:
            model = json.load(fp)

        profiles = np.load(path / cls._profiles_fname)
        model["flatfield"] = profiles[..., 0]
        model["darkfield"] = profiles[..., 1]

        return BaSiC(**model)
