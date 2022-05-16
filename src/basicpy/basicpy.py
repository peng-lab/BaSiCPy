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
import time
import logging

# 3rd party modules
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from scipy.fftpack import dct
from skimage.transform import resize

# Package modules
from basicpy.tools import inexact_alm_rspca_l1
from basicpy.types import ArrayLike

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


class EstimationMode(Enum):

    l0: str = "l0"


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
    estimation_mode: EstimationMode = Field(
        "l0",
        description="Flatfield offset for weight updates.",
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
        description="Maximum number of iterations.",
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
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    reweighting_tol: float = Field(
        1e-3,
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
    _alm_settings = {
        "lambda_darkfield",
        "lambda_flatfield",
        "get_darkfield",
        "optimization_tol",
        "max_iterations",
    }

    class Config:

        arbitrary_types_allowed = True

    def __init__(self, **kwargs) -> None:
        """Initialize BaSiC with the provided settings."""

        log_str = f"Initializing BaSiC {id(self)} with parameters: \n"
        for k, v in kwargs.items():
            log_str += f"{k}: {v}\n"
        logger.info(log_str)

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

        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()
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

        D = np.sort(D, axis=2)

        X_A_offset = np.zeros(D.shape[:2])
        weight = np.ones(D.shape)

        # TODO: The original implementation includes a segmentation argument.
        # if segmentation is not None:
        #     segmentation = np.array(segmentation)
        #     segmentation = np.transpose(segmentation, (1, 2, 0))
        #     for i in range(weight.shape[2]):
        #         weight[segmentation] = 1e-6
        #     # weight[options.segmentation] = 1e-6

        flatfield_last = np.ones(D.shape[:2])
        darkfield_last = np.random.randn(*D.shape[:2])

        for reweighting_iter in range(self.max_reweight_iterations):
            logger.info(f"reweighting iteration {reweighting_iter}")
            reweighting_iter += 1

            # TODO: Included in the original code
            # if initial_flatfield:
            #     # TODO: implement inexact_alm_rspca_l1_intflat?
            #     raise IOError("Initial flatfield option not implemented yet!")
            # else:
            X_k_A, X_k_E, X_k_A_offset, self._score = inexact_alm_rspca_l1(
                D, weight=weight, **self.dict(include=self._alm_settings)
            )

            X_A = np.reshape(X_k_A, D.shape[:2] + (-1,), order="F")
            X_E = np.reshape(X_k_E, D.shape[:2] + (-1,), order="F")
            X_A_offset = np.reshape(X_k_A_offset, D.shape[:2], order="F")
            X_E_norm = X_E / np.mean(X_A, axis=(0, 1))

            # Update the weights:
            weight = np.ones_like(X_E_norm) / (np.abs(X_E_norm) + self.epsilon)

            # TODO: Included in the original code
            # if segmentation is not None:
            #     weight[segmentation] = 0

            weight = weight * weight.size / np.sum(weight)

            temp = np.mean(X_A, axis=2) - X_A_offset
            flatfield_current = temp / np.mean(temp)
            darkfield_current = X_A_offset
            mad_flatfield = np.sum(np.abs(flatfield_current - flatfield_last)) / np.sum(
                np.abs(flatfield_last)
            )
            temp_diff = np.sum(np.abs(darkfield_current - darkfield_last))
            if temp_diff < 1e-7:
                mad_darkfield = 0
            else:
                mad_darkfield = temp_diff / np.maximum(
                    np.sum(np.abs(darkfield_last)), 1e-6
                )
            flatfield_last = flatfield_current
            darkfield_last = darkfield_current
            self._reweight_score = np.maximum(mad_flatfield, mad_darkfield)
            logger.info(f"Iteration {reweighting_iter} finished.")
            logger.info(f"reweighting score: {self._reweight_score}")
            logger.info(f"elapsed time: {time.monotonic() - start_time} seconds")
            if self._reweight_score <= self.reweighting_tol:
                logger.info("Reweighting converged.")
                break
            if reweighting_iter == self.max_reweight_iterations - 1:
                logger.warning("Reweighting did not converge.")

        shading = np.mean(X_A, 2) - X_A_offset
        self.flatfield = shading / shading.mean()

        if self.get_darkfield:
            self.darkfield = X_A_offset

        self._darkfield = self.darkfield
        self._flatfield = self.flatfield
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

        logger.info(f"unshading in {self.max_workers} threads")
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
