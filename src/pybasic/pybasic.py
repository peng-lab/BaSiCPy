"""Contains the PyBaSiC class."""

import functools
import os
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

PathLike = Union[str, bytes, os.PathLike]
ESTIMATION_MODES = ["l0"]  # NOTE convert to enum?


@dataclass
class Settings:
    """A class to hold BaSiC settings.

    Args:
        lambda_flatfield
        estimation_mode
        max_iterations: maximum number of iterations allowed in the optimization
        darkfield: whether to estimate a darkfield correction
        optimization_tolerance: error tolerance in the optimization
        lambda_darkfield
        working_size
        max_reweight_iterations
        eplson
        varying_coeff
        reweight_tolerance

    Todo:
        * Fill in parameters descriptions
    """

    lambda_flatfield: float = 0
    estimation_mode: str = "l0"
    max_iterations: int = 500
    optimization_tolerance: float = 1e-6
    darkfield: bool = False
    lambda_darkfield: float = 0
    working_size: int = 0
    max_reweight_iterations: int = 10
    eplson: float = 0.1  # NOTE rename to epslon?
    varying_coeff: bool = True
    reweight_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        if self.estimation_mode not in ESTIMATION_MODES:
            raise ValueError(
                f"Estimation mode '{self.estimation_mode}' is not valid. "
                f"Please select mode from {ESTIMATION_MODES}."
            )


class Model:
    """A class to hold illumination profiles."""

    def __init__(
        self, profile: np.ndarray, type_: str = "flatfield", op: Callable = None
    ):
        """Init the Model class.

        Args:
            profile: illumination correction profile
            type_: profile type (options are flatfield, darkfield)
            op: operation to apply profile to input image

        Raises:
            ValueError: profile type cannot be identified
        """
        self.profile = profile
        self.type_ = type_

        if op is None:
            if self.type_ == "flatfield":
                self.op = functools.partial(np.multiply, self.profile)
            elif self.type_ == "darkfield":
                self.op = functools.partial(np.add, -self.profile)
            else:
                raise ValueError(
                    "Unidentified profile type. Cannot determine application operation."
                )
        else:
            self.op = op

    def apply(self, input: np.ndarray) -> np.ndarray:
        """Apply illumination profile to input image.

        Args:
            input: input image

        Returns:
            illumination corrected image
        """
        return self.op(input)


# NOTE how should multiple channels be handled?
# NOTE if settings need to be updated, should user reinitialize class?
class BaSiC:
    """A class for fitting and applying BaSiC correction models."""

    def __init__(
        self, settings: Settings, use_gpu: bool = False, use_tpu: bool = False
    ) -> None:
        """Inits the BaSiC class with the provided settings.

        Args:
            settings: :meth:`pybasic.Settings` object
            use_gpu: wheter to use gpu device
            use_tpu: wheter to use tpu device
        """
        self._settings = settings
        ...

    def fit(self, images: np.ndarray) -> List[Model]:
        """Generate illumination correction profiles.

        Args:
            images: input images to predict illumination model

        Returns:
            correction profile model :meth:`pybasic.Model` objects

        Todo:
            * encourage use of generator to provide images to reduce memory usage
        """
        ...
        return

    def predict(self, images: np.ndarray, models: List[np.ndarray]) -> np.ndarray:
        """Apply model.

        Args:
            images: input images to correct
            models: illumination correction profiles

        Returns:
            ndarray same size as input with illumination correction applied
        """
        ...
        return

    # NOTE: move to function outside of class for easier testing?
    def _inexact_alm_rspca_l1(self):
        ...

    # NOTE: move to function outside of class for easier testing?
    def _dct(self):
        if self._use_gpu:
            return self._dct_gpu()
        else:
            return self._dct_cpu()

    # NOTE: move to function outside of class for easier testing?
    def _idct(self):
        if self._use_gpu:
            return self._idct_gpu()
        else:
            return self._idct_cpu()

    # NOTE: move to function outside of class for easier testing?
    def _dct_gpu(self):
        ...

    # NOTE: move to function outside of class for easier testing?
    def _dct_cpu(self):
        ...

    # NOTE: move to function outside of class for easier testing?
    def _idct_gpu(self):
        ...

    # NOTE: move to function outside of class for easier testing?

    def _idct_cpu(self):
        ...

    def __repr__(self):
        """Return details of the BaSiC object."""
        return self._settings.__repr__()

    def __enter__(self):
        # for context management
        ...

    def __exit__(self):
        # for context management
        ...
