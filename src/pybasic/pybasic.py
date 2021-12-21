"""Contains the PyBaSiC class."""

from dataclasses import dataclass

import numpy as np

ESTIMATION_MODES = ["l0"]  # NOTE convert to enum?


@dataclass
class Settings:
    """A class to hold BaSiC settings.

    Attributes:
        lambda_flatfield (float)
        estimation_mode (str)
        max_iterations (int): Maximum number of iterations allowed in the
            optimization.
        optimization_tolerance (float): Tolerance of error in the optimization.
        darkfield (bool): Whether to estimate a darkfield correction.
        lambda_darkfield (float)
        working_size (int)
        max_reweight_iterations (int)
        eplson (float)
        varying_coeff (bool)
        reweight_tolerance (float)
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
                f"Estimation mode '{self.estimation_mode}' not in {ESTIMATION_MODES}."
            )


# NOTE how should multiple channels be handled?
# NOTE if settings need to be updated, should user reinitialize class?
class BaSiC:
    """A class for fitting and applying BaSiC correction models."""

    _use_gpu: bool

    def __init__(self, settings: Settings) -> None:
        """Inits the BaSiC class with the provided settings."""
        self._settings = settings
        ...

    def fit(self):
        """Generate profiles."""
        ...

    def predict(self, image: np.ndarray):
        """Apply model."""
        ...

    def _dct(self):
        if self._use_gpu:
            return self._dct_gpu()
        else:
            return self._dct_cpu()

    def _idct(self):
        if self._use_gpu:
            return self._idct_gpu()
        else:
            return self._idct_cpu()

    def _dct_gpu(self):
        ...

    def _dct_cpu(self):
        ...

    def _idct_gpu(self):
        ...

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


class Model:
    # include input image, channel/scene, training date/time, training duration
    ...


class FlatField(Model):
    ...


class DarkField(Model):
    ...


class TimeLapse(Model):
    ...
