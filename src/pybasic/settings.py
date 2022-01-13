"""Settings dataclass for BaSiC."""

from dataclasses import dataclass

# NOTE convert to enum
ESTIMATION_MODES = ["l0"]


@dataclass
class Settings:
    """A class to hold BaSiC settings.

    Args:
        darkfield: whether to estimate a darkfield correction
        epsilon
        estimation_mode
        lambda_darkfield
        lambda_flatfield
        max_iterations: maximum number of iterations allowed in the optimization
        max_reweight_iterations
        optimization_tol: error tolerance in the optimization
        reweighting_tol
        timelapse: whether to estimate photobleaching effect
        varying_coeff
        working_size

    Todo:
        * Fill in parameters descriptions
    """

    darkfield: bool = False
    epsilon: float = 0.1
    estimation_mode: str = "l0"
    lambda_darkfield: float = 0
    lambda_flatfield: float = 0
    max_iterations: int = 500
    max_reweight_iterations: int = 10
    optimization_tol: float = 1e-6
    reweighting_tol: float = 1e-3
    timelapse: bool = False
    varying_coeff: bool = True
    working_size: int = 128

    def __post_init__(self) -> None:
        """Validate input.

        Raises:
            ValueError: if invalid estimation mode
        """
        if self.estimation_mode not in ESTIMATION_MODES:
            raise ValueError(
                f"Estimation mode '{self.estimation_mode}' is not valid. "
                f"Please select mode from {ESTIMATION_MODES}."
            )
