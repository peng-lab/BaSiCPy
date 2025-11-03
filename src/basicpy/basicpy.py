"""Main BaSiC class."""

from pydantic import BaseModel, Field, PrivateAttr, model_validator
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import logging
import dask.array as da
import torch

from skimage.transform import resize as skimage_resize
import torch.nn.functional as F
import time
import torch_dct as dct
import copy

from basicpy._torch_routines import ApproximateFit, LadmapFit
from basicpy.metrics import autotune_cost
from basicpy.metrics_numpy import autotune_cost_numpy
from basicpy.utils import maybe_tqdm, make_overlap_chunks
import tqdm

try:
    from hyperactive import Hyperactive
    from hyperactive.optimizers import HillClimbingOptimizer
except ImportError:
    Hyperactive = None
    HillClimbingOptimizer = None

from pathlib import Path
import json
import math
import gc
from skimage.filters import threshold_otsu


# initialize logger with the package name
logger = logging.getLogger(__name__)

_SETTINGS_FNAME = "settings.json"
_PROFILES_FNAME = "profiles.npz"


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
    fitting_mode: str = Field(
        "approximate", description="Must be one of ['ladmap', 'approximate']"
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
    smoothness_flatfield: float = Field(
        None, description="Weight of the flatfield term in the Lagrangian."
    )
    smoothness_darkfield: float = Field(
        None, description="Weight of the darkfield term in the Lagrangian."
    )
    sparse_cost_darkfield: float = Field(
        0.01, description="Weight of the darkfield sparse term in the Lagrangian."
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
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    mu_coef: float = Field(12.5, description="Coefficient for initial mu value.")
    max_mu_coef: float = Field(
        1e7, description="Maximum allowed value of mu, divided by the initial value."
    )
    optimization_tol: float = Field(
        1e-3,
        description="Optimization tolerance.",
    )
    optimization_tol_diff: float = Field(
        1e-2,
        description="Optimization tolerance for update diff.",
    )
    resize_params: Dict = Field(
        {},
        description="Parameters for the resize function when downsampling images.",
    )
    reweighting_tol: float = Field(
        1e-2,
        description="Reweighting tolerance in mean absolute difference of images.",
    )
    sort_intensity: bool = Field(
        False,
        description="Whether or not to sort the intensities of the image.",
    )
    working_size: Optional[Union[int, List[int]]] = Field(
        128,
        description="Size for running computations. None means no rescaling.",
    )
    device: str = Field(
        "none",
        description="Must be one of ['cpu', 'cuda', 'none']",
    )

    # Private attributes for internal processing
    _score: float = PrivateAttr(None)
    _reweight_score: float = PrivateAttr(None)
    _weight: float = PrivateAttr(None)
    _weight_dark: float = PrivateAttr(None)
    _residual: float = PrivateAttr(None)
    _S: float = PrivateAttr(None)
    _B: float = PrivateAttr(None)
    _D_R: float = PrivateAttr(None)
    _D_Z: float = PrivateAttr(None)
    _smoothness_flatfield: float = PrivateAttr(None)
    _smoothness_darkfield: float = PrivateAttr(None)
    _sparse_cost_darkfield: float = PrivateAttr(None)
    _flatfield_small: float = PrivateAttr(None)
    _darkfield_small: float = PrivateAttr(None)
    _converge_flag: bool = PrivateAttr(None)

    class Config:
        """Pydantic class configuration."""

        arbitrary_types_allowed = True
        extra = "forbid"

    @model_validator(mode="before")
    def debug_log_values(cls, values: Dict[str, Any]):
        """Use a validator to echo input values."""
        logger.debug("Initializing BaSiC with parameters:")
        for k, v in values.items():
            logger.debug(f"{k}: {v}")
        return values

    def __call__(
        self,
        images: Union[np.ndarray, da.core.Array, torch.Tensor],
        fitting_weight: Optional[Union[np.ndarray, torch.Tensor, da.core.Array]] = None,
        skip_shape_warning=False,
        is_timelapse: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Shortcut for `BaSiC.fit_transform`."""
        out = self.fit_transform(
            images,
            fitting_weight,
            skip_shape_warning,
            is_timelapse,
        )
        gc.collect()
        for _ in range(10):
            torch.cuda.empty_cache()
        return out

    def _resize(
        self,
        Im,
        target_shape,
        method="bilinear",
    ):
        if isinstance(Im, da.core.Array):
            assert np.array_equal(target_shape[:-2], Im.shape[:-2])
            Im2 = (
                da.from_array(
                    [
                        skimage_resize(
                            np.array(Im[tuple(inds)]).astype(np.float32),
                            target_shape[-2:],
                            preserve_range=True,
                            **self.resize_params,
                        )
                        for inds in np.ndindex(Im.shape[:-2])
                    ]
                )
                .reshape((*Im.shape[:-2], *target_shape[-2:]))
                .compute()
            )
        elif isinstance(Im, torch.Tensor):
            if Im.is_cuda:
                Im2 = F.interpolate(
                    Im.float(),
                    target_shape[-2:],
                    mode=method,
                    align_corners=True if method != "nearest" else None,
                    antialias=True if method != "nearest" else False,
                )
            else:
                Im2 = torch.empty(target_shape, dtype=torch.float32, device=self.device)
                for i in range(Im.shape[0]):
                    Im2[i] = F.interpolate(
                        Im[i : i + 1].float().to(self.device),
                        target_shape[-2:],
                        mode=method,
                        align_corners=True if method != "nearest" else None,
                        antialias=True if method != "nearest" else False,
                    )
        elif isinstance(Im, np.ndarray):
            Im2 = torch.empty(target_shape, dtype=torch.float32, device=self.device)
            for i in range(Im.shape[0]):
                Im2[i] = F.interpolate(
                    torch.from_numpy(Im[i : i + 1].astype(np.float32)).to(self.device),
                    target_shape[-2:],
                    mode=method,
                    align_corners=True if method != "nearest" else None,
                    antialias=True if method != "nearest" else False,
                )
        else:
            raise ValueError(
                "Input must be either numpy.ndarray, dask.core.Array, or torch.Tensor."
            )
        return Im2

    def _resize_to_working_size(
        self,
        Im,
        method="bilinear",
    ):
        """Resize the images to the working size."""
        if np.isscalar(self.working_size):
            working_shape = [self.working_size] * (Im.ndim - 2)
        else:
            if not Im.ndim - 2 == len(self.working_size):
                raise ValueError(
                    "working_size must be a scalar or match the image dimensions"
                )
            else:
                working_shape = self.working_size
        target_shape = [*Im.shape[:2], *working_shape]
        Im = self._resize(Im, target_shape, method)

        return Im

    def fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
        for_autotune=False,
    ) -> None:
        self._fit(images, fitting_weight, skip_shape_warning, for_autotune)
        gc.collect()
        for _ in range(10):
            torch.cuda.empty_cache()

    def _fit(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning=False,
        for_autotune=False,
    ) -> None:
        """Generate illumination correction profiles from images.

        Args:
            images: Input images to fit shading model.
                    Must be 3-dimensional or 4-dimensional array
                    with dimension of (T,Y,X) or (T,Z,Y,X).
                    T can be either of time or mosaic position.
                    Multichannel images should be
                    independently corrected for each channel.
            fitting_weight: Relative fitting weight for each pixel.
                    Higher value means more contribution to fitting.
                    Must has the same shape as images.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.

        Example:
            >>> from basicpy import BaSiC
            >>> from basicpy import datasets as bdata
            >>> images = bdata.wsi_brain()
            >>> basic = BaSiC()  # use default settings
            >>> basic.fit(images)

        """
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, None, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, None, ...]
        elif images.ndim == 4:
            if self.fitting_mode == "approximate":
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )

        if images.shape[-1] < 10 and not skip_shape_warning:
            logger.warning(
                "Image last dimension is less than 10. "
                + "Are you supplying images with the channel dimension?"
                + "Multichannel images should be "
                + "independently corrected for each channel."
            )

        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        logger.info("=== BaSiC fit started ===")
        start_time = time.monotonic()

        if isinstance(images, torch.Tensor):
            if images.is_cuda:
                self.device = "cuda"

        if self.device == "none":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        Im = self._resize_to_working_size(images)

        if isinstance(Im, np.ndarray):
            Im = torch.from_numpy(Im.astype(np.float32)).to(self.device)
        else:
            Im = Im.to(self.device)

        if fitting_weight is not None:
            flag_segmentation = True
            Ws = self._resize_to_working_size(fitting_weight, "nearest") > 0
            if isinstance(Ws, np.ndarray):
                Ws = torch.from_numpy(Ws).to(self.device)
            else:
                Ws = Ws.to(self.device)
        else:
            flag_segmentation = False
            Ws = torch.ones_like(Im)

        # Im2 and Ws2 will possibly be sorted
        if self.sort_intensity:
            inds = torch.argsort(Im, dim=0)
            Im2 = torch.take_along_dim(Im, inds, dim=0)
            Ws2 = torch.take_along_dim(Ws, inds, dim=0)
        else:
            Im2 = Im
            Ws2 = Ws

        if self.smoothness_flatfield is None:
            meanD = Im.mean(0)
            meanD = meanD / meanD.mean()
            W_meanD = dct.dct_2d(meanD, norm="ortho")
            self._smoothness_flatfield = torch.sum(torch.abs(W_meanD)) / (400) * 0.5
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
        if self.smoothness_darkfield is None:
            self._smoothness_darkfield = self._smoothness_flatfield * 0.1
        else:
            self._smoothness_darkfield = self.smoothness_darkfield
        if self.sparse_cost_darkfield is None:
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        logger.debug(f"_smoothness_flatfield set to {self._smoothness_flatfield}")
        logger.debug(f"_smoothness_darkfield set to {self._smoothness_darkfield}")
        logger.debug(f"_sparse_cost_darkfield set to {self._sparse_cost_darkfield}")

        _temp = torch.linalg.svd(Im2.reshape((Im2.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp[1][0]

        if self.fitting_mode == "approximate":
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im2.shape)
        fit_params = {}
        fit_params.update(
            dict(
                epsilon=self.epsilon,
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                rho=self.rho,
                optimization_tol=self.optimization_tol,
                optimization_tol_diff=self.optimization_tol_diff,
                get_darkfield=self.get_darkfield,
                max_iterations=self.max_iterations,
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=torch.min(Im2),
                image_norm=torch.linalg.norm(Im2),
            )
        )

        # Initialize variables
        W = torch.ones_like(Im2, dtype=torch.float32) * Ws2
        if flag_segmentation:
            W[Ws2 == 0] = self.epsilon
        W = W * W.numel() / W.sum()
        W_D = torch.zeros(
            Im2.shape[1:],
            dtype=torch.float32,
            device=self.device,
        )
        last_S = None
        last_D = None
        S = None
        D = None
        B = None

        if self.fitting_mode == "ladmap":
            fit_params.update(
                dict(
                    sparse_cost_darkfield=self.sparse_cost_darkfield,
                )
            )
            fitting_step = LadmapFit(**fit_params).to(self.device)
        else:
            fitting_step = ApproximateFit(**fit_params).to(self.device)

        self._converge_flag = True

        for i in range(self.max_reweight_iterations):
            logger.debug(f"reweighting iteration {i}")
            if self.fitting_mode == "approximate":
                S = torch.ones(Im2.shape[1:], dtype=torch.float32, device=self.device)
            else:
                S = torch.median(Im2, dim=0)[0]
            S_hat = dct.dct_2d(S, norm="ortho")
            D_R = torch.zeros(Im2.shape[1:], dtype=torch.float32, device=self.device)
            D_Z = 0.0
            if self.fitting_mode == "approximate":
                B = copy.deepcopy(Im2)
                B[Ws2 == 0] = torch.nan
                B = torch.squeeze(torch.nanmean(B, dim=(-2, -1))) / torch.nanmean(B)
                B = torch.nan_to_num(B)
            else:
                B = torch.ones(Im2.shape[0], dtype=torch.float32, device=self.device)

            I_R = torch.zeros(Im2.shape, dtype=torch.float32, device=self.device)
            I_B = (S * B[:, None, None])[:, None, ...] + D_R[None, ...]

            S, S_hat, D_R, D_Z, I_B, I_R, B, norm_ratio, converged = fitting_step.fit(
                Im2,
                W,
                W_D,
                S,
                S_hat,
                D_R,
                D_Z,
                B,
                I_B,
                I_R,
            )

            D_R = D_R + D_Z * S
            S = I_B.mean(dim=0) - D_R
            mean_S = torch.mean(S)
            S = S / torch.mean(S)  # flatfields
            logger.debug(f"single-step optimization score: {norm_ratio}.")
            logger.debug(f"mean of S: {float(torch.mean(S))}.")
            self._score = norm_ratio
            if not converged:
                logger.debug("single-step optimization did not converge.")
            if S.max() == 0:
                logger.error(
                    "Estimated flatfield is zero. "
                    + "Please try to decrease smoothness_darkfield."
                )
                raise RuntimeError(
                    "Estimated flatfield is zero. "
                    + "Please try to decrease smoothness_darkfield."
                )
            self._S = S
            self._D_R = D_R
            self._B = B
            self._D_Z = D_Z

            D = fitting_step.calc_darkfield(S, D_R, D_Z)  # darkfield
            W = fitting_step.calc_weights(
                I_B,
                I_R,
                Ws2,
                self.epsilon,
            )
            W_D = fitting_step.calc_dark_weights(D_R)

            self._W = W.cpu().data.numpy()

            self._weight = W
            self._weight_dark = W_D
            self._residual = I_R

            logger.debug(f"Iteration {i} finished.")

            if last_S is not None:
                mad_flatfield = torch.sum(torch.abs(S - last_S)) / torch.sum(
                    torch.abs(last_S)
                )
                temp_diff = torch.sum(torch.abs(S - last_S))
                if temp_diff < 1e-7:
                    mad_darkfield = 0
                else:
                    mad_darkfield = temp_diff / (
                        max(torch.sum(torch.abs(last_S)), 1e-6)
                    )
                self._reweight_score = max(mad_flatfield, mad_darkfield)
                logger.debug(f"reweighting score: {self._reweight_score}")
                logger.info(
                    f"Iteration {i} elapsed time: "
                    + f"{time.monotonic() - start_time} seconds"
                )

                if self._reweight_score <= self.reweighting_tol:
                    logger.info("Reweighting converged.")
                    break
            last_S = S
            last_D = D

        if (i == self.max_reweight_iterations - 1) and (not converged):
            self._converge_flag = False
            if not for_autotune:
                logger.warning("Reweighting did not converge.")

        assert S is not None
        assert D is not None
        assert B is not None

        # if self.sort_intensity:
        # for i in range(self.max_reweight_iterations_baseline):
        #     B = torch.ones(Im.shape[0], dtype=torch.float32, device=self.device)
        #     if self.fitting_mode == "approximate":
        #         B = torch.mean(Im, dim=(1, 2, 3))
        #     I_R = torch.zeros(Im.shape, dtype=torch.float32, device=self.device)
        #     logger.debug(f"reweighting iteration for baseline {i}")
        #     I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
        #         Im,
        #         W,
        #         S,
        #         D,
        #         B,
        #         I_R,
        #     )

        #     I_B = B[:, None, None, None] * S[None, ...] + D[None, ...]
        #     W = fitting_step.calc_weights_baseline(I_B, I_R, Ws, self.epsilon) * Ws
        #     self._weight = W
        #     self._residual = I_R
        #     logger.debug(f"Iteration {i} finished.")

        self._flatfield_small = S
        self._darkfield_small = D

        self.flatfield = F.interpolate(
            S[None],
            images.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )[0]
        self.darkfield = F.interpolate(
            D[None],
            images.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )[0]
        if ndim == 3:
            self.flatfield = self.flatfield[0]
            self.darkfield = self.darkfield[0]
            self._flatfield_small = self._flatfield_small[0]
            self._darkfield_small = self._darkfield_small[0]
        self.baseline = B * mean_S

        self.flatfield = self.flatfield.cpu().numpy()
        self.darkfield = self.darkfield.cpu().numpy()
        self.baseline = self.baseline.cpu().numpy()

        logger.info(
            f"=== BaSiC fit finished in {time.monotonic()-start_time} seconds ==="
        )

    def fit_only_baseline(
        self,
        images,
        fitting_weight,
        S,
        D,
    ):
        ndim = images.ndim
        if images.ndim == 3:
            images = images[:, None, ...]
            if fitting_weight is not None:
                fitting_weight = fitting_weight[:, None, ...]
        elif images.ndim == 4:
            if self.fitting_mode == "approximate":
                raise ValueError(
                    "Only 2-dimensional images are accepted for the approximate mode."
                )
        else:
            raise ValueError(
                "Images must be 3 or 4-dimensional array, "
                + "with dimension of (T,Y,X) or (T,Z,Y,X)."
            )
        if fitting_weight is not None and fitting_weight.shape != images.shape:
            raise ValueError("fitting_weight must have the same shape as images.")

        if S.ndim == 3:
            S = S[None]
            D = D[None]
        elif S.ndim == 2:
            S = S[None, None]
            D = D[None, None]
        else:
            raise ValueError("S and D must be 2D or 3D")

        if isinstance(images, torch.Tensor):
            if images.is_cuda:
                self.device = "cuda"

        if self.device == "none":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Im = self._resize_to_working_size(images)
        if isinstance(Im, np.ndarray):
            Im = torch.from_numpy(Im.astype(np.float32)).to(self.device)
        else:
            Im = Im.to(self.device)

        S = self._resize_to_working_size(
            torch.from_numpy(S.astype(np.float32)).to(self.device)
        )[
            0,
        ]
        D = self._resize_to_working_size(
            torch.from_numpy(D.astype(np.float32)).to(self.device)
        )[
            0,
        ]

        if fitting_weight is not None:
            flag_segmentation = True
            Ws = self._resize_to_working_size(fitting_weight, "nearest") > 0
            if isinstance(Ws, np.ndarray):
                Ws = torch.from_numpy(Ws).to(self.device)
            else:
                Ws = Ws.to(self.device)
        else:
            flag_segmentation = True
            Ws = torch.ones_like(Im)
            Ws = Ws * (
                Im
                < torch.quantile(Im.reshape(Im.shape[0], -1), 0.1, dim=-1)[
                    :,
                    None,
                    None,
                    None,
                ]
            )
        # np.save("Im.npy", Im.squeeze().cpu().data.numpy())
        # Im = torch.from_numpy(DD[:, None]).cuda()

        if self.smoothness_flatfield is None:
            meanD = Im.mean(0)
            meanD = meanD / meanD.mean()
            W_meanD = dct.dct_2d(meanD, norm="ortho")
            self._smoothness_flatfield = torch.sum(torch.abs(W_meanD)) / (400) * 0.5
        else:
            self._smoothness_flatfield = self.smoothness_flatfield
        if self.smoothness_darkfield is None:
            self._smoothness_darkfield = self._smoothness_flatfield * 0.1
        else:
            self._smoothness_darkfield = self.smoothness_darkfield
        if self.sparse_cost_darkfield is None:
            self._sparse_cost_darkfield = (
                self._smoothness_darkfield * self.sparse_cost_darkfield * 100
            )
        else:
            self._sparse_cost_darkfield = self.sparse_cost_darkfield

        logger.debug(f"_smoothness_flatfield set to {self._smoothness_flatfield}")
        logger.debug(f"_smoothness_darkfield set to {self._smoothness_darkfield}")
        logger.debug(f"_sparse_cost_darkfield set to {self._sparse_cost_darkfield}")

        _temp = torch.linalg.svd(Im.reshape((Im.shape[0], -1)), full_matrices=False)
        spectral_norm = _temp[1][0]

        if self.fitting_mode == "approximate":
            init_mu = self.mu_coef / spectral_norm
        else:
            init_mu = self.mu_coef / spectral_norm / np.prod(Im.shape)
        fit_params = {}
        fit_params.update(
            dict(
                epsilon=self.epsilon,
                smoothness_flatfield=self._smoothness_flatfield,
                smoothness_darkfield=self._smoothness_darkfield,
                sparse_cost_darkfield=self._sparse_cost_darkfield,
                rho=self.rho,
                optimization_tol=self.optimization_tol,
                optimization_tol_diff=self.optimization_tol_diff,
                get_darkfield=self.get_darkfield,
                max_iterations=self.max_iterations,
                init_mu=init_mu,
                max_mu=init_mu * self.max_mu_coef,
                D_Z_max=torch.min(Im),
                image_norm=torch.linalg.norm(Im.ravel(), ord=2),
            )
        )

        # Initialize variables
        W = torch.ones_like(Im, dtype=torch.float32) * Ws
        if flag_segmentation:
            W[Ws == 0] = self.epsilon
        W_D = torch.zeros(
            Im.shape[1:],
            dtype=torch.float32,
            device=self.device,
        )

        if self.fitting_mode == "ladmap":
            fit_params.update(
                dict(
                    sparse_cost_darkfield=self.sparse_cost_darkfield,
                )
            )
            fitting_step = LadmapFit(**fit_params).to(self.device)
        else:
            fitting_step = ApproximateFit(**fit_params).to(self.device)

        for i in range(
            self.max_reweight_iterations_baseline
        ):  # self.max_reweight_iterations_baseline

            if self.fitting_mode == "approximate":
                B = copy.deepcopy(Im)
                B[Ws == 0] = torch.nan
                B = torch.squeeze(torch.nanmean(B, dim=(-2, -1)))
                B = torch.nan_to_num(B)
            else:
                B = torch.ones(Im.shape[0], dtype=torch.float32, device=self.device)

            I_R = torch.zeros(Im.shape, dtype=torch.float32, device=self.device)
            logger.debug(f"reweighting iteration for baseline {i}")
            I_R, B, norm_ratio, converged = fitting_step.fit_baseline(
                Im,
                W,
                S,
                D,
                B,
                I_R,
            )

            W = fitting_step.calc_weights_baseline(B, I_R, Ws, self.epsilon) * Ws
            self._weight = W
            self._residual = I_R
            logger.debug(f"Iteration {i} finished.")

        return B

    def transform(
        self,
        images: Union[np.ndarray, torch.Tensor, da.core.Array],
        fitting_weight=None,
        is_timelapse: Union[bool, str] = False,
        frames: Optional[Sequence[Union[int, np.int_]]] = None,
        use_tqdm: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        out = self._transform(
            images,
            fitting_weight,
            is_timelapse,
            frames,
            use_tqdm=use_tqdm,
        )
        gc.collect()
        for _ in range(10):
            torch.cuda.empty_cache()
        return out

    def _transform(
        self,
        images: Union[np.ndarray, torch.Tensor, da.core.Array],
        fitting_weight,
        is_timelapse: Union[bool, str] = False,
        frames: Optional[Sequence[Union[int, np.int_]]] = None,
        use_tqdm: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Apply profile to images.

        Args:
            images: input images to correct. See `fit`.
            is_timelapse: If `True`, corrects the timelapse/photobleaching offsets,
                       assuming that the residual is the product of flatfield and
                       the object fluorescence. Also accepts "multiplicative"
                       (the same as `True`) or "additive" (residual is the object
                       fluorescence).
            frames: Frames to use for transformation. Defaults to None (all frames).

        Returns:
            corrected images

        Example:
            >>> basic.fit(images)
            >>> corrected = basic.transform(images)
        """
        # if self.baseline is None:
        #     raise RuntimeError("BaSiC object is not initialized")
        logger.info("=== BaSiC transform started ===")
        start_time = time.monotonic()

        s = 100

        chunk_inds = make_overlap_chunks(images.shape[0], s, overlap=1)

        if isinstance(images, torch.Tensor):
            output = torch.zeros(
                images.shape,
                device=images.device,
                dtype=torch.float32,
            )
        elif isinstance(images, np.ndarray):
            output = np.zeros(images.shape, dtype=np.float32)
        elif isinstance(images, da.core.Array):
            output = []
        else:
            raise ValueError(
                "Input must be either numpy.ndarray, dask.core.Array, or torch.Tensor."
            )

        baseline_previous = None
        for i, images_inds in enumerate(
            maybe_tqdm(
                chunk_inds,
                use_tqdm=use_tqdm,
                desc="Transforming: ",
                leave=False,
            )
        ):
            images_chunk = images[min(images_inds) : max(images_inds) + 1]
            if fitting_weight is not None:
                fitting_weight_chunk = fitting_weight[
                    min(images_inds) : max(images_inds) + 1,
                ]
            else:
                fitting_weight_chunk = None
            # Convert to the correct format
            if isinstance(images, torch.Tensor):
                flatfield = torch.from_numpy(self.flatfield).to(images.device)
                darkfield = torch.from_numpy(self.darkfield).to(images.device)
                im_float = images_chunk.to(torch.float)
            elif isinstance(images, np.ndarray):
                flatfield = self.flatfield
                darkfield = self.darkfield
                im_float = images_chunk.astype(np.float32)
            elif isinstance(images, da.core.Array):
                flatfield = self.flatfield
                darkfield = self.darkfield
                im_float = images_chunk.compute().astype(np.float32)
            else:
                raise ValueError(
                    "Input must be either numpy.ndarray, dask.core.Array, or torch.Tensor."
                )

            if is_timelapse:
                baseline = self.fit_only_baseline(
                    im_float,
                    fitting_weight_chunk,
                    self.flatfield,
                    self.darkfield,
                )
                if isinstance(im_float, torch.Tensor):
                    baseline = baseline.to(im_float.device)
                else:
                    baseline = baseline.cpu().data.numpy()

                if baseline_previous is not None:
                    baseline = baseline - baseline[0] + baseline_previous[-1]
                else:
                    pass

                output_chunks = (im_float - darkfield[None]) / flatfield[
                    None
                ] - baseline[:, None, None]

                baseline_previous = baseline

            else:
                output_chunks = (im_float - darkfield[None]) / flatfield[None]
            if isinstance(images, da.core.Array):
                if i != len(chunk_inds) - 1:
                    output.append(output_chunks[:-1])
                else:
                    output.append(output_chunks)
            else:
                output[min(images_inds) : max(images_inds) + 1] = output_chunks
        if isinstance(output, list):
            output = da.concatenate(output, axis=0)

        logger.info(
            f"=== BaSiC transform finished in {time.monotonic()-start_time} seconds ==="
        )

        return output

    def fit_transform(
        self,
        images: Union[np.ndarray, da.core.Array, torch.Tensor],
        fitting_weight: Optional[Union[np.ndarray, torch.Tensor, da.core.Array]] = None,
        skip_shape_warning=False,
        is_timelapse: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Fit and transform on data.

        Args:
            images: input images to fit and correct. See `fit`.

        Returns:
            corrected images

        Example:
            >>> corrected = basic.fit_transform(images)
        """
        self.fit(
            images,
            fitting_weight=fitting_weight,
            skip_shape_warning=skip_shape_warning,
        )
        corrected = self.transform(
            images,
            fitting_weight,
            is_timelapse,
            use_tqdm=False,
        )

        gc.collect()
        for _ in range(10):
            torch.cuda.empty_cache()

        return corrected

    def autotune(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning: bool = False,
        search_space_flatfield=None,
        init_params=None,
        is_timelapse: bool = False,
        histogram_qmin: float = 0.01,
        histogram_qmax: float = 0.99,
        vmin_factor: float = 0.6,
        vrange_factor: float = 1.5,
        histogram_bins: int = 1000,
        histogram_use_fitting_weight: bool = True,
        fourier_l0_norm_image_threshold: float = 0.1,
        fourier_l0_norm_fourier_radius=10,
        fourier_l0_norm_threshold=0.0,
        fourier_l0_norm_cost_coef=30,
    ) -> None:
        self._autotune(
            np.asarray(images) if isinstance(images, da.core.Array) else images,
            fitting_weight,
            skip_shape_warning,
            search_space_flatfield,
            init_params,
            is_timelapse,
            histogram_qmin,
            histogram_qmax,
            vmin_factor,
            vrange_factor,
            histogram_bins,
            histogram_use_fitting_weight,
            fourier_l0_norm_image_threshold,
            fourier_l0_norm_fourier_radius,
            fourier_l0_norm_threshold,
            fourier_l0_norm_cost_coef,
        )
        gc.collect()
        for _ in range(10):
            torch.cuda.empty_cache()

    def _autotune(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning: bool = False,
        search_space_flatfield=None,
        init_params=None,
        is_timelapse: bool = False,
        histogram_qmin: float = 0.01,
        histogram_qmax: float = 0.99,
        vmin_factor: float = 0.6,
        vrange_factor: float = 1.5,
        histogram_bins: int = 1000,
        histogram_use_fitting_weight: bool = True,
        fourier_l0_norm_image_threshold: float = 0.1,
        fourier_l0_norm_fourier_radius=10,
        fourier_l0_norm_threshold=0.0,
        fourier_l0_norm_cost_coef=30,
    ) -> None:
        """Automatically tune the parameters of the model.

        Args:
            images: input images to fit and correct. See `fit`.
            fitting_weight: Relative fitting weight for each pixel. See `fit`.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.
            optimizer: optimizer to use. Defaults to
                    `hyperactive.optimizers.HillClimbingOptimizer`.
            n_iter: number of iterations for the optimizer. Defaults to 100.
            search_space: search space for the optimizer.
                    Defaults to a reasonable range for each parameter.
            init_params: initial parameters for the optimizer.
                    Defaults to a reasonable initial value for each parameter.
            is_timelapse: if True, corrects the timelapse/photobleaching offsets.
            histogram_qmin: the minimum quantile to use for the histogram.
                    Defaults to 0.01.
            histogram_qmax: the maximum quantile to use for the histogram.
                    Defaults to 0.99.
            histogram_bins: the number of bins to use for the histogram.
                    Defaults to 100.
            hisogram_use_fitting_weight: if True, uses the weight for the histogram.
                    Defaults to True.
            fourier_l0_norm_image_threshold : float
                The threshold for image values for the fourier L0 norm calculation.
            fourier_l0_norm_fourier_radius : float
                The Fourier radius for the fourier L0 norm calculation.
            fourier_l0_norm_threshold : float
                The maximum preferred value for the fourier L0 norm.
            fourier_l0_norm_cost_coef : float
                The cost coefficient for the fourier L0 norm.
            early_stop: if True, stops the optimization when the change in
                    entropy is less than `early_stop_torelance`.
                    Defaults to True.
            early_stop_n_iter_no_change: the number of iterations for early
                    stopping. Defaults to 10.
            early_stop_torelance: the absolute value torelance
                    for early stopping.
            random_state: random state for the optimizer.

        """
        # if is_timelapse:
        #     if histogram_qmax == 0.99:
        #         images_mask = images < threshold_otsu(images[:, ::3, ::3])
        #         histogram_qmax = images_mask.sum() / (
        #             images.size if isinstance(images, np.ndarray) else images.numel()
        #         )
        #         print(histogram_qmax)

        if self.fitting_mode == "ladmap":
            print(
                "Autotune is not applicable to LADMAP mode, please try autotune_hillclimbing instead."
            )
            return

        flatfield_pool = np.array(
            [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7, 8, 10]
        )
        flatfield_pool_coarse = np.array([0.01, 0.1, 0.5, 2, 8, 10])
        second_flag = True
        if search_space_flatfield is None:
            pass
        else:
            search_space_flatfield = np.asarray(search_space_flatfield)
            if min(search_space_flatfield) <= 0.01:
                a = 0
            else:
                a = np.where(flatfield_pool < min(search_space_flatfield))[0][-1]
            if max(search_space_flatfield) >= 10:
                b = None
            else:
                b = np.where(flatfield_pool > max(search_space_flatfield))[0][0] + 1
            flatfield_pool = flatfield_pool[a:b]
            flatfield_pool_coarse = flatfield_pool_coarse[
                (flatfield_pool_coarse >= flatfield_pool.min())
                * (flatfield_pool_coarse <= flatfield_pool.max())
            ]
            flatfield_pool_coarse = np.concatenate(
                (
                    np.array([flatfield_pool[0]]),
                    flatfield_pool_coarse,
                    np.array([flatfield_pool[-1]]),
                )
            )
            flatfield_pool_coarse = np.unique(flatfield_pool_coarse)
        if search_space_flatfield is not None:
            search_space_flatfield_outlier = search_space_flatfield[
                (search_space_flatfield < 0.01) + (search_space_flatfield > 10)
            ]
        else:
            search_space_flatfield_outlier = np.array([])
        if len(search_space_flatfield_outlier) == 0:
            pass
        else:
            flatfield_pool_coarse = np.concatenate(
                (flatfield_pool_coarse, search_space_flatfield_outlier)
            )
            flatfield_pool = np.concatenate(
                (flatfield_pool, search_space_flatfield_outlier)
            )
            flatfield_pool_coarse = np.sort(flatfield_pool_coarse)
            flatfield_pool = np.sort(flatfield_pool)

        if init_params is None:
            init_params = {
                "smoothness_flatfield": sum(flatfield_pool) / len(flatfield_pool),
                "get_darkfield": self.get_darkfield,
            }
            if self.get_darkfield:
                init_params.update(
                    {
                        "smoothness_darkfield": init_params["smoothness_flatfield"]
                        * 0.1,
                        "sparse_cost_darkfield": 1e-3,
                    }
                )
        basic = self.model_copy(update=init_params)

        images = images[:: max(images.shape[0] // 50, 1), ::]
        if fitting_weight is not None:
            fitting_weight = fitting_weight[
                :: max(fitting_weight.shape[0] // 50, 1), ::
            ]

        if isinstance(images, torch.Tensor):
            images = images.to(torch.float)
        else:
            images = torch.from_numpy(images.astype(np.float32))
        r = images[0].numel() / (1024 * 1024)
        if basic.device == "none":
            basic.device = "cuda" if torch.cuda.is_available() else "cpu"
        if r > 1:
            images = basic._resize(
                images[:, None],
                (
                    images.shape[0],
                    1,
                    int(images.shape[1] / r),
                    int(images.shape[2] / r),
                ),
            )[:, 0]
            if fitting_weight is not None:
                fitting_weight = basic._resize(
                    fitting_weight[:, None],
                    (
                        fitting_weight.shape[0],
                        1,
                        int(fitting_weight.shape[1] / r),
                        int(fitting_weight.shape[2] / r),
                    ),
                )[:, 0]

        device_available = 1 if torch.cuda.is_available() else 0
        if device_available:
            free, _ = torch.cuda.mem_get_info()
            if (images.numel() * 4) / free < 0.1:
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = "cpu"

        images = images.to(device).to(torch.float)
        size_r = images.numel() / 2**24

        if size_r < 1:
            size_r = 1
        else:
            size_r = math.ceil(size_r)
        # images_numpy = images.cpu().data.numpy()
        if fitting_weight is None:
            pass
        else:
            if isinstance(fitting_weight, torch.Tensor):
                pass
            else:
                fitting_weight = torch.from_numpy(fitting_weight)
            fitting_weight = fitting_weight.to(device)

        basic.fit(
            images,
            fitting_weight=fitting_weight,
            skip_shape_warning=skip_shape_warning,
            for_autotune=True,
        )

        transformed = basic.transform(
            images,
            fitting_weight=fitting_weight,
            is_timelapse=is_timelapse,
            use_tqdm=False,
        )
        # vmin, vmax = np.percentile(
        #     transformed.cpu().data.numpy(), [histogram_qmin, histogram_qmax]
        # )

        # transformed_sorted, _ = torch.sort(transformed.flatten())
        # vmin = transformed_sorted[int(transformed.numel()*histogram_qmin/100)]
        # vmax = transformed_sorted[int(transformed.numel()*histogram_qmax/100)]

        vmin, vmax = torch.quantile(
            transformed.flatten()[::size_r],
            torch.tensor([histogram_qmin / 1, histogram_qmax / 1])
            .to(torch.float)
            .to(device),
        )

        # vmin, vmax = np.quantile(
        #     transformed.cpu().data.numpy(),
        #     [histogram_qmin / 1, histogram_qmax / 1],
        # )

        val_range = (
            vmax - vmin * vmin_factor
        ) * vrange_factor  # fix the value range for histogram

        if fitting_weight is None or not histogram_use_fitting_weight:
            weights = None
        else:
            weights = fitting_weight.to(images.dtype)

        def fit_and_calc_entropy(params):
            # try:
            basic = self.model_copy(update=params)
            basic.fit(
                images,
                fitting_weight=fitting_weight,
                skip_shape_warning=skip_shape_warning,
                for_autotune=True,
            )

            transformed = basic.transform(
                images,
                fitting_weight=fitting_weight,
                is_timelapse=is_timelapse,
                use_tqdm=False,
            )
            if torch.isnan(transformed).sum():
                return np.inf

            vmin_new = (
                torch.quantile(transformed.flatten()[::size_r], histogram_qmin / 1)
                * vmin_factor
            )

            # vmin_new = np.quantile(
            #     transformed.cpu().data.numpy(),
            #     histogram_qmin,
            # )

            # transformed_sorted, _ = torch.sort(transformed.flatten())
            # vmin_new = transformed_sorted[int(transformed.numel()*histogram_qmin/100)] * vmin_factor

            if np.allclose(basic.flatfield, np.ones_like(basic.flatfield)):
                return np.inf  # discard the case where flatfield is all ones

            r = autotune_cost(
                transformed,
                basic._flatfield_small,
                entropy_vmin=vmin_new,
                entropy_vmax=vmin_new + val_range,
                histogram_bins=histogram_bins,
                fourier_l0_norm_cost_coef=fourier_l0_norm_cost_coef,
                fourier_l0_norm_image_threshold=fourier_l0_norm_image_threshold,
                fourier_l0_norm_fourier_radius=fourier_l0_norm_fourier_radius,
                fourier_l0_norm_threshold=fourier_l0_norm_threshold,
                weights=weights,
            )

            return r if basic._converge_flag else np.inf
            # except RuntimeError:
            #     return np.inf

        cost_coarse = []
        flatfield_coarse = []
        for i in tqdm.tqdm(flatfield_pool_coarse, desc="coarse-level search: "):
            params = {
                "smoothness_flatfield": i,
                "smoothness_darkfield": i * 0.1,  # 0.1 * i if self.get_darkfield else 0
                "get_darkfield": self.get_darkfield,
            }
            a = fit_and_calc_entropy(params)
            cost_coarse.append(a)
            flatfield_coarse.append(basic.flatfield)

        cost_coarse = torch.tensor(cost_coarse)
        flatfield_coarse = np.stack(flatfield_coarse, 0)

        best_ind = torch.argmin(cost_coarse)
        if best_ind == len(cost_coarse) - 1:
            second_best_ind = best_ind - 1
        elif best_ind == 0:
            second_best_ind = 1
        else:
            # if cost_coarse[best_ind - 1] < cost_coarse[best_ind + 1]:
            #     second_best_ind = best_ind - 1
            # else:
            #     second_best_ind = best_ind + 1
            second_best_ind = best_ind + 1
            best_ind = best_ind - 1
        best = flatfield_pool_coarse[best_ind]
        second_best = flatfield_pool_coarse[second_best_ind]

        if second_best < best:
            flatfield_pool_narrow = flatfield_pool[
                (flatfield_pool >= second_best) * (flatfield_pool <= best)
            ]
        else:
            flatfield_pool_narrow = flatfield_pool[
                (flatfield_pool >= best) * (flatfield_pool <= second_best)
            ]
        if len(flatfield_pool_narrow) == 2:
            flatfield_pool_narrow = (
                [flatfield_pool_narrow[0]]
                + [(flatfield_pool_narrow[0] + flatfield_pool_narrow[1]) / 2]
                + [flatfield_pool_narrow[1]]
            )

        cost_narrow = np.zeros(len(flatfield_pool_narrow))
        cost_narrow[0] = cost_coarse[flatfield_pool_narrow[0] == flatfield_pool_coarse][
            0
        ]
        cost_narrow[-1] = cost_coarse[
            flatfield_pool_narrow[-1] == flatfield_pool_coarse
        ][0]

        flatfield_narrow = np.zeros(
            (
                len(flatfield_pool_narrow),
                basic.flatfield.shape[-2],
                basic.flatfield.shape[-1],
            ),
            dtype=np.float32,
        )
        flatfield_narrow[0] = flatfield_coarse[
            flatfield_pool_narrow[0] == flatfield_pool_coarse
        ]
        flatfield_narrow[-1] = flatfield_coarse[
            flatfield_pool_narrow[-1] == flatfield_pool_coarse
        ]

        ind = 1
        for i in tqdm.tqdm(flatfield_pool_narrow[1:-1], desc="fine-level search: "):
            params = {
                "smoothness_flatfield": i,
                "smoothness_darkfield": i * 0.1,
                "get_darkfield": self.get_darkfield,
            }
            a = fit_and_calc_entropy(params)
            cost_narrow[ind] = a
            flatfield_narrow[ind] = basic.flatfield
            ind += 1

        self.__dict__.update(
            {
                "smoothness_flatfield": flatfield_pool_narrow[np.argmin(cost_narrow)],
                "smoothness_darkfield": flatfield_pool_narrow[np.argmin(cost_narrow)]
                * 0.1,
            }
        )

        self.flatfield = flatfield_narrow[0]

        if not self.get_darkfield:
            print("\nAutotune is done.")
            print("Best smoothness_flatfield = {}.".format(self.smoothness_flatfield))
        else:
            print("Autotune is done.")
            print("Best smoothness_flatfield = {}.".format(self.smoothness_flatfield))
            print("Best smoothness_darkfield = {}.".format(self.smoothness_darkfield))

    def autotune_hillclimbing(
        self,
        images: np.ndarray,
        fitting_weight: Optional[np.ndarray] = None,
        skip_shape_warning: bool = False,
        *,
        optmizer=None,
        n_iter=100,
        search_space=None,
        init_params=None,
        is_timelapse: bool = False,
        histogram_qmin: float = 0.01,
        histogram_qmax: float = 0.99,
        vmin_factor: float = 0.6,
        vrange_factor: float = 1.5,
        histogram_bins: int = 1000,
        histogram_use_fitting_weight: bool = True,
        fourier_l0_norm_image_threshold: float = 0.1,
        fourier_l0_norm_fourier_radius=10,
        fourier_l0_norm_threshold=0.0,
        fourier_l0_norm_cost_coef=30,
        early_stop: bool = True,
        early_stop_n_iter_no_change: int = 15,
        early_stop_torelance: float = 1e-6,
        random_state: Optional[int] = None,
    ) -> None:
        """Automatically tune the parameters of the model.

        Args:
            images: input images to fit and correct. See `fit`.
            fitting_weight: Relative fitting weight for each pixel. See `fit`.
            skip_shape_warning: if True, warning for last dimension
                    less than 10 is suppressed.
            optimizer: optimizer to use. Defaults to
                    `hyperactive.optimizers.HillClimbingOptimizer`.
            n_iter: number of iterations for the optimizer. Defaults to 100.
            search_space: search space for the optimizer.
                    Defaults to a reasonable range for each parameter.
            init_params: initial parameters for the optimizer.
                    Defaults to a reasonable initial value for each parameter.
            is_timelapse: if True, corrects the timelapse/photobleaching offsets.
            histogram_qmin: the minimum quantile to use for the histogram.
                    Defaults to 0.01.
            histogram_qmax: the maximum quantile to use for the histogram.
                    Defaults to 0.99.
            histogram_bins: the number of bins to use for the histogram.
                    Defaults to 100.
            hisogram_use_fitting_weight: if True, uses the weight for the histogram.
                    Defaults to True.
            fourier_l0_norm_image_threshold : float
                The threshold for image values for the fourier L0 norm calculation.
            fourier_l0_norm_fourier_radius : float
                The Fourier radius for the fourier L0 norm calculation.
            fourier_l0_norm_threshold : float
                The maximum preferred value for the fourier L0 norm.
            fourier_l0_norm_cost_coef : float
                The cost coefficient for the fourier L0 norm.
            early_stop: if True, stops the optimization when the change in
                    entropy is less than `early_stop_torelance`.
                    Defaults to True.
            early_stop_n_iter_no_change: the number of iterations for early
                    stopping. Defaults to 10.
            early_stop_torelance: the absolute value torelance
                    for early stopping.
            random_state: random state for the optimizer.

        """

        if search_space is None:
            search_space = {
                "smoothness_flatfield": list(np.logspace(-3, 1, 15)),
            }
            if self.get_darkfield:
                search_space.update(
                    {
                        "smoothness_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                        "sparse_cost_darkfield": [0] + list(np.logspace(-3, 1, 15)),
                    }
                )

        if init_params is None:
            init_params = {
                "smoothness_flatfield": 0.1,
            }
            if self.get_darkfield:
                init_params.update(
                    {
                        "smoothness_darkfield": 1e-3,
                        "sparse_cost_darkfield": 1e-3,
                    }
                )

        # calculate the histogram range
        device = images.device
        if isinstance(images, torch.Tensor):
            images_numpy = images.cpu().numpy()
        else:
            images_numpy = images
        init_params["fitting_mode"] = "approximate"
        basic = self.model_copy(update=init_params)
        basic.fit(
            images,
            fitting_weight=fitting_weight,
            skip_shape_warning=skip_shape_warning,
        )
        transformed = basic.transform(
            images_numpy,
            is_timelapse=is_timelapse,
            use_tqdm=False,
        )

        vmin, vmax = np.quantile(transformed, [histogram_qmin, histogram_qmax])

        val_range = (
            vmax - vmin * vmin_factor
        ) * vrange_factor  # fix the value range for histogram

        if fitting_weight is None or not histogram_use_fitting_weight:
            weights = None
        else:
            weights = fitting_weight

        init_params["fitting_mode"] = self.fitting_mode

        def fit_and_calc_entropy(params):
            try:
                basic = self.model_copy(update=params)
                basic.fit(
                    images,
                    fitting_weight=fitting_weight,
                    skip_shape_warning=skip_shape_warning,
                )
                transformed = basic.transform(
                    images_numpy,
                    is_timelapse=is_timelapse,
                    use_tqdm=False,
                )
                vmin_new = np.quantile(transformed, histogram_qmin) * vmin_factor

                if np.allclose(basic.flatfield, np.ones_like(basic.flatfield)):
                    return -np.inf  # discard the case where flatfield is all ones

                return -1.0 * autotune_cost_numpy(
                    transformed,
                    basic._flatfield_small.cpu().data.numpy(),
                    entropy_vmin=vmin_new,
                    entropy_vmax=vmin_new + val_range,
                    histogram_bins=histogram_bins,
                    fourier_l0_norm_cost_coef=fourier_l0_norm_cost_coef,
                    fourier_l0_norm_image_threshold=fourier_l0_norm_image_threshold,
                    fourier_l0_norm_fourier_radius=fourier_l0_norm_fourier_radius,
                    fourier_l0_norm_threshold=fourier_l0_norm_threshold,
                    weights=weights,
                )
            except RuntimeError:
                return -np.inf

        if optmizer is None:
            optimizer = HillClimbingOptimizer(
                epsilon=0.1,
                distribution="laplace",
                n_neighbours=4,
                rand_rest_p=0.1,
            )

        hyper = Hyperactive()

        params = dict(
            optimizer=optimizer,
            n_iter=n_iter,
            initialize=dict(warm_start=[init_params]),
            random_state=random_state,
        )

        if early_stop:
            params.update(
                dict(
                    early_stopping=dict(
                        n_iter_no_change=early_stop_n_iter_no_change,
                        tol_abs=early_stop_torelance,
                    )
                )
            )

        hyper.add_search(
            fit_and_calc_entropy,
            search_space,
            **params,
        )
        hyper.run()
        best_params = hyper.best_para(fit_and_calc_entropy)

        print("Autotune is done.")
        for key, value in best_params.items():
            print(f"Best {key}: {value}")

        self.__dict__.update(best_params)

    @property
    def score(self):
        """The BaSiC fit final score."""
        return self._score

    @property
    def reweight_score(self):
        """The BaSiC fit final reweighting score."""
        return self._reweight_score

    @property
    def settings(self) -> Dict:
        """Current settings.

        Returns:
            current settings
        """
        return self.model_dump()

    def save_model(
        self,
        model_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """Save current model to folder.

        Args:
            model_dir: path to model directory

        Raises:
            FileExistsError: if model directory already exists
        """
        path = Path(model_dir)

        try:
            path.mkdir()
        except FileExistsError:
            if not overwrite:
                raise FileExistsError("Model folder already exists.")

        # save settings
        with open(path / _SETTINGS_FNAME, "w") as fp:
            # see pydantic docs for output options
            fp.write(self.model_dump_json())

        # NOTE emit warning if profiles are all zeros? fit probably not run
        # save profiles
        np.savez(
            path / _PROFILES_FNAME,
            flatfield=np.array(self.flatfield),
            darkfield=np.array(self.darkfield),
            baseline=np.array(self.baseline),
        )

    @classmethod
    def load_model(
        cls,
        model_dir: Union[str, Path],
    ):
        """Create a new instance from a model folder."""
        path = Path(model_dir)

        if not path.exists():
            raise FileNotFoundError("Model directory not found.")

        with open(path / _SETTINGS_FNAME) as fp:
            model = json.load(fp)

        profiles = np.load(
            path / _PROFILES_FNAME,
            allow_pickle=True,
        )
        model["flatfield"] = profiles["flatfield"]
        model["darkfield"] = profiles["darkfield"]
        model["baseline"] = profiles["baseline"]

        return BaSiC(**model)
