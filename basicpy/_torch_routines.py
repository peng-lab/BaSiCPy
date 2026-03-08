import torch.nn as nn
import torch
import numpy as np
import torch_dct as dct
from typing import Tuple
import math


def _tshrinkage(x, thresh):
    return torch.sign(x) * torch.clip(torch.abs(x) - thresh, 0)


class BaseFit(nn.Module):
    def __init__(
        self,
        epsilon: float = 0.1,
        max_mu: float = 0,
        init_mu: float = 0,
        D_Z_max: float = 0,
        image_norm: float = 0,
        rho: float = 1.5,
        optimization_tol: float = 1e-6,
        optimization_tol_diff: float = 1e-6,
        smoothness_darkfield: float = 0.0,
        sparse_cost_darkfield: float = 0.0,
        smoothness_flatfield: float = 0.0,
        get_darkfield: bool = False,
        max_iterations: int = 500,
    ):
        super(BaseFit, self).__init__()
        """
        epsilon: Weight regularization term
        max_mu: The maximum value of mu
        init_mu: Initial value for mu
        D_Z_max: Maximum value for D_Z
        image_norm: The 2nd order norm for the images
        rho: Parameter rho for mu update
        optimization_tol: Optimization tolerance
        optimization_tol_diff: Optimization tolerance for update diff
        smoothness_darkfield: Darkfield smoothness weight for sparse reguralization
        sparse_cost_darkfield: Darkfield sparseness weight for sparse reguralization
        smoothness_flatfield: Flatfield smoothness weight for sparse reguralization
        get_darkfield: When True, will estimate the darkfield shading component
        max_iterations: Maximum number of iterations for single optimization
        """
        # kwargs = locals()
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.epsilon = epsilon
        self.max_mu = max_mu
        self.init_mu = init_mu
        self.D_Z_max = D_Z_max
        self.image_norm = image_norm
        self.rho = rho
        self.optimization_tol = optimization_tol
        self.optimization_tol_diff = optimization_tol_diff
        self.smoothness_darkfield = smoothness_darkfield
        self.sparse_cost_darkfield = sparse_cost_darkfield
        self.smoothness_flatfield = smoothness_flatfield
        self.get_darkfield = get_darkfield
        self.max_iterations = max_iterations

    def _cond(self, vals):
        k = vals[0]

        fit_residual = vals[-2]
        norm_ratio = torch.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return (norm_ratio > self.optimization_tol) * (k < self.max_iterations)

    def fit(
        self,
        Im,
        W,
        W_D,
        S,
        S_hat,
        D_R,
        D_Z,
        B,
        I_B,
        I_R,
    ):
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D_R.shape != Im.shape[1:]:
            raise ValueError("D_R must have the same shape as images.shape[1:]")
        if not np.isscalar(D_Z):
            raise ValueError("D_Z must be a scalar.")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        if W_D.shape != Im.shape[1:]:
            raise ValueError(
                "darkfield weight must have the same shape as images.shape[1:]"
            )

        # initialize values
        Y = torch.zeros_like(Im)
        mu = self.init_mu
        fit_residual = torch.ones_like(Im) * torch.inf
        value_diff = torch.inf

        vals = [0, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff]

        self.register_buffer("Im", Im)
        self.register_buffer("W", W)
        self.register_buffer("W_D", W_D)

        while self._cond(vals):
            vals = self._step(vals)

        k, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = torch.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return S, S_hat, D_R, D_Z, I_B, I_R, B, norm_ratio, k < self.max_iterations

    def fit_baseline(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, bool]:
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D.shape != Im.shape[1:]:
            raise ValueError("D must have the same shape as images.shape[1:]")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")

        # initialize values
        Y = torch.zeros_like(Im)
        mu = self.init_mu
        fit_residual = torch.ones_like(Im) * torch.inf
        value_diff = torch.inf

        vals = [
            0,
            I_R,
            B,
            Y,
            mu,
            fit_residual,
            value_diff,
        ]

        self.register_buffer("Im", Im)
        self.register_buffer("W", W)
        self.register_buffer("S", S)
        self.register_buffer("D", D)

        while self._cond(vals):
            vals = self._step_only_baseline(vals)

        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        norm_ratio = torch.linalg.norm(fit_residual.ravel(), ord=2) / self.image_norm
        return I_R, B, norm_ratio, k < self.max_iterations


class ApproximateFit(BaseFit, nn.Module):
    def __init__(self, **kwargs):
        super(ApproximateFit, self).__init__(**kwargs)
        self._ent1 = 1.0
        self._ent2 = 10.0

    def _step(
        self,
        vals,
    ):
        # approximate fitting only accepts two-dimensional images.

        k, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, _ = vals

        s_s, _, s_m, s_n = self.Im.shape
        Im = self.Im[:, 0, ...].reshape(s_s, -1)
        weight = self.W[:, 0, ...].reshape(s_s, -1)

        D_R = D_R[0, ...]
        I_R = I_R[:, 0, ...].reshape(s_s, -1)
        Y = Y[:, 0, ...].reshape(s_s, -1)

        S = dct.idct_2d(S_hat, norm="ortho")
        I_B = S * B[:, None, None] + D_R[None, ...]
        I_B = I_B.reshape(s_s, -1)
        temp_W = (Im - I_R - I_B + Y / mu) / self._ent1
        temp_W = torch.mean(temp_W, dim=0)
        S_hat = S_hat + dct.dct_2d(temp_W.reshape(s_m, s_n), norm="ortho")
        S_hat = _tshrinkage(S_hat, self.smoothness_flatfield / (self._ent1 * mu))
        S = dct.idct_2d(S_hat, norm="ortho")
        I_B = S[None, ...] * B[:, None, None] + D_R[None, ...]
        I_B = I_B.reshape(s_s, -1)
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _tshrinkage(I_R, weight / (self._ent1 * mu))
        R = Im - I_R
        B = torch.mean(R, dim=1) / torch.mean(R)
        # B = torch.mean(R, dim=1) - torch.mean(D_R)
        B = torch.clip(B, 0)
        I_B = S[None, ...] * B[:, None, None] + D_R[None, ...]
        I_B = I_B.reshape(s_s, -1)

        if self.get_darkfield:
            validA1coeff_idx = B < 1
            S_inmask = S.reshape(-1) >= torch.mean(S)
            S_outmask = S.reshape(-1) < torch.mean(S)
            R_0 = torch.where(
                S_inmask[None, ...] * validA1coeff_idx[..., None],
                R,
                torch.nan,
            )
            R_1 = torch.where(
                S_outmask[None, ...] * validA1coeff_idx[..., None],
                R,
                torch.nan,
            )
            B1_coeff = (torch.nanmean(R_0, 1) - torch.nanmean(R_1, 1)) / (
                torch.mean(R) + 1e-6
            )
            k = validA1coeff_idx.sum()
            B_nan = torch.where(validA1coeff_idx, B, torch.nan)

            temp1 = torch.nansum(B_nan**2)
            temp2 = torch.nansum(B_nan)
            temp3 = torch.nansum(B1_coeff)
            temp4 = torch.nansum(B_nan * B1_coeff)
            temp1 = torch.nan_to_num(temp1)
            temp2 = torch.nan_to_num(temp2)
            temp3 = torch.nan_to_num(temp3)
            temp4 = torch.nan_to_num(temp4)
            temp5 = temp2 * temp3 - k * temp4

            if temp5 == 0:
                D_Z = 0
            else:
                D_Z = (temp1 * temp3 - temp2 * temp4) / temp5
            D_Z = max(D_Z, 0)
            D_Z = min(D_Z, self.D_Z_max / torch.mean(S))

            Z = D_Z * torch.mean(S) - D_Z * S.reshape(-1)

            R_nan = torch.where(validA1coeff_idx[:, None], R, torch.nan)
            A1_offset = (
                torch.nanmean(R_nan, 0)
                - torch.nanmean(B_nan[..., None]) * S.reshape(-1)[None, ...]
            )
            A1_offset = A1_offset - torch.nanmean(A1_offset)

            D_R = A1_offset - torch.mean(A1_offset) - Z
            D_R = dct.dct_2d(D_R.reshape(s_m, s_n), norm="ortho")
            D_R = _tshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = dct.idct_2d(D_R, norm="ortho")
            D_R = _tshrinkage(D_R, self.smoothness_darkfield / (self._ent2 * mu))
            D_R = D_R + Z.reshape(s_m, s_n)

        fit_residual = Im - I_B - I_R
        Y = Y + mu * fit_residual
        mu = torch.minimum(mu * self.rho, self.max_mu)

        # put the variables back to 4-dim input array
        D_R = D_R[None, ...]
        I_R = I_R[:, None, ...]
        I_B = I_B[:, None, ...]
        Y = Y[:, None, ...]
        fit_residual = fit_residual[:, None, ...]

        return (
            k + 1,
            S,
            S_hat,
            D_R,
            D_Z,
            I_B.reshape(s_s, 1, s_m, s_n),
            I_R.reshape(s_s, 1, s_m, s_n),
            B,
            Y.reshape(s_s, 1, s_m, s_n),
            mu,
            fit_residual.reshape(s_s, 1, s_m, s_n),
            0.0,
        )

    def _step_only_baseline(
        self,
        vals,
    ):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        Im = self.Im[:, 0, ...]
        weight = self.W[:, 0, ...]
        S = self.S[0]
        D = self.D[0]
        I_R = I_R[:, 0, ...]
        Y = Y[:, 0, ...]

        I_B = S[None, ...] * B[:, None, None] + D[None, ...]

        # update I_R using approximated l0 norm
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _tshrinkage(I_R, weight / (self._ent1 * mu))

        R1 = Im - I_R
        B = torch.mean(R1, dim=(1, 2)) - torch.mean(D)
        B = torch.clip(B, 0, None)

        # eps = 1e-8
        # num = (weight * S * (R1 - D)).sum(dim=(1, 2))
        # den = (weight * (S**2)).sum(dim=(1, 2)).clamp_min(eps)
        # B = num / den
        # B = torch.clamp(B, min=0)

        fit_residual = Im - I_B - I_R
        Y = Y + mu * fit_residual
        mu = torch.minimum(mu * self.rho, self.max_mu)

        I_R = I_R[:, None, ...]
        Y = Y[:, None, ...]
        fit_residual = fit_residual[:, None, ...]
        return [k + 1, I_R, B, Y, mu, fit_residual, 0.0]

    def calc_weights(
        self,
        I_B,
        I_R,
        Ws2,
        epsilon,
    ):
        I_B = I_B[:, 0, ...]
        I_R = I_R[:, 0, ...]
        XE_norm = I_R / (torch.mean(I_B, dim=(1, 2))[:, None, None] + 1e-6)
        weight = torch.ones_like(I_R) / (torch.abs(XE_norm) + self.epsilon)
        weight[Ws2[:, 0, ...] == 0] *= epsilon
        weight = weight * weight.numel() / weight.sum()

        return weight[:, None, ...]

    def calc_dark_weights(self, D_R):
        return torch.ones_like(D_R)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R

    def calc_weights_baseline(
        self,
        B,
        I_R,
        Ws2,
        epsilon,
    ):
        I_R = I_R[:, 0, ...]
        XE_norm = I_R / B[:, None, None]
        weight = 1.0 / (torch.abs(XE_norm) + 0.1)
        weight[Ws2[:, 0, ...] == 0] *= epsilon
        weight = weight / torch.mean(weight)
        return weight[:, None, ...]


class LadmapFit(BaseFit, nn.Module):
    def __init__(self, **kwargs):
        super(LadmapFit, self).__init__(**kwargs)

    def _step(
        self,
        vals,
    ):
        k, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = self.Im.shape[0]

        I_B = S[None, ...] * B[:, None, None, None] + D_R[None, ...] + D_Z
        eta_S = torch.sum(B**2) * 1.02 + 0.01
        S_new = (
            S
            + torch.sum(B[:, None, None, None] * (self.Im - I_B - I_R + Y / mu), dim=0)
            / eta_S
        )
        S_new = dct.idct_3d(
            _tshrinkage(
                dct.dct_3d(S_new, norm="ortho"),
                self.smoothness_flatfield / (eta_S * mu),
            ),
            norm="ortho",
        )
        if S_new.min() < 0:
            S_new = S_new - S_new.min()
        dS = S_new - S
        S = S_new

        I_B = S[None, ...] * B[:, None, None, None] + D_R[None, ...] + D_Z
        I_R_new = _tshrinkage(self.Im - I_B + Y / mu, self.W / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = self.Im - I_R
        S_sq = torch.sum(S**2)
        B_new = torch.sum(S[None, ...] * (R + Y / mu), dim=(1, 2, 3)) / S_sq
        if S_sq <= 0:
            B_new = B
        B_new = torch.clip(B_new, 0)

        mean_B = torch.mean(B_new)
        if mean_B > 0:
            B_new = B_new / mean_B
            S = S * mean_B

        dB = B_new - B
        B = B_new

        BS = S[None, ...] * B[:, None, None, None]

        if self.get_darkfield:
            D_Z_new = torch.mean(self.Im - BS - D_R[None, ...] - I_R + Y / 2.0 / mu)
            D_Z_new = torch.clip(D_Z_new, 0, self.D_Z_max)
            dD_Z = D_Z_new - D_Z
            D_Z = D_Z_new

            eta_D = self.Im.shape[0] * 1.02
            D_R_new = D_R + 1.0 / eta_D * torch.sum(
                self.Im - BS - D_R[None, ...] - D_Z - I_R + Y / mu, dim=0
            )
            D_R_new = dct.idct_3d(
                _tshrinkage(dct.dct_3d(D_R_new), self.smoothness_darkfield / eta_D / mu)
            )
            D_R_new = _tshrinkage(
                D_R_new, self.sparse_cost_darkfield * self.W_D / eta_D / mu
            )
            dD_R = D_R_new - D_R
            D_R = D_R_new

        I_B = BS + D_R[None, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = max(
            [
                torch.linalg.norm(dS.ravel()) * torch.sqrt(eta_S),
                torch.linalg.norm(dI_R.ravel()) * math.sqrt(1.0),
                torch.linalg.norm(dB.ravel()),
            ]
        )

        if self.get_darkfield:
            value_diff = max(
                [
                    value_diff,
                    torch.linalg.norm(dD_R.ravel()) * math.sqrt(eta_D),
                    dD_Z**2,
                ]
            )

        value_diff = value_diff / self.image_norm
        mu = torch.minimum(mu * self.rho, self.max_mu)

        return [k + 1, S, S_hat, D_R, D_Z, I_B, I_R, B, Y, mu, fit_residual, value_diff]

    def _step_only_baseline(self, vals):
        k, I_R, B, Y, mu, fit_residual, value_diff = vals
        T_max = self.Im.shape[0]

        I_B = self.S[None, ...] * B[:, None, None, None] + self.D[None, ...]
        I_R_new = _tshrinkage(self.Im - I_B + Y / mu, self.W / mu / T_max)
        dI_R = I_R_new - I_R
        I_R = I_R_new

        R = self.Im - I_R
        B_new = torch.sum(self.S[None, ...] * (R + Y / mu), dim=(1, 2, 3)) / torch.sum(
            self.S**2
        )
        B_new = torch.clip(B_new, 0)
        dB = B_new - B
        B = B_new

        I_B = self.S[None, ...] * B[:, None, None, None] + self.D[None, ...]
        fit_residual = R - I_B
        Y = Y + mu * fit_residual

        value_diff = max(
            [
                torch.linalg.norm(dI_R.ravel()) * math.sqrt(1.0),
                torch.linalg.norm(dB.ravel()),
            ]
        )
        value_diff = value_diff / self.image_norm

        mu = torch.minimum(mu * self.rho, self.max_mu)

        return (k + 1, I_R, B, Y, mu, fit_residual, value_diff)

    def calc_weights(
        self,
        I_B,
        I_R,
        Ws2,
        epsilon,
    ):
        Ws = torch.ones_like(I_R) / (
            torch.abs(I_R / (I_B + self.epsilon)) + self.epsilon
        )
        return Ws / torch.mean(Ws)

    def calc_dark_weights(self, D_R):
        Ws = torch.ones_like(D_R) / (torch.abs(D_R) + self.epsilon)
        return Ws / torch.mean(Ws)

    def calc_weights_baseline(
        self,
        I_B,
        I_R,
        Ws2,
        epsilon,
    ):
        return self.calc_weights(
            I_B,
            I_R,
            Ws2,
            epsilon,
        )

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z
