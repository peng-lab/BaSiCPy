from functools import partial
from typing import Tuple

import numpy as np
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from pydantic import BaseModel, Field, PrivateAttr

from basicpy.tools.dct2d_tools import JaxDCT

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d
newax = jnp.newaxis


@jit
def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


class BaseFit(BaseModel):
    epsilon: float = Field(
        0.1,
        description="Weight regularization term.",
    )
    max_mu: float = Field(0, description="The maximum value of mu.")
    init_mu: float = Field(0, description="Initial value for mu.")
    D_Z_max: float = Field(0, description="Maximum value for D_Z.")
    image_norm: float = Field(0, description="The 2nd order norm for the images.")
    rho: float = Field(1.5, description="Parameter rho for mu update.")
    optimization_tol: float = Field(
        1e-6,
        description="Optimization tolerance.",
    )
    lambda_darkfield: float = Field(
        0.0,
        description="Darkfield offset for weight updates.",
    )
    lambda_flatfield: float = Field(
        0.0,
        description="Flatfield offset for weight updates.",
    )
    get_darkfield: bool = Field(
        False,
        description="When True, will estimate the darkfield shading component.",
    )
    max_iterations: int = Field(
        500,
        description="Maximum number of iterations for single optimization.",
    )

    class Config:
        extra = "ignore"

    def _cond(self, vals):
        k = vals[0]
        fit_residual = vals[-1]
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return jnp.all(
            jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
        )

    @jit
    def _fit_jit(
        self,
        Im,
        W,
        S,
        D_R,
        D_Z,
        B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        Y_S = 0
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

        vals = (0, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual)
        step = partial(
            self._step,
            Im,
            W,
        )
        #        while self._cond(vals):
        #            vals = step(vals)
        vals = lax.while_loop(self._cond, step, vals)
        k, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations

    @jit
    def _fit_baseline_jit(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ):
        # initialize values
        Y = jnp.zeros_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

        vals = (0, I_R, B, Y, mu, fit_residual)
        step = partial(
            self._step_only_baseline,
            Im,
            W,
            S,
            D,
        )

        #        while self._cond(vals):
        #            vals = step(vals)
        vals = lax.while_loop(self._cond, step, vals)
        k, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return I_R, B, norm_ratio, k < self.max_iterations

    def fit(
        self,
        Im,
        W,
        S,
        D_R,
        D_Z,
        B,
        I_R,
    ):
        if S.shape != Im.shape[1:]:
            raise ValueError("S must have the same shape as images.shape[1:]")
        if D_R.shape != Im.shape[1:]:
            raise ValueError("D_R must have the same shape as images.shape[1:]")
        if not jnp.isscalar(D_Z):
            raise ValueError("D_Z must be a scalar.")
        if B.shape != Im.shape[:1]:
            raise ValueError("B must have the same shape as images.shape[:1]")
        if I_R.shape != Im.shape:
            raise ValueError("I_R must have the same shape as images.shape")
        if W.shape != Im.shape:
            raise ValueError("weight must have the same shape as images.shape")
        return self._fit_jit(Im, W, S, D_R, D_Z, B, I_R)

    def fit_baseline(
        self,
        Im,
        W,
        S,
        D,
        B,
        I_R,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool]:
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
        return self._fit_baseline_jit(Im, W, S, D, B, I_R)

    def tree_flatten(self):
        # all of the fields are treated as "static" values for JAX
        children = []
        aux_data = self.dict()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, _children):
        return cls(**aux_data)


@register_pytree_node_class
class LadmapFit(BaseFit):
    @jit
    def _step(
        self,
        Im,
        weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual = vals
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        eta = (jnp.sum(B**2) + 1) * 1.02
        S_incl1 = jnp.sum(B[:, newax, newax] * (Im - I_B - I_R + Y / mu), axis=0)
        S_incl2 = (jnp.mean(S) - 1 + Y_S / mu) / jnp.product(jnp.array(S.shape))
        S = S + (S_incl1 + S_incl2) / eta
        S = idct2d(_jshrinkage(dct2d(S), self.lambda_flatfield / (eta * mu)))

        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        I_R = _jshrinkage(Im - I_B + Y / mu, weight / mu)

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
        Y_S = Y_S + mu * (jnp.mean(S) - 1)
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual)

    @jit
    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual = vals
        I_B = S[newax, ...] * B[:, newax, newax] + D[newax, ...]
        I_R = _jshrinkage(Im - I_B + Y / mu, weight / mu)

        R = Im - I_R
        B = jnp.sum(S[newax, ...] * (R + Y / mu), axis=(1, 2)) / jnp.sum(S**2)
        B = jnp.maximum(B, 0)

        I_B = S[newax, ...] * B[:, newax, newax] + D[newax, ...]
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, I_R, B, Y, mu, fit_residual)

    def calc_weights(self, I_B, I_R):
        return jnp.ones_like(I_R, dtype=jnp.float32) / (
            jnp.abs(I_R / I_B) + self.epsilon
        )

    def calc_weights_baseline(self, I_B, I_R):
        return self.calc_weights(I_B, I_R)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z


@register_pytree_node_class
class ApproximateFit(BaseFit):
    _ent1: float = PrivateAttr(1.0)
    _ent2: float = PrivateAttr(10.0)

    @jit
    def _step(
        self,
        Im,
        weight,
        vals,
    ):
        k, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual = vals
        S_hat = dct2d(S)
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        temp_W = (Im - I_B - I_R + Y / mu) / self._ent1
        #    plt.imshow(temp_W[0]);plt.show()
        #    print(type(temp_W))
        temp_W = jnp.mean(temp_W, axis=0)
        S_hat = S_hat + dct2d(temp_W)
        S_hat = _jshrinkage(S_hat, self.lambda_flatfield / (self._ent1 * mu))
        S = idct2d(S_hat)
        I_B = S[newax, ...] * B[:, newax, newax] + D_R[newax, ...] + D_Z
        I_R = (Im - I_B + Y / mu) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))
        R = Im - I_R
        B = jnp.mean(R, axis=(1, 2)) / jnp.mean(R)
        B = jnp.maximum(B, 0)

        if self.get_darkfield:
            B_valid = B < 1

            S_inmask = S > jnp.mean(S) * (1 - 1e-6)
            S_outmask = S < jnp.mean(S) * (1 + 1e-6)
            A = (
                jnp.sum(R * S_inmask[newax, ...], axis=(1, 2))
                / jnp.sum(S_inmask * R.shape[0])
                - jnp.sum(R * S_outmask[newax, ...], axis=(1, 2))
                / jnp.sum(S_outmask * R.shape[0])
            ) / jnp.mean(R)
            A = jnp.where(jnp.isnan(A), 0, A)

            # temp1 = jnp.sum(p['A1_coeff'][validA1coeff_idx]**2)
            B_sq_sum = jnp.sum(B**2 * B_valid)
            B_sum = jnp.sum(B * B_valid)
            A_sum = jnp.sum(A * B_valid)
            BA_sum = jnp.sum(B * A * B_valid)
            denominator = B_sum * A_sum - BA_sum * jnp.sum(B_valid)
            # limit B1_offset: 0<B1_offset<B1_uplimit

            D_Z = jnp.clip(
                (B_sq_sum * A_sum - B_sum * BA_sum) / (denominator + 1e-6),
                0,
                self.D_Z_max / jnp.mean(S),
            )

            Z = D_Z * (np.mean(S) - S)

            D_R = (R * B_valid[:, newax, newax]).sum(axis=0) / B_valid.sum() - (
                B * B_valid
            ).sum() / B_valid.sum() * S
            D_R = D_R - jnp.mean(D_R) - Z

            # smooth A_offset
            D_R = dct2d(D_R)
            D_R = _jshrinkage(D_R, self.lambda_darkfield / (self._ent2 * mu))
            D_R = idct2d(D_R)
            D_R = _jshrinkage(D_R, self.lambda_darkfield / (self._ent2 * mu))
            D_R = D_R + Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, Y_S, mu, fit_residual)

    @jit
    def _step_only_baseline(self, Im, weight, S, D, vals):
        k, I_R, B, Y, mu, fit_residual = vals
        I_B = S[newax, ...] * B[:, newax, newax] + D[newax, ...]

        # update I_R using approximated l0 norm
        I_R = I_R + (Im - I_B - I_R + (1 / mu) * Y) / self._ent1
        I_R = _jshrinkage(I_R, weight / (self._ent1 * mu))

        R1 = Im - I_R
        # A1_coeff = mean(R1)-mean(A_offset);
        B = jnp.mean(R1, axis=(1, 2)) - jnp.mean(D)
        # A1_coeff(A1_coeff<0) = 0;
        B = jnp.maximum(B, 0)
        # Z1 = D - A1_hat - E1_hat;
        fit_residual = Im - I_B - I_R
        # Y1 = Y1 + mu*Z1;
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)
        return (k + 1, I_R, B, Y, mu, fit_residual)

    def calc_weights(self, I_B, I_R):
        XE_norm = I_R / (jnp.mean(I_B, axis=(1, 2))[:, newax, newax] + 1e-6)
        weight = jnp.ones_like(I_R) / (jnp.abs(XE_norm) + self.epsilon)
        return weight / jnp.mean(weight)

    def calc_weights_baseline(self, I_B, I_R):
        mean_vec = jnp.mean(I_B, axis=(1, 2))
        XE_norm = mean_vec[:, newax, newax] / (I_R + 1e-6)
        weight = 1.0 / (jnp.abs(XE_norm) + self.epsilon)
        return weight / jnp.mean(weight)

    def calc_darkfield(_self, S, D_R, D_Z):
        return D_R + D_Z * (1 + S)
