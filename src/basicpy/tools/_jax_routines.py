from jax import numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from basicpy.tools.dct2d_tools import JaxDCT
from functools import partial
from pydantic import BaseModel

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d


@jit
def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


class BaseFit(BaseModel):
    def __call__(
        self,
    ):
        pass


@register_pytree_node_class
class LadmapFit(BaseFit):
    @jit
    def _step(
        self,
        Im,
        weight,
        vals,
    ):
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
        B = jnp.sum(S[jnp.newaxis, ...] * (R + Y / mu), axis=(1, 2)) / jnp.sum(S**2)
        B = jnp.maximum(B, 0)

        BS = S[jnp.newaxis, ...] * B[:, jnp.newaxis, jnp.newaxis]
        if self.get_darkfield:
            D_Z = jnp.mean(Im - BS - D_R[jnp.newaxis, ...] - I_R + Y / 2.0 / mu)
            D_Z = jnp.clip(D_Z, 0, self.D_Z_max)
            eta_D = Im.shape[0] * 1.02
            D_R = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[jnp.newaxis, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R = idct2d(_jshrinkage(dct2d(D_R), self.lambda_darkfield / eta_D / mu))
            D_R = _jshrinkage(D_R, self.lambda_darkfield / eta_D / mu)

        I_B = BS + D_R[jnp.newaxis, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)

    @jit
    def _cond(self, vals):
        k, _, _, _, _, _, _, _, fit_residual = vals
        norm_ratio = (
            jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.init_image_norm
        )
        return jnp.all(
            jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
        )

    @jit
    def __call__(
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
        Y = jnp.ones_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)
        ladmap_step = partial(
            self._step,
            Im,
            W,
        )
        vals = lax.while_loop(self._ladmap_cond, ladmap_step, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations


@register_pytree_node_class
class ApproximateFit(BaseFit):
    @jit
    def _step(
        self,
        Im,
        weight,
        vals,
    ):
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
        B = jnp.sum(S[jnp.newaxis, ...] * (R + Y / mu), axis=(1, 2)) / jnp.sum(S**2)
        B = jnp.maximum(B, 0)

        BS = S[jnp.newaxis, ...] * B[:, jnp.newaxis, jnp.newaxis]
        if self.get_darkfield:
            D_Z = jnp.mean(Im - BS - D_R[jnp.newaxis, ...] - I_R + Y / 2.0 / mu)
            D_Z = jnp.clip(D_Z, 0, self.D_Z_max)
            eta_D = Im.shape[0] * 1.02
            D_R = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[jnp.newaxis, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R = idct2d(_jshrinkage(dct2d(D_R), self.lambda_darkfield / eta_D / mu))
            D_R = _jshrinkage(D_R, self.lambda_darkfield / eta_D / mu)

        I_B = BS + D_R[jnp.newaxis, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * self.rho, self.max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)

    @jit
    def _cond(self, vals):
        k, _, _, _, _, _, _, _, fit_residual = vals
        norm_ratio = (
            jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.init_image_norm
        )
        return jnp.all(
            jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
        )

    @jit
    def __call__(
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
        Y = jnp.ones_like(Im, dtype=jnp.float32)
        mu = self.init_mu
        fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

        vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)
        ladmap_step = partial(
            self._step,
            Im,
            W,
        )
        vals = lax.while_loop(self._ladmap_cond, ladmap_step, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations
