from jax import numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from basicpy.tools.dct2d_tools import JaxDCT
from functools import partial
from pydantic import BaseModel, Field

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d


@jit
def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


class BaseFit(BaseModel):
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
        vals = lax.while_loop(self._cond, ladmap_step, vals)
        k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return S, D_R, D_Z, I_R, B, norm_ratio, k < self.max_iterations

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
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / self.image_norm
        return jnp.all(
            jnp.array([norm_ratio > self.optimization_tol, k < self.max_iterations])
        )


#    @classmethod
#    def tree_unflatten(cls, aux_data, children):
#        super(LadmapFit,cls,)._tree_unflatten(aux_data, children)


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
