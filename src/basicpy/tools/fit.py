import numpy as np
from jax import numpy as jnp
from jax import jit, lax, device_put
from basicpy.tools.dct2d_tools import JaxDCT

idct2d, dct2d = JaxDCT.idct2d, JaxDCT.dct2d


def _jshrinkage(x, thresh):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - thresh, 0)


def _fit_ladmap_single(
    images,
    weight,
    lambda_darkfield,
    lambda_flatfield,
    get_darkfield,
    optimization_tol,
    max_iterations,
    rho,
    mu_coef,
    max_mu_coef,
):
    # image dimension ... (time, Z, Y, X)
    assert np.array_equal(images.shape, weight.shape)

    # matrix 2-norm (largest sing. value)
    spectral_norm = np.linalg.norm(images.reshape((images.shape[0], -1)), ord=2)
    mu = mu_coef / spectral_norm
    max_mu = mu * max_mu_coef

    init_image_norm = np.linalg.norm(images.flatten(), ord=2)

    Im = device_put(images).astype(jnp.float32)
    D_Z_max = jnp.min(Im)

    # initialize values
    S = jnp.zeros(images.shape[1:], dtype=jnp.float32)
    D_R = jnp.zeros(images.shape[1:], dtype=jnp.float32)
    D_Z = 0
    B = jnp.ones(images.shape[0], dtype=jnp.float32)
    I_R = jnp.zeros(Im.shape, dtype=jnp.float32)
    Y = jnp.ones_like(Im, dtype=jnp.float32)
    fit_residual = jnp.ones(Im.shape, dtype=jnp.float32) * jnp.inf

    @jit
    def basic_step_ladmap(vals):
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
        S = idct2d(_jshrinkage(dct2d(S), lambda_flatfield / (eta * mu)))

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
        if get_darkfield:
            D_Z = jnp.mean(Im - BS - D_R[jnp.newaxis, ...] - I_R + Y / 2.0 / mu)
            D_Z = jnp.clip(D_Z, 0, D_Z_max)
            eta_D = Im.shape[0] * 1.02
            D_R = D_R + 1.0 / eta_D * jnp.sum(
                Im - BS - D_R[jnp.newaxis, ...] - D_Z - I_R + Y / mu, axis=0
            )
            D_R = idct2d(_jshrinkage(dct2d(D_R), lambda_darkfield / eta_D / mu))
            D_R = _jshrinkage(D_R, lambda_darkfield / eta_D / mu)

        I_B = BS + D_R[jnp.newaxis, ...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * rho, max_mu)

        return (k + 1, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)

    @jit
    def continuing_cond(vals):
        k, _, _, _, _, _, _, _, fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / init_image_norm
        #            print(i,norm_ratio,D_Z)
        return jnp.all(jnp.array([norm_ratio > optimization_tol, k < max_iterations]))

    vals = (0, S, D_R, D_Z, I_R, B, Y, mu, fit_residual)
    vals = lax.while_loop(continuing_cond, basic_step_ladmap, vals)
    k, S, D_R, D_Z, I_R, B, Y, mu, fit_residual = vals
    norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) / init_image_norm
    return S, D_R, D_Z, I_R, B, norm_ratio, k < max_iterations
