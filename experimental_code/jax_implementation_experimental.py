#%%
"""jax_implementation_experimental.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KyM-VBLB3uZ8qjaOyN0V7-xBmwP3Oux-

# install and load data
"""


# copy from basicpy.data

import pooch
from os import path
from skimage.io import imread
import glob

EXPERIMENTAL_TEST_DATA_PROPS = {
    "cell_culture": {
        "filename": "Cell_culture.zip",
        "hash": "md5:797bbc4c891e5fe59f4200e771b55c3a",
    },
    "timelapse_brightfield": {
        "filename": "Timelapse_brightfield.zip",
        "hash": "md5:460e5f78ac69856705704fedad9f9e59",
    },
    "timelapse_nanog": {
        "filename": "Timelapse_nanog.zip.zip",
        "hash": "md5:815d53cac35b671269b17bd627d7baa7",
    },
    "timelapse_pu1": {
        "filename": "Timelapse_Pu1.zip.zip",
        "hash": "md5:bee97561e87c51e90b46da9b439e8b7b",
    },
    "wsi_brain": {
        "filename": "WSI_Brain.zip",
        "hash": "md5:6e163786ddec2a690aa4bb47a64bcded",
    },
}

POOCH = pooch.create(
    path=pooch.os_cache("basicpy"),
    # Use the Zenodo DOI
    base_url="doi:10.5281/zenodo.6334810/",
    registry={v["filename"]: v["hash"] for v in EXPERIMENTAL_TEST_DATA_PROPS.values()},
)

def fetch(data_name: str):
    """Fetch a sample dataset from Zenodo.
    Args:
        data_name: The name of the dataset. Must be one of ["cell_culture",
            "timelapse_brightfield", "timelapse_nanog", "timelapse_pu1",
            "wsi_brain"].

    Returns:
        Iterable[ndarray]: An iterable of uncorrected images.
        Iterable[ndarray]: An iterable of corrected images by the reference
            implementation (Tingying Peng et al., Nature Communication 8:14836 (2017).)

    Raises:
        ValueError: If the dataset name is not one of the allowed values.
    """
    if data_name not in EXPERIMENTAL_TEST_DATA_PROPS.keys():
        raise ValueError(f"{data_name} is not a valid test data name")
    file_name = EXPERIMENTAL_TEST_DATA_PROPS[data_name]["filename"]
    test_file_paths = POOCH.fetch(file_name, processor=pooch.Unzip())
    assert all(path.exists(f) for f in test_file_paths)
    basedir = path.commonpath(test_file_paths)
    uncorrected_paths = sorted(
        glob.glob(path.join(basedir, "Uncorrected*", "**", "*.tif"), recursive=True)
    )
    if len(uncorrected_paths) == 0:
        uncorrected_paths = sorted(
            glob.glob(path.join(basedir, "Uncorrected*", "**", "*.png"), recursive=True)
        )
    corrected_paths = sorted(
        glob.glob(path.join(basedir, "Corrected*", "**", "*.tif"), recursive=True)
    )
    if "WSI_Brain" in file_name:
        uncorrected_paths = list(
            filter(lambda p: "BrainSection" in p, uncorrected_paths)
        )
        corrected_paths = list(filter(lambda p: "BrainSection" in p, corrected_paths))

    assert len(uncorrected_paths) > 0
    assert len(uncorrected_paths) == len(corrected_paths)
    uncorrected = (imread(f) for f in uncorrected_paths)
    corrected = (imread(f) for f in corrected_paths)

    return uncorrected, corrected
#%%
import numpy as np
from matplotlib import pyplot as plt

images=np.array(list(fetch("wsi_brain")[0]))
print(images.shape)
# %%
plt.imshow(images[10])

"""# test original implementation"""

import basicpy
meanD = np.mean(images, axis=2)
meanD = meanD / np.mean(meanD)
W_meanD = basicpy.tools.dct2d_tools.dct2d(meanD.T)
lambda_flatfield=np.sum(np.abs(W_meanD)) / 400 * 0.5
lambda_darkfield=lambda_flatfield*0.2

images2=np.swapaxes(images,0,-1).astype(np.float32)
W=np.ones_like(images2,dtype=np.float32)
print(images2.shape,W.shape,lambda_flatfield,lambda_darkfield)
#%%
%%time
A1_hat, E1_hat, A_offset, stopCriterion=basicpy.tools.inexact_alm.inexact_alm_rspca_l1(
    images2,
    weight=W,
    lambda_flatfield=lambda_flatfield,
    lambda_darkfield=lambda_darkfield,
    get_darkfield=False,
    optimization_tol=1e-4,
    max_iterations=500,
)
print(A1_hat.shape,E1_hat.shape,A_offset.shape,stopCriterion)
X_A = np.reshape(A1_hat, images2.shape[:2] + (-1,), order="F")
X_E = np.reshape(E1_hat, images2.shape[:2] + (-1,), order="F")
X_A_offset = np.reshape(A_offset, images2.shape[:2], order="F")
print(X_A.shape,X_E.shape,X_A_offset.shape)
flatfield_flatonly_original = np.mean(X_A, axis=2) - X_A_offset

#%%
%%time
A1_hat, E1_hat, A_offset, stopCriterion=basicpy.tools.inexact_alm.inexact_alm_rspca_l1(
    images2,
    weight=W,
    lambda_flatfield=lambda_flatfield,
    lambda_darkfield=lambda_darkfield,
    get_darkfield=True,
    optimization_tol=1e-4,
    max_iterations=500,
)
print(A1_hat.shape,E1_hat.shape,A_offset.shape,stopCriterion)
X_A = np.reshape(A1_hat, images2.shape[:2] + (-1,), order="F")
X_E = np.reshape(E1_hat, images2.shape[:2] + (-1,), order="F")
X_A_offset = np.reshape(A_offset, images2.shape[:2], order="F")
print(X_A.shape,X_E.shape,X_A_offset.shape)
flatfield_withdark_original = np.mean(X_A, axis=2) - X_A_offset
darkfield_withdark_original = X_A_offset

#%%
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title("flat only flatfield")
plt.imshow(flatfield_flatonly_original)
plt.colorbar()
plt.subplot(132)
plt.title("with dark flatfield")
plt.imshow(flatfield_withdark_original)
plt.colorbar()
plt.subplot(133)
plt.title("with dark darkfield")
plt.imshow(darkfield_withdark_original)
plt.colorbar()
plt.suptitle("original implementation")

#%%
"""# test Tingying's implementation (in numpy)"""

from jax import numpy as jnp
from basicpy.tools.dct2d_tools import idct2d, dct2d, SciPyDCT
idct2d, dct2d = SciPyDCT.idct2d, SciPyDCT.dct2d

def shrinkage(x, thresh):
    return np.maximum(x-thresh,0) + np.minimum(x+thresh,0)

def basic_step_approximate(
        I,S,D_R,D_Z,I_R,B,Y,
        mu,weight,
        rho,ent1,ent2,max_mu,
        max_D_Z,
        lambda_flatfield,
        lambda_darkfield,
        get_darkfield):
    S_hat = dct2d(S)
    I_B = S[np.newaxis,...]*B[:,np.newaxis,np.newaxis] + D_R[np.newaxis,...] + D_Z
    temp_W = (I - I_B - I_R + Y/mu) / ent1
#    plt.imshow(temp_W[0]);plt.show()
#    print(type(temp_W))
    temp_W = np.mean(temp_W, axis=0)
    S_hat = S_hat + dct2d(temp_W)
    S_hat = shrinkage(S_hat, lambda_flatfield/(ent1*mu))
    S = idct2d(S_hat)
    I_B = S[np.newaxis,...]*B[:,np.newaxis,np.newaxis]+D_R[np.newaxis,...] + D_Z
    I_R = (I - I_B + Y/mu) / ent1
    I_R = shrinkage(I_R, weight / (ent1 * mu))
    R = I - I_R
    B = np.mean(R, axis =(1,2)) / np.mean(R)
    B = np.maximum(B, 0)

    if get_darkfield:
        B_valid = B<1
        _B2 = B[B_valid]

        S_inmask = (S > np.mean(S)*(1-1e-6))
        S_outmask = (S < np.mean(S)*(1+1e-6))
        A = (np.mean(R[B_valid][:,S_inmask],axis=1)-\
             np.mean(R[B_valid][:,S_outmask],axis=1))/np.mean(R)

        # temp1 = np.sum(p['A1_coeff'][validA1coeff_idx]**2)
        B_sq_sum = np.sum(_B2**2)
        B_sum = np.sum(_B2)
        A_sum = np.sum(A)
        BA_sum = np.sum(_B2 * A)
        denominator = B_sum * A_sum - BA_sum * np.sum(B_valid)
            # limit B1_offset: 0<B1_offset<B1_uplimit

        D_Z = np.clip((B_sq_sum * A_sum - B_sum * BA_sum) / (denominator+1e-6),\
                       0,max_D_Z/np.mean(S))

        Z = D_Z*(np.mean(S)-S)

        D_R = (R*B_valid[:,np.newaxis,np.newaxis]).sum(axis=0)/B_valid.sum()-_B2.sum()/B_valid.sum()*S
        D_R = D_R - np.mean(D_R) - Z

        # smooth A_offset
        D_R = dct2d(D_R)
        D_R = shrinkage(D_R, lambda_darkfield/(ent2*mu))
        D_R = idct2d(D_R)
        D_R = shrinkage(D_R, lambda_darkfield/(ent2*mu))
        D_R = D_R + Z
    fit_residual = R - I_B
    Y = Y + mu * fit_residual
    mu = np.minimum(mu * rho, max_mu)

    return S, D_R,D_Z, I_R, B, Y, mu, fit_residual

def basic_fit_approximate(images,
                    weight,
                    lambda_darkfield,
                    lambda_flatfield,
                    get_darkfield,
                    optimization_tol,
                    max_iterations,
                    rho=1.5,
                    ent1 = 1,
                    ent2 = 10,
                    mu_coef = 12.5,
                    max_mu_coef = 1e7
                    ):
    ## image dimension ... (time, Z, Y, X)
    assert np.array_equal(images.shape,weight.shape)

    # matrix 2-norm (largest sing. value)
    spectral_norm = np.linalg.norm(
        images.reshape((images.shape[0],-1)),ord=2)
    mu = mu_coef / spectral_norm
    max_mu = mu * max_mu_coef

    init_image_norm = np.linalg.norm(images.flatten(), ord=2)

    # initialize values
    I = images.copy()
    max_D_Z=np.min(I)
    S = np.zeros(images.shape[1:])
    B = np.ones(images.shape[0])
    D_R = np.zeros(images.shape[1:])
    D_Z = 0.
    I_R = np.zeros(I.shape)
    Y=np.ones_like(I)
    converged=False
    for i in range(max_iterations):
        S,D_R,D_Z,I_R,B,Y,mu,fit_residual = basic_step_approximate(I,S,D_R,D_Z,I_R,B,Y,mu,weight,
                                                rho,ent1,ent2,max_mu,
                                                max_D_Z,
                                                lambda_flatfield,
                                                lambda_darkfield,
                                                get_darkfield)
        # Stop Criterion
        norm_ratio = np.linalg.norm(fit_residual.flatten(), ord=2) \
                        / init_image_norm
        #print(i,norm_ratio)
        if norm_ratio < optimization_tol:
            converged = True
            break

    return S, D_R,D_Z, I_R, B, norm_ratio, converged

#%%
%%time
S,D_R,D_Z, I_R, B, norm_ratio, converged=basic_fit_approximate(
    images,
    weight=np.ones_like(images),
    lambda_darkfield=lambda_darkfield,
    lambda_flatfield=lambda_flatfield,
    get_darkfield=False,
    optimization_tol=1e-4,
    max_iterations=500,)
flatfield_flatonly_tingying=S

#%%
%%time
S, D_R,D_Z, I_R, B, norm_ratio, converged=basic_fit_approximate(
    images,
    weight=np.ones_like(images),
    lambda_darkfield=lambda_darkfield,
    lambda_flatfield=lambda_flatfield,
    get_darkfield=True,
    optimization_tol=1e-4,
    max_iterations=500,)
flatfield_withdark_tingying=S
darkfield_withdark_tingying=D_R+D_Z*S

#%%
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title("flat only flatfield")
plt.imshow(flatfield_flatonly_tingying)
plt.colorbar()
plt.subplot(132)
plt.title("with dark flatfield")
plt.imshow(flatfield_withdark_tingying)
plt.colorbar()
plt.subplot(133)
plt.title("with dark darkfield")
plt.imshow(darkfield_withdark_tingying)
plt.colorbar()
plt.suptitle("Tingying implementation")

#%%
"""# LADMAP implementation (in JAX)

## dct implementation by Tim
"""

"""A custom JAX idct method."""

import jax.numpy as jnp
import scipy.fft as osp_fft
from jax import lax
from jax._src.numpy.util import _wraps
from jax._src.util import canonicalize_axis


# `_W4` copied from jax._src.scipy.fft
def _W4(N, k):
    return jnp.exp(-0.5j * jnp.pi * k / N)


def _slice_ref_in_dim(x, start, stop, stride, axis):
    return tuple(
        slice(start, stop, stride) if dim == axis else slice(None, None)
        for dim in range(x.ndim)
    )


def _dct_interleave_inverse(x, axis):
    inverse = True
    if inverse:
        N = x.shape[axis]
        v0 = lax.slice_in_dim(x, None, (N + 1) // 2, 1, axis)
        v1 = lax.rev(lax.slice_in_dim(x, (N + 1) // 2, None, 1, axis), (axis,))
        ref0 = _slice_ref_in_dim(x, None, None, 2, axis)
        ref1 = _slice_ref_in_dim(x, 1, None, 2, axis)
        out = jnp.zeros(x.shape, dtype=x.dtype)
        out = out.at[ref0].set(v0)
        out = out.at[ref1].set(v1)
        return out


# REFACTOR I'm sure there is a much cleaner way of doing this
@_wraps(osp_fft.idct)
def idct(x, norm=None, axis=-1):
    axis = canonicalize_axis(axis, x.ndim)
    N = x.shape[axis]
    k = lax.expand_dims(jnp.arange(N), [a for a in range(x.ndim) if a != axis])
    V = _W4(N, -k) * x

    x0 = lax.slice_in_dim(x, None, 1, 1, axis) / 2
    V = V.at[_slice_ref_in_dim(V, None, 1, 1, axis)].set(x0)

    if norm == "ortho":
        factor = lax.concatenate(
            [
                lax.full((1,), 2 * jnp.sqrt(N), V.dtype),
                lax.full((N - 1,), jnp.sqrt(2 * N), V.dtype),
            ],
            0,
        )
        factor = lax.expand_dims(factor, [a for a in range(V.ndim) if a != axis])

        V = V * factor

    v = jnp.fft.ifft(V, axis=axis)

    xrev = lax.slice_in_dim(x, 1, None, 1, axis)
    xrev = lax.rev(xrev, (axis,))

    out = _dct_interleave_inverse(v, axis)
    return out.real

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:58

@author: Tingying
"""
import jax.scipy.fft as jax_fft
import numpy as np

def dct2d(mtrx: np.array):
    """
    Calculates 2D discrete cosine transform.

    Parameters
    ----------
    mtrx
        Input matrix.

    Returns
    -------
    Discrete cosine transform of the input matrix.
    """

    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")

    return jax_fft.dct(jax_fft.dct(mtrx.T, norm='ortho').T, norm='ortho')

def idct2d(mtrx: np.array):
    """
    Calculates 2D inverse discrete cosine transform.

    Parameters
    ----------
    mtrx
        Input matrix.

    Returns
    -------
    Inverse of discrete cosine transform of the input matrix.
    """

    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")

    return idct(idct(mtrx.T, norm='ortho').T, norm='ortho')

"""## shadow correction implementation"""
"""## loop jit"""
#%%
from jax import numpy as jnp
from jax import jit, lax, device_put

def jshrinkage(x, thresh):
    return jnp.sign(x)*jnp.maximum(jnp.abs(x)-thresh,0)

def basic_fit_ladmap(
                    images,
                    weight,
                    lambda_darkfield,
                    lambda_flatfield,
                    get_darkfield,
                    optimization_tol,
                    max_iterations,
                    rho=1.5,
                    mu_coef = 12.5,
                    max_mu_coef = 1e7
                    ):
    ## image dimension ... (time, Z, Y, X)
    assert np.array_equal(images.shape,weight.shape)

    # matrix 2-norm (largest sing. value)
    spectral_norm = np.linalg.norm(
        images.reshape((images.shape[0],-1)),ord=2)
    mu = mu_coef / spectral_norm
    max_mu = mu * max_mu_coef

    init_image_norm = np.linalg.norm(images.flatten(), ord=2)

    #FIXME fix I definition for jax
    I = device_put(images).astype(jnp.float32)

    # initialize values
    S = jnp.zeros(images.shape[1:],dtype=jnp.float32)
    D_R = jnp.zeros(images.shape[1:],dtype=jnp.float32)
    D_Z = 0
    B = jnp.ones(images.shape[0],dtype=jnp.float32)
    I_R = jnp.zeros(I.shape,dtype=jnp.float32)
    Y=jnp.ones_like(I,dtype=jnp.float32)
    fit_residual = jnp.ones(I.shape,dtype=jnp.float32)*jnp.inf

    @jit
    def basic_step_ladmap(vals):
        i,S,D_R,D_Z,I_R,B,Y,mu,fit_residual = vals
        I_B = S[jnp.newaxis,...]*B[:,jnp.newaxis,jnp.newaxis] + D_R[jnp.newaxis,...] + D_Z
        eta=jnp.sqrt(jnp.sum(B**2))*1.02
        S_prime = S+jnp.sum(B[:,jnp.newaxis,jnp.newaxis]*(I - I_B - I_R + Y/mu), axis=0)/eta
        S_hat = dct2d(S_prime)
        S = idct2d(jshrinkage(S_hat, lambda_flatfield/(mu)))

        I_B = S[jnp.newaxis,...]*B[:,jnp.newaxis,jnp.newaxis] + D_R[jnp.newaxis,...] + D_Z
        I_R = jshrinkage(I - I_B + Y/mu, weight / mu)

        R = I - I_R
        B = jnp.sum(S[jnp.newaxis,...]*(R+Y/mu), axis =(1,2))/jnp.sum(S**2)
        B = jnp.maximum(B, 0)

        I_B = S[jnp.newaxis,...]*B[:,jnp.newaxis,jnp.newaxis] + D_R[jnp.newaxis,...] + D_Z
        fit_residual = R - I_B
        Y = Y + mu * fit_residual
        mu = jnp.minimum(mu * rho, max_mu)

        return (i+1,S,D_R,D_Z,I_R, B, Y, mu, fit_residual)

    @jit
    def stopping_cond(vals):
        i,S,D_R,D_Z,I_R,B,Y,mu,fit_residual = vals
        norm_ratio = jnp.linalg.norm(fit_residual.flatten(), ord=2) \
                        / init_image_norm
        return jnp.all(jnp.array([norm_ratio > optimization_tol, i < max_iterations]))

    vals=(0,S,D_R,D_Z,I_R,B,Y,mu,fit_residual)
    vals = lax.while_loop(stopping_cond, basic_step_ladmap, vals)
    i,S,D_R,D_Z,I_R,B,Y,mu,fit_residual = vals

    return S,D_R,D_Z,I_R, B, norm_ratio, i<max_iterations

S, D_R,D_Z,I_R, B, norm_ratio, converged=basic_fit_ladmap(
    images,
    weight=np.ones_like(images),
    lambda_darkfield=lambda_darkfield,
    lambda_flatfield=lambda_flatfield,
    get_darkfield=False,
    optimization_tol=1e-4,
    max_iterations=500,)

plt.plot(B)
plt.show()
plt.imshow(S)
plt.colorbar()
plt.show()
plt.imshow(I_R[0])
plt.colorbar()
plt.show()
plt.imshow((images-S[np.newaxis]*B[:,np.newaxis,np.newaxis]-I_R)[0])
plt.colorbar()

"""# Implementation with darkfield"""

# %%
