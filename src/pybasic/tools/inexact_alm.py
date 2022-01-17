#!/usr/bin/env python3
"""
Created on Thu Jun 25 14:04:46 2020

@author: mohammad.mirkazemi

Modified on Sun Jan 16, 2021

@author: nicholas-schaub
@email nick.schaub@nih.gov
"""


import numpy as np
from scipy.fftpack import dct, idct


def dct2d(x: np.ndarray) -> np.ndarray:

    return dct(dct(x, norm="ortho").T, norm="ortho")


def idct2d(x: np.ndarray) -> np.ndarray:
    idct(idct(x, norm="ortho").T, norm="ortho")


def _shrinkageOperator(matrix: np.ndarray, epsilon):
    temp1 = matrix - epsilon
    temp1[temp1 < 0] = 0
    temp2 = matrix + epsilon
    temp2[temp2 > 0] = 0
    res = temp1 + temp2
    return res


def inexact_alm_rspca_l1(
    images: np.ndarray,
    lambda_darkfield: float,
    lambda_flatfield: float,
    get_darkfield: bool,
    optimization_tol: float,
    max_iterations: int,
    weight: np.ndarray = None,
):

    p = images.shape[0]
    q = images.shape[1]
    m = p * q
    n = images.shape[2]
    images = np.reshape(images, (m, n), order="F")

    if weight is not None:
        weight = np.reshape(weight, (m, n), order="F")
    else:
        weight = np.ones_like(images)
    svd = np.linalg.svd(
        images, False, False
    )  # TODO: Is there a more efficient implementation of SVD?
    norm_two = svd[0]
    Y1 = 0
    ent1 = 1
    ent2 = 10

    A1_hat = np.zeros_like(images)
    A1_coeff = np.ones((1, images.shape[1]))

    E1_hat = np.zeros_like(images)
    W_hat = dct2d(np.zeros((p, q)).T)
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(images, ord="fro")

    A_offset = np.zeros((m, 1))
    B1_uplimit = np.min(images)
    B1_offset = 0

    A_inmask = np.zeros((p, q))
    A_inmask[
        int(np.round(p / 6) - 1) : int(np.round(p * 5 / 6)),
        int(np.round(q / 6) - 1) : int(np.round(q * 5 / 6)),
    ] = 1

    # main iteration loop starts
    iter = 0
    converged = False

    while not converged:
        iter += 1

        if len(A1_coeff.shape) == 1:
            A1_coeff = np.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = np.expand_dims(A_offset, 1)
        W_idct_hat = idct2d(W_hat.T)
        A1_hat = np.dot(np.reshape(W_idct_hat, (-1, 1), order="F"), A1_coeff) + A_offset

        temp_W = (images - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        temp_W = np.reshape(temp_W, (p, q, n), order="F")
        temp_W = np.mean(temp_W, axis=2)
        W_hat = W_hat + dct2d(temp_W.T)
        W_hat = np.maximum(W_hat - lambda_flatfield / (ent1 * mu), 0) + np.minimum(
            W_hat + lambda_flatfield / (ent1 * mu), 0
        )
        W_idct_hat = idct2d(W_hat.T)
        if len(A1_coeff.shape) == 1:
            A1_coeff = np.expand_dims(A1_coeff, 0)
        if len(A_offset.shape) == 1:
            A_offset = np.expand_dims(A_offset, 1)
        A1_hat = np.dot(np.reshape(W_idct_hat, (-1, 1), order="F"), A1_coeff) + A_offset
        E1_hat = images - A1_hat + (1 / mu) * Y1 / ent1
        E1_hat = _shrinkageOperator(E1_hat, weight / (ent1 * mu))
        R1 = images - E1_hat
        A1_coeff = np.mean(R1, 0) / np.mean(R1)
        A1_coeff[A1_coeff < 0] = 0

        if get_darkfield:
            validA1coeff_idx = np.where(A1_coeff < 1)

            B1_coeff = (
                np.mean(
                    R1[
                        np.reshape(W_idct_hat, -1, order="F")
                        > np.mean(W_idct_hat) - 1e-6
                    ][:, validA1coeff_idx[0]],
                    0,
                )
                - np.mean(
                    R1[
                        np.reshape(W_idct_hat, -1, order="F")
                        < np.mean(W_idct_hat) + 1e-6
                    ][:, validA1coeff_idx[0]],
                    0,
                )
            ) / np.mean(R1)
            k = np.array(validA1coeff_idx).shape[1]
            temp1 = np.sum(A1_coeff[validA1coeff_idx[0]] ** 2)
            temp2 = np.sum(A1_coeff[validA1coeff_idx[0]])
            temp3 = np.sum(B1_coeff)
            temp4 = np.sum(A1_coeff[validA1coeff_idx[0]] * B1_coeff)
            temp5 = temp2 * temp3 - temp4 * k
            if temp5 == 0:
                B1_offset = 0
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit

            B1_offset = np.maximum(B1_offset, 0)
            B1_offset = np.minimum(B1_offset, B1_uplimit / np.mean(W_idct_hat))

            B_offset = B1_offset * np.reshape(W_idct_hat, -1, order="F") * (-1)

            B_offset = B_offset + np.ones_like(B_offset) * B1_offset * np.mean(
                W_idct_hat
            )
            A1_offset = np.mean(R1[:, validA1coeff_idx[0]], axis=1) - np.mean(
                A1_coeff[validA1coeff_idx[0]]
            ) * np.reshape(W_idct_hat, -1, order="F")
            A1_offset = A1_offset - np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset

            # smooth A_offset
            W_offset = dct2d(np.reshape(A_offset, (p, q), order="F").T)
            W_offset = np.maximum(
                W_offset - lambda_darkfield / (ent2 * mu), 0
            ) + np.minimum(W_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = idct2d(W_offset.T)
            A_offset = np.reshape(A_offset, -1, order="F")

            # encourage sparse A_offset
            A_offset = np.maximum(
                A_offset - lambda_darkfield / (ent2 * mu), 0
            ) + np.minimum(A_offset + lambda_darkfield / (ent2 * mu), 0)
            A_offset = A_offset + B_offset

        Z1 = images - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = np.minimum(mu * rho, mu_bar)

        # Stop Criterion
        stopCriterion = np.linalg.norm(Z1, ord="fro") / d_norm
        if stopCriterion < optimization_tol:
            converged = True

        if not converged and iter >= max_iterations:
            print("Maximum iterations reached")
            converged = True

    A_offset = np.squeeze(A_offset)
    A_offset = A_offset + B1_offset * np.reshape(W_idct_hat, -1, order="F")

    return A1_hat, E1_hat, A_offset
