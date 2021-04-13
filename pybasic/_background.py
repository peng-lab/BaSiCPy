#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:53:07 2020

@author: mohammad.mirkazemi
"""
from typing import List
#from skimage.transform import resize as skresize
import numpy as np
from ._settings import settings
from .tools._resize import _resize_images_list, _resize_image
from .tools._dct2d_tools import dct2d
from .tools.inexact_alm_rspca_l1 import inexact_alm_rspca_l1


def background_timelapse(
        images_list: List,
        flatfield: np.ndarray = None,
        darkfield: np.ndarray = None,
        **kwargs
        ):
    #TODO: Rename s.t. fluorescence is included? E.g. background_fluorescence?

    """
    Estimation of background fluoresence signal for time-lapse movie. 
    Used in conjunction with BaSiC.
    """
    
    for _key, _value in kwargs.items():
        setattr(settings, _key, _value)

    nrows = ncols = _working_size = settings.working_size

    # Reszing
    # cv2.INTER_LINEAR is not exactly the same method as 'bilinear' in MATLAB
    
    resized_images = np.stack(_resize_images_list(images_list=images_list, side_size=_working_size))
    print("defore reshape", resized_images.shape, resized_images[0][0][:10])
    resized_images = resized_images.reshape([-1, nrows * nrows], order = 'F')

    resized_flatfield = _resize_image(image = flatfield, side_size = _working_size)
    
    if darkfield is not None:
        resized_darkfield = _resize_image(image = darkfield, side_size = _working_size)
    else:
        resized_darkfield = np.zeros(resized_flatfield.shape, np.uint8)
            
    print('resized_images', resized_images.shape)        
    print('resized_flatfield', resized_flatfield.shape)        
    print('resized_darkfield', resized_darkfield.shape)        
    print("0:10", resized_images[0][:10])        
    # reweighting     
    _weights = np.ones(resized_images.shape)
    eplson = 0.1
    tol = 1e-6
    for reweighting_iter in range(1,6):
        W_idct_hat = np.reshape(resized_flatfield, (1,-1), order='F')
        A_offset = np.reshape(resized_darkfield, (1,-1), order='F')
        A1_coeff = np.mean(resized_images, 1).reshape([-1,1])

        # main iteration loop starts:
        # The first element of the second array of np.linalg.svd
        _temp = np.linalg.svd(resized_images, full_matrices=False)[1]
        norm_two = _temp[0]

        mu = 12.5/norm_two # this one can be tuned
        mu_bar = mu * 1e7
        rho = 1.5 # this one can be tuned
        print('resized_images',resized_images.shape)
        d_norm = np.linalg.norm(resized_images, ord = 'fro')
        ent1 = 1
        _iter = 0
        total_svd = 0
        converged = False;
        A1_hat = np.zeros(resized_images.shape)
        E1_hat = np.zeros(resized_images.shape)
        Y1 = 0
            
        while not converged:
            _iter = _iter + 1;
            A1_hat = W_idct_hat * A1_coeff + A_offset

            # update E1 using l0 norm
            E1_hat = E1_hat + np.divide((resized_images - A1_hat - E1_hat + (1/mu)*Y1), ent1)
            E1_hat = np.maximum(E1_hat - _weights/(ent1*mu), 0) +\
                     np.minimum(E1_hat + _weights/(ent1*mu), 0)
            # update A1_coeff, A2_coeff and A_offset
            #if coeff_flag
            
            R1 = resized_images - E1_hat
            A1_coeff = np.mean(R1,1).reshape(-1,1) - np.mean(A_offset,1)

            A1_coeff[A1_coeff<0] = 0
                
            Z1 = resized_images - A1_hat - E1_hat

            Y1 = Y1 + mu*Z1

            mu = min(mu*rho, mu_bar)
                
            # stop Criterion  
            stopCriterion = np.linalg.norm(Z1, ord = 'fro') / d_norm
            # print(stopCriterion, tol)
            print('Iteration', _iter, ':', stopCriterion)
            if stopCriterion < tol:
                converged = True
            # if total_svd % 10 == 0:
            #     print('stop')
                
        # updating weight
        # XE_norm = E1_hat / np.mean(A1_hat)
        XE_norm = E1_hat
        mean_vec = np.mean(A1_hat, axis=1)
        XE_norm = np.transpose(np.tile(mean_vec, (16384, 1))) / XE_norm
        _weights = 1./(abs(XE_norm)+eplson)

        _weights = np.divide( np.multiply(_weights, _weights.shape[0] * _weights.shape[1]), np.sum(_weights))

    return A1_coeff


def basic(images_list: List, segmentation: List = None,  **kwargs):
    """
    Estimation of flatfield for optical microscopy. Apply to a collection of monochromatic images. Multi-channel images
    should be separated, and each channel corrected separately.



    #TODO: Explain possible inputs?
    :param images: array with shape [N,M,L], with [N,M] image dimensions, and L number of images
    """

    for _key, _value in kwargs.items():
        setattr(settings, _key, _value)

    nrows = ncols = _working_size = settings.working_size
    
    _saved_size = images_list[0].shape

    D = np.dstack(_resize_images_list(images_list=images_list, side_size=_working_size))
    print(D.shape)
    '''
    if images.shape[0] != nrows or images.shape[1] != ncols:
        D = np.array([skresize(images[:,:,i],
                               (nrows, ncols),
                               order = _resize_order,
                               mode = _resize_mode)
                      for i in range(images.shape[2])])
        D = np.transpose(D, (1, 2, 0))
        print(D.shape)
    else:
        D = images.copy()
    '''
    meanD = np.mean(D, axis=2)
    meanD = meanD / np.mean(meanD)
    W_meanD = dct2d(meanD.T)
    if settings.lambda_s == 0:
        setattr(settings, 'lambda_s', np.sum(np.abs(W_meanD)) / 400 * 0.5)
    if settings.lambda_darkfield == 0:
        setattr(settings, 'lambda_darkfield', settings.lambda_s * 0.2)

    # TODO: Ask Tingying whether to keep sorting? I remember the sorting caused some problems with some data.
    D = np.sort(D, axis=2)

    XAoffset = np.zeros((nrows, ncols))
    weight = np.ones(D.shape)

    if segmentation is not None:
        segmentation = np.array(segmentation)
        segmentation = np.transpose(segmentation, (1, 2, 0))
        for i in range(weight.shape[2]):
            weight[segmentation] = 1e-6
        # weight[options.segmentation] = 1e-6

    reweighting_iter = 0
    flag_reweighting = True
    flatfield_last = np.ones((nrows, ncols))
    darkfield_last = np.random.randn(nrows, ncols)

    while flag_reweighting:
        reweighting_iter += 1
        print('Reweighting Iteration', reweighting_iter)

        initial_flatfield = False
        if initial_flatfield:
            # TODO: implement inexact_alm_rspca_l1_intflat?
            raise IOError('Initial flatfield option not implemented yet!')
        else:
            X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, weight=weight);
        XA = np.reshape(X_k_A, [nrows, ncols, -1], order='F')
        XE = np.reshape(X_k_E, [nrows, ncols, -1], order='F')
        XAoffset = np.reshape(X_k_Aoffset, [nrows, ncols], order='F')
        XE_norm = XE / np.mean(XA, axis=(0, 1))

        # Update the weights:
        weight = np.ones_like(XE_norm) / (np.abs(XE_norm) + settings.eplson)
        if segmentation is not None:
            weight[segmentation] = 0

        weight = weight * weight.size / np.sum(weight)

        temp = np.mean(XA, axis=2) - XAoffset
        flatfield_current = temp / np.mean(temp)
        darkfield_current = XAoffset
        mad_flatfield = np.sum(np.abs(flatfield_current - flatfield_last)) / np.sum(np.abs(flatfield_last))
        temp_diff = np.sum(np.abs(darkfield_current - darkfield_last))
        if temp_diff < 1e-7:
            mad_darkfield = 0
        else:
            mad_darkfield = temp_diff / np.maximum(np.sum(np.abs(darkfield_last)), 1e-6)
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if np.maximum(mad_flatfield,
                      mad_darkfield) <= settings.reweight_tolerance or \
                reweighting_iter >= settings.max_reweight_iterations:
            flag_reweighting = False

    shading = np.mean(XA, 2) - XAoffset

    flatfield = _resize_image(
        image = shading, 
        x_side_size = _saved_size[0], 
        y_side_size = _saved_size[1]
    )
    flatfield = flatfield / np.mean(flatfield)

    if settings.darkfield:
        print('tttt', type(XAoffset), XAoffset.shape)
        darkfield = _resize_image(
            image = XAoffset, 
            x_side_size = _saved_size[0], 
            y_side_size = _saved_size[1]
        )
    else:
        darkfield = np.zeros_like(flatfield)

    return flatfield, darkfield

def correct_illumination(images_list: List, flatfield: np.ndarray = None, darkfield: np.ndarray = None):
    _saved_size = images_list[0].shape
    
    if not flatfield.shape == _saved_size:
        flatfield = _resize_image(
            image = flatfield, 
            x_side_size = _saved_size[0], 
            y_side_size = _saved_size[1]
        )
    
    if darkfield is None:
        return [_im / flatfield for _im in images_list]
    else:
        if not darkfield.shape == _saved_size:
            darkfield = _resize_image(
                image = darkfield, 
                x_side_size = _saved_size[0], 
                y_side_size = _saved_size[1]
            )
        return [(_im  - darkfield)/ flatfield for _im in images_list]
