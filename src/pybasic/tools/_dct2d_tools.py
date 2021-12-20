#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:25:46 2019

@author: Mohammad Mirkazemi
"""
from scipy.fftpack import dct, idct
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

    return dct(dct(mtrx.T, norm='ortho').T, norm='ortho')

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