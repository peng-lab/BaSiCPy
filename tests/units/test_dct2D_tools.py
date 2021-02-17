#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:07:42 2019

@author: Mohammad Mirkazemi
"""

import numpy as np
from pybasic.tools import dct2d, idct2d
import pytest
import os



@pytest.fixture
def dct2_data():
    infiles = [
        'dcttool_mirt_dct2_in1.txt',
        'dcttool_mirt_dct2_in2.txt',
        'dcttool_mirt_dct2_in3.txt'
    ]
    outfiles = [
        'dcttool_mirt_dct2_out1.txt',
        'dcttool_mirt_dct2_out2.txt',
        'dcttool_mirt_dct2_out3.txt'
    ]
    
    in_mtrx = [np.loadtxt(open(os.path.join(
        os.path.dirname(__file__), 
        '../test_data', infile)
        ), delimiter=",")  for infile in infiles]
    
    out_mtrx = [np.loadtxt(open(os.path.join(
        os.path.dirname(__file__), 
        '../test_data', outfile)
        ), delimiter=",")  for outfile in outfiles]
    
    return in_mtrx, out_mtrx


def test_dct2d(dct2_data):
    in_mtrx, out_mtrx = dct2_data
    np.testing.assert_array_almost_equal(out_mtrx[0], dct2d(in_mtrx[0]), 9)
    np.testing.assert_array_almost_equal(out_mtrx[1], dct2d(in_mtrx[1]), 9)
    np.testing.assert_array_almost_equal(out_mtrx[2], dct2d(in_mtrx[2]), 9)

def test_idct2d(dct2_data):
    in_mtrx, out_mtrx = dct2_data
    np.testing.assert_array_almost_equal(in_mtrx[0], idct2d(out_mtrx[0]), 9)
    np.testing.assert_array_almost_equal(in_mtrx[1], idct2d(out_mtrx[1]), 9)
    np.testing.assert_array_almost_equal(in_mtrx[2], idct2d(out_mtrx[2]), 9)    
    