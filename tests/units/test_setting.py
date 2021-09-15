#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:36:37 2020

@author: mohammad.mirkazemi
"""

import pytest
from pybasic._settings import PyBasicConfig

@pytest.fixture
def pybasic_config():
    return PyBasicConfig()


def test_pybasic_config1(pybasic_config):
    assert pybasic_config.lambda_flatfield == 0
    assert pybasic_config.estimation_mode == 'l0'
    assert pybasic_config.max_iterations == 500
    assert pybasic_config.optimization_tolerance == 1e-6
    assert pybasic_config.darkfield == False
    assert pybasic_config.lambda_darkfield == 0
    assert pybasic_config.working_size == 128
    assert pybasic_config.max_reweight_iterations == 10
    assert pybasic_config.eplson == 0.1
    assert pybasic_config.varying_coeff == True
    assert pybasic_config.reweight_tolerance == 1e-3

def test_pybasic_config2(pybasic_config):
    _pybasic_config = pybasic_config
    _pybasic_config.lambda_flatfield = 1.5
    _pybasic_config.lambda_darkfield = 2.5
    _pybasic_config.varying_coeff = True
    _pybasic_config.reweight_tolerance = 1e-3 
    
    assert _pybasic_config.lambda_flatfield == 1.5
    assert _pybasic_config.lambda_darkfield == 2.5
    assert _pybasic_config.varying_coeff == True
    assert _pybasic_config.reweight_tolerance == 1e-3 
    
    
def test_pybasic_config3(pybasic_config):
    with pytest.raises(AttributeError):
        pybasic_config.abc = 'abc'
        
        
