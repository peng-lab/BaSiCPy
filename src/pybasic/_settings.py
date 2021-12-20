#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:55:36 2020

@author: mohammad.mirkazemi
"""
            
def auto_property(attr_storage_name):
    '''
    Automatically decorate the attribute with @property
    '''
    def get_attr(instance):
        return instance.__dict__[attr_storage_name]

    def set_attr(instance, value):
        instance.__dict__[attr_storage_name] = value
    
    return property(get_attr, set_attr)
       
class FrozenClass():
    '''
    For child classes
    '''
    __is_frozen = False 
    def __setattr__(self, key, value):
        if self.__is_frozen and not hasattr(self, key):
            raise AttributeError("%r is a frozen class" % self)
            #TODO: the API user should know which argument is typed wrong
        super().__setattr__(key, value)
        
    def _frozen(self):
        self.__is_frozen = True
        
class PyBasicConfig(FrozenClass):
    """
    Manages the configuration of the PyBasic
    """

    lambda_flatfield                = auto_property('_lambda_flatfield')
    estimation_mode         = auto_property('_estimation_mode')   
    max_iterations          = auto_property('_max_iterations')   
    optimization_tolerance  = auto_property('_optimization_tolerance')   
    darkfield               = auto_property('_darkfield')   
    lambda_darkfield        = auto_property('_lambda_darkfield')   
    working_size            = auto_property('_working_size')   
    max_reweight_iterations = auto_property('_max_reweight_iterations')   
    eplson                  = auto_property('_eplson')   
    varying_coeff           = auto_property('_varying_coeff')   
    reweight_tolerance      = auto_property('_reweight_tolerance')  

    def __init__(
            self, 
            lambda_flatfield: float = 0,
            estimation_mode: str = 'l0',
            max_iterations: int = 500,
            optimization_tolerance: float = 1e-6,
            darkfield: bool = False,
            lambda_darkfield: float = 0,
            working_size: int = 128,
            max_reweight_iterations: int = 10,
            eplson: float = 0.1,
            varying_coeff: bool = True,
            reweight_tolerance: float = 1e-3,
        ):
        self.lambda_flatfield                = lambda_flatfield
        self.estimation_mode         = estimation_mode
        self.max_iterations          = max_iterations   
        self.optimization_tolerance  = optimization_tolerance  
        self.darkfield               = darkfield  
        self.lambda_darkfield        = lambda_darkfield
        self.working_size            = working_size
        self.max_reweight_iterations = max_reweight_iterations
        self.eplson                  = eplson
        self.varying_coeff           = varying_coeff
        self.reweight_tolerance      = reweight_tolerance 
        self._frozen()

settings = PyBasicConfig()
