"""Contains the PyBaSiC class."""

import numpy as np


class BaSiC:
    def __init__(self):
        ...

    def fit(self):
        """Generate profiles"""
        ...

    def predict(self, image: np.ndarray):
        """Apply model"""
        ...

    def __repr__(self):
        # print out current settings and model if available
        ...

    def __enter__(self):
        # for context management
        ...

    def __exit__(self):
        # for context management
        ...


class Model:
    ...


class FlatField(Model):
    ...


class DarkField(Model):
    ...


class TimeLapse(Model):
    ...
