# -*- coding: utf-8 -*-

"""Top-level package for LSFM FUSE."""

__author__ = "Yu Liu"
__email__ = "liuyu9671@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.0.1"

import os
import logging


def get_module_version():
    return __version__


from .basicpy import BaSiC
from .datasets import *

# Set logger level from environment variable
logging_level = os.getenv("BASIC_LOG_LEVEL", default="WARNING").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

__all__ = ["BaSiC", "datasets", "metrics"]
