"""Compute an illumination correction model for mosaic microscope images."""

import os
import logging

from pybasic.pybasic import BaSiC

# Set logger level from environment variable
logging_level = os.getenv("BASIC_LOG_LEVEL", default="INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


__all__ = ["BaSiC"]
