"""Compute an illumination correction model for mosaic microscope images."""

import logging
import os

from basicpy import data
from basicpy.basicpy import BaSiC

# Set logger level from environment variable
logging_level = os.getenv("BASIC_LOG_LEVEL", default="INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)


__all__ = ["BaSiC", "data"]
