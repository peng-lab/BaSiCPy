"""Compute an illumination correction model for mosaic microscope images."""

from os import getenv
from logging import getLogger

from pybasic.pybasic import BaSiC

# Set logger level from environment variable
logging_level = getenv("BASIC_LOG_LEVEL", default="INFO").upper()
logger = getLogger(__name__)
logger.setLevel(logging_level)


__all__ = ["BaSiC"]
