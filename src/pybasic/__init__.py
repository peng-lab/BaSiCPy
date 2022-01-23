"""Compute an illumination correction model for mosaic microscope images."""

from os import getenv
import logging

logging_level=getenv("BASIC_LOG_LEVEL", default="INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)

from pybasic.pybasic import BaSiC

__all__ = ["BaSiC"]
