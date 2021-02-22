from setuptools import setup, find_packages
from pathlib import Path

setup(
      name = 'pybasic',
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ]
)
