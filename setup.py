#!/usr/bin/env python

from os import walk
from os.path import join

from setuptools import find_packages, setup

from torchmoon.__version__ import VERSION
from torchmoon import __author__, __email__

def package_files(directory):
    paths = []
    for (path, _, filenames) in walk(directory):
        for filename in filenames:
            paths.append(join(path, filename))
    return paths

extra_files = package_files('configs')

setup(name="torchmoon",
      version=VERSION,
      description="Crater detection",
      author=__author__,
      author_email=__email__,
      url='https://github.com/afeldman/TorchMoon',
      packages=find_packages(),
      package_data={'': extra_files},)
