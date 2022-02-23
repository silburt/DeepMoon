from setuptools import find_packages, setup

from deepmoon.__version__ import VERSION
from deepmoon import __author__, __email__

setup(name="DeepMoon",
      version=VERSION,
      description="Crater detection",
      author=__author__,
      author_email=__email__,
      url='https://github.com/afeldman/deepmoon',
      py_modules=['deepmoon'],
      packages=find_packages(),
      include_package_data=True,
      install_requires=[])
