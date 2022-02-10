from setuptools import find_packages, setup

setup(name="DeepMoon",
      version='.'.join([str(v) for v in (0,0,1)]),
      description="Crater learning",
      author="anton.feldmann@gmail.com",
      author_email="anton.feldmann@gmail.com",
      url='http://',
      py_modules=['deepmoon'],
      packages=find_packages(),
      scripts=['./bin/crater_data.py'],
      include_package_data=True,
      install_requires=[])
