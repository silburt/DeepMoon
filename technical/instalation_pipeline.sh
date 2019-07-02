#!/usr/bin/env bash
virtualenv deep_crater_env
source deep_crater_env/bin/activate
conda install Cartopy
pip install h5py==2.6.0
pip install Keras==1.2.2
pip install numpy
pip install opencv-python==3.2.0.6
pip install pandas==0.19.1
pip install Pillow
pip install scikit-image==0.12.3
pip install tables==3.4.2
pip install tensorflow==1.10.0rc0
