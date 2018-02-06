# DeepMoon - Lunar Crater Counting Through Deep Learning
Center for Planetary Sciences / Department of Astronomy & Astrophysics / Canadian Institute for Theoretical Astrophysics
University of Toronto

DeepMoon is a TensorFlow-based pipeline for training a convolutional neural
network (CNN) to recognize craters on the Moon, and determine their positions and
radii.  It is the companion repo to Silburt et al. in preparation, which
describes the motivation and development of the code, as well as initial
results.

## Getting Started

### Overview

The DeepMoon pipeline trains a neural net using data derived from a global
digital elevation map (DEM) and catalogue of craters.  The code is divided into
three parts.  The first generates a set images of the Moon randomly cropped
from the DEM, with corresponding crater positions and radii.  The second
trains a convnet using this data.  The third validates the convnet's
predictions.

To first order, our CNN activates regions with high negative gradients, i.e.
pixels that decrease in value as you move across the image. Below illustrates 
two examples of this, the first is a genuine DEM Lunar image from our dataset, 
the second is a sample image taken from the web.
![CNN1](docs/CNN1.png?raw=true)
![CNN2](docs/CNN2.png?raw=true)

### Dependences

DeepMoon requires the following packages to function:

- [Python](https://www.python.org/) version 2.7 or 3.5+
- [Cartopy](http://scitools.org.uk/cartopy/) >= 0.14.2.  Cartopy itself has a
number of [dependencies](http://scitools.org.uk/cartopy/docs/latest/installing.html#installing),
including the GEOS and Proj.4.x libraries.  (For Ubuntu systems, these can be
installed through the `libgeos++-dev` and `libproj-dev` packages,
respectively.)
- [h5py](http://www.h5py.org/) >= 2.6.0
- [Keras](https://keras.io/) 1.2.2 [(documentation)](https://faroit.github.io/keras-docs/1.2.2/);
  also tested with Keras >= 2.0.2
- [Numpy](http://www.numpy.org/) >= 1.12
- [OpenCV](https://pypi.python.org/pypi/opencv-python) >= 3.2.0.6
- [*pandas*](https://pandas.pydata.org/) >= 0.19.1
- [Pillow](https://python-pillow.org/) >= 3.1.2
- [PyTables](http://www.pytables.org/) >=3.4.2
- [TensorFlow](https://www.tensorflow.org/) 0.10.0rc0, also tested with
  TensorFlow >= 1.0

This list can also be found in the `requirements.txt`.

### Data Sources
Our train, validation and test datasets, global DEM, post-processed
crater distribution on the test set, best model, and sample output
images can be found [on Zenodo](https://doi.org/10.5281/zenodo.1133969).

Examples of how to read these data can be found in the
`docs/Using Zenodo Data.ipynb` IPython notebook.

#### Digital Elevation Maps

We use the [LRO-Kaguya merged 59 m/pixel DEM][lola dem].  The DEM was
downsampled to 118 m/pixel and converted to 16-bit GeoTiff with the USGS
Astrogeology Cloud Processing service, and then rescaled to 8-bit PNG using
the [GDAL](http://www.gdal.org/) library:

```
gdal_translate -of PNG -scale -21138 21138 -co worldfile=no 
    LunarLROLrocKaguya_118mperpix_int16.tif LunarLROLrocKaguya_118mperpix.png
```

#### Crater Catalogues

For the ground truth longitude / latitude locations and sizes of craters, we
combine the [LROC Craters 5 to 20 km diameter][lroc cat] dataset with the
[Head et al. 2010 >= 20 km diameter][head cat] one ([alternate download
link][head cat2]).  The LROC dataset was converted from ESRI shapefile to .csv.
They can be found under the `catalogues` folder of the repo, and have had their
formatting slightly modified to be read into *pandas*.

During initial testing, we also used the [Salamunićcar LU78287GT
catalogue][sala cat].

### Running DeepMoon

Each stage of DeepMoon has a corresponding script: `run_input_data_gen.py` for
generating input data, `run_model_training.py` to build and train the convnet, 
and `run_get_unique_craters.py` to validate predictions and generate a crater
atlas.  User-defined parameters, and instructions on on how to use each script,
can be found in the scripts themselves.

We recommend copying these scripts into a new working directory (and appending
this repo to your Python path) instead of modifying them in the repo.

Our model with default parameters was trained on a 16GB Tesla P100 GPU, however
12GB GPUs are more standard. Therefore, our default model may not run on many 
systems without reducing the batch size, number of filters, etc., which can 
affect final model convergence. 

### Quick Usage

See `docs/Using Zenodo Data.ipynb` for basic examples on generating sample
datasets, loading a pre-trained CNN and using it to make predictions on
samples.

## Authors

* **Ari Silburt** - convnet architecture, crater extraction and post-processing
  [silburt](https://github.com/silburt)
* **Charles Zhu** - input image generation, data ingestion and post-processing
  [cczhu](https://github.com/cczhu)

### Contributors

* Mohamad Ali-Dib - [malidib](https://github.com/malidib/)
* Kristen Menou - [kmenou](https://www.kaggle.com/kmenou)
* Alan Jackson

## License

Copyright 2018 Ari Silburt, Charles Zhu and contributors.

DeepMoon is free software made available under the MIT License. For details see
the LICENSE.md file.

[lola dem]: https://astrogeology.usgs.gov/search/map/Moon/LRO/LOLA/Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd
[lroc cat]: http://wms.lroc.asu.edu/lroc/rdr_product_select?filter%5Btext%5D=&filter%5Blat%5D=&filter%5Blon%5D=&filter%5Brad%5D=&filter%5Bwest%5D=&filter%5Beast%5D=&filter%5Bsouth%5D=&filter%5Bnorth%5D=&filter%5Btopographic%5D=either&filter%5Bprefix%5D%5B%5DSHAPEFILE&show_thumbs=0&per_page=100&commit=Search
[head cat]: http://science.sciencemag.org/content/329/5998/1504/tab-figures-data
[head cat2]: http://www.planetary.brown.edu/html_pages/LOLAcraters.html
[sala cat]: https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/GoranSalamuniccar_MoonCraters
