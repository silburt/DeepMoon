# DeepMoon - Lunar Crater Counting Through Deep Learning
Center for Planetary Sciences / Department of Astronomy & Astrophysics / Canadian Institute for Theoretical Astrophysics
University of Toronto

DeepMoon is a TensorFlow-based pipeline for training a convolutional neural
network to recognize craters on the Moon, and determine their positions and
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

### Dependences

DeepMoon requires the following packages to function:

- [Python](https://www.python.org/) version 2.7 or 3.5+
- [Cartopy](http://scitools.org.uk/cartopy/) >= 0.14.2; Cartopy requires
  installing the **XXXX** and **YYYY** packages.
- [h5py](http://www.h5py.org/) >= 2.6.0
- [Keras](https://keras.io/) 1.2.2 [documentation](https://faroit.github.io/keras-docs/1.2.2/);
  also tested with Keras >= 2.0
- [Numpy](http://www.numpy.org/) >= 1.12
- [OpenCV](https://pypi.python.org/pypi/opencv-python) >= 3.2.0.6
- [*pandas*](https://pandas.pydata.org/) >= 0.19.1
- [Pillow](https://python-pillow.org/) >= 3.1.2
- [TensorFlow](https://www.tensorflow.org/) 0.10.0rc0, also tested with
  TensorFlow >= 1.0

This list can also be found in the `requirements.txt`.

### Data Sources

#### Digital Elevation Maps

We use the [LRO-Kaguya merged 59 m/pixel DEM][lola dem].  The DEM was
downsampled to 118 m/pixel and converted to 16-bit GeoTiff with the USGS
Astrogeology Cloud Processing service, and then rescaled to 8-bit PNG using
the [GDAL](http://www.gdal.org/) library:

```
gdal_translate -of PNG -scale -21138 21138 -co worldfile=no 
    LunarLROLrocKaguya_118mperpix_int16.tif LunarLROLrocKaguya_118mperpix.png
```

A copy of our global DEM can be found at **XXXXX**.

#### Crater Catalogues

For the ground truth longitude / latitude locations and sizes of craters, we
combine the [LROC Craters 5 to 20 km diameter][lroc cat] dataset with the
[Head et al. 2010 >= 20 km diameter][head cat] one.  The LROC dataset was
converted from ESRI shapefile to .csv.  They can be found under the
`catalogues` folder of the repo, and have had their formatting slightly
modified to be read into *pandas*.

During initial testing, we also used the [Salamunićcar LU78287GT
catalogue][sala cat].

### Running DeepMoon

Each stage of DeepMoon has a corresponding script: `run_input_data_gen.py` for
generating input data, `run_model_training.py` to train the convnet, and
`run_get_unique_craters.py` to validate predictions and generate a crater
atlas.  User-defined parameters, and instructions on on how to use each script,
can be found in the scripts themselves.

We recommend copying these scripts into a new working directory (and appending
this repo to your Python path) instead of modifying them in the repo.

## Authors

* **Ari Silburt** - convnet architecture, crater extraction and post-processing
  [silburt](https://github.com/silburt)
* **Charles Zhu** - input image generation, data ingestion and post-processing
  [cczhu](https://github.com/cczhu)

## License

**CC or MIT?**

[lola dem]: https://astrogeology.usgs.gov/search/map/Moon/LRO/LOLA/Lunar_LRO_LrocKaguya_DEMmerge_60N60S_512ppd
[lroc cat]: http://wms.lroc.asu.edu/lroc/rdr_product_select?filter%5Btext%5D=&filter%5Blat%5D=&filter%5Blon%5D=&filter%5Brad%5D=&filter%5Bwest%5D=&filter%5Beast%5D=&filter%5Bsouth%5D=&filter%5Bnorth%5D=&filter%5Btopographic%5D=either&filter%5Bprefix%5D%5B%5DSHAPEFILE&show_thumbs=0&per_page=100&commit=Search
[head cat]: http://science.sciencemag.org/content/329/5998/1504/tab-figures-data
[sala cat]: https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/GoranSalamuniccar_MoonCraters