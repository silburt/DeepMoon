# DeepMoon Test Suite

## Dependences

Running the DeepMoon test suite requires the following packages (in addition to
those required by DeepMoon itself):

- [pytest](https://pypi.python.org/pypi/pytest) >= 3.11

## Running the Test Suite

Each file in the folder is a self-contained test script for the main functions
and utilities of DeepMoon.  To run them, use `pytest` on the command line, eg.

```
pytest test_get_unique_craters.py
```

## Sample Files Used By the Test Suite

- `LunarLROLrocKaguya_1180mperpix_downsamp.png`: the LROC-Kaguya DEM,
downsampled to 9216 x 3072 pixels, or 1180 m per pixel.
- `sample.hdf5`: **ARI**
- `sample_crater_csv.hdf5`: Pandas HDFStore file containing the crater
data table for the 6th image (`img_00005`) in the test dataset.
- `sample_crater_csv_metadata.hdf5`: metadata for the 6th image in the test
dataset.