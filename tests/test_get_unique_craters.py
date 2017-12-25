from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import pandas as pd
import sys
sys.path.append('../')
import get_unique_craters as guc


class TestLongLatEstimation(object):

    def setup(self):
        ctrs = pd.HDFStore('./sample_crater_csv.hdf5', 'r')
        ctrs_meta = h5py.File('./sample_crater_csv_metadata.hdf5', 'r')

        self.craters = ctrs['craters']
        self.dim = (256, 256)
        self.llbd = ctrs_meta['longlat_bounds'][...]
        self.dc = ctrs_meta['pix_distortion_coefficient'][...]

        ctrs.close()
        ctrs_meta.close()

    def test_estimate_longlatdiamkm(self):
        coords = self.craters[['x', 'y', 'Radius (pix)']].as_matrix()
        craters_unique = guc.estimate_longlatdiamkm(
            self.dim, self.llbd, self.dc, coords)
        # Check that estimate is same as predictions in sample_crater_csv.hdf5.
        assert np.all(np.isclose(craters_unique[:, 0],
                                 self.craters['Predicted Long'],
                                 atol=0., rtol=1e-10))
        assert np.all(np.isclose(craters_unique[:, 1],
                                 self.craters['Predicted Lat'],
                                 atol=0., rtol=1e-10))
        assert np.all(np.isclose(craters_unique[:, 2],
                                 self.craters['Predicted Radius (km)'],
                                 atol=0., rtol=1e-10))
        # Check that estimate is within expected tolerance from ground truth
        # values in sample_crater_csv.hdf5.
        assert np.all(abs(craters_unique[:, 0] - self.craters['Long']) /
                      (self.llbd[1] - self.llbd[0]) < 0.01)
        assert np.all(abs(craters_unique[:, 1] - self.craters['Lat']) /
                      (self.llbd[3] - self.llbd[2]) < 0.02)
        # Radius is exact, since we use the inverse estimation from km to pix
        # to get the ground truth crater pixel radii/diameters in
        # input_data_gen.py.
        assert np.all(np.isclose(craters_unique[:, 2],
                                 self.craters['Radius (km)'],
                                 atol=0., rtol=1e-10))
