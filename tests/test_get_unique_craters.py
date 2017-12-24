from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import sys
sys.path.append('../')
import utils.template_match_target as tmt


class TestCraterExtraction(object):
    def setup(self):
        sample = h5py.File('sample.hdf5', 'r')
        csv = sample['csv'][...].T
        self.pred = sample['pred'][...]
        # self.coordcsv = np.array((csv[0], csv[1], csv[2] / 2.)).T
        self.pixcsv = np.array((csv[3], csv[4], csv[5] / 2.)).T

    def test_extract(self):
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, csv_duplicates) = tmt.template_match_t2c(
            self.pred, self.pixcsv, minrad=8, maxrad=11)
        assert N_match == 1
        assert elo < 0.1
        assert ela < 0.1
        assert er < 0.1
