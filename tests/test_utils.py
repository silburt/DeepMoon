from __future__ import absolute_import, division, print_function
import numpy as np
import h5py
import pytest
import sys
sys.path.append('../')
import utils.template_match_target as tmt
import utils.transform as trf


class TestCraterExtraction(object):
    def setup(self):
        sample = h5py.File('sample_template_match.hdf5', 'r')
        csv = sample['csv'][...].T
        self.pred = sample['pred'][...]
        # self.coordcsv = np.array((csv[0], csv[1], csv[2] / 2.)).T
        self.pixcsv = np.array((csv[3], csv[4], csv[5] / 2.)).T
        sample.close()

    def test_extract(self):
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, csv_duplicates) = tmt.template_match_t2c(
            self.pred, self.pixcsv, minrad=8, maxrad=11)
        assert N_match == 1
        assert elo < 0.1
        assert ela < 0.1
        assert er < 0.1


class TestCoordinateTransforms(object):
    """Tests pix2coord and coord2pix."""

    def setup(self):
        np.random.seed(9590)
        origin = np.random.uniform(-30, 30, 1000)
        extent = np.random.uniform(0, 45, 1000)
        self.cdim = [origin[0], origin[0] + extent[0],
                     origin[1], origin[1] + extent[1]]
        self.imgdim = np.random.randint(100, high=200, size=1000)

        self.cx = np.array(
            [self.cdim[1], np.random.uniform(self.cdim[0] + 1, self.cdim[1])])
        self.cy = np.array(
            [self.cdim[3], np.random.uniform(self.cdim[2] + 1, self.cdim[3])])

    @pytest.mark.parametrize('origin', ('lower', 'upper'))
    def test_coord2pix(self, origin):
        x_gt = (self.imgdim[0] *
                (self.cx - self.cdim[0]) / (self.cdim[1] - self.cdim[0]))
        y_gt = (self.imgdim[1] *
                (self.cy - self.cdim[2]) / (self.cdim[3] - self.cdim[2]))
        yi_gt = (self.imgdim[1] *
                 (self.cdim[3] - self.cy) / (self.cdim[3] - self.cdim[2]))

        x, y = trf.coord2pix(self.cx, self.cy, self.cdim,
                             self.imgdim, origin=origin)
        if origin == "upper":
            y_gt_curr = yi_gt
        else:
            y_gt_curr = y_gt
        xy = np.r_[x, y]
        xy_gt = np.r_[x_gt, y_gt_curr]
        assert np.all(np.isclose(xy, xy_gt, rtol=1e-7, atol=1e-10))

    @pytest.mark.parametrize('origin', ('lower', 'upper'))
    def test_pix2coord(self, origin):
        x, y = trf.coord2pix(self.cx, self.cy, self.cdim,
                             self.imgdim, origin=origin)
        cx, cy = trf.pix2coord(x, y, self.cdim, self.imgdim,
                               origin=origin)
        cxy = np.r_[cx, cy]
        cxy_gt = np.r_[self.cx, self.cy]
        assert np.all(np.isclose(cxy, cxy_gt, rtol=1e-7, atol=1e-10))

    @pytest.mark.parametrize("imgheight, latextent, dc", [
        (1500., 180., 0.5),
        (312., 17.1, 0.7),
        (1138., 15.3, 0.931),
        (6500., 34.5, 0.878),
        (3317., 22.8, 0.946),
        (2407., 45.9832, 0.7324809721)])
    def test_km2pix(self, imgheight, latextent, dc):
        mypixperkm = (180. / (np.pi * 1737.4)) * (imgheight * dc / latextent)
        pixperkm = trf.km2pix(imgheight, latextent, dc=dc, a=1737.4)
        assert np.isclose(mypixperkm, pixperkm, rtol=1e-10, atol=0.)
        # degperpix used in get_unique_craters.
        degperpix = (180. / (np.pi * 1737.4)) / pixperkm
        mydegperpix = latextent / imgheight / dc
        assert np.isclose(degperpix, mydegperpix, rtol=1e-10, atol=0.)
