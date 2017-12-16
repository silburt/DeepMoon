from __future__ import absolute_import, division, print_function
import sys
import pytest
import re
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
from PIL import Image
from scipy import signal
sys.path.append('../')
import input_data_gen as igen
import utils.transform as trf


class TestCatalogue(object):
    """Tests crater catalogues."""

    def setup(self):
        # Head et al. dataset.
        head = pd.read_csv('../catalogues/HeadCraters.csv', header=0,
                           names=['Long', 'Lat', 'Diameter (km)'])
        lroc = pd.read_csv('../catalogues/LROCCraters.csv',
                           usecols=range(2, 5), header=0)

        self.lrochead_t = pd.concat([lroc, head], axis=0, ignore_index=True,
                                    copy=True)
        self.lrochead_t.sort_values(by='Lat', inplace=True)
        self.lrochead_t.reset_index(inplace=True, drop=True)

    def test_dataframes_equal(self):
        lrochead = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=True)
        lrochead_nosort = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=False)

        assert np.all(lrochead == self.lrochead_t)
        assert not np.all(lrochead == lrochead_nosort)


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

    def test_km2pix(self):
        mykmppix = 1500. / (np.pi * 1737.4) * 0.5
        kmppix = trf.km2pix(1500., 180., dc=0.5, a=1737.4)
        assert np.isclose(mykmppix, kmppix, rtol=1e-7, atol=1e-10)


class InputImgTest(object):

    def setup(self):
        self.img = Image.open(
            'LunarLROLrocKaguya_1180mperpix_downsamp.png').convert("L")
        self.craters = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=True)
        self.rawlen_range=[256, 512]
        self.npseed = 1337

        np.random.seed(self.npseed)

        rawlen_min = np.log10(self.rawlen_range[0])
        rawlen_max = np.log10(self.rawlen_range[1])

        rawlen = int(10**np.random.uniform(rawlen_min, rawlen_max))
        xc = np.random.randint(0, self.img.size[0] - rawlen)
        yc = np.random.randint(0, self.img.size[1] - rawlen)
        box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')
        # THIS WILL BECOME WRONG!
        assert np.all(box == np.array([860, 1256, 1166, 1562]))

        # Take top edge of long-lat bounds.
        ix = np.array([0, 300])
        iy = np.array([0, 300])
        cdim = [-180, 180, -60, 60]
        llong, llat = trf.pix2coord(ix, iy, cdim, 
                            imgsize, origin="upper")
        self.llbd = np.r_[llong, llat[::-1]]

        self.iglobe = ccrs.Globe(semimajor_axis=1737400, 
                                 semiminor_axis=1737400,
                                 ellipse=None)

        self.geoproj = ccrs.Geodetic(globe=self.iglobe)
        self.iproj = ccrs.PlateCarree(globe=self.iglobe)
        self.oproj = ccrs.Orthographic(central_longitude=np.mean(self.llbd[:2]), 
                                central_latitude=np.mean(self.llbd[2:]), 
                                globe=self.iglobe)

        xllr = np.array([self.llbd[0], np.mean(self.llbd[:2]), 
                            self.llbd[1]])
        yllr = np.array([self.llbd[2], np.mean(self.llbd[2:]), 
                            self.llbd[3]])
        xll, yll = np.meshgrid(xllr, yllr)
        xll = xll.ravel(); yll = yll.ravel()

        # [:,:2] becaus we don't need elevation data
        res = self.iproj.transform_points(x=xll, y=yll,
                                src_crs=self.geoproj)[:,:2]
        self.iextent = [min(res[:,0]), max(res[:,0]), 
                    min(res[:,1]), max(res[:,1])]

        res = self.oproj.transform_points(x=xll, y=yll,
                                src_crs=self.geoproj)[:,:2]
        self.oextent = [min(res[:,0]), max(res[:,0]), 
                    min(res[:,1]), max(res[:,1])]

        self.craters = pd.DataFrame(np.vstack([xllr, yllr]).T, 
                                columns=["Long", "Lat"])
        self.craters["Diameter (km)"] = [10, 10, 10]

    def test_regrid_shape_aspect(self):
        # Wide image with x / y aspect ratio of 2.25.
        target_extent = 1737 * np.array([0.7, 1.6, 0.4, 0.8])
        regrid_shape = igen.regrid_shape_aspect(256, target_extent)
        assert int(regrid_shape[1]) == 256
        assert regrid_shape[0] / regrid_shape[1] == (
            (target_extent[1] - target_extent[0]) /
            (target_extent[3] - target_extent[2]))

        # Tall image with x / y aspect ratio of 
        target_extent = 1377 * np.array([0.2, 1.1, 0.4, 3.8])
        regrid_shape = igen.regrid_shape_aspect(256, target_extent)
        assert int(regrid_shape[0]) == 256
        regrid_aspect = regrid_shape[0] / regrid_shape[1]
        expected_aspect = ((target_extent[1] - target_extent[0]) /
                           (target_extent[3] - target_extent[2]))
        assert abs(regrid_aspect - expected_aspect) / expected_aspect < 1e-8

    def test_warpimage(self):

        img = np.asanyarray(self.img)
        img = img[::-1]

        regrid_shape = 1.2*min(img.shape)
        regrid_shape = igen.regrid_shape_aspect(regrid_shape,
                                         self.oextent)

        imgout, ext = cimg.warp_array(np.asanyarray(self.img),
                             source_proj=self.iproj,
                             source_extent=self.iextent,
                             target_proj=self.oproj,
                             target_res=regrid_shape,
                             target_extent=self.oextent,
                             mask_extrapolated=True)

        imgout = np.ma.filled(imgout[::-1], fill_value=0)

        imgoutigen = igen.WarpImage(img, self.iproj, 
                        self.iextent, self.oproj, self.oextent,
                        origin="upper", rgcoeff=1.2)
        imgoutigen = np.ma.filled(imgoutigen, fill_value=0)

        self.assertTrue( np.all(np.isclose(imgout.ravel(), 
                        imgoutigen.ravel(), 
                        rtol=1e-6, atol=1e-10)) )

    def test_warpcraters(self):

        # Not the real image dimensions, but whatever
        imgdim = [250, 300]

        ilong = self.craters["Long"].as_matrix()
        ilat = self.craters["Lat"].as_matrix()
        res = self.oproj.transform_points(x=ilong, y=ilat,
                                src_crs=self.geoproj)[:,:2]

        # Get output
        x, y = trf.coord2pix(res[:,0], 
                        res[:,1], self.oextent, imgdim, 
                        origin="upper")

        ctr_sub = igen.WarpCraterLoc(self.craters, self.geoproj, self.oproj, 
                        self.oextent, imgdim, llbd=None,
                        origin="upper")

        xy_gt = np.r_[x, y]
        xy = np.r_[ctr_sub["x"].as_matrix(), ctr_sub["y"].as_matrix()]

        self.assertTrue( np.all(np.isclose(xy, xy_gt, 
                        rtol=1e-7, atol=1e-10)) )

    def test_pctoortho(self):

        imgo, imgwshp, offset = igen.WarpImagePad(self.img, self.iproj, self.iextent, 
                        self.oproj, self.oextent, origin="upper", rgcoeff=1.2, 
                        fillbg="black")

        ctr_xy = igen.WarpCraterLoc(self.craters, self.geoproj, self.oproj, 
                        self.oextent, imgwshp, llbd=None,
                        origin="upper")

        ctr_xy.loc[:, "x"] += offset[0]
        ctr_xy.loc[:, "y"] += offset[1]

        Cd = 1.
        pxperkm = trf.km2pix(imgo.size[1], self.llbd[3] - self.llbd[2], \
                            dc=Cd, a=1737.4)
        ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pxperkm

        imgo2, ctr_xy2 = igen.PlateCarree_to_Orthographic(self.img, None, self.llbd, 
                                self.craters, iglobe=self.iglobe, ctr_sub=True, 
                                origin="upper", rgcoeff=1.2,
                                dontsave=True, slivercut=0.)

        imgo = np.asanyarray(imgo)
        imgo2 = np.asanyarray(imgo2)

        self.assertTrue( np.all(np.isclose(imgo.ravel(), 
                        imgo2.ravel(), 
                        rtol=1e-6, atol=1e-10)) )
        self.assertTrue( ctr_xy.equals(ctr_xy2) )


class DensMapTest(object):

    def setup(self):

        self.img =  np.asanyarray(Image.open("moonmap_tiny.png").convert("L"))
        self.craters = igen.ReadCombinedCraterCSV(dropfeatures=True)
        cx, cy = trf.coord2pix(self.craters["Long"].as_matrix(), 
                    self.craters["Lat"].as_matrix(), 
                    [-180, 180, -90, 90],
                    [self.img.shape[1], self.img.shape[0]])
        self.craters["x"] = cx
        self.craters["y"] = cy
        self.craters["Diameter (pix)"] = \
            self.craters["Diameter (km)"]*trf.km2pix(self.img.shape[0], 180)
        self.craters.drop( np.where(self.craters["Diameter (pix)"] < 15.)[0], inplace=True )
        self.craters.reset_index(inplace=True)

        self.img2 = np.zeros([200,200])
        crat_x_list = np.array([100, 50, 2, 4, 167, 72, 198, 1])
        crat_y_list = np.array([100, 50, 1, 191, 3, 199, 198, 199])
        self.craters2 = pd.DataFrame([crat_x_list, crat_y_list]).T.rename(columns={0 : "x", 1 : "y"})

        for i in range(len(crat_x_list)):
            self.img2[crat_y_list[i],crat_x_list[i]] +=1 # add one at crater location


    @staticmethod
    def gkern(l=5, sig=1.):
        """
        Creates Gaussian kernel with side length l and a sigma of sig
        """
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
        return kernel / np.sum(kernel)

    def test_make_dens_map(self):

        # Tiny Moon image
        kernel_sig = 4
        kernel_extent = 8

        dmap_delta = np.zeros(self.img.shape)

        for i in range(self.craters.shape[0]):
            dmap_delta[int(self.craters.loc[i,"y"]), int(self.craters.loc[i,"x"])] +=1 # add one at crater location

        # keep kernel support odd number (for comparison with my function)
        kernel_support = int(kernel_extent*kernel_sig/2)*2 + 1
        kernel = self.gkern(kernel_support, kernel_sig)

        img_dm_c2 = signal.convolve2d(dmap_delta, kernel, boundary='fill', mode='same')
        img_dm = igen.make_density_map(self.craters, self.img.shape, k_sig=kernel_sig, k_support=kernel_extent)

        self.assertTrue( np.isclose(img_dm_c2, img_dm, rtol=1e-05, atol=1e-08).sum() / img_dm.size )

        # Edge cases image
        kernel_sig = 12
        kernel_extent = 5
        kernel_support = int(kernel_extent * kernel_sig/2)*2 + 1
        kernel = self.gkern(kernel_support, kernel_sig)

        img2_dm_c2 = signal.convolve2d(self.img2, kernel, boundary='fill', mode='same')
        img2_dm = igen.make_density_map(self.craters2, self.img2.shape, k_sig=kernel_sig, k_support=kernel_extent)

        self.assertTrue( np.isclose(img2_dm_c2, img2_dm, rtol=1e-05, atol=1e-06).sum()/ img2_dm.size )
