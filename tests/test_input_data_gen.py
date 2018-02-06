from __future__ import absolute_import, division, print_function
import sys
import pytest
import pandas as pd
import numpy as np
import cv2
import h5py
import cartopy.crs as ccrs
import cartopy.img_transform as cimg
from PIL import Image
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


class TestImageTransforms(object):
    """Tests image transform functions."""

    def setup(self):
        # Image length.
        self.imlen = 256

        # Image.
        self.img = Image.open(
            "LunarLROLrocKaguya_1180mperpix_downsamp.png").convert("L")
        self.imgsize = self.img.size

        # Crater catalogue.
        self.craters = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=True)

        # Long/lat limits
        self.cdim = [-180., 180., -60., 60.]

        # Coordinate systems.
        self.iglobe = ccrs.Globe(semimajor_axis=1737400,
                                 semiminor_axis=1737400,
                                 ellipse=None)
        self.geoproj = ccrs.Geodetic(globe=self.iglobe)
        self.iproj = ccrs.PlateCarree(globe=self.iglobe)

    def get_llbd(self, box):
        llong, llat = trf.pix2coord(box[::2], box[1::2], self.cdim,
                                    self.imgsize, origin="upper")
        return np.r_[llong, llat[::-1]]

    def get_extents(self, llbd, geoproj, iproj, oproj):
        """Calculates image boundaries within projections' coordinates."""

        # Get min, mean and max long and lat.
        longbd = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
        latbd = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
        # Make a grid of these, then unravel.
        longbd, latbd = np.meshgrid(longbd, latbd)
        longbd = longbd.ravel()
        latbd = latbd.ravel()

        # [:,:2] because we don't need elevation data.
        ires = iproj.transform_points(
            x=longbd, y=latbd, src_crs=geoproj)[:, :2]
        iextent = [min(ires[:, 0]), max(ires[:, 0]),
                   min(ires[:, 1]), max(ires[:, 1])]
        # Remove zeros from iextent.
        iextent = [1e-8 if iext == 0 else iext for iext in iextent]

        ores = oproj.transform_points(
            x=longbd, y=latbd, src_crs=geoproj)[:, :2]
        oextent = [min(ores[:, 0]), max(ores[:, 0]),
                   min(ores[:, 1]), max(ores[:, 1])]

        return iextent, oextent, ores

    def test_regrid_shape_aspect(self):
        # Wide image with x / y aspect ratio of 2.25.
        target_extent = 1737 * np.array([0.7, 1.6, 0.4, 0.8])
        regrid_shape = igen.regrid_shape_aspect(self.imlen, target_extent)
        assert int(regrid_shape[1]) == self.imlen
        assert regrid_shape[0] / regrid_shape[1] == (
            (target_extent[1] - target_extent[0]) /
            (target_extent[3] - target_extent[2]))

        # Tall image with x / y aspect ratio of 3.777...
        target_extent = 1377 * np.array([0.2, 1.1, 0.4, 3.8])
        regrid_shape = igen.regrid_shape_aspect(self.imlen, target_extent)
        assert int(regrid_shape[0]) == self.imlen
        regrid_aspect = regrid_shape[0] / regrid_shape[1]
        expected_aspect = ((target_extent[1] - target_extent[0]) /
                           (target_extent[3] - target_extent[2]))
        assert np.isclose(regrid_aspect, expected_aspect, rtol=1e-8, atol=0.)

    @pytest.mark.parametrize("box", ([1197, 430, 1506, 739],
                                     [5356, 603, 5827, 1074],
                                     [813, 916, 1118, 1221],
                                     [5662, 2287, 6018, 2643],
                                     [420, 1627, 814, 2021]))
    def test_warpimage(self, box):
        """Test image warping and padding.

        Output of this function was tested by visual inspection.
        """
        box = np.array(box, dtype='int32')
        # Crop image.
        img = np.asanyarray(self.img.crop(box))

        # Determine long/lat and output projection.
        llbd = self.get_llbd(box)
        oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                                  central_latitude=np.mean(llbd[2:]),
                                  globe=self.iglobe)

        # Determine coordinates of image limits in input and output projection
        # coordinates.
        iextent, oextent, ores = self.get_extents(llbd, self.geoproj,
                                                  self.iproj, oproj)

        regrid_shape = 1.2 * min(img.shape)
        regrid_shape = igen.regrid_shape_aspect(regrid_shape, oextent)

        imgout, ext = cimg.warp_array(img[::-1],
                                      source_proj=self.iproj,
                                      source_extent=iextent,
                                      target_proj=oproj,
                                      target_res=regrid_shape,
                                      target_extent=oextent,
                                      mask_extrapolated=True)
        imgout = np.ma.filled(imgout[::-1], fill_value=0)

        # Obtain image from igen.WarpImage.
        imgout_WarpImage = igen.WarpImage(img, self.iproj, iextent, oproj,
                                          oextent, origin="upper", rgcoeff=1.2)
        imgout_WarpImage = np.ma.filled(imgout_WarpImage, fill_value=0)

        # Test that WarpImage gives the same result as this function.
        assert np.all(imgout == imgout_WarpImage)

        # Pad image.
        imgw = Image.fromarray(imgout, mode="L")
        imgw_loh = imgw.size[0] / imgw.size[1]
        if imgw_loh > (img.shape[1] / img.shape[0]):
            imgw = imgw.resize([img.shape[0],
                                int(np.round(img.shape[0] / imgw_loh))],
                               resample=Image.NEAREST)
        else:
            imgw = imgw.resize([int(np.round(imgw_loh * img.shape[0])),
                                img.shape[0]], resample=Image.NEAREST)
        imgpad = Image.new('L', (img.shape[1], img.shape[0]), (0))
        offset = ((imgpad.size[0] - imgw.size[0]) // 2,
                  (imgpad.size[1] - imgw.size[1]) // 2)
        imgpad.paste(imgw, offset)

        # Obtain image from igen.WarpImagePad.
        imgout_WarpImagePad, WIPsize, WIPoffset = igen.WarpImagePad(
            img, self.iproj, iextent, oproj, oextent, origin="upper",
            rgcoeff=1.2, fillbg="black")

        # Test that WarpImagePad gives the same result as this function.
        assert np.all(np.asanyarray(imgpad) ==
                      np.asanyarray(imgout_WarpImagePad))
        assert WIPsize == imgw.size
        assert offset == WIPoffset

    @pytest.mark.parametrize("box", ([1197, 430, 1506, 739],
                                     [5356, 603, 5827, 1074],
                                     [813, 916, 1118, 1221],
                                     [5662, 2287, 6018, 2643],
                                     [420, 1627, 814, 2021]))
    def test_warpcraters(self, box):
        """Test image warping and padding.

        Output of this function was tested by visual inspection.
        """
        box = np.array(box, dtype='int32')
        # Crop image.
        img = np.asanyarray(self.img.crop(box))

        # Determine long/lat and output projection.
        llbd = self.get_llbd(box)
        oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                                  central_latitude=np.mean(llbd[2:]),
                                  globe=self.iglobe)

        # Determine coordinates of image limits in input and output projection
        # coordinates.
        iextent, oextent, ores = self.get_extents(llbd, self.geoproj,
                                                  self.iproj, oproj)

        # Obtain image from igen.WarpImagePad.
        imgout_WarpImagePad, WIPsize, WIPoffset = igen.WarpImagePad(
            img, self.iproj, iextent, oproj, oextent, origin="upper",
            rgcoeff=1.2, fillbg="black")

        ctr_xlim = ((self.craters["Long"] >= llbd[0]) &
                    (self.craters["Long"] <= llbd[1]))
        ctr_ylim = ((self.craters["Lat"] >= llbd[2]) &
                    (self.craters["Lat"] <= llbd[3]))
        ctr_wrp = self.craters.loc[ctr_xlim & ctr_ylim, :].copy()

        # Get output projection coords.
        # [:,:2] becaus we don't need elevation data
        # If statement is in case ctr_wrp has nothing in it
        if ctr_wrp.shape[0]:
            ilong = ctr_wrp["Long"].as_matrix()
            ilat = ctr_wrp["Lat"].as_matrix()
            res = oproj.transform_points(x=ilong, y=ilat,
                                         src_crs=self.geoproj)[:, :2]

            # Get output
            ctr_wrp["x"], ctr_wrp["y"] = trf.coord2pix(res[:, 0], res[:, 1],
                                                       oextent, WIPsize,
                                                       origin="upper")
        else:
            ctr_wrp["x"] = []
            ctr_wrp["y"] = []

        ctr_wrpctrloc = igen.WarpCraterLoc(self.craters, self.geoproj, oproj,
                                           oextent, WIPsize, llbd=llbd,
                                           origin="upper")

        assert np.all(ctr_wrp == ctr_wrpctrloc)

    @pytest.mark.parametrize("box", ([1197, 430, 1506, 739],
                                     [5356, 603, 5827, 1074],
                                     [813, 916, 1118, 1221],
                                     [5662, 2287, 6018, 2643],
                                     [420, 1627, 814, 2021]))
    def test_platecarreetoorthographic(self, box):
        """Test full Plate Carree to orthographic transform.

        Assumes input_data_gen's image and crater position warping
        functions are correct. Output of this function was tested by visual
        inspection.
        """

        box = np.array(box, dtype='int32')
        # Crop image.
        img = np.asanyarray(self.img.crop(box))

        # Determine long/lat and output projection.
        llbd = self.get_llbd(box)
        oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                                  central_latitude=np.mean(llbd[2:]),
                                  globe=self.iglobe)

        # Determine coordinates of image limits in input and output projection
        # coordinates.
        iextent, oextent, ores = self.get_extents(llbd, self.geoproj,
                                                  self.iproj, oproj)

        # Obtain image from igen.WarpImagePad.
        imgo, imgwshp, offset = igen.WarpImagePad(
            img, self.iproj, iextent, oproj, oextent, origin="upper",
            rgcoeff=1.2, fillbg="black")

        ctr_xy = igen.WarpCraterLoc(self.craters, self.geoproj, oproj,
                                    oextent, imgwshp, llbd=llbd,
                                    origin="upper")
        ctr_xy.loc[:, "x"] += offset[0]
        ctr_xy.loc[:, "y"] += offset[1]

        distortion_coefficient = ((ores[7, 1] - ores[1, 1]) /
                                  (oextent[3] - oextent[2]))
        pixperkm = trf.km2pix(imgo.size[1], llbd[3] - llbd[2],
                              dc=distortion_coefficient, a=1737.4)
        ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm


        # Determine x, y position of central lat/long.
        centrallonglat = pd.DataFrame({"Long": [np.mean(llbd[:2])],
                                       "Lat": [np.mean(llbd[2:])]})
        centrallonglat_xy = igen.WarpCraterLoc(centrallonglat, self.geoproj,
                                               oproj, oextent, imgwshp,
                                               llbd=llbd, origin="upper")
        centrallonglat_xy.loc[:, "x"] += offset[0]
        centrallonglat_xy.loc[:, "y"] += offset[1]

        img_pc, ctr_pc, dc_pc, cll_pc = igen.PlateCarree_to_Orthographic(
            self.img.crop(box), None, llbd, self.craters, iglobe=self.iglobe,
            ctr_sub=False, arad=1737.4, origin="upper", rgcoeff=1.2,
            slivercut=0.)

        assert np.all(np.asanyarray(img_pc) == np.asanyarray(imgo))
        assert np.all(ctr_pc == ctr_xy)
        assert dc_pc == distortion_coefficient
        assert np.all(cll_pc == centrallonglat_xy)


class TestMaskMaking(object):
    """Tests mask making functions."""

    def setup(self):
        # Image length.
        self.imlen = 256

        # Dummy image.
        self.img = np.ones([self.imlen, self.imlen])

        # Fake craters.
        crat_x = np.array([128, 50, 2, 4, 167, 72, 254, 1])
        crat_y = np.array([128, 50, 1, 191, 3, 255, 254, 255])
        crat_d = np.array([131, 7, 12, 38, 64, 4, 3, 72])
        self.craters = pd.DataFrame([crat_x, crat_y, crat_d]).T
        self.craters.rename(columns={0: "x", 1: "y", 2: "Diameter (pix)"},
                            inplace=True)

    @pytest.mark.parametrize("r", (2, 17, 240, 1))
    def test_circlemaker(self, r):
        circle = igen.circlemaker(r=r)
        midpt = circle.shape[0] // 2
        nx, ny = np.mgrid[-midpt:midpt + 1, -midpt:midpt + 1]
        circle_comp = (nx**2 + ny**2 <= r**2)
        assert np.all(circle_comp == circle)

    @pytest.mark.parametrize("r, dr", ((2, 1), (17, 4), (240, 3), (1, 1)))
    def test_ringmaker(self, r, dr):
        ring = igen.ringmaker(r=r, dr=dr)
        midpt_corr = ring.shape[0] // 2
        ring_comp = cv2.circle(np.zeros_like(ring), (midpt_corr, midpt_corr),
                               int(np.round(r)), 1, thickness=dr)
        assert np.all(ring == ring_comp)

    def test_makerings(self):
        mask = np.zeros_like(self.img)
        radius = self.craters["Diameter (pix)"].values / 2.

        for i in range(self.craters.shape[0]):
            radius = self.craters.loc[i, "Diameter (pix)"] / 2.
            kernel = igen.ringmaker(r=radius, dr=1)
            # "Dummy values" so we can use get_merge_indices
            kernel_support = kernel.shape[0]
            ks_half = kernel_support // 2

            # Calculate indices on image where kernel should be added
            [imxl, imxr, gxl, gxr] = igen.get_merge_indices(
                self.craters.loc[i, "x"], self.img.shape[1], ks_half,
                kernel_support)
            [imyl, imyr, gyl, gyr] = igen.get_merge_indices(
                self.craters.loc[i, "y"], self.img.shape[0], ks_half,
                kernel_support)

            # Add kernel to image
            mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

        mask = (mask > 0).astype(float)

        mask_mm = igen.make_mask(self.craters, self.img, binary=True,
                                 rings=True, ringwidth=1, truncate=False)

        assert np.all(mask == mask_mm)


class TestAuxFunctions(object):
    """Tests helper functions for run_input_data_gen.py."""

    def setup(self):
        # Image length.
        self.imlen = 256

        # Image.
        self.img = Image.open(
            "LunarLROLrocKaguya_1180mperpix_downsamp.png").convert("L")
        self.imgsize = self.img.size

        # Crater catalogue.
        self.craters = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=True)

        # Long/lat limits
        self.cdim = [-180., 180., -60., 60.]

        # Coordinate systems.
        self.iglobe = ccrs.Globe(semimajor_axis=1737400,
                                 semiminor_axis=1737400,
                                 ellipse=None)

    @pytest.mark.parametrize("llbd, minpix",
                             (([-30., 30., -30., 30.], 0),
                              ([-127., 0., 35., 87.], 3),
                              ([-117.54, 120., -18., 90.], 2),
                              ([117.54, 120., -90., 0.], 0),
                              ([-180., 180., -60., 60.], 10)))
    def test_resamplecraters(self, llbd, minpix):
        ctr_xlim = ((self.craters['Long'] >= llbd[0]) &
                    (self.craters['Long'] <= llbd[1]))
        ctr_ylim = ((self.craters['Lat'] >= llbd[2]) &
                    (self.craters['Lat'] <= llbd[3]))
        ctr_sub = self.craters.loc[ctr_xlim & ctr_ylim, :]
        imgheight = int(3000. * (llbd[3] - llbd[2]) / 180.)
        pixperkm = trf.km2pix(imgheight, llbd[3] - llbd[2])
        minkm = minpix / pixperkm
        ctr_sub = ctr_sub[ctr_sub['Diameter (km)'] >= minkm]
        if minkm > 0.:      # Silly that we only do it when minkm > 0!
            ctr_sub.reset_index(drop=True, inplace=True)
        ctr_rs = igen.ResampleCraters(self.craters, llbd, imgheight,
                                      minpix=minpix)
        assert np.all(ctr_rs == ctr_sub)

    @pytest.mark.parametrize("newcdim, newpixdim",
                             (([-180., 0., 0., 60.], (4608, 1536)),
                              ([-60., 60., -60., 60.], (3072, 3072))))
    def test_initialimagecrop(self, newcdim, newpixdim):
        cropped_img = igen.InitialImageCut(self.img, self.cdim, newcdim)
        assert cropped_img.size == newpixdim


class TestGenDataset(object):
    """Test dataset generation function."""

    def setup(self):
        # Seed.
        self.seed = 1337

        # Image length.
        self.imlen = 256

        # Image.
        self.img = Image.open(
            "LunarLROLrocKaguya_1180mperpix_downsamp.png").convert("L")
        self.imgsize = self.img.size

        # Crater catalogue.
        self.craters = igen.ReadLROCHeadCombinedCraterCSV(
            filelroc="../catalogues/LROCCraters.csv",
            filehead="../catalogues/HeadCraters.csv",
            sortlat=True)

        # Long/lat limits
        self.cdim = [-180., 180., -60., 60.]

        # Coordinate systems.
        self.iglobe = ccrs.Globe(semimajor_axis=1737400,
                                 semiminor_axis=1737400,
                                 ellipse=None)

    @pytest.mark.parametrize("ringwidth", (1, 2))
    def test_gendataset(self, tmpdir, ringwidth):
        amt = 10
        zeropad = 2
        outhead = str(tmpdir.join('gentest'))

        igen.GenDataset(self.img, self.craters, outhead,
                        rawlen_range=[300, 1000], rawlen_dist='log',
                        ilen=self.imlen, tglen=self.imlen, cdim=self.cdim,
                        minpix=1, ringwidth=ringwidth, amt=amt, istart=0,
                        seed=self.seed)

        imgs_h5 = h5py.File(outhead + '_images.hdf5', 'r')
        craters_h5 = pd.HDFStore(outhead + '_craters.hdf5', 'r')

        for i in range(amt):
            # Find image number.
            img_number = "img_{i:0{zp}d}".format(i=i, zp=zeropad)

            # Load box.
            box = np.array(imgs_h5['pix_bounds'][img_number][...])

            im = self.img.crop(box)
            im.load()

            # Obtain long/lat bounds for coordinate transform.
            ix = box[::2]
            iy = box[1::2]
            llong, llat = trf.pix2coord(ix, iy, self.cdim, list(self.img.size),
                                        origin='upper')
            llbd = np.r_[llong, llat[::-1]]

            # Downsample image.
            im = im.resize([self.imlen, self.imlen], resample=Image.NEAREST)

            # Remove all craters that are too small to be seen in image.
            ctr_sub = igen.ResampleCraters(self.craters, llbd, im.size[1],
                                           minpix=1)

            # Convert Plate Carree to Orthographic.
            [imgo, ctr_xy, distortion_coefficient, clonglat_xy] = (
                igen.PlateCarree_to_Orthographic(
                    im, None, llbd, ctr_sub, iglobe=self.iglobe, ctr_sub=True,
                    slivercut=0.5))
            imgo_arr = np.asanyarray(imgo)

            # Make target mask.
            tgt = np.asanyarray(imgo.resize((self.imlen, self.imlen),
                                            resample=Image.BILINEAR))
            mask = igen.make_mask(ctr_xy, tgt, binary=True, rings=True,
                                  ringwidth=ringwidth, truncate=True)

            assert np.all(imgo_arr == imgs_h5['input_images'][i, ...])
            assert np.all(mask == imgs_h5['target_masks'][i, ...])
            assert np.all(llbd == imgs_h5['longlat_bounds'][img_number][...])
            assert (distortion_coefficient ==
                    imgs_h5['pix_distortion_coefficient'][img_number][0])
            assert np.all(clonglat_xy[["x", "y"]] ==
                          imgs_h5['cll_xy'][img_number][...])
            assert np.all(ctr_xy == craters_h5[img_number])

        imgs_h5.close()
        craters_h5.close()
