import numpy as np
import pandas as pd

from PIL import Image
import cartopy.crs as ccrs

from deepmoon.image.wrap import WarpImagePad
from deepmoon.craters.wrap import WarpCraterLoc
from deepmoon.coord.transformation import km2pix

def PlateCarree_to_Orthographic(img, llbd, craters, iglobe=None,
                                ctr_sub=False, arad=1737.4, origin="upper",
                                rgcoeff=1.2, slivercut=0.):
    """Transform Plate Carree image and associated csv file into Orthographic.

    Parameters
    ----------
    img : PIL.Image.image or str
        File or filename.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    craters : pandas.DataFrame
        Craters catalogue.
    iglobe : cartopy.crs.Geodetic instance
        Globe for images.  If False, defaults to spherical Moon.
    ctr_sub : bool, optional
        If `True`, assumes craters dataframe includes only craters within
        image. If `False` (default_, llbd used to cut craters from outside
        image out of (copy of) dataframe.
    arad : float
        World radius in km.  Default is Moon (1737.4 km).
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper"
        (default) means that [0,0] is upper-left corner of image; "lower" means
        it is bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  By default set
        to 1.2 to prevent loss of fidelity during transform (though warping can
        be so extreme that this might be meaningless).
    slivercut : float from 0 to 1, optional
        If transformed image aspect ratio is too narrow (and would lead to a
        lot of padding, return null images).

    Returns
    -------
    imgo : PIL.Image.image
        Transformed, padded image in PIL.Image format.
    ctr_xy : pandas.DataFrame
        Craters with transformed x, y pixel positions and pixel radii.
    distortion_coefficient : float
        Ratio between the central heights of the transformed image and original
        image.
    centrallonglat_xy : pandas.DataFrame
        xy position of the central longitude and latitude.
    """

    # If user doesn't provide Moon globe properties.
    if not iglobe:
        iglobe = ccrs.Globe(semimajor_axis=arad*1000.,
                            semiminor_axis=arad*1000., ellipse=None)

    # Set up Geodetic (long/lat), Plate Carree (usually long/lat, but not when
    # globe != WGS84) and Orthographic projections.
    geoproj = ccrs.Geodetic(globe=iglobe)
    iproj = ccrs.PlateCarree(globe=iglobe)
    oproj = ccrs.Orthographic(central_longitude=np.mean(llbd[:2]),
                              central_latitude=np.mean(llbd[2:]),
                              globe=iglobe)

    # Create and transform coordinates of image corners and edge midpoints.
    # Due to Plate Carree and Orthographic's symmetries, max/min x/y values of
    # these 9 points represent extrema of the transformed image.
    xll = np.array([llbd[0], np.mean(llbd[:2]), llbd[1]])
    yll = np.array([llbd[2], np.mean(llbd[2:]), llbd[3]])
    xll, yll = np.meshgrid(xll, yll)
    xll = xll.ravel()
    yll = yll.ravel()

    # [:,:2] because we don't need elevation data.
    res = iproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    iextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    res = oproj.transform_points(x=xll, y=yll, src_crs=geoproj)[:, :2]
    oextent = [min(res[:, 0]), max(res[:, 0]), min(res[:, 1]), max(res[:, 1])]

    # Sanity check for narrow images; done before the most expensive part of
    # the function.
    oaspect = (oextent[1] - oextent[0]) / (oextent[3] - oextent[2])
    if oaspect < slivercut:
        return [None, None]

    if type(img) != Image.Image:
        img = Image.open(img).convert("L")

    # Warp image.
    imgo, imgwshp, offset = WarpImagePad(img, iproj, iextent, oproj, oextent,
                                         origin=origin, rgcoeff=rgcoeff,
                                         fillbg="black")

    # Convert crater x, y position.
    if ctr_sub:
        llbd_in = None
    else:
        llbd_in = llbd
    ctr_xy = WarpCraterLoc(craters, geoproj, oproj, oextent, imgwshp,
                           llbd=llbd_in, origin=origin)
    # Shift crater x, y positions by offset (origin doesn't matter for y-shift,
    # since padding is symmetric).
    ctr_xy.loc[:, "x"] += offset[0]
    ctr_xy.loc[:, "y"] += offset[1]

    # Pixel scale for orthographic determined (for images small enough that
    # tan(x) approximately equals x + 1/3x^3 + ...) by l = R_moon*theta,
    # where theta is the latitude extent of the centre of the image.  Because
    # projection transform doesn't guarantee central vertical axis will keep
    # its pixel resolution, we need to calculate the conversion coefficient
    #   C = (res[7,1]- res[1,1])/(oextent[3] - oextent[2]).
    #   C0*pix height/C = theta
    # Where theta is the latitude extent and C0 is the theta per pixel
    # conversion for the Plate Carree image).  Thus
    #   l_ctr = R_moon*C0*pix_ctr/C.
    distortion_coefficient = ((res[7, 1] - res[1, 1]) /
                              (oextent[3] - oextent[2]))
    if distortion_coefficient < 0.7:
        raise ValueError("Distortion Coefficient cannot be"
                         " {0:.2f}!".format(distortion_coefficient))
    pixperkm = km2pix(imgo.size[1], llbd[3] - llbd[2],
                          dc=distortion_coefficient, a=arad)
    ctr_xy["Diameter (pix)"] = ctr_xy["Diameter (km)"] * pixperkm

    # Determine x, y position of central lat/long.
    centrallonglat = pd.DataFrame({"Long": [xll[4]], "Lat": [yll[4]]})
    centrallonglat_xy = WarpCraterLoc(centrallonglat, geoproj, oproj, oextent,
                                      imgwshp, llbd=llbd_in, origin=origin)

    # Shift central long/lat
    centrallonglat_xy.loc[:, "x"] += offset[0]
    centrallonglat_xy.loc[:, "y"] += offset[1]

    return [imgo, ctr_xy, distortion_coefficient, centrallonglat_xy]