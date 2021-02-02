from deepmoon.coord.transformation import coord2pix


def WarpCraterLoc(craters,
                  geoproj,
                  oproj,
                  oextent,
                  imgdim,
                  llbd=None,
                  origin="upper"):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    geoproj : cartopy.crs.Geodetic instance
        Input lat/long coordinate system
    oproj : cartopy.crs.Projection instance
        Output coordinate system
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max)
        of output
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    llbd : list-like
        Long/lat limits (long_min, long_max,
        lat_min, lat_max) of image
    origin : "lower" or "upper"
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.

    Returns
    -------
    ctr_wrp : pandas.DataFrame
        DataFrame that includes pixel x, y positions
    """

    # Get subset of craters within llbd limits
    if llbd is None:
        ctr_wrp = craters
    else:
        ctr_xlim = ((craters["Long"] >= llbd[0]) &
                    (craters["Long"] <= llbd[1]))
        ctr_ylim = ((craters["Lat"] >= llbd[2]) & (craters["Lat"] <= llbd[3]))
        ctr_wrp = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    # Get output projection coords.
    # [:,:2] becaus we don't need elevation data
    # If statement is in case ctr_wrp has nothing in it
    if ctr_wrp.shape[0]:
        ilong = ctr_wrp["Long"].to_numpy()
        ilat = ctr_wrp["Lat"].to_numpy()
        res = oproj.transform_points(x=ilong, y=ilat, src_crs=geoproj)[:, :2]

        # Get output
        ctr_wrp["x"], ctr_wrp["y"] = coord2pix(res[:, 0],
                                               res[:, 1],
                                               oextent,
                                               imgdim,
                                               origin=origin)
    else:
        ctr_wrp["x"] = []
        ctr_wrp["y"] = []

    return ctr_wrp