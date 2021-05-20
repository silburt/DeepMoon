from deepmoon.coord.transformation import km2pix

def ResampleCraters(craters, llbd, imgheight, arad=1737.4, minpix=0):
    """Crops crater file, and removes craters smaller than some minimum value.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater dataframe.
    llbd : list-like
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    imgheight : int
        Pixel height of image.
    arad : float, optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimium crater pixel size to be included in output.  Default is 0
        (equvalent to no cutoff).

    Returns
    -------
    ctr_sub : pandas.DataFrame
        Cropped and filtered dataframe.
    """

    # Get subset of craters within llbd limits.
    ctr_xlim = (craters["Long"] >= llbd[0]) & (craters["Long"] <= llbd[1])
    ctr_ylim = (craters["Lat"] >= llbd[2]) & (craters["Lat"] <= llbd[3])
    ctr_sub = craters.loc[ctr_xlim & ctr_ylim, :].copy()

    if minpix > 0:
        # Obtain pixel per km conversion factor.  Use latitude because Plate
        # Carree doesn't distort along this axis.
        pixperkm = km2pix(imgheight, llbd[3] - llbd[2], dc=1., a=arad)
        minkm = minpix / pixperkm

        # Remove craters smaller than pixel limit.
        ctr_sub = ctr_sub[ctr_sub["Diameter (km)"] >= minkm]

    ctr_sub.reset_index(inplace=True, drop=True)

    return ctr_sub