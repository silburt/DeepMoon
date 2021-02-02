from deepmoon.coord.transformation import coord2pix


def AddPlateCarree_XY(craters,
                      imgdim,
                      cdim=[-180., 180., -90., 90.],
                      origin="upper"):
    """Adds x and y pixel locations to craters dataframe.

    Parameters
    ----------
    craters : pandas.DataFrame
        Crater info
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        [-180., 180., -90., 90.].
    origin : "upper" or "lower", optional
        Based on imshow convention for displaying image y-axis.
        "upper" means that [0,0] is upper-left corner of image;
        "lower" means it is bottom-left.
    """
    x, y = coord2pix(craters["Long"].to_numpy(),
                     craters["Lat"].to_numpy(),
                     cdim,
                     imgdim,
                     origin=origin)
    craters["x"] = x
    craters["y"] = y