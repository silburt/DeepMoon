"""Coordinate Transform Functions

Functions for coordinate transforms, used by input_data_gen.py functions.
"""
import numpy as np

########## Coordinates to pixels projections ##########

def coord2pix(cx, cy, cdim, imgdim, origin="upper"):
    """Converts coordinate x/y to image pixel locations.

    Parameters
    ----------
    cx : float or ndarray
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    """

    x = imgdim[0] * (cx - cdim[0]) / (cdim[1] - cdim[0])

    if origin == "lower":
        y = imgdim[1] * (cy - cdim[2]) / (cdim[3] - cdim[2])
    else:
        y = imgdim[1] * (cdim[3] - cy) / (cdim[3] - cdim[2])

    return x, y


def pix2coord(x, y, cdim, imgdim, origin="upper"):
    """Converts image pixel locations to Plate Carree lat/long.  Assumes
    central meridian is at 0 (so long in [-180, 180)).

    Parameters
    ----------
    x : float or ndarray
        Pixel x positions.
    y : float or ndarray
        Pixel y positions.
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    imgdim : list, tuple or ndarray
        Length and height of image, in pixels.
    origin : 'upper' or 'lower', optional
        Based on imshow convention for displaying image y-axis. 'upper' means
        that [0, 0] is upper-left corner of image; 'lower' means it is
        bottom-left.

    Returns
    -------
    cx : float or ndarray
        Coordinate x.
    cy : float or ndarray
        Coordinate y.
    """

    cx = (x / imgdim[0]) * (cdim[1] - cdim[0]) + cdim[0]

    if origin == "lower":
        cy = (y / imgdim[1]) * (cdim[3] - cdim[2]) + cdim[2]
    else:
        cy = cdim[3] - (y / imgdim[1]) * (cdim[3] - cdim[2])

    return cx, cy

########## Metres to pixels conversion ##########

def km2pix(imgheight, latextent, dc=1., a=1737.4):
    """Returns conversion from km to pixels (i.e. pix / km).

    Parameters
    ----------
    imgheight : float
        Height of image in pixels.
    latextent : float
        Latitude extent of image in degrees.
    dc : float from 0 to 1, optional
        Scaling factor for distortions.
    a : float, optional
        World radius in km.  Default is Moon (1737.4 km).

    Returns
    -------
    km2pix : float
        Conversion factor pix/km
    """
    return (180. / np.pi) * imgheight * dc / latextent / a
