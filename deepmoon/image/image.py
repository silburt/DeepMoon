import numpy as np
from deepmoon.coord.transformation import coord2pix


def InitialImageCut(img, cdim, newcdim):
    """Crops image, so that the crop output can be used in GenDataset.

    Parameters
    ----------
    img : PIL.Image.Image
        Image
    cdim : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of image.
    newcdim : list-like
        Crop boundaries (x_min, x_max, y_min, y_max).  There is
        currently NO CHECK that newcdim is within cdim!

    Returns
    -------
    img : PIL.Image.Image
        Cropped image
    """
    x, y = coord2pix(np.array(newcdim[:2]),
                     np.array(newcdim[2:]),
                     cdim,
                     img.size,
                     origin="upper")

    # y is backward since origin is upper!
    box = [x[0], y[1], x[1], y[0]]
    img = img.crop(box)
    img.load()

    return img