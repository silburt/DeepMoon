import numpy as np
import cv2

def ringmaker(r=10., dr=1):
    """
    Creates ring of radius r and thickness dr.

    Parameters
    ----------
    r : float
        Ring radius
    dr : int
        Ring thickness (cv2.circle requires int)
    """
    # See <http://docs.opencv.org/2.4/modules/core/doc/
    # drawing_functions.html#circle>, supplemented by
    # <http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html>
    # and <https://github.com/opencv/opencv/blob/
    # 05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/imgproc/
    # src/drawing.cpp>.

    # mask grid extent (dr/2 +1 to ensure we capture ring width
    # and radius); same philosophy as above
    rhext = int(np.ceil(r + dr / 2.)) + 1

    # cv2.circle requires integer radius
    mask = np.zeros([2 * rhext + 1, 2 * rhext + 1], np.uint8)

    # Generate ring
    ring = cv2.circle(mask, (rhext, rhext), int(np.round(r)), 1, thickness=int(dr))

    return ring.astype(float)

def circlemaker(r=10.):
    """
    Creates circle mask of radius r.
    """
    # Based on <https://stackoverflow.com/questions/10031580/
    # how-to-write-simple-geometric-shapes-into-numpy-arrays>

    # Mask grid extent (+1 to ensure we capture radius).
    rhext = int(r) + 1

    xx, yy = np.mgrid[-rhext:rhext + 1, -rhext:rhext + 1]
    circle = (xx**2 + yy**2) <= r**2

    return circle.astype(float)

def get_merge_indices(cen, imglen, ks_h, ker_shp):
    """Helper function that returns indices for merging stencil with base
    image, including edge case handling.  x and y are identical, so code is
    axis-neutral.

    Assumes INTEGER values for all inputs!
    """

    left = cen - ks_h
    right = cen + ks_h + 1

    # Handle edge cases.  If left side of stencil is beyond the left end of
    # the image, for example, crop stencil and shift image index to lefthand
    # side.
    if left < 0:
        img_l = 0
        g_l = -left
    else:
        img_l = left
        g_l = 0
    if right > imglen:
        img_r = imglen
        g_r = ker_shp - (right - imglen)
    else:
        img_r = right
        g_r = ker_shp

    return [img_l, img_r, g_l, g_r]

def make_mask(craters, img, rings=False, ringwidth=1,
              truncate=True):
    """Makes crater mask binary image (does not yet consider crater overlap).

    Parameters
    ----------
    craters : pandas.DataFrame
        Craters catalogue that includes pixel x and y columns.
    img : numpy.ndarray
        Original image; assumes colour channel is last axis (tf standard).
    binary : bool, optional
        If True, returns a binary image of crater masks.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.

    Returns
    -------
    mask : numpy.ndarray
        Target mask image.
    """

    # Load blank density map
    imgshape = img.shape[:2]
    mask = np.zeros(imgshape)
    cx = craters["x"].values.astype('int')
    cy = craters["y"].values.astype('int')
    radius = craters["Diameter (pix)"].values / 2.

    for i in range(craters.shape[0]):
        if rings:
            kernel = ringmaker(r=radius[i], dr=ringwidth)
        else:
            kernel = circlemaker(r=radius[i])
        # "Dummy values" so we can use get_merge_indices
        kernel_support = kernel.shape[0]
        ks_half = kernel_support // 2

        # Calculate indices on image where kernel should be added
        [imxl, imxr, gxl, gxr] = get_merge_indices(cx[i], imgshape[1],
                                                   ks_half, kernel_support)
        [imyl, imyr, gyl, gyr] = get_merge_indices(cy[i], imgshape[0],
                                                   ks_half, kernel_support)

        # Add kernel to image
        mask[imyl:imyr, imxl:imxr] += kernel[gyl:gyr, gxl:gxr]

    if truncate:
        if img.ndim == 3:
            mask[img[:, :, 0] == 0] = 0
        else:
            mask[img == 0] = 0

    return mask