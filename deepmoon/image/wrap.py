import collections
import numpy as np
from PIL import Image
import cartopy.img_transform as cimg

def regrid_shape_aspect(regrid_shape, target_extent):
    """Helper function copied from cartopy.img_transform for resizing an image
    without changing its aspect ratio.

    Parameters
    ----------
    regrid_shape : int or float
        Target length of the shorter axis (in units of pixels).
    target_extent : some
        Width and height of the target image (generally not in units of
        pixels).

    Returns
    -------
    regrid_shape : tuple
        Width and height of the target image in pixels.
    """
    if not isinstance(regrid_shape, collections.Sequence):
        target_size = int(regrid_shape)
        x_range, y_range = np.diff(target_extent)[::2]
        desired_aspect = x_range / y_range
        if x_range >= y_range:
            regrid_shape = (target_size * desired_aspect, target_size)
        else:
            regrid_shape = (target_size, target_size / desired_aspect)
    return regrid_shape

def WarpImage(img, iproj, iextent, oproj, oextent,
              origin="upper", rgcoeff=1.2):
    """Warps images with cartopy.img_transform.warp_array, then plots them with
    imshow.  Based on cartopy.mpl.geoaxes.imshow.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    """

    if iproj == oproj:
        raise Warning("Input and output transforms are identical!"
                      "Returing input!")
        return img

    if origin == 'upper':
        # Regridding operation implicitly assumes origin of image is
        # 'lower', so adjust for that here.
        img = img[::-1]

    # rgcoeff is padding when we rescale the image later.
    regrid_shape = rgcoeff * min(img.shape)
    regrid_shape = regrid_shape_aspect(regrid_shape,
                                       oextent)

    # cimg.warp_array uses cimg.mesh_projection, which cannot handle any
    # zeros being used in iextent.  Create iextent_nozeros to fix.
    iextent_nozeros = np.array(iextent)
    iextent_nozeros[iextent_nozeros == 0] = 1e-8
    iextent_nozeros = list(iextent_nozeros)

    imgout, _ = cimg.warp_array(img,
                                     source_proj=iproj,
                                     source_extent=iextent_nozeros,
                                     target_proj=oproj,
                                     target_res=list(map(int, regrid_shape)),
                                     target_extent=oextent,
                                     mask_extrapolated=True)

    if origin == 'upper':
        # Transform back.
        imgout = imgout[::-1]

    return imgout


def WarpImagePad(img,
                 iproj,
                 iextent,
                 oproj,
                 oextent,
                 origin="upper",
                 rgcoeff=1.2,
                 fillbg="black"):
    """Wrapper for WarpImage that adds padding to warped image to make it the
    same size as the original.

    Parameters
    ----------
    img : numpy.ndarray
        Image as a 2D array.
    iproj : cartopy.crs.Projection instance
        Input coordinate system.
    iextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of input.
    oproj : cartopy.crs.Projection instance
        Output coordinate system.
    oextent : list-like
        Coordinate limits (x_min, x_max, y_min, y_max) of output.
    origin : "lower" or "upper", optional
        Based on imshow convention for displaying image y-axis.  "upper" means
        [0,0] is in the upper-left corner of the image; "lower" means it's in
        the bottom-left.
    rgcoeff : float, optional
        Fractional size increase of transformed image height.  Generically set
        to 1.2 to prevent loss of fidelity during transform (though some of it
        is inevitably lost due to warping).
    fillbg : 'black' or 'white', optional.
        Fills padding with either black (0) or white (255) values.  Default is
        black.

    Returns
    -------
    imgo : PIL.Image.Image
        Warped image with padding
    imgw.size : tuple
        Width, height of picture without padding
    offset : tuple
        Pixel width of (left, top)-side padding
    """
    # Based off of <https://stackoverflow.com/questions/2563822/
    # how-do-you-composite-an-image-onto-another-image-with-pil-in-python>

    if type(img) == Image.Image:
        img = np.asanyarray(img)

    # Check that we haven't been given a corrupted image.
    assert img.sum() > 0, "Image input to WarpImagePad is blank!"

    # Set background colour
    if fillbg == "white":
        bgval = 255
    else:
        bgval = 0

    # Warp image.
    imgw = WarpImage(img,
                     iproj,
                     iextent,
                     oproj,
                     oextent,
                     origin=origin,
                     rgcoeff=rgcoeff)

    # Remove mask, turn image into Image.Image.
    imgw = np.ma.filled(imgw, fill_value=bgval)
    imgw = Image.fromarray(imgw, mode="L")

    # Resize to height of original, maintaining aspect ratio.  Note
    # img.shape = height, width, and imgw.size and imgo.size = width, height.
    imgw_loh = imgw.size[0] / imgw.size[1]

    # If imgw is stretched horizontally compared to img.
    if imgw_loh > (img.shape[1] / img.shape[0]):
        imgw = imgw.resize(
            [img.shape[0],
             int(np.round(img.shape[0] / imgw_loh))],
            resample=Image.NEAREST)
    # If imgw is stretched vertically.
    else:
        imgw = imgw.resize(
            [int(np.round(imgw_loh * img.shape[0])), img.shape[0]],
            resample=Image.NEAREST)

    # Make background image and paste two together.
    imgo = Image.new('L', (img.shape[1], img.shape[0]), (bgval))
    offset = ((imgo.size[0] - imgw.size[0]) // 2,
              (imgo.size[1] - imgw.size[1]) // 2)
    imgo.paste(imgw, offset)

    return imgo, imgw.size, offset
