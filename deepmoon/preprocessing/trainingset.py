from pathlib import Path
import json

from functools import partial
from tqdm import tqdm
from multiprocessing import Lock, Pool, cpu_count

import numpy as np
import cartopy.crs as ccrs
from PIL import Image

from deepmoon.craters.resample import ResampleCraters
from deepmoon.coord.plateCarree_xy import AddPlateCarree_XY
from deepmoon.coord.transformation import pix2coord
from deepmoon.craters.carree2ortographic import PlateCarree_to_Orthographic
from deepmoon.image.mask import make_mask

lock = Lock()

def GenTrainingSet(img,
                   craters,
                   output,
                   rawlen_range=[1_000, 2_000],
                   rawlen_dist='log',
                   ilen=256,
                   cdim=[-180., 180., -60., 60.],
                   arad=1_737.4,
                   minpix=0,
                   tglen=256,
                   rings=True,
                   ringwidth=1,
                   truncate=True,
                   amt=100,
                   seed=None,
                   verbose=False,
                   processes=cpu_count()):
    """Generates random dataset from a global DEM and crater catalogue.

    The function randomly samples small images from a global digital elevation
    map (DEM) that uses a Plate Carree projection, and converts the small
    images to Orthographic projection.  Pixel coordinates and radii of craters
    from the catalogue that fall within each image are placed in a
    corresponding Pandas dataframe.  Images and dataframes are saved to disk in
    hdf5 format.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image.
    craters : pandas.DataFrame
        Crater catalogue .csv.
    outhead : str
        Filepath and file prefix of the image and crater table hdf5 files.
    rawlen_range : list-like, optional
        Lower and upper bounds of raw image widths, in pixels, to crop from
        source.  To always crop the same sized image, set lower bound to the
        same value as the upper.  Default is [300, 4000].
    rawlen_dist : 'uniform' or 'log'
        Distribution from which to randomly sample image widths.  'uniform' is
        uniform sampling, and 'log' is loguniform sampling.
    ilen : int, optional
        Input image width, in pixels.  Cropped images will be downsampled to
        this size.  Default is 256.
    cdim : list-like, optional
        Coordinate limits (x_min, x_max, y_min, y_max) of image.  Default is
        LRO-Kaguya's [-180., 180., -60., 60.].
    arad : float. optional
        World radius in km.  Defaults to Moon radius (1737.4 km).
    minpix : int, optional
        Minimum crater diameter in pixels to be included in crater list.
        Useful when the smallest craters in the catalogue are smaller than 1
        pixel in diameter.
    tglen : int, optional
        Target image width, in pixels.
    rings : bool, optional
        If True, mask uses hollow rings rather than filled circles.
    ringwidth : int, optional
        If rings is True, ringwidth sets the width (dr) of the ring.
    truncate : bool
        If True, truncate mask where image truncates.
    amt : int, optional
        Number of images to produce.  100 by default.
    istart : int
        Output file starting number, when creating datasets spanning multiple
        files.
    seed : int or None
        np.random.seed input (for testing purposes).
    verbose : bool
        If True, prints out number of image being generated.
    """

    # just in case we ever make this user-selectable...
    origin = "upper"

    # Seed random number generator.
    np.random.seed(seed)

    # Get craters.
    AddPlateCarree_XY(craters, list(img.size), cdim=cdim, origin=origin)

    iglobe = ccrs.Globe(semimajor_axis=arad * 1000.,
                        semiminor_axis=arad * 1000.,
                        ellipse=None)

    # Create random sampler (either uniform or loguniform).
    if rawlen_dist == 'log':
        rawlen_min = np.log10(rawlen_range[0])
        rawlen_max = np.log10(rawlen_range[1])

        def random_sampler():
            return int(10**np.random.uniform(rawlen_min, rawlen_max))
    else:

        def random_sampler():
            return np.random.randint(rawlen_range[0], rawlen_range[1] + 1)

    data_collection = list()
    with Pool(processes=processes) as pros:
        for i in tqdm(range(amt)):
            data_collection.append(
                run_element(i, img, cdim, origin, verbose, ilen, tglen,
                            random_sampler, craters, arad, minpix, output,
                            iglobe, rings, ringwidth, truncate))

    with open(f"{output}/data_rec.json", 'w', encoding='utf-8') as json_file:
        json.dump(data_collection, json_file, ensure_ascii=False, indent=4)


def run_element(i, img, cdim, origin, verbose, ilen, tglen, random_sampler,
                craters, arad, minpix, output, iglobe, rings, ringwidth,
                truncate):
    # Current image number.
    img_number = f"{i}"
    if verbose:
        print(f"Generating {img_number}", end="\r")

    # Determine image size to crop.
    rawlen = random_sampler()
    xc = np.random.randint(0, img.size[0] - rawlen)
    yc = np.random.randint(0, img.size[1] - rawlen)
    box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')

    # Load necessary because crop may be a lazy operation; im.load() should
    # copy it.  See <http://pillow.readthedocs.io/en/3.1.x/
    # reference/Image.html>.
    im = img.crop(box)
    im.load()

    # Obtain long/lat bounds for coordinate transform.
    ix = box[::2]
    iy = box[1::2]
    llong, llat = pix2coord(ix, iy, cdim, list(img.size), origin=origin)
    llbd = np.r_[llong, llat[::-1]]

    # Downsample image.
    im = im.resize([ilen, ilen], resample=Image.NEAREST)

    # Remove all craters that are too small to be seen in image.
    ctr_sub = ResampleCraters(craters,
                              llbd,
                              im.size[1],
                              arad=arad,
                              minpix=minpix)

    # Convert Plate Carree to Orthographic.
    [imgo, ctr_xy, distortion_coefficient,
     clonglat_xy] = (PlateCarree_to_Orthographic(im,
                                                 llbd,
                                                 ctr_sub,
                                                 iglobe=iglobe,
                                                 ctr_sub=True,
                                                 arad=arad,
                                                 origin=origin,
                                                 rgcoeff=1.2,
                                                 slivercut=0.5))
    if imgo is None:
        print("Discarding narrow image")
        return None

    imgo_arr = np.asanyarray(imgo)
    assert imgo_arr.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                "an error in projecting the cropped "
                                "image.")

    # Make target mask.  Used Image.BILINEAR resampling because
    # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
    # leaves artifacts).
    tgt = np.asanyarray(imgo.resize((tglen, tglen), resample=Image.BILINEAR))
    mask = make_mask(ctr_xy,
                     tgt,
                     rings=rings,
                     ringwidth=ringwidth,
                     truncate=truncate)

    # Output everything to file.
    image_file = f"{img_number}.png"
    output_path = Path(output)

    image_out = output_path / image_file
    imgo_arr = Image.fromarray(imgo_arr)
    imgo_arr.save(image_out)

    mask_out = output_path / "mask"
    if not mask_out.is_dir():
        mask_out.mkdir(parents=True, exist_ok=True)
    mask_out /= image_file
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask.save(mask_out)

    combined_out = output_path / "comp"
    if not combined_out.is_dir():
        combined_out.mkdir(parents=True, exist_ok=True)
    combined_out /= image_file
    imgo_arr.paste(mask, (0, 0), mask)
    imgo_arr.save(combined_out)

    crater_out = output_path / "crater"
    if not crater_out.is_dir():
        crater_out.mkdir(parents=True, exist_ok=True)
    ctr_xy.to_pickle(f"{crater_out}/{img_number}.pkl")

    with lock:
        return {
            "name": img_number,
            "distortion_coefficient": distortion_coefficient,
            "llbd": llbd.tolist(),
            "clonglat_xy": clonglat_xy.to_dict('records')[0],
            "box": box.tolist(),
        }
