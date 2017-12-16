#!/usr/bin/env python
"""Input Image Dataset Generator

Script for generating input datasets from Lunar global digital elevation maps 
(DEMs) and crater catalogs.

This script is designed to use the LRO-Kaguya DEM and a combination of the
LOLA-LROC 5 - 20 km and Head et al. 2010 >=20 km crater catalogs.  It
generates a randomized set of small (projection-corrected) images and
corresponding crater targets.  The input and target image sets are stored as
hdf5 files.  The longitude and latitude limits of each image is included in the
input set file, and tables of the craters in each image are stored in a
separate Pandas HDFStore hdf5 file.

The script's parameters are located under the Global Variables.  We recommend
making a copy of this script when generating a dataset.

MPI4py can be used to generate more images - each thread is given `amt` number
of images to generate.  Comment out the MPI code block below to run on systems
where it's not installed.
"""

########## Imports ##########

# Past-proofing
from __future__ import absolute_import, division, print_function

from PIL import Image
import make_input_data as mkin
import time

########## Global Variables ##########

# Use MPI4py?  Set this to False if it's not supposed by the system.
use_mpi4py = False

# Source image path.
source_image_path = "/home/cczhu/public_html/LunarLROLrocKaguya_118mperpix.png"

# LROC crater catalog csv path.
lroc_csv_path = "./LROCCraters.csv"

# Head et al. catalog csv path.
head_csv_path = "./HeadCraters.csv"

# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5" and
# "./out/train_targets.hdf5"
outhead = "./input_data/train"

# Number of images to make (if using MPI4py, number of image per thread to
# make).
amt = 30000

# Range of image widths, in pixels, to crop from source image (input images
# will be scaled down to ilen). For Orthogonal projection, larger images are
# distorted at their edges, so there is some trade-off between ensuring images
# have minimal distortion, and including the largest craters in the image.
rawlen_range = [300, 4000]

# Distribution to sample from rawlen_range - "uniform" for uniform, and "log"
# for loguniform.
rawlen_dist = 'log'

# Size of input images.
ilen = 256

# Size of target images.
tglen = 256

# [Min long, max long, min lat, max lat] dimensions of source image.
source_cdim = [-180., 180., -60., 60.]

# [Min long, max long, min lat, max lat] dimensions of the region of the source
# to use when randomly cropping.  Used to distinguish training from test sets.
sub_cdim = [-180., 180., -60., 60.]

# Minimum pixel diameter of craters to include in in the target.
minpix = 1.

# Radius of the world in km (1737.4 for Moon).
R_km = 1737.4

### Target mask arguments. ###

# If True, truncate mask where image has padding.
truncate = True

# If rings = True, thickness of ring in pixels.
ringwidth = 1

# If True, script prints out the image it's currently working on.
verbose = True

########## Script ##########

if __name__ == '__main__':

    start_time = time.time()

    # Utilize mpi4py for multithreaded processing.
    if use_mpi4py:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print("Thread {0} of {1}".format(rank, size))
        istart = rank * amt
    else:
        istart = 0

    # Read source image and crater catalogs.
    img = Image.open(source_image_path).convert("L")
    craters = mkin.ReadLROCHeadCombinedCraterCSV(filelroc=lroc_csv_path,
                                                 filehead=head_csv_path)

    # Sample subset of image.  Co-opt mkin.ResampleCraters to remove all
    # craters beyond cdim (either sub or source).
    if sub_cdim != source_cdim:
        img = mkin.InitialImageCut(img, source_cdim, sub_cdim)
    # This always works, since sub_cdim < source_cdim.
    craters = mkin.ResampleCraters(craters, sub_cdim, None, arad=R_km)

    # Generate input images.
    mkin.GenDataset(img, craters, outhead, rawlen_range=rawlen_range,
                    rawlen_dist=rawlen_dist, ilen=ilen, cdim=sub_cdim,
                    arad=R_km, minpix=minpix, tglen=tglen, binary=True,
                    rings=True, ringwidth=ringwidth, truncate=truncate,
                    amt=amt, istart=istart, verbose=verbose)

    elapsed_time = time.time() - start_time
    if verbose:
        print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))
