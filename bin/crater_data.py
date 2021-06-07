#!/usr/bin/env python

import os
import sys
import ast
import click
import pathlib
import h5py
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

if __debug__:
    scriptPath = os.path.realpath(os.path.dirname(__file__))
    sys.path.append(scriptPath + "/..")

from deepmoon.craters.reader import ReadLROCHeadCombinedCraterCSV
from deepmoon.craters.resample import ResampleCraters
from deepmoon.image.image import InitialImageCut

from deepmoon.preprocessing.trainingset import GenTrainingSet


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--number",
              "-n",
              help="number of images to create",
              type=int,
              default=30_000)
@click.option("--dist_crop",
              "-c",
              help="Range of image witdh. will be corped in the raw image \
                    For the ortogonal projection a lager number can occure \
                     distortions",
              default="500,6500",
              type=str)
@click.option("--image_size", "-i", help="image size", default=256, type=int)
@click.option("--target_size",
              "-e",
              default=256,
              type=int,
              help="size of target image")
@click.option("--distribution",
              "-d",
              help="distribution to sample from the raw image",
              type=click.Choice(['log', 'uniform']))
@click.option("--diameter_of_crater",
              "-c",
              help="minimal pixel diameter of crater in the images",
              default=1,
              type=float)
@click.option("--planet_radius",
              "-r",
              help='radius of planet in km',
              default=1_737.4,
              type=float)
@click.option("--truncate",
              "-t",
              help="trancate mask where image has pending",
              is_flag=True,
              default=True)
@click.option("--ringwidth",
              "-w",
              help="ring width in pixel",
              default=1,
              type=float)
@click.option("--source_cdim",
              "-y",
              type=str,
              help="source image dimension",
              default="-180,180,-60,60")
@click.option("--sub_cdim",
              "-x",
              type=str,
              help="image region source dimension",
              default="-180,180,-60,60")
@click.option("--ring",
              "-r",
              is_flag=True,
              help="filled circles or rings",
              default=True)
@click.option("--processes",
              "-p",
              help="number of processes",
              default=2,
              type=int)
@click.option('--verbose',
              "-v",
              is_flag=True,
              help='Enables verbose mode',
              default=False)
@click.argument("image_file", type=click.File("rb"))
@click.argument("crater_lroc", type=click.File("rb"))
@click.argument("crater_head", type=click.File("rb"))
@click.argument("output_folder", type=click.Path())
def main(number, dist_crop, target_size, image_size, distribution,
         diameter_of_crater, planet_radius, truncate, ringwidth, source_cdim,
         sub_cdim, ring, processes, verbose, image_file, crater_lroc,
         crater_head, output_folder):
         
    # first check outputfolder
    output_folder = pathlib.Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)

    # parse dist_crop
    dist_crop = list(map(int, [dc.strip() for dc in dist_crop.split(",")]))
    if 0 < len(dist_crop) <= 2:
        if len(dist_crop) == 1:
            dist_crop = tuple(dist_crop[0], dist_crop[0])
        else:
            dist_crop = tuple(dist_crop)
    else:
        raise SystemExit(f"dist_crop need one/two elements")

    # parse source_cdim
    source_cdim = list(
        map(float, [dc.strip() for dc in source_cdim.split(",")]))
    if 2 == len(source_cdim):
        source_cdim = tuple(-1 * source_cdim[0], source_cdim[0],
                            +1 * source_cdim[1], source_cdim[1])
    elif 4 == len(source_cdim):
        source_cdim = tuple(source_cdim)
    else:
        raise SystemExit(f"source_cdim need two/for elements")

    # parse sub_cdim
    sub_cdim = list(map(float, [dc.strip() for dc in sub_cdim.split(",")]))
    if 2 == len(sub_cdim):
        sub_cdim = tuple(-1 * sub_cdim[0], sub_cdim[0], +1 * sub_cdim[1],
                         sub_cdim[1])
    elif 4 == len(sub_cdim):
        sub_cdim = tuple(sub_cdim)
    else:
        raise SystemExit(f"sub_cdim need two/for elements")

    # crater information
    craters = ReadLROCHeadCombinedCraterCSV(filelroc=crater_lroc,
                                            filehead=crater_head)

    # image
    img = Image.open(image_file).convert("L")

    # subdimenson check
    if sub_cdim != source_cdim:
        img = InitialImageCut(img, source_cdim, sub_cdim)

    # This always works, since sub_cdim < source_cdim.
    craters = ResampleCraters(craters, sub_cdim, None, arad=planet_radius)

    GenTrainingSet(img=img,
                   craters=craters,
                   output=output_folder,
                   rawlen_range=dist_crop,
                   rawlen_dist=distribution,
                   ilen=image_size,
                   tglen=target_size,
                   cdim=sub_cdim,
                   arad=planet_radius,
                   minpix=diameter_of_crater,
                   rings=ring,
                   ringwidth=ringwidth,
                   truncate=truncate,
                   amt=number,
                   verbose=verbose,
                   processes=processes)


if __name__ == '__main__':
    main()
