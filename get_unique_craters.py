#!/usr/bin/env python
"""Unique Crater Distribution Functions

Functions for extracting craters from model target predictions and filtering
out duplicates.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import h5py
import sys
import utils.template_match_target as tmt
import utils.processing as proc
import utils.transform as trf
from keras.models import load_model

#########################
def get_model_preds(CP):
    """Reads in or generates model predictions.

    Parameters
    ----------
    CP : dict
        Containins directory locations for loading data and storing
        predictions.

    Returns
    -------
    craters : h5py
        Model predictions.
    """
    n_imgs, dtype = CP['n_imgs'], CP['datatype']

    data = h5py.File(CP['dir_data'], 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    proc.preprocess(Data)

    model = load_model(CP['dir_model'])
    preds = model.predict(Data[dtype][0])

    # save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    print("Successfully generated and saved model predictions.")
    return preds

#########################
def add_unique_craters(craters, craters_unique, thresh_longlat2, thresh_rad2):
    """Generates unique crater distribution by filtering out duplicates.

    Parameters
    ----------
    craters : array
        Crater tuples from a single image in the form (long, lat, radius).
    craters_unique : array
        Master array of unique crater tuples in the form (long, lat, radius)
    thresh_longlat2 : float.
        Hyperparameter that controls the minimum squared longitude/latitude
        difference between craters to be considered unique entries.
    thresh_rad2 : float
        Hyperparaeter that controls the minimum squared radius difference
        between craters to be considered unique entries.

    Returns
    -------
    craters_unique : array
        Modified master array of unique crater tuples with new crater entries.
    """
    km_to_deg = 180. / (np.pi * 1737.4)
    Long, Lat, Rad = craters_unique.T
    for j in range(len(craters)):
        lo, la, r = craters[j].T
        # Fractional long/lat change
        diff_longlat = ((Long - lo)**2 + (Lat - la)**2) / (r * km_to_deg)**2
        Rad_ = Rad[diff_longlat < thresh_longlat2]
        if len(Rad_) > 0:
            # Fractional radius change
            diff_rad = ((Rad_ - r) / r)**2
            index = diff_rad < thresh_rad2
            if len(np.where(index == True)[0]) == 0:
                craters_unique = np.vstack((craters_unique, craters[j]))
        else:
            craters_unique = np.vstack((craters_unique, craters[j]))
    return craters_unique

#########################
def estimate_longlatdiamkm(dim, llbd, distcoeff, coords):
    """First-order estimation of long/lat, and radius (km) from
    (Orthographic) x/y position and radius (pix).

    For images transformed from ~6000 pixel crops of the 30,000 pixel
    LROC-Kaguya DEM, this results in < ~0.4 degree latitude, <~0.2
    longitude offsets (~2% and ~1% of the image, respectively) and ~2% error in
    radius. Larger images thus may require an exact inverse transform,
    depending on the accuracy demanded by the user.

    Parameters
    ----------
    dim : tuple or list
        (width, height) of input images.
    llbd : tuple or list
        Long/lat limits (long_min, long_max, lat_min, lat_max) of image.
    distcoeff : float
        Ratio between the central heights of the transformed image and original
        image.
    coords : numpy.ndarray
        Array of crater x coordinates, y coordinates, and pixel radii.

    Returns
    -------
    craters_longlatdiamkm : numpy.ndarray
        Array of crater longitude, latitude and radii in km.
    """
    # Expand coords.
    long_pix, lat_pix, radii_pix = coords.T

    # Determine radius (km).
    km_per_pix = 1. / trf.km2pix(dim[1], llbd[3] - llbd[2], dc=distcoeff)
    radii_km = radii_pix * km_per_pix

    # Determine long/lat.
    deg_per_pix = km_per_pix * 180. / (np.pi * 1737.4)
    long_central = 0.5 * (llbd[0] + llbd[1])
    lat_central = 0.5 * (llbd[2] + llbd[3])

    # Iterative method for determining latitude.
    lat_deg_firstest = lat_central - deg_per_pix * (lat_pix - dim[1] / 2.)
    latdiff = abs(lat_central - lat_deg_firstest)
    lat_deg = lat_central - (deg_per_pix * (lat_pix - dim[1] / 2.) *
                             (np.pi * latdiff / 180.) /
                             np.sin(np.pi * latdiff / 180.))
    # Determine longitude using determined latitude.
    long_deg = long_central + (deg_per_pix * (long_pix - dim[0] / 2.) /
                               np.cos(np.pi * lat_deg / 180.))

    # Return combined long/lat/radius array.
    return np.column_stack((long_deg, lat_deg, radii_km))


def extract_unique_craters(CP, craters_unique):
    """Top level function that extracts craters from model predictions,
    converts craters from pixel to real (degree, km) coordinates, and filters
    out duplicate detections across images.

    Parameters
    ----------
    CP : dict
        Crater Parameters needed to run the code.
    craters_unique : array
        Empty master array of unique crater tuples in the form 
        (long, lat, radius).

    Returns
    -------
    craters_unique : array
        Filled master array of unique crater tuples.
    """

    # Load/generate model preds
    try:
        preds = h5py.File(CP['dir_preds'], 'r')[CP['datatype']]
        print("Loaded model predictions successfully")
    except:
        print("Couldnt load model predictions, generating")
        preds = get_model_preds(CP)

    # need for long/lat bounds
    P = h5py.File(CP['dir_data'], 'r')
    llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds',
                            'pix_distortion_coefficient')
    #r_moon = 1737.4
    dim = float(CP['dim'])

    N_matches_tot = 0
    for i in range(CP['n_imgs']):
        id = proc.get_id(i)

        # sloped minrad
        rawlen = P[pbd][proc.get_id(i)][2] - P[pbd][proc.get_id(i)][0]
        if rawlen < 4000:
            minrad = int((3. / 1000.) * rawlen - 3)
            coords = tmt.template_match_t(preds[i], minrad=max(minrad, 3))
        elif rawlen >= 4000:
            coords = tmt.template_match_t(preds[i], minrad=9)

        # convert, add to master dist
        if len(coords) > 0:

            new_craters_unique = estimate_longlatdiamkm(
                dim, P[llbd][id], P[distcoeff][id][0], coords)
            N_matches_tot += len(coords)

            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique = add_unique_craters(new_craters_unique,
                                                    craters_unique,
                                                    CP['llt2'], CP['rt2'])
            else:
                craters_unique = np.concatenate((craters_unique,
                                                 new_craters_unique))

    np.save(CP['dir_result'], craters_unique)
    return craters_unique
