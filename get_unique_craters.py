#!/usr/bin/env python
"""Unique Crater Distribution Functions

Functions for extracting craters from model target predictions and filtering
out duplicates.
"""

import numpy as np
import h5py
from utils.template_match_target import *
from utils.processing import *
import sys
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
    preprocess(Data)

    model = load_model(CP['model'])
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
            if len(np.where(index is True)[0]) == 0:
                craters_unique = np.vstack((craters_unique, craters[j]))
        else:
            craters_unique = np.vstack((craters_unique, craters[j]))
    return craters_unique

#########################
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
    r_moon = 1737.4
    dim = float(CP['dim'])

    N_matches_tot = 0
    for i in range(CP['n_imgs']):
        print i, len(craters_unique)
        id = get_id(i)

        # sloped minrad
        rawlen = P[pbd][get_id(i)][2] - P[pbd][get_id(i)][0]
        if rawlen < 4000:
            minrad = int((3. / 1000.) * rawlen - 3)
            coords = template_match_t(preds[i], minrad=max(minrad, 3))
        elif rawlen >= 4000:
            coords = template_match_t(preds[i], minrad=9)

        # convert, add to master dist
        if len(coords) > 0:
            pix_to_km = ((P[llbd][id][3] - P[llbd][id][2]) *
                         (np.pi / 180.0) * r_moon / dim)
            long_pix, lat_pix, radii_pix = coords.T
            radii_km = radii_pix * pix_to_km
            long_central = 0.5 * (P[llbd][id][0] + P[llbd][id][1])
            lat_central = 0.5 * (P[llbd][id][2] + P[llbd][id][3])
            deg_per_pix = ((P[llbd][id][3] - P[llbd][id][2]) / dim /
                           P[distcoeff][id][0])
            lat_deg = lat_central - deg_per_pix * (lat_pix - 128.)
            long_deg = long_central + (deg_per_pix * (long_pix - 128.) /
                                       np.cos(np.pi * lat_deg / 180.))
            tuple_ = np.column_stack((long_deg, lat_deg, radii_km))
            N_matches_tot += len(coords)

            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique = add_unique_craters(tuple_, craters_unique,
                                                    CP['llt2'], CP['rt2'])
            else:
                craters_unique = np.concatenate((craters_unique, tuple_))

    np.save(CP['dir_result'], craters_unique)
    return craters_unique

#########################
if __name__ == '__main__':
    # Arguments
    CP = {}
    CP['datatype'] = 'dev'
    CP['n_imgs'] = 30000
    CP['dim'] = 256

    # Hyperparameters
    CP['llt2'] = float(sys.argv[1])    # D_{L,L} from Silburt et. al (2017)
    CP['rt2'] = float(sys.argv[2])     # D_{R} from Silburt et. al (2017)

    CP['dir_data'] = 'datasets/HEAD/%s_images_final.hdf5' % CP['datatype']
    CP['dir_preds'] = 'datasets/HEAD/HEAD_%spreds_n%d_final.hdf5' % (
                      (CP['datatype'], CP['n_imgs']))
    CP['dir_result'] = 'datasets/HEAD/HEAD_%s_craterdist_llt%.2f_rt%.2f_' \
                       'final.npy' % (CP['datatype'], CP['llt2'], CP['rt2'])

    # Needed to generate model_preds if they don't exist yet
    CP['model'] = 'models/HEAD_final.h5'

    craters_unique = np.empty([0, 3])
    craters_unique = extract_unique_craters(CP, craters_unique)
