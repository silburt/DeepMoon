#The point of this script is to take the outputted numpy files generated from crater_distribution_extract_*.py and generate a list of unique craters, i.e. no duplicates. The key hyperparameters to tune are thresh_longlat2 and thresh_rad2, which is guided by comparing the unique distirbution to the ground truth (alanalldata.csv) data.
#First you need to generate predictions from crater_distribution_get_modelpreds.py

import numpy as np, h5py
from utils.template_match_target import *
from utils.processing import *
import glob
from keras.models import load_model
import os

#########################
def get_model_preds(CP):
    dim, n_imgs, dtype = CP['dim'], CP['n_imgs'], CP['datatype']

    data = h5py.File('%s/%s_images.hdf5'%(CP['dir_data'],dtype), 'r')

    Data = {
        dtype: [data['input_images'][:n_imgs].astype('float32'),
                data['target_masks'][:n_imgs].astype('float32')]
    }
    data.close()
    preprocess(Data)

    model = load_model(CP['model'])
    preds = model.predict(Data[dtype][0])
    
    #save
    h5f = h5py.File(CP['dir_preds'], 'w')
    h5f.create_dataset(dtype, data=preds)
    print "Successfully generated and saved model predictions."
    return preds

#########################
def add_unique_craters(tuple, crater_dist, thresh_longlat2, thresh_rad2):
    Long, Lat, Rad = crater_dist.T
    for j in range(len(tuple)):
        lo,la,r = tuple[j].T
        diff_longlat = (Long - lo)**2 + (Lat - la)**2
        Rad_ = Rad[diff_longlat < thresh_longlat2]
        if len(Rad_) > 0:
            diff_rad = ((Rad_ - r)/r)**2                #fractional radius change
            index = diff_rad < thresh_rad2
            if len(np.where(index==True)[0]) == 0:      #unique value determined from long/lat, then rad
                crater_dist = np.vstack((crater_dist,tuple[j]))
        else:                                           #unique value determined from long/lat alone
            crater_dist = np.vstack((crater_dist,tuple[j]))
    return crater_dist

#########################
def extract_crater_dist(CP, pred_crater_dist):
    
    #load/generate model preds
    try:
        preds = h5py.File(CP['dir_preds'],'r')[CP['datatype']]
        print "Loaded model predictions successfully"
    except:
        print "Couldnt load model predictions, generating"
        preds = get_model_preds(CP)
    
    # need for long/lat bounds
    P = h5py.File('%s/%s_images.hdf5'%(CP['dir_data'],CP['datatype']), 'r')
    llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds',
                            'pix_distortion_coefficient')

    # Radius of the moon (km).
    r_moon = 1737.4
    # Image dimension (pixels, assume dim=height=width), needs to be float.
    dim = float(CP['dim'])

    N_matches_tot = 0
    for i in range(CP['n_imgs']):
        print i, len(pred_crater_dist)
        coords = template_match_target(preds[i])
        if len(coords) > 0:
            id = get_id(i)
            pix_to_km = ((P[llbd][id][3] - P[llbd][id][2]) *
                         (np.pi / 180.0) * r_moon / dim)
            long_pix, lat_pix, radii_pix = coords.T
            radii_km = radii_pix * pix_to_km
            long_central = 0.5 * (P[llbd][id][0] + P[llbd][id][1])
            lat_central = 0.5 * (P[llbd][id][2] + P[llbd][id][3])
            deg_per_pix = ((P[llbd][id][3] - P[llbd][id][2]) / dim /
                           distortion_coeff)
            lat_deg = lat_central - deg_per_pix * (lat_pix - 128.)
            long_deg = long_central + (deg_per_pix * (long_pix - 128.) /
                                       np.cos(np.pi * lat_deg / 180.))
            tuple_ = np.column_stack((long_deg, lat_deg, radii_km))
            N_matches_tot += len(coords)

            #only add unique (non-duplicate) values to the master pred_crater_dist
            if len(pred_crater_dist) > 0:
                pred_crater_dist = add_unique_craters(tuple_, pred_crater_dist, CP['llt2'], CP['rt2'])
            else:
                pred_crater_dist = np.concatenate((pred_crater_dist,tuple_))

    np.save(CP['dir_result'],pred_crater_dist)
    return pred_crater_dist

#########################
if __name__ == '__main__':
    # Arguments
    CP = {}
    #CP['dir_data'] = '/scratch/m/mhvk/czhu/newscripttest_for_ari'     #exclude final '/' in path
    CP['dir_data'] = 'datasets/HEAD'
    
    CP['datatype'] = 'test'
    CP['n_imgs'] = 30000
    CP['dir_preds'] = 'datasets/HEAD/HEAD_%spreds_n%d.hdf5'%(CP['datatype'],CP['n_imgs'])
    CP['dir_result'] = 'datasets/HEAD/HEAD_%s_craterdist_n%d.npy'%(CP['datatype'], CP['n_imgs'])
    
    #Needed to generate model_preds if they don't exist yet
    CP['model'] = 'models/HEAD.h5'
    CP['dim'] = 256
    
    # Tuned Hyperparameters - Shouldn't really change
    CP['llt2'] = 0.6    #D_{L,L} from Silburt et. al (2017)
    CP['rt2'] = 0.6     #D_{R} from Silburt et. al (2017)

    pred_crater_dist = np.empty([0,3])
    pred_crater_dist = extract_crater_dist(CP, pred_crater_dist)
