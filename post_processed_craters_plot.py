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
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#np.random.seed(42)

#########################
def get_GT(truth_datatype, minrad=0, maxrad=100):
    #prepare GT
    truthalan = pd.read_csv('catalogues/LROCCraters.csv')
    truthhead = pd.read_csv('catalogues/HeadCraters.csv')
    truthhead = truthhead[(truthhead['Lat']>=-60)&(truthhead['Lat']<=60)&(truthhead['Diam_km']>2*minrad)&(truthhead['Diam_km']<2*maxrad)]
    truthalan = truthalan[(truthalan['Lat']>=-60)&(truthalan['Lat']<=60)&(truthalan['Diameter (km)']>2*minrad)&(truthalan['Diameter (km)']<2*maxrad)]
    if truth_datatype == 'train':
        truthalan = truthalan[truthalan['Long']<-60]        #region of train data
        truthhead = truthhead[(truthhead['Lon']<-60)&(truthhead['Diam_km']>20.)]
    elif truth_datatype == 'test':
        truthalan = truthalan[truthalan['Long']>60]        #region of test data
        truthhead = truthhead[(truthhead['Lon']>60)&(truthhead['Diam_km']>20.)]
    elif truth_datatype == 'dev':
        truthalan = truthalan[(truthalan['Long']>-60)&(truthalan['Long']<60)]        #region of test data
        truthhead = truthhead[(truthhead['Lon']<60)&(truthhead['Lon']>-60)&(truthhead['Diam_km']>20.)]
    return np.array((np.concatenate((truthalan['Long'].values, truthhead['Lon'].values)),
                     np.concatenate((truthalan['Lat'].values, truthhead['Lat'].values)),
                     np.concatenate((truthalan['Diameter (km)'].values/2.,truthhead['Diam_km'].values/2.)))).T

#########################
def preprocess(imgs, dim=256, low=0.1, hi=1.0):
    for img in imgs:
        minn, maxx = np.min(img[img > 0]), np.max(img[img > 0])
        img[img > 0] = low + (img[img > 0] - minn) * (hi - low) / (maxx - minn)
    return imgs

#########################
def new_crater_check(lo, la, r, GTLong, GTLat, GTRad, thresh_longlat2, thresh_rad2):
    km_to_deg = 180. / (np.pi * 1737.4)
    dL = ((GTLong - lo)**2 + (GTLat - la)**2) / (r * km_to_deg)**2
    dR = ((GTRad - r) / r)**2
    index = (dL < thresh_longlat2) & (dR < thresh_rad2)
    N_match = len(np.where(index == True)[0])
    return N_match

#########################
def add_unique_craters(craters, craters_unique, GT, thresh_longlat2, thresh_rad2):
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
    craters_unique_index = []
    
    km_to_deg = 180. / (np.pi * 1737.4)
    Long, Lat, Rad = craters_unique.T
    GTLong, GTLat, GTRad = GT.T
    for j in range(len(craters)):
        lo, la, r = craters[j].T
        # Fractional long/lat change
        dL = ((Long - lo)**2 + (Lat - la)**2) / (r * km_to_deg)**2
        Rad_ = Rad[(dL < thresh_longlat2)]
        if len(Rad_) > 0:
            # Fractional radius change
            dR = ((Rad_ - r) / r)**2
            index = dR < thresh_rad2
            if len(np.where(index == True)[0]) == 0:
                craters_unique = np.vstack((craters_unique, craters[j]))
                N_match = new_crater_check(lo, la, r, GTLong, GTLat, GTRad, thresh_longlat2, thresh_rad2)
                if N_match == 0:
                    craters_unique_index.append(j)
        else:
            craters_unique = np.vstack((craters_unique, craters[j]))
            N_match = new_crater_check(lo, la, r, GTLong, GTLat, GTRad, thresh_longlat2, thresh_rad2)
            if N_match == 0:
                craters_unique_index.append(j)
    return craters_unique, craters_unique_index

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
    # Protect against latdiff = 0 situation.
    latdiff[latdiff < 1e-7] = 1e-7
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
    preds = h5py.File(CP['dir_preds'], 'r')[CP['datatype']]
    print("Loaded model predictions successfully")
    
    # need for long/lat bounds
    P = h5py.File(CP['dir_data'], 'r')
    llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds',
                            'pix_distortion_coefficient')
    dim = (float(CP['dim']), float(CP['dim']))
                            
    # prepare images, detect craters
    imgs = preprocess(P['input_images'][:CP['n_imgs']].astype('float32'))
    GT = get_GT(CP['datatype'])
    rand = np.random.randint(0,CP['n_imgs'],100)
        
    N_matches_tot = 0
    for i in range(CP['n_imgs']):
        id = proc.get_id(i)
                                    
        coords = tmt.template_match_t(preds[i], minrad=CP['mr'])
                                        
        # convert, add to master dist
        if len(coords) > 0:
            
            new_craters_unique = estimate_longlatdiamkm(dim, P[llbd][id], P[distcoeff][id][0], coords)
            N_matches_tot += len(coords)
                    
            # Only add unique (non-duplicate) craters
            if len(craters_unique) > 0:
                craters_unique, craters_unique_index = add_unique_craters(new_craters_unique, craters_unique, GT, CP['llt2'], CP['rt2'])
                coords_unique = coords[craters_unique_index]
            else:
                craters_unique = np.concatenate((craters_unique, new_craters_unique))
                coords_unique = []
                                            
            if len(coords_unique) > 0:
                f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=[25, 10])
                img = imgs[i].reshape(256,256)
                ax1.imshow(img, origin='upper', cmap="Greys_r")
                ax2.imshow(img, origin='upper', cmap="Greys_r")
                                
                x, y, r = coords_unique.T
                for k in range(len(x)):
                    circle = plt.Circle((x[k], y[k]), r[k], color='blue', fill=False, linewidth=2, alpha=0.5)
                    ax2.add_artist(circle)
                ax3.imshow(preds[i], origin='upper', cmap="Greys_r")
                fontsize = 30
                ax1.set_title('Moon Image', fontsize=fontsize)
                ax2.set_title('%d new CNN-Detected Craters'%len(x), fontsize=fontsize)
                ax3.set_title('CNN-Pred', fontsize=fontsize)
                plt.savefig('post_processed_imgs/%d.png'%i)
                plt.close()

    np.save(CP['dir_result'], craters_unique)
    return craters_unique

if __name__ == '__main__':
    # Crater Parameters
    CP = {}
    
    # Image width/height, assuming square images.
    CP['dim'] = 256
    
    # Data type - train, dev, test
    CP['datatype'] = 'test'
    
    # Number of images to extract craters from
    CP['n_imgs'] = 100
    
    # Hyperparameters
    CP['llt2'] = 1.80    # D_{L,L} from Silburt et. al (2017)
    CP['rt2'] = 0.40     # D_{R} from Silburt et. al (2017)
    CP['mr'] = 5
    
    # Location of where hdf5 data images are stored
    CP['dir_data'] = '../moon-craters/datasets/HEAD/%s_images_final.hdf5' % CP['datatype']
    
    # Location of where model predictions are/will be stored
    CP['dir_preds'] = '../moon-craters/datasets/HEAD/HEAD_%spreds_n30000_final.hdf5'%CP['datatype']
    
    # Location of where final unique crater distribution will be stored
    CP['dir_result'] = 'datasets/HEAD/HEAD_%s_craterdist_llt%.2f_rt%.2f_mr%d_fin2.npy' % (CP['datatype'], CP['llt2'], CP['rt2'], CP['mr'])
    
    craters_unique = np.empty([0, 3])
    craters_unique = extract_unique_craters(CP, craters_unique)
