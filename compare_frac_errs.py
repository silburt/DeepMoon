import numpy as np, h5py, pandas as pd
import os
import cv2
from utils.template_match_target import *
import utils.transform as trf

def get_id(i, zeropad=5):
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)

def estimate_longlatdiamkm(llbd, distcoeff, long_pix, lat_pix, radii_pix):
    dim = (256., 256.)
    
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
    latdiff = max(latdiff, 1e-7)
    lat_deg = lat_central - (deg_per_pix * (lat_pix - dim[1] / 2.) *
                             (np.pi * latdiff / 180.) /
                             np.sin(np.pi * latdiff / 180.))
    # Determine longitude using determined latitude.
    long_deg = long_central + (deg_per_pix * (long_pix - dim[0] / 2.) /
                               np.cos(np.pi * lat_deg / 180.))
                             
    # Return combined long/lat/radius array.
    return long_deg, lat_deg, radii_km

dir = '../moon-craters/datasets/HEAD'
dtype = 'test'

preds = h5py.File('../moon-craters/datasets/HEAD/HEAD_%spreds_n30000_final.hdf5'%(dtype), 'r')[dtype]
imgs = h5py.File('/scratch/m/mhvk/czhu/moondata/final_data/%s_images.hdf5'%(dtype), 'r')
craters = pd.HDFStore('%s/%s_craters.hdf5'%(dir,dtype), 'r')

llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds', 'pix_distortion_coefficient')

longlat_thresh2 = 70
minrad, maxrad = 3, 40
rad_thresh = 1.0
template_thresh = 0.5
target_thresh = 0.1

i = -1
while i < 2:
    i += 1
    templ_coords = template_match_t(preds[i], minrad, maxrad, longlat_thresh2,
                                    rad_thresh, template_thresh, target_thresh)
    # get csv coords
    cutrad, n_csvs = 0.8, 50
    csv = craters[get_id(i)]
    csv_coords = np.asarray((csv['x'], csv['y'], csv['Diameter (pix)'] / 2.)).T
    csv_real = np.asarray((csv['Long'], csv['Lat'], csv['Diameter (km)'] / 2.)).T
        
    # compare template-matched results to ground truth csv input data
    N_match = 0
    csv_duplicates = []
    err_lo_pix, err_la_pix, err_r_pix = [], [], []
    err_lo_real, err_la_real, err_r_real = [], [], []
    N_csv, N_detect = len(csv_coords), len(templ_coords)
    for lo, la, r in templ_coords:
        csvLong, csvLat, csvRad = csv_coords.T
        diff_longlat = (csvLong - lo)**2 + (csvLat - la)**2
        diff_rad = abs(csvRad - r)
        index = (diff_rad < max(1.01, rad_thresh * r)) & (diff_longlat < longlat_thresh2)
        index_True = np.where(index == True)[0]
        N = len(index_True)
        if N > 1:
            cratervals = np.array((lo, la, r))
            id_keep = index_True[0]
            diff = np.sum((csv_coords[id_keep] - cratervals)**2)
            csv_duplicates.append(csv_coords[id_keep])
            for id in index_True[1:]:
                dupevals = csv_coords[id]
                index[id] = False
                csv_duplicates.append(dupevals)
                diff_ = np.sum((dupevals - cratervals)**2)
                if diff_ < diff:
                    id_keep = id
                    diff = diff_
            index[id_keep] = True       # keep only closest match as true
            Lo, La, R = csv_coords[id_keep].T
            Lo_, La_, R_ = csv_real[id_keep].T
            lo_, la_, r_ = estimate_longlatdiamkm(imgs[llbd][get_id(id_keep)], imgs[distcoeff][get_id(id_keep)][0], lo, la, r)
            err_lo_pix.append(abs(Lo - lo) / r)
            err_la_pix.append(abs(La - la) / r)
            err_r_pix.append(abs(R - r) / r)
            err_lo_real.append(abs(Lo_ - lo_) / r_)
            err_la_real.append(abs(La_ - la_) / r_)
            err_r_real.append(abs(R_ - r_) / r_)
            print("""%d GT entries matched to (%d,%d,%d) ring... counted
                (%f,%f,%f) as the match.""" % (N, lo, la, r, Lo, La, r))
            print(csv_duplicates)
        elif N == 1:
            Lo, La, R = csv_coords[index_True[0]].T
            Lo_, La_, R_ = csv_real[index_True[0]].T
            lo_, la_, r_ = estimate_longlatdiamkm(imgs[llbd][get_id(index_True[0])], imgs[distcoeff][get_id(index_True[0])][0], lo, la, r)
            err_lo_pix.append(abs(Lo - lo) / r)
            err_la_pix.append(abs(La - la) / r)
            err_r_pix.append(abs(R - r) / r)
            err_lo_real.append(abs(Lo_ - lo_) / r_)
            err_la_real.append(abs(La_ - la_) / r_)
            err_r_real.append(abs(R_ - r_) / r_)
        N_match += min(1, N)
        # remove csv so it can't be re-matched again
        csv_coords = csv_coords[np.where(index == False)]
        if len(csv_coords) == 0:
            break

print(list(zip(err_lo_pix, err_lo_real)))
print(list(zip(err_la_pix, err_la_real)))
print(list(zip(err_r_pix, err_r_real)))
