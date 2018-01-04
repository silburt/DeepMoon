import numpy as np, h5py, pandas as pd
import os
import cv2
from utils.template_match_target import *
import utils.processing as proc
import utils.transform as trf
import get_unique_craters as guc

#from keras.models import load_model

#os.system("sshfs silburt@rein005.utsc.utoronto.ca:/data_local/silburt/moon-craters/datasets/HEAD /Users/silburt/remotemount/")
#dir = '/Users/silburt/remotemount'
dir = '../moon-craters/datasets/HEAD'

dtype = 'test'
n_imgs = 1000

#preds = h5py.File('../moon-craters/datasets/HEAD/HEAD_%spreds_n30000_final.hdf5'%(dtype), 'r')[dtype]
#imgs = h5py.File('/scratch/m/mhvk/czhu/moondata/final_data/%s_images.hdf5'%(dtype), 'r')
#craters = pd.HDFStore('%s/%s_craters.hdf5'%(dir,dtype), 'r')
preds = h5py.File('%s/HEAD_%spreds_n30000_final.hdf5'%(dir,dtype), 'r')[dtype]
imgs = h5py.File('%s/%s_images_final.hdf5'%(dir,dtype), 'r')
craters = pd.HDFStore('%s/%s_craters_final.hdf5'%(dir,dtype), 'r')

llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds', 'pix_distortion_coefficient')
dim = (float(256), float(256))

longlat_thresh2 = 70
minrad, maxrad = 3, 40
rad_thresh = 1.0
template_thresh = 0.5
target_thresh = 0.1

#model = load_model('models/DeepMoon_final.h5')
#preds = model.predict(imgs['input_images'][0:n_imgs].reshape(n_imgs,256,256,1))

err_lo_pix, err_la_pix, err_r_pix = [], [], []
err_lo_deg, err_la_deg, err_r_deg = [], [], []
err_lo_csv, err_la_csv, err_r_csv = [], [], []
km_to_deg = 180. / (np.pi * 1737.4)
i = -1
while i < n_imgs-1:
    i += 1
    print(i)
    id = proc.get_id(i)
    llbd_val, dist_val = imgs[llbd][id], imgs[distcoeff][id][0]
    
    # get csv coords
    csv = craters[id]
    csv_coords = np.asarray((csv['x'], csv['y'], csv['Diameter (pix)'] / 2.)).T
    csv_real = np.asarray((csv['Long'], csv['Lat'], csv['Diameter (km)'] / 2.)).T
    csv_conv = guc.estimate_longlatdiamkm(dim, llbd_val, dist_val, csv_coords)
    
    rawlen = imgs[pbd][id][2] - imgs[pbd][id][0]
    if rawlen < 4000:
        minrad = max(int((3. / 1000.) * rawlen - 3), 3)
    elif rawlen >= 4000:
        minrad=9
    
    coords = template_match_t(preds[i], minrad, maxrad, longlat_thresh2, rad_thresh, template_thresh, target_thresh)
    coords_conv = guc.estimate_longlatdiamkm(dim, llbd_val, dist_val, coords)
    
    # compare template-matched results to ground truth csv input data
    N_match = 0
    csv_duplicates = []
    N_csv, N_detect = len(csv_coords), len(coords)
    for j in range(len(coords)):
        lo, la, r = coords[j]
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
        elif N == 1:
            Lo, La, R = csv_coords[index_True[0]].T
            
            lo_, la_, r_ = coords_conv[j].T
            Lo_, La_, R_ = csv_real[index_True[0]].T
            Loo_, Laa_, Rr_ = csv_conv[index_True[0]].T
            
            dL_pix = abs(Lo - lo) / r
            dL_deg = abs(Lo_ - lo_) / (r_* km_to_deg)
            dL_csv = abs(Lo_ - Loo_) / (R_* km_to_deg)
            #print("dL_pix=%f, dL_deg=%f, dL_csv=%f"%(dL_pix, dL_deg, dL_csv))
            
            err_lo_pix.append(dL_pix)
            err_la_pix.append(abs(La - la) / r)
            err_r_pix.append(abs(R - r) / r)
            err_lo_deg.append(dL_deg)
            err_la_deg.append(abs(La_ - la_) / (r_* km_to_deg))
            err_r_deg.append(abs(R_ - r_) / r_)
            err_lo_csv.append(dL_csv)
            err_la_csv.append(abs(La_ - Laa_) / (R_* km_to_deg))
            err_r_csv.append(abs(R_ - Rr_) / R_)
        N_match += min(1, N)
        # remove csv so it can't be re-matched again
        csv_coords = csv_coords[np.where(index == False)]
        csv_real = csv_real[np.where(index == False)]
        csv_conv = csv_conv[np.where(index == False)]
        if len(csv_coords) == 0:
            break



# printing stuff
print("Stats:")
print("Mean err Longitude (pix) = %f +/- %f"%(np.mean(err_lo_pix), np.std(err_lo_pix)))
print("Mean err Longitude (deg) = %f +/- %f"%(np.mean(err_lo_deg), np.std(err_lo_deg)))
print("Mean err Longitude (csv) = %f +/- %f"%(np.mean(err_lo_csv), np.std(err_lo_csv)))
#print(list(zip(err_lo_pix, err_lo_deg)))

print("Mean err Latitude (pix) = %f +/- %f"%(np.mean(err_la_pix), np.std(err_la_pix)))
print("Mean err Latitude (deg) = %f +/- %f"%(np.mean(err_la_deg), np.std(err_la_deg)))
print("Mean err Latitude (csv) = %f +/- %f"%(np.mean(err_la_csv), np.std(err_la_csv)))
#print(list(zip(err_la_pix, err_la_deg)))

print("Mean err Radius (pix) = %f +/- %f"%(np.mean(err_r_pix), np.std(err_r_pix)))
print("Mean err Radius (deg) = %f +/- %f"%(np.mean(err_r_deg), np.std(err_r_deg)))
print("Mean err Radius (csv) = %f +/- %f"%(np.mean(err_r_csv), np.std(err_r_csv)))
#print(list(zip(err_r_pix, err_r_deg)))

"""
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes.flatten()
_,_,_=ax0.hist(err_lo_pix)
_,_,_=ax1.hist(err_la_pix)
_,_,_=ax2.hist(err_r_pix)
ax0.set_title('err_lo_pix', fontsize=7)
ax1.set_title('err_la_pix', fontsize=7)
ax2.set_title('err_r_pix', fontsize=7)

_,_,_=ax3.hist(err_lo_deg)
_,_,_=ax4.hist(err_la_deg)
_,_,_=ax5.hist(err_r_deg)
ax3.set_title('err_lo_deg', fontsize=7)
ax4.set_title('err_la_deg', fontsize=7)
ax5.set_title('err_r_deg', fontsize=7)

_,_,_=ax6.hist(err_lo_csv)
_,_,_=ax7.hist(err_la_csv)
_,_,_=ax8.hist(err_r_csv)
ax6.set_title('err_lo_csv', fontsize=7)
ax7.set_title('err_la_csv', fontsize=7)
ax8.set_title('err_r_csv', fontsize=7)
plt.savefig('images/compare_frac_errs.png')
plt.show()
"""
