import utils.processing as proc
import utils.transform as trf
import get_unique_craters as guc
import numpy as np
import h5py
import pandas as pd
import os
import matplotlib.pyplot as plt

#os.system("sshfs silburt@login.scinet.utoronto.ca:/scratch/m/mhvk/czhu/moondata/final_data/ /Users/silburt/remotemount/")
#os.system("sshfs silburt@rein005.utsc.utoronto.ca:/data_local/silburt/moon-craters/datasets/HEAD /Users/silburt/remotemount/")
dir = '/Users/silburt/remotemount'
#dir = '/scratch/m/mhvk/czhu/moondata/final_data/'
dtype = 'test'

P = h5py.File('%s/%s_images_final.hdf5'%(dir,dtype), 'r')
craters = pd.HDFStore('%s/%s_craters_final.hdf5'%(dir,dtype), 'r')
dim = (256., 256.)
llbd, pbd, distcoeff = ('longlat_bounds', 'pix_bounds', 'pix_distortion_coefficient')

diff_Lo, diff_La, diff_R = [], [], []

n_imgs = 15
for i in range(n_imgs):
    print(i)
    id = proc.get_id(i)
    
    #prepare csvs
    csv = craters[id]
    csv_pix = np.asarray((csv['x'], csv['y'], csv['Diameter (pix)'] / 2.)).T
    csv_real = np.asarray((csv['Long'], csv['Lat'], csv['Diameter (km)'] / 2.)).T

    csv_convert = guc.estimate_longlatdiamkm(dim, P[llbd][id], P[distcoeff][id][0], csv_pix)
    Lo, La, R = csv_convert.T
    Lo_r, La_r, R_r = csv_real.T
    
    dLo = np.abs(Lo - Lo_r)/R_r
    dLa = np.abs(La - La_r)/R_r
    dR = np.abs(R - R_r)/R_r

    diff_Lo += list(dLo)
    diff_La += list(dLa)
    diff_R += list(dR)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(13, 5))
ax0, ax1, ax2 = axes.flatten()
n, bins, patches = ax0.hist(diff_Lo)
ax0.set_xlabel('fractional longitude difference')
n, _, _ = ax1.hist(diff_La)
ax1.set_xlabel('fractional latitude difference')
n, _, _ = ax2.hist(diff_R)
ax2.set_xlabel('fractional radius difference')
plt.title('ground truth craters')
plt.show()
