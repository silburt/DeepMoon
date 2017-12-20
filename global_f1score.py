## Global metrics when comparing across the datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from progress.bar import Bar

def get_GT(truth_datatype, radcut):
    #prepare GT
    truthalan = pd.read_csv('catalogues/LROCCraters.csv')
    truthhead = pd.read_csv('catalogues/HeadCraters.csv')
    truthhead = truthhead[(truthhead['Lat']>=-60)&(truthhead['Lat']<=60)&(truthhead['Diam_km']>2*radcut)]
    truthalan = truthalan[(truthalan['Lat']>=-60)&(truthalan['Lat']<=60)&(truthalan['Diameter (km)']>2*radcut)]
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

def get_stats(filename, csv_coords, thresh_longlat2, thresh_rad2, radcut):
    pred = np.load(filename)
    pred = pred[pred.T[2] > radcut]
    
    csv_duplicates = []
    N_match, err_lo, err_la, err_r, beta = 0, 0, 0, 0, 1
    N_csv, N_detect = len(csv_coords), len(pred)
    km_to_deg = 180. / (np.pi * 1737.4)
    for lo, la, r in pred:
        Long, Lat, Rad = csv_coords.T
        
        diff_longlat = ((Long - lo)**2 + (Lat - la)**2) / (r * km_to_deg)**2
        index = (diff_longlat < thresh_longlat2) & (((Rad - r) / r)**2 < thresh_rad2)
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
            err_lo += abs(Lo - lo) / (r * km_to_deg)
            err_la += abs(La - la) / (r * km_to_deg)
            err_r += abs(R - r) / r
        elif N == 1:
            Lo, La, R = csv_coords[index_True[0]].T
            err_lo += abs(Lo - lo) / (r * km_to_deg)
            err_la += abs(La - la) / (r * km_to_deg)
            err_r += abs(R - r) / r
        N_match += min(1, N)
        # remove csv so it can't be re-matched again
        csv_coords = csv_coords[np.where(index == False)]
        if len(csv_coords) == 0:
            break

    p = float(N_match) / float(N_match + (N_detect - N_match))
    r = float(N_match) / float(N_csv)
    f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
    return p, r, f, err_lo/N_detect, err_la/N_detect, err_r/N_detect, (N_detect - N_match)/float(N_csv)


if __name__ == '__main__':
    dtype = 'dev'
    radcut = 0
    
    files = glob.glob('../moon-craters/datasets/HEAD/HEAD_%s*final.npy'%dtype)
    csv_coords = get_GT(dtype, radcut)
    
    llt2, rt2, precision, recall, f1 = [], [], [], [], []
    err_lo, err_la, err_r, N_new = [], [], [], []
    bar = Bar("Running", max=len(files))
    for f in files:
        longlat_thresh2 = float(f.split('_')[3].split('llt')[1])
        rad_thresh2 = float(f.split('_')[4].split('rt')[1])
        p, r, f, elo, ela, er, frac_new = get_stats(f, csv_coords, longlat_thresh2, rad_thresh2, radcut)
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
        llt2.append(longlat_thresh2)
        rt2.append(rad_thresh2)
        err_lo.append(elo)
        err_la.append(ela)
        err_r.append(er)
        N_new.append(N_new)
        if longlat_thresh2 == 1.6 and rad_thresh2 == 0.3:
            print(elo, ela, er, frac_new)
        
        bar.next()
    bar.finish()

    fig = plt.figure()
    pp = plt.scatter(llt2, rt2, c=f1)
    plt.xlabel('longlat_thresh2')
    plt.ylabel('rad_thresh2')
    plt.title('tuning hypers on %s set'%dtype)
    cbar = fig.colorbar(pp, label='f1')
    for i in range(len(llt2)):
        plt.text(llt2[i], rt2[i], "r=%.3f"%(recall[i]), fontsize=8)

    plt.savefig('images/global_f1score_%s_radcut%d.png'%(dtype,radcut))
    plt.show()

