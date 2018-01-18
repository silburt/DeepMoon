## Global metrics when comparing across the datasets.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from progress.bar import Bar

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

def get_stats(filename, csv_coords, thresh_longlat2, thresh_rad, minrad, maxrad):
    pred = np.load(filename)
    pred = pred[(pred.T[2] > minrad)&(pred.T[2] < maxrad)]
    
    csv_duplicates = []
    beta = 1
    N_match, err_lo, err_la, err_r = 0, [], [], []
    N_csv, N_detect = len(csv_coords), len(pred)
    k2d = 180. / (np.pi * 1737.4)
    for lo, la, r in pred:
        Long, Lat, Rad = csv_coords.T

        la_m = (la + Lat) / 2.
        minr = np.minimum(r, Rad)
        
        dL = (((Long - lo)/(minr * k2d / np.cos(np.pi * la_m / 180.)))**2
              + ((Lat - la)/(minr * k2d))**2)
        dR = np.abs(Rad - r) / minr
        index = (dL < thresh_longlat2) & (dR < thresh_rad)
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
            err_lo.append(abs(Lo - lo) / (r * k2d))
            err_la.append(abs(La - la) / (r * k2d))
            err_r.append(abs(R - r) / r)
        elif N == 1:
            Lo, La, R = csv_coords[index_True[0]].T
            err_lo.append(abs(Lo - lo) / (r * k2d))
            err_la.append(abs(La - la) / (r * k2d))
            err_r.append(abs(R - r) / r)
        N_match += min(1, N)
        # remove csv so it can't be re-matched again
        csv_coords = csv_coords[np.where(index == False)]
        if len(csv_coords) == 0:
            break

    p = float(N_match) / float(N_match + (N_detect - N_match))
    r = float(N_match) / float(N_csv)
    f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
    return p, r, f, err_lo, err_la, err_r, (N_detect - N_match)/(float(N_csv) + (N_detect - N_match))


if __name__ == '__main__':
    dtype = 'test'
    min_csv_rad = 2
    
    minrad = 0
    #minrad = min_csv_rad
    
    maxrad = 35
    csv_coords = get_GT(dtype, minrad+0.25, maxrad-2)
    
    files = glob.glob('datasets/HEAD/HEAD_%s*fin3.npy'%dtype)
    #files = ['../moon-craters/datasets/HEAD/HEAD_test_craterdist_llt1.60_rt0.30_final.npy']
    
    llt2, rt2, precision, recall, f1 = [], [], [], [], []
    err_lo, err_la, err_r, N_new = [], [], [], []
    bar = Bar("Running", max=len(files))
    for f in files:
        longlat_thresh2 = float(f.split('_')[3].split('llt')[1])
        rad_thresh = float(f.split('_')[4].split('rt')[1])
        if longlat_thresh2 == 1.4 and rad_thresh == 0.6:
            p, r, f, elo, ela, er, frac_new = get_stats(f, csv_coords, longlat_thresh2, rad_thresh, minrad, maxrad)
            
            precision.append(p)
            recall.append(r)
            f1.append(f)
            llt2.append(longlat_thresh2)
            rt2.append(rad_thresh)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            N_new.append(N_new)
            #if longlat_thresh2 == 1.6 and rad_thresh == 0.3:
            print(longlat_thresh2, rad_thresh)
            print(frac_new, p, r, f)
            print("\n")
            print("error stats:")
            print("long: median=%.5f, median - quartiles(5, 25, 75, 95) = %.5f, %.5f, %.5f, %.5f"%(np.median(elo), np.median(elo)-np.percentile(elo, 5), np.median(elo)-np.percentile(elo, 25), np.percentile(elo, 75)-np.median(elo), np.percentile(elo, 95)-np.median(elo)))
            print("lat: median=%.5f, median - quartiles(5, 25, 75, 95) = %.5f, %.5f, %.5f, %.5f"%(np.median(ela), np.median(ela)-np.percentile(ela, 5), np.median(ela)-np.percentile(ela, 25), np.percentile(ela, 75)-np.median(ela), np.percentile(ela, 95)-np.median(ela)))
            print("rad: median=%.5f, median - quartiles(5, 25, 75, 95) = %.5f, %.5f, %.5f, %.5f"%(np.median(er), np.median(er)-np.percentile(er, 5), np.median(er)-np.percentile(er, 25), np.percentile(er, 75)-np.median(er), np.percentile(er, 95)-np.median(er)))

# plot
#    fig = plt.figure()
#    pp = plt.scatter(llt2, rt2, c=f1)
#    plt.xlabel('longlat_thresh2')
#    plt.ylabel('rad_thresh2')
#    plt.title('tuning hypers on %s set'%dtype)
#    cbar = fig.colorbar(pp, label='f1')
#    for i in range(len(llt2)):
#        plt.text(llt2[i], rt2[i], "f=%.3f"%(f1[i]), fontsize=6)
#
#    plt.savefig('images/global_f1score_%s_minrad%d_maxrad%d.png'%(dtype,minrad,maxrad))
#    plt.show()

