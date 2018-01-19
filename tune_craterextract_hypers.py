# The purpose of this macro is to tune the hyperparameters of the crater detection algorithm
# on the validation set, namely:
# match_thresh2, template_thresh and target_thresh. 

import numpy as np
import sys
import h5py
import pandas as pd
from utils.template_match_target import *

def get_id(i, zeropad=5):
    return 'img_{i:0{zp}d}'.format(i=i, zp=zeropad)

def prep_csvs(craters, nimgs):
    # get csvs
    csvs = []
    cutrad, dim, minrad, maxrad = 1, 256, 5, 40
    for i in range(nimgs):
        csv = craters[get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv['Diameter (pix)'] < 2*maxrad) & (csv['Diameter (pix)'] > 2*minrad)]
        csv = csv[(csv['x']+cutrad*csv['Diameter (pix)']/2 <= dim)]
        csv = csv[(csv['y']+cutrad*csv['Diameter (pix)']/2 <= dim)]
        csv = csv[(csv['x']-cutrad*csv['Diameter (pix)']/2 > 0)]
        csv = csv[(csv['y']-cutrad*csv['Diameter (pix)']/2 > 0)]
        if len(csv) < 3:    #exclude csvs with tiny crater numbers
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'],csv['y'],csv['Diameter (pix)']/2)).T
            csvs.append(csv_coords)
    return csvs

def get_metrics(preds, csvs, nimgs, longlat_thresh2, rad_thresh, template_thresh, target_thresh):
    recall, precision, f1 = [], [], []
    err_lo, err_la, err_r = [], [], []
    duplicate_match_frac = []
    for i in range(nimgs):
        if len(csvs[i]) < 3:
            continue
        N_match, N_csv, N_templ, maxr, elo, ela, er, dupe_frac = template_match_t2c(preds[i], csvs[i],
                                                                                    longlat_thresh2=longlat_thresh2,
                                                                                    rad_thresh=rad_thresh,
                                                                                    template_thresh=template_thresh,
                                                                                    target_thresh=target_thresh,
                                                                                    rmv_oor_csvs=0)
        if N_match > 0:
            print(i, N_match, N_csv, N_templ, maxr, dupe_frac)
            p = float(N_match)/float(N_match + (N_templ-N_match))   #assumes unmatched detected circles are FP
            r = float(N_match)/float(N_csv)                         #N_csv = tp + fn, i.e. total ground truth matches
            recall.append(r); precision.append(p); f1.append(2*r*p/(r+p))
            err_lo.append(elo); err_la.append(ela); err_r.append(er)
            duplicate_match_frac.append(dupe_frac)
        else:
            print("skipping iteration %d,N_csv=%d,N_templ=%d,N_match=%d"%(i,N_csv,N_templ,N_match))

    print("longlat_thresh2=%f, rad_thresh=%f, template_thresh=%f, target_thresh=%f"%(longlat_thresh2, rad_thresh, template_thresh, target_thresh))
    print("mean and std of N_match/N_csv (recall) = %f, %f"%(np.mean(recall), np.std(recall)))
    print("mean and std of N_match/(N_match + (N_templ-N_match)) (precision) = %f, %f"%(np.mean(precision), np.std(precision)))
    print("mean and std of 2rp/(r+p) (F1 score) = %f, %f"%(np.mean(f1), np.std(f1)))
    print("mean and std longitude fractional difference between pred and GT craters: %f, %f"%(np.mean(err_lo),np.std(err_lo)))
    print("mean and std latitude fractional difference between pred and GT craters: %f, %f"%(np.mean(err_la),np.std(err_la)))
    print("mean and std radius fractional difference between pred and GT craters: %f, %f"%(np.mean(err_r),np.std(err_r)))
    print("mean and std duplicate match fraction: %f, %f"%(np.mean(duplicate_match_frac),np.std(duplicate_match_frac)))

if __name__ == '__main__':
    #data parameters
    dir = '../moon-craters/datasets/HEAD/'    #location of model predictions. Exclude final '/' in path.
    datatype = 'dev'
    nimgs = 5000              #1000, 10016, 30016
    
    #load hyperparameters
    longlat_thresh2 = float(sys.argv[1])
    rad_thresh = float(sys.argv[2])
    template_thresh = float(sys.argv[3])
    target_thresh = float(sys.argv[4])
    
    #load data
    file = '%sHEAD_%spreds_n30000_final.hdf5'%(dir,datatype)
    preds = h5py.File(file,'r')[datatype]
    craters = pd.HDFStore('%s%s_craters_final.hdf5'%(dir,datatype),'r')

    csvs = prep_csvs(craters, nimgs)

    get_metrics(preds, csvs, nimgs, longlat_thresh2, rad_thresh, template_thresh, target_thresh)
    print("finished successfully")
