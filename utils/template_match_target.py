##########################
#template match functions#
########################################################################

import numpy as np
from skimage.feature import match_template
import cv2

###########HYPERPARAMETERS###########
# LONGLAT_THRESH2, RAD_THRESH: for template matching, if (x1-x2)^2 + (y1-y2)^2 < longlat_thresh2 AND abs(r1-r2) < max(1.01,rad_thresh*r1), remove (x2,y2,r2) circle (it is a duplicate of another crater candidate). In addition, during predicted target -> csv matching (i.e. template_match_target_to_csv), the same criteria is used to match CNN craters with csv craters (increasing the recall). Maybe these should technically be separate parameters, but to first order they should be the same...
# TEMPLATE_THRESH: 0-1 range, if scikit-image's template matching probability > template_thresh, count as detection
# TARGET_THRESH: 0-1 range, set pixel values > target_thresh to 1, and pixel values < target_thresh -> 0
# MINRAD/MAXRAD are the radii to search over during template matching. For minrad, keep in mind that if the predicted target has thick rings, a small ring of diameter ~ ring_thickness could be detected by match_filter.
minrad_= 3
maxrad_= 50
longlat_thresh2_= 70
rad_thresh_= 1
template_thresh_= 0.6
target_thresh_= 0.1
#####################################

def template_match_target(target, minrad=minrad_, maxrad=maxrad_, longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_, template_thresh=template_thresh_, target_thresh=target_thresh_):

    # thickness of rings for the templates. 2 works well, just hardcode.
    ring_thickness = 2
    
    # target - can be predicted or ground truth
    target[target >= target_thresh] = 1
    target[target < target_thresh] = 0
    
    radii = np.arange(minrad,maxrad+1,1,dtype=int)
    coords = []     #coordinates extracted from template matching
    corr = []       #correlation coefficient for coordinates set
    for r in radii:
        # template
        n = 2*(r+ring_thickness+1)
        template = np.zeros((n,n))
        cv2.circle(template, (r+ring_thickness+1,r+ring_thickness+1), r, 1, ring_thickness)
        
        # template match - result is nxn array of probabilities
        result = match_template(target, template, pad_input=True)   #skimage
        index_r = np.where(result > template_thresh)
        coords_r = np.asarray(zip(*index_r))
        corr_r = np.asarray(result[index_r])
        
        # store x,y,r
        try:
            for c in coords_r:
                coords.append([c[1],c[0],r])
            for l in corr_r:
                corr.append(np.abs(l))
        except:
            pass

    # remove duplicates from template matching at neighboring radii/locations
    coords, corr = np.asarray(coords), np.asarray(corr)
    i, N = 0, len(coords)
    while i < N:
        Long, Lat, Rad = coords.T
        lo, la, r = coords[i]
        diff_longlat = (Long - lo)**2 + (Lat - la)**2
        diff_rad = abs(Rad - r)
        index = (diff_rad < max(1.01,rad_thresh*r))&(diff_longlat < longlat_thresh2)
        if len(np.where(index==True)[0]) > 1:
            #replace current coord with highest match probability coord in duplicate list
            coords_i, corr_i = coords[np.where(index==True)], corr[np.where(index==True)]
            coords[i] = coords_i[corr_i == np.max(corr_i)][0]
            index[i] = False
            coords = coords[np.where(index==False)]
        N, i = len(coords), i+1

    return coords


def template_match_target_to_csv(target, csv, minrad=minrad_, maxrad=maxrad_, longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_, template_thresh=template_thresh_, target_thresh=target_thresh_, remove_large_craters_csv=0):

    # get coordinates from template matching
    templ_coords = template_match_target(target, minrad, maxrad, longlat_thresh2, rad_thresh, template_thresh, target_thresh)

    # find max detected crater radius
    maxr = 0
    if len(templ_coords > 0):
        x,y,r = templ_coords.T
        maxr = np.max(r)
    
    # If remove_large_craters_csv == 1, see how recall improves when large craters are excluded.
    if remove_large_craters_csv == 1:
        index = np.where((csv.T[2] < maxr)&(csv.T[2] > minrad))
        if len(index[0]) > 0:
            csv_coords = csv[index]
        else:
            print("all craters are larger than max detected radius")
            csv_coords = csv
    else:
        csv_coords = csv

    # compare template-matched results to ground truth csv input data
    N_match = 0
    csv_duplicate_flag = 0
    N_csv, N_templ = len(csv_coords), len(templ_coords)
    for lo,la,r in templ_coords:
        csvLong, csvLat, csvRad = csv_coords.T
        diff_longlat = (csvLong - lo)**2 + (csvLat - la)**2
        diff_rad = abs(csvRad - r)
        index = (diff_rad < max(1.01,rad_thresh*r))&(diff_longlat < longlat_thresh2)
        index_True = np.where(index==True)[0]
        N = len(index_True)
        if N > 1:
            csv_duplicate_flag = 1
            print("%d GT entries matched to CNN-predicted ring... only counting first match."%N)
            for i,id in enumerate(index_True):
                print(csv_coords[id])
                if i > 0:                               #keep only first match as true
                    index[id] = False
        N_match += min(1,N)                             #count up to one match in recall
        csv_coords = csv_coords[np.where(index==False)] #remove csv so it can't be re-matched again
        if len(csv_coords) == 0:
            break

    return N_match, N_csv, N_templ, maxr, csv_duplicate_flag


