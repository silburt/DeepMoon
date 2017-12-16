import numpy as np, pandas as pd, h5py

def preprocess(Data, dim=256, low=0.1, hi=1):
    #rescaling and inverting images
    #https://www.mathworks.com/help/vision/ref/contrastadjustment.html
    #Since maxpooling is used, we want the interesting stuff (craters) to be 1, not 0.
    #But ignore null background pixels, keep them at 0.
    for key in Data:
        Data[key][0] = Data[key][0].reshape(len(Data[key][0]),dim,dim,1)
        for i,img in enumerate(Data[key][0]):
            img = img/255.
            #img[img > 0.] = 1. - img[img > 0.]      #inv color
            minn, maxx = np.min(img[img>0]), np.max(img[img>0])
            img[img>0] = low + (img[img>0] - minn)*(hi - low)/(maxx - minn) #linear re-scaling
            Data[key][0][i] = img
