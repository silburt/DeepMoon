#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py

from keras.models import Model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
K.set_image_dim_ordering('tf')

import utils.template_match_target as tmt
import utils.processing as proc

# Check Keras version - code will switch API if needed.
from keras import __version__ as keras_version
k2 = True if keras_version[0] == '2' else False

# If Keras is v2.x.x, create Keras 1-syntax wrappers.
if not k2:
    from keras.layers import merge, Input
    from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                            UpSampling2D)

else:
    from keras.layers import Concatenate, Input
    from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                            UpSampling2D)

    def merge(layers, mode=None, concat_axis=None):
        """Wrapper for Keras 2's Concatenate class (`mode` is discarded)."""
        return Concatenate(axis=concat_axis)(list(layers))

    def Convolution2D(n_filters, FL, FLredundant, activation=None,
                      init=None, W_regularizer=None, border_mode=None):
        """Wrapper for Keras 2's Conv2D class."""
        return Conv2D(n_filters, FL, activation=activation,
                      kernel_initializer=init,
                      kernel_regularizer=W_regularizer,
                      padding=border_mode)


########################
def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]

########################
def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.

    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.

    Yields
    ------
    Manipulated images and targets.
        
    """
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
                                                              npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)

########################
def get_metrics(data, craters, dim, model, beta=1):
    """Function that prints pertinent metrics at the end of each epoch. 

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data. 
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]

    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[proc.get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    preds = model.predict(X)
    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = tmt.template_match_t2c(preds[i], csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    print("binary XE score = %f" % model.evaluate(X, Y))
    if len(recall) > 3:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")

########################
def build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network. 

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter. 
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1)
    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)

    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2)
    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)

    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                       W_regularizer=l2(lmbda), border_mode='same')(a3)
    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)

    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(a3P)
    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a3, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a2, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    u = UpSampling2D((2, 2))(u)
    u = merge((a1, u), mode='concat', concat_axis=3)
    u = Dropout(drop)(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)

    # Final output
    final_activation = 'sigmoid'
    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
                      W_regularizer=l2(lmbda), border_mode='same')(u)
    u = Reshape((dim, dim))(u)
    if k2:
        model = Model(inputs=img_input, outputs=u)
    else:
        model = Model(input=img_input, output=u)

    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    print(model.summary())

    return model

########################
def train_and_test_model(Data, Craters, MP, i_MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model. 

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Build model
    model = build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters)

    # Main loop
    n_samples = MP['n_train']
    for nb in range(nb_epoch):
        if k2:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                steps_per_epoch=n_samples/bs, epochs=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                validation_steps=n_samples,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
        else:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                samples_per_epoch=n_samples, nb_epoch=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                nb_val_samples=n_samples,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0)])

        get_metrics(Data['dev'], Craters['dev'], dim, model)

    if MP['save_models'] == 1:
        model.save(MP['save_dir'])

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))
    get_metrics(Data['test'], Craters['test'], dim, model)
    print("###################################")
    print("###################################")

########################
def get_models(MP):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']

    # Load data
    train = h5py.File('%strain_images.hdf5' % dir, 'r')
    dev = h5py.File('%sdev_images.hdf5' % dir, 'r')
    test = h5py.File('%stest_images.hdf5' % dir, 'r')
    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'dev': [dev['input_images'][:n_dev].astype('float32'),
                  dev['target_masks'][:n_dev].astype('float32')],
        'test': [test['input_images'][:n_test].astype('float32'),
                 test['target_masks'][:n_test].astype('float32')]
    }
    train.close()
    dev.close()
    test.close()

    # Rescale, normalize, add extra dim
    proc.preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': pd.HDFStore('%strain_craters.hdf5' % dir, 'r'),
        'dev': pd.HDFStore('%sdev_craters.hdf5' % dir, 'r'),
        'test': pd.HDFStore('%stest_craters.hdf5' % dir, 'r')
    }

    # Iterate over parameters
    for i in range(MP['N_runs']):
        train_and_test_model(Data, Craters, MP, i)
