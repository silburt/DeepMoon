from __future__ import absolute_import, division, print_function
import numpy as np
from keras import backend as K
import sys
sys.path.append('../')
import model_train as mt


class TestModelTrain():
    """Tests model building."""

    def test_build_model(self):
        dim = 256
        FL = 3
        learn_rate = 0.0001
        n_filters = 112
        init = 'he_normal'
        lmbda = 1e-06
        drop = 0.15

        model = mt.build_model(dim, learn_rate, lmbda, drop, FL, init,
                               n_filters)

        # Following https://stackoverflow.com/questions/45046525/keras-number-of-trainable-parameters-in-model
        trainable_count = int(np.sum([K.count_params(p) for p in
                                      set(model.trainable_weights)]))
        non_trainable_count = int(np.sum([K.count_params(p) for p in
                                          set(model.non_trainable_weights)]))
        assert trainable_count + non_trainable_count == 10278017
        assert trainable_count == 10278017
        assert non_trainable_count == 0
