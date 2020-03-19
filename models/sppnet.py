import math

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense,
    Flatten, MaxPooling2D, concatenate
)

from . import Backbone


def spatial_pyramid_pool(x, prev_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        h, w = prev_conv_size
        h_window = math.ceil(h / out_pool_size[i])
        w_window = math.ceil(w / out_pool_size[i])
        h_stride = math.floor(h / out_pool_size[i])
        w_stride = math.floor(w / out_pool_size[i])

        max_pool = MaxPooling2D((h_window, w_window), strides=(h_stride, w_stride))(x)
        if (i == 0):
            spp = Flatten()(max_pool)
        else:
            spp = concatenate([spp, Flatten()(max_pool)])
  
    return spp


class SPPNet(Backbone):

    @staticmethod
    def _stem(inputs):
        x = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=(2, 2))(x)
        return x
    
    @staticmethod
    def _learner(x):
        x = Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=(2, 2))(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    @staticmethod
    def _spp(x):
        spp_inp_shape = x.shape
        out_pool_size = [4, 2, 1]
        x = spatial_pyramid_pool(x, [spp_inp_shape[1], spp_inp_shape[2]], out_pool_size)
        return x

    @staticmethod
    def _classifier(x, n_classes):
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(n_classes, activation='softmax')(x)
        return x

    def construct(self, n_classes):
        inputs = Input((224, 224, 3))

        x = self._stem(inputs)
        x = self._learner(x)
        x = self._spp(x)
        outputs = self._classifier(x, n_classes)

        model = Model(inputs, outputs)
        return model
