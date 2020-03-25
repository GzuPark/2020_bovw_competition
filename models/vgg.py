import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from . import Backbone


class VGG(Backbone):
    
    def _stem(self, inputs):
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
        return x
    
    def _learner(self, x, blocks):
        for n_layers, n_filters in blocks:
            x = self._group(x, n_layers, n_filters)
        return x

    @staticmethod
    def _group(x, n_layers, n_filters):
        for n in range(n_layers):
            x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=(2, 2))(x)
        return x
    
    def _classifier(self, x, n_classes):
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(n_classes, activation='softmax')(x)
        return x

    def construct(self, n_classes, group=16):
        _groups = {
            16 : [ (1, 64), (2, 128), (3, 256), (3, 512), (3, 512) ],
            19 : [ (1, 64), (2, 128), (4, 256), (4, 512), (4, 512) ],
        }

        inputs = Input((224, 224, 3))
        x = self._stem(inputs)
        x = self._learner(x, _groups[group])
        outputs = self._classifier(x, n_classes)

        model = Model(inputs, outputs)
        return model
 