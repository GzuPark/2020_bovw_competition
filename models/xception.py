import tensorflow as tf

from . import Backbone


class Xception(Backbone):
 
    def pretrained(self, input_shape):
        base_model = tf.keras.applications.Xception(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet',
        )
        return base_model
