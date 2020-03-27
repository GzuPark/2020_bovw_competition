import tensorflow as tf

from . import Backbone


class Inception(Backbone):
 
    def pretrained(self, input_shape):
        base_model = tf.keras.applications.InceptionV3(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet',
        )
        return base_model
