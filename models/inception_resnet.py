import tensorflow as tf

from . import Backbone


class InceptionResNet(Backbone):
 
    def pretrained(self, input_shape):
        base_model = tf.keras.applications.InceptionResNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet',
        )
        return base_model
