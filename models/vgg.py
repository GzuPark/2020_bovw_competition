import tensorflow as tf

from . import Backbone


class VGG(Backbone):
 
    def pretrained(self, input_shape):
        if '16' in self.backbone:
            base_model = tf.keras.applications.VGG16(
                input_shape=input_shape,
                include_top=False, 
                weights='imagenet',
            )
        elif '19' in self.backbone:
            base_model = tf.keras.applications.VGG19(
                input_shape=input_shape,
                include_top=False, 
                weights='imagenet',
            )
        return base_model
