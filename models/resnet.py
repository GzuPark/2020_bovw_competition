import tensorflow as tf

from . import Backbone


class ResNet(Backbone):
 
    def pretrained(self, input_shape):
        if '50' in self.backbone:
            base_model = tf.keras.applications.ResNet50V2(
                input_shape=input_shape,
                include_top=False, 
                weights='imagenet',
            )
        elif '101' in self.backbone:
            base_model = tf.keras.applications.ResNet101V2(
                input_shape=input_shape,
                include_top=False, 
                weights='imagenet',
            )
        elif '152' in self.backbone:
            base_model = tf.keras.applications.ResNet152V2(
                input_shape=input_shape,
                include_top=False, 
                weights='imagenet',
            )
        return base_model
