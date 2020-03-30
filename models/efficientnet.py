import keras

from .custom.keras_applications import efficientnet
# from keras_applications.efficientnet import EfficientNetB0
from .custom.keras_applications.efficientnet import preprocess_input, decode_predictions
kwargs = {
    'backend': keras.backend,
    'layers': keras.layers,
    'models': keras.models,
    'utils': keras.utils
}

# model = EfficientNetB0(weights='imagenet', **kwargs)
# img = load_img('cat.png', target_size=256, crop_size=224)
# preds = model.predict(preprocess_input(img, **kwargs))
# print(decode_predictions(preds, top=3, **kwargs)[0])


from . import Backbone


class EfficientNet(Backbone):
 
    def pretrained(self, input_shape):
        base_model = efficientnet.EfficientNetB5(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet',
            **kwargs,
        )
        return base_model
