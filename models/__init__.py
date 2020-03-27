import math

from tensorflow.keras.layers import MaxPooling2D, Flatten, concatenate

class Backbone(object):

    def __init__(self, backbone):
        self.backbone = backbone


def backbone(backbone_name):
    if 'vgg' in backbone_name:
        from .vgg import VGG as model
    elif 'resnet' in backbone_name:
        from .resnet import ResNet as model
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return model(backbone_name)


def spatial_pyramid_pool(x, num_sample, previous_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        h, w = previous_conv_size
        h_window = math.ceil(h/out_pool_size[i])
        w_window = math.ceil(w/out_pool_size[i])
        h_stride = math.floor(h/out_pool_size[i])
        w_stride = math.floor(w/out_pool_size[i])
        
        max_pool = MaxPooling2D((h_window, w_window), strides=(h_stride, w_stride))(x)
        if (i == 0):
            spp = Flatten()(max_pool)
        else:
            spp = concatenate([spp, Flatten()(max_pool)])
  
    return spp
