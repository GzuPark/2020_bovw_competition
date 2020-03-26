#https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/models/__init__.py
#https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/tree/master/zoo/vgg
#https://github.com/peace195/sppnet/blob/master/alexnet_spp.py
#https://github.com/gunooknam/SPPNet
#https://n1094.tistory.com/30

class Backbone(object):

    def __init__(self, backbone):
        self.backbone = backbone


def backbone(backbone_name):
    if 'vgg' in backbone_name:
        from .vgg import VGG as model
    elif 'sppnet' in backbone_name:
        from .sppnet import SPPNet as model
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return model(backbone_name)
