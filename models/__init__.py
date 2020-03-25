class Backbone(object):

    def __init__(self, backbone):
        self.backbone = backbone


def backbone(backbone_name):
    if 'vgg' in backbone_name:
        from .vgg import VGG as model
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone_name))

    return model(backbone_name)
