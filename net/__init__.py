from .resnet_v2 import resnet_v2_50, resnet_v2_101, resnet_v2_152, resnet_v2_200
from .resnet_v2 import resnet_arg_scope
from .inception_resnet_v2 import inception_resnet_v2_base, inception_resnet_v2
from .inception_resnet_v2 import inception_resnet_v2_arg_scope

nets = {
    'resnet_v2_50': {
        'fun': resnet_v2_50,
        'ckpt': 'PreTrains/resnet_v2_50/resnet_v2_50.ckpt',
        'arg_scope': resnet_arg_scope,
        'prediction_name': 'predictions'
    },

    'resnet_v2_101': {
        'fun': resnet_v2_101,
        'ckpt': '',
        'arg_scope': resnet_arg_scope,
        'prediction_name': 'predictions'
    },

    'resnet_v2_152': {
        'fun': resnet_v2_152,
        'ckpt': '',
        'arg_scope': resnet_arg_scope,
        'prediction_name': 'predictions'
    },

    'resnet_v2_200': {
        'fun': resnet_v2_200,
        'ckpt': '',
        'arg_scope': resnet_arg_scope,
        'prediction_name': 'predictions'
    },

    'InceptionResnetV2': {
        'fun': inception_resnet_v2,
        'ckpt': 'PreTrains/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt',
        'arg_scope': inception_resnet_v2_arg_scope,
        'prediction_name': 'Predictions'
    }
}


def check_need_squeeze(network_name):
    assert network_name in nets.keys()
    if network_name in ['resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200']:
        return True
    else:
        return False


def get_exclude_names(network_name):
    assert network_name in nets.keys()
    if network_name in ['resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200']:
        names = ['{}/logits/weights:0'.format(network_name),
                 '{}/logits/biases:0'.format(network_name)]
    else:
        names = ['{}/AuxLogits'.format(network_name),
                 '{}/Logits'.format(network_name)]
    return names


def get_logits_names(network_name):
    assert network_name in nets.keys()
    if network_name in ['resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'resnet_v2_200']:
        names = ['{}/logits/weights:0'.format(network_name),
                 '{}/logits/biases:0'.format(network_name)]
    else:
        names = ['{}/AuxLogits/Logits/'.format(network_name),
                 '{}/Logits/Logits/'.format(network_name)]
    return names