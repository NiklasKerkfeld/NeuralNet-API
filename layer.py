from Layer.Activation import Activation
from Layer.BatchNorm import BatchNormalization
from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
from Layer.Pooling import Pooling
from Layer.Dropout import Dropout
from Layer.Reshape import Reshape
from Layer.Flatten import Flatten


def give_layer_dict():
    layer_dict = {
        'dense': Dense,
        'dropout': Dropout,
        'conv2d': Conv2D,
        'pooling': Pooling,
        'reshape': Reshape,
        'flatten': Flatten,
        'activation': Activation,
        'batchnorm': BatchNormalization,
        'batchnormalization': BatchNormalization
    }
    return layer_dict