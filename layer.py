from Layer.Activation import Activation
from Layer.BatchNorm import BatchNormalization
from Layer.Dense import Dense
from Layer.Conv2D import Conv2D
from Layer.Pooling import Pooling
from Layer.Dropout import Dropout
from Layer.Reshape import Reshape
from Layer.Flatten import Flatten
from Layer.GaussianNoise import GaussianNoise
from Layer.BatchNormGaussianNoise import BatchNormGaussianNoise


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
        'batchnormalization': BatchNormalization,
        'gaussiannoise': GaussianNoise,
        'batchnormgaussiannoise': BatchNormGaussianNoise
    }
    return layer_dict