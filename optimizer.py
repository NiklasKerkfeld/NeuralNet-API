import numpy as np
from Optimizer.Adam import Adam
from Optimizer.AdaGrad import AdaGrad
from Optimizer.SGD import SGD
from Optimizer.RMSprop import RMSprop
from Optimizer.Default import Default


def give_optimizer_dict():
    functions = {
        'adam': Adam,
        'default': Default,
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': AdaGrad
    }
    return functions
