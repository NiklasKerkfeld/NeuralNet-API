import numpy as np
from helperfunctions import *


class Dropout:
    def __init__(self, input_shape, dropout=.1, layer_before=None, next_layer=None, trainings_mode=True):
        """
        randomly sets values of Inputs to zero
        :param input_size: size of the input tensor
        :param dropout: proportion of values turned to zero (default: .1)
        """
        self.input_shape = input_shape                          # shape of the input vector
        self.output_shape = input_shape
        # self.drop_count = round(input_size * dropout)          # count of neurons set to zero
        self.dropout = dropout
        self.drop_vec = np.ones(self.input_shape)               # vector for multiplication

        self.trainings_mode = trainings_mode                    # if False layer does nothing
        self.first_mode = False                                 # if True backward does not return anything

        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer,  layer_before=None):    # I think taking an param and not use it is not the best
        """
        takes the params as a dict and retuns the Layer
        the param 'optimizer' is just for standardization
        :param params: dictionary with all params ('input_shape', 'ratio' for dropout, 'trainingsmode')
        :param optimizer:
        :return:
        """
        trainings_mode = True if not 'trainings_mode' in params.keys() else params['trainingsmode']
        dropout = .1 if not 'ratio' in params.keys() else params['ratio']
        return cls(params['input_shape'], dropout=dropout, layer_before=layer_before, trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, dropout = load
        return cls(retuple(input_shape), float(dropout), layer_before=layer_before)

    def set_weights_biases(self, params):
        """
        no trainable params
        this function is just for standardisation
        :param params:
        :return:
        """
        pass

    def set_trainingsmode(self, mode=True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def forward(self, input, target=None, give_return=False):

        if not self.trainings_mode:
            self.drop_vec = np.ones(self.input_shape)                           # important for backpropagation

        else:                                                                   # input = output
            self.drop_vec = np.random.binomial(1, 1-self.dropout, self.input_shape)

        if self.next_layer is None or give_return:
            return input * self.drop_vec                                        # by multiplication by one and zero
        else:
            self.next_layer.forward(input * self.drop_vec, target=target)

    def backward(self, error):
        """
        Backpropagation
        there are no trainable params so just returns delta
        :param error: delta from next layer
        :return: delta for layer before
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return error * self.drop_vec                                    # retuns detla of not droped out neurons
        else:
            self.layer_before.backward(error * self.drop_vec)

    def save(self):
        hyperparameter = ['Dropout', seperate_tuple(self.input_shape), seperate_tuple(self.dropout)]
        params = [np.array([]), np.array([]), np.array([]), np.array([])]
        return hyperparameter, params

    def take_snapshot(self):
        pass

    def load_snapshot(self, nr=-1):
        pass

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons (0), trainable-params (0), activation-function (-)
        """
        trainable_params = 0
        neurons = 0

        return 'Dropout', self.input_shape, self.output_shape, neurons, trainable_params, '-'
