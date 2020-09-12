import numpy as np
from typing import Union
from helperfunctions import *


class Dropout:
    def __init__(self, input_shape: tuple, dropout: float = .1, layer_before: Union[None, object] = None,
                 next_layer: Union[None, object] = None, trainings_mode: bool = True):
        """
        randomly sets values of Inputs to zero
        :param input_size: size of the input tensor
        :param dropout: proportion of values turned to zero (default: .1)
        """
        # shape of input and output
        self.input_shape = input_shape
        self.output_shape = input_shape

        # dropout and dropout-tensor
        self.dropout = dropout
        self.drop_tensor = np.ones(self.input_shape)               # vector for multiplication

        # modes
        self.trainings_mode = trainings_mode                    # if False layer does nothing
        self.first_mode = False                                 # if True backward does not return anything

        # links to next layer and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

        self.Output = np.zeros(input_shape)

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

    def save(self):
        hyperparameter = ['Dropout', seperate_tuple(self.input_shape), seperate_tuple(self.dropout)]
        params = [np.array([]), np.array([]), np.array([]), np.array([])]
        return hyperparameter, params

    def set_weights_biases(self, params):
        """
        no trainable params
        this function is just for standardisation
        :param params:
        :return:
        """
        pass

    def set_firstmode(self, mode: bool = False):
        self.first_mode = mode

    def set_trainingsmode(self, mode: bool = True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def forward(self, input: np.ndarray, target: Union[None, np.ndarray] = None, give_return: bool = False):
        """
        forward propagation of the layer
        :param input: input tensor of the layer
        :param target: target(s) of the batch
        :param give_return: returns Output if True
        :return:
        """
        # just set values to zero if trainings-mode is True
        if self.trainings_mode:
            # drop tensor important for backpropagation
            self.drop_tensor = np.random.binomial(1, 1-self.dropout, self.input_shape)
            self.Output = input * self.drop_tensor
        else:
            # else just pass the input through the layer
            self.Output = input

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        Backpropagation
        there are no trainable params so just returns delta
        :param error: delta from next layer
        :return: delta for layer before
        :param return_input_error: returns error for network input if True
        """
        # return if first layer in network
        if self.first_mode:
            return None

        if self.trainings_mode:
            new_error = error * self.drop_tensor
        else:
            new_error = error

        if self.layer_before is None:
            return new_error                                 # retuns detla of not droped out neurons
        else:
            if return_input_error:
                return self.layer_before.backward(new_error, return_input_error)
            else:
                self.layer_before.backward(new_error)

    def take_snapshot(self):
        """
        no trainable params to save
        :return:
        """
        pass

    def load_snapshot(self, nr=-1):
        """
        no trainable params to load
        :return:
        """
        pass

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons (0), trainable-params (0), activation-function (-)
        """
        trainable_params = 0
        neurons = 0

        return 'Dropout', self.input_shape, self.output_shape, neurons, trainable_params, '-'
