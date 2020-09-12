import numpy as np
from typing import Union
from helperfunctions import *


class Reshape:
    def __init__(self, input_shape: tuple, output_shape: tuple, layer_before: Union[None, object] = None,
                 next_layer: Union[None, object] = None, trainings_mode: bool = True):
        """
        reshapes the input data to a given shape
        :param input_shape: shape of the input
        :param output_shape: shape of the output
        :param layer_before: link to the layer before
        :param next_layer:  link to the next layer
        :param trainings_mode:
        """
        # input- and output shape
        self.input_shape = input_shape
        self.output_shape = output_shape

        # modes
        self.trainings_mode = trainings_mode
        self.first_mode = False

        # links to the next layer and the layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init layer from a dict
        :param params: dict of hyperparams
        :param optimizer: just for standardization
        :param layer_before: link to layer before
        :return:
        """
        trainings_mode = True if not 'trainings_mode' in params.keys() else params['trainingsmode']
        return cls(params['input_shape'], params['output_shape'], layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        """
        init layer from a save file
        :param load: List of hyperparams
        :param optimizer: just for standardization
        :param layer_before: link to layer before
        :return:
        """
        input_shape, output_shape = load
        return cls(retuple(input_shape), retuple(output_shape), layer_before=layer_before)

    def save(self):
        hyperparameter = ['Reshape', seperate_tuple(self.input_shape), seperate_tuple(self.output_shape)]
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
        :param input: input-tensor
        :param target: target(s) of the batch
        :param give_return: return Output if True
        :return:
        """
        if self.next_layer is None or give_return:
            return np.reshape(input, self.output_shape)
        else:
            self.next_layer.forward(np.reshape(input, self.output_shape), target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        make the flatt error into the input-shape
        :param error: error of output
        :param return_input_error: returns the error of the network input if True
        :return:
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return np.reshape(error, self.input_shape)
        else:
            self.layer_before.backward(np.reshape(error, self.input_shape))

    def take_snapshot(self):
        """
        no trainable params to save
        :return:
        """
        pass

    def load_snapshot(self, nr: int = -1):
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

        return 'Reshape', self.input_shape, self.output_shape, neurons, trainable_params, '-'