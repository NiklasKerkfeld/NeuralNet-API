import numpy as np
from typing import Union
from helperfunctions import *


class Flatten:
    def __init__(self, input_shape: tuple, layer_before: Union[None, object] = None,
                 next_layer: Union[None, object] = None, trainings_mode: bool = True):
        """
        flattens the input data to Vector
        :param input_shape: shape of the input data
        """
        # input and output shape
        self.input_shape = input_shape
        self.output_shape = (np.prod([input_shape]),)

        # trainings- and fist mode
        self.trainings_mode = trainings_mode        # has no effect
        self.first_mode = False

        # link to next layer and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init layer form a dict
        :param params: dict of hyperparams
        :param optimizer: just for standardization
        :param layer_before: link to layer before
        :return: instance of the layer
        """
        trainings_mode = True if not 'trainings_mode' in params.keys() else params['trainingsmode']
        return cls(params['input_shape'], layer_before=layer_before, trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        """
        inits layer from save-files
        :param load: list of hyperparams
        :param optimizer: just for standardization
        :param layer_before: link to layer before
        :return: instance of the layer
        """
        input_shape = load
        return cls(retuple(input_shape[0]), layer_before=layer_before)

    def save(self):
        """
        returns hyperparameter for saving the model
        :return:
        """
        hyperparameter = ['Flatten', seperate_tuple(self.input_shape)]
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
        make input flatt
        :param input: input tensor for the layer
        :param target: target(s) of the batch
        :param give_return: return Output if True
        :return: flatt input
        """
        if self.next_layer is None or give_return:
            return np.reshape(input, self.output_shape)
        else:
            self.next_layer.forward(np.reshape(input, ((input.shape[0],) + self.output_shape)), target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        make the flatt error into the input-shape
        :param error: error of the output
        :param return_input_error: returns error of network input  if True
        :return:
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return np.reshape(error, ((error.shape[0],) + self.input_shape))
        else:
            if return_input_error:
                return self.layer_before.backward(np.reshape(error, ((error.shape[0],) + self.input_shape)),
                                                  return_input_error)
            else:
                self.layer_before.backward(np.reshape(error, ((error.shape[0],) + self.input_shape)))

    def take_snapshot(self):
        pass

    def load_snapshot(self, nr: int = -1):
        pass

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons (0), trainable-params (0), activation-function (-)
        """
        trainable_params = 0
        neurons = 0

        return 'Flatten', self.input_shape, self.output_shape, neurons, trainable_params, '-'