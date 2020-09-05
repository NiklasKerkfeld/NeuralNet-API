import numpy as np
from helperfunctions import *


class Reshape:
    def __init__(self, input_shape, output_shape, layer_before=None, next_layer=None, trainings_mode=True):
        """
        flattens the input data to Vector
        :param input_shape: shape of the input data
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.trainings_mode = trainings_mode
        self.first_mode = False

        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        trainings_mode = True if not 'trainings_mode' in params.keys() else params['trainingsmode']
        return cls(params['input_shape'], params['output_shape'], layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, output_shape = load
        return cls(retuple(input_shape), retuple(output_shape), layer_before=layer_before)

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
        """
        make input flatt
        :param input: input data from Layer before or data
        :return: Vector
        """
        if self.next_layer is None or give_return:
            return np.reshape(input, self.output_shape)
        else:
            self.next_layer.forward(np.reshape(input, self.output_shape), target=target)

    def backward(self, error):
        """
        make the flatt error into the input-shape
        :param error: error form Layer after
        :return: error with input-shape
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return np.reshape(error, self.input_shape)
        else:
            self.layer_before.backward(np.reshape(error, self.input_shape))

    def save(self):
        hyperparameter = ['Reshape', seperate_tuple(self.input_shape), seperate_tuple(self.output_shape)]
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

        return 'Reshape', self.input_shape, self.output_shape, neurons, trainable_params, '-'