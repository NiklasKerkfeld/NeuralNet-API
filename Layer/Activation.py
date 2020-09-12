from typing import Union
from activationfunctions import *
from helperfunctions import *


class Activation:
    def __init__(self, input_shape: np.ndarray, activation: str, layer_before=None, next_layer=None,
                 trainings_mode: bool = True):
        """
        Layer with an activation-function
        :param input_shape: shape of the input data
        :param activation: function to use
        :param layer_before: layer in the network before
        :param next_layer: next layer in the network
        :param trainings_mode: does nothing in this Layer
        """
        # define input- and output-shape
        self.input_shape = input_shape
        self.output_shape = input_shape

        # define activation function and the derivative
        self.activation_name = activation                               # safe the name of the used activation-function
        activation_dict = give_activation_dict()                        # get dict of activation-functions
        self.function, self.derivative = activation_dict[activation.lower()]    # activation-function and its derivative

        # define trainings- and first-mode
        self.trainings_mode = trainings_mode                        # trainings-mode has no effect on this layer
        self.first_mode = False                                     # if True backward-function does not return error

        # links for the next layer and the layer before
        self.layer_before = layer_before                            # link the layer before
        self.next_layer = next_layer                                # link the next layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the Activation-Layer-class with params in a dictionary
        :param params: dict with prams for init
        :param optimizer: this layer need no optimizer
        :param layer_before: link to the layer before
        :return: cls-object                                             # I'm not sure what exactly it is
        """
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        return cls(params['input_shape'], activation=params['activation'].lower(), layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, activation = load
        return cls(retuple(input_shape), activation=activation, layer_before=layer_before)

    def save(self):
        # input_shape, neurons, activation
        hyperparameter = ['Activation', seperate_tuple(self.input_shape), self.activation_name]
        params = [np.array([]), np.array([]), np.array([]), np.array([])]
        return hyperparameter, params

    def set_weights_biases(self, params):
        """
        just for standardization no use in this Layer
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
        forward_propagation [activation(input * weights + bias)]
        :param input: input to the Layer
        :return: the output of the layer
        """
        self.Input = input
        self.Output = self.function(input)

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        Backpropagation
        :param error: delta from next layer
        :return: delta for layer before
        """

        if self.first_mode:                                                                 # stop backprop if no more layer
            return None

        new_error = self.derivative(self.Input, error)

        if self.layer_before is None:
            return new_error                                                                # next layer
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
        :param nr:
        :return:
        """
        pass

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons, trainable-params, activation-function
        """
        trainable_params = 0        # count of trainable params
        neurons = 0                                                        # count of neurons

        return 'Activation', self.input_shape, self.output_shape, neurons, trainable_params, self.activation_name
