import numpy as np
from typing import Union
from activationfunctions import *
from helperfunctions import *


class Dense:
    def __init__(self, input_shape: tuple, neurons: int, optimizer: Union[str, object],
                 activation: Union[None, str] = None, use_bias: bool = True, layer_before: object = None,
                 next_layer: object = None, trainings_mode: bool = True):
        """
        a fully connected dense Layer
        :param input_size: size of the input-vector
        :param neurons: count of the neurons in this layer
        :param activation: activation function for this layer
        :param optimizer: optimizer for weight and bias update
        :param trainings_mode: do not update trainable params if False
        """
        # define neurons (count), input- and output-shape
        self.neurons = neurons
        self.input_shape = input_shape
        self.output_shape = (neurons)

        # iniit the trainable params
        self.weights = np.random.randn(neurons, np.prod(np.array(input_shape)))
        self.bias = np.random.randn(neurons)

        # init the activation function if not None
        self.activation_name = activation                                # safe the name of the used activation-function
        if activation is not None:
            activation_dict = give_activation_dict()                        # get dict of activation-functions
            self.function, self.derivative = activation_dict[activation.lower()]    # activation-function and its derivative

        # var for saving Output-, Input and Z-tensor
        self.Output = None                                              # saves the last output of the neurons
        self.Input = None                                               # save the last input to the layer
        self.Z = None                                                   # last output without the activation-function

        # saves the calculated change for update function
        self.change_weights = np.zeros_like(self.weights.T)             # saves the sum of all calculated changes for
        self.change_bias = np.zeros_like(self.bias)                     # one batch

        # define the optimizer and creates an instance for every trainable tensor
        self.optimizer = optimizer                                      # safes the optimizer
        self.weights_update = self.optimizer.Update(self.optimizer, self.weights.shape)     # creates in instance of the optimizer (inner
        self.bias_update = self.optimizer.Update(self.optimizer, self.bias.shape)        # class) for optimization of weights and biases

        # modes
        self.trainings_mode = trainings_mode                            # False => no updates for trainable params
        self.first_mode = False                                         # if True backward-function does not return

        # use an bias
        self.use_bias = use_bias

        # links to next layer and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

        # list of snapshots
        self.snapshot = []

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the Dense-Layer-class with params in a dictionary
        :param params: dict with prams for init
        :return: cls-object                                             # I'm not sure what exactly it is
        """
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        use_bias = True if not 'use_bias' in params.keys() else params['use_bias']
        activation = None if not 'activation' in params.keys() else params['activation'].lower()
        return cls(params['input_shape'], params['neurons'], optimizer=optimizer, activation=activation,
                   use_bias=use_bias, layer_before=layer_before, trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, neurons, activation, use_bias = load
        activation = None if activation is '' else activation
        return cls(retuple(input_shape), int(neurons), optimizer=optimizer, activation=activation, use_bias=use_bias,
                   layer_before=layer_before)

    def save(self):
        # input_shape, neurons, activation
        hyperparameter = ['Dense', seperate_tuple(self.input_shape), seperate_tuple(self.neurons), self.activation_name,
                          self.use_bias]
        params = [self.weights, self.bias, np.array([]), np.array([])]
        return hyperparameter, params

    def set_firstmode(self, mode=False):
        self.first_mode = mode

    def set_weights_biases(self, params):
        """
        just load some values for weights and bias
        :param params:
        :return:
        """
        self.weights = params[0]  # set filter-params and
        self.bias = params[1]  # biases to the loaded values

    def set_trainingsmode(self, mode=True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def forward(self, input, target=None, give_return=False):
        """
        forward_propagation [activation(input * weights + bias)]
        :param input: input to the Layer
        :return: the output of the layer
        """
        self.Input = np.array(input)                                            # save input in self.Input
        self.Z = np.tensordot(self.weights, input.T, axes=([1], [0])).T         # Z = w * I + b

        if self.use_bias:
            self.Z += self.bias

        if self.activation_name is not None:
            self.Output = self.function(self.Z)                                 # Output = activation-function(Z)
        else:
            self.Output = self.Z

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error, return_input_error: bool = False):
        """
        Backpropagation
        :param error: delta from next layer
        :return: delta for layer before
        """
        # backprop of activation
        if self.activation_name is not None:
            delta = self.derivative(self.Z, error)
        else:
            delta = error

        # calculate delta of weights
        self.change_weights = np.tensordot(self.Input.T, delta, axes=([1], [0])).T

        # calculate delta for bias
        if self.use_bias:
            self.change_bias = np.sum(delta, axis=0).T

        # update trainable params
        if self.trainings_mode:
            self.update(error.shape[0])

        # stop backprop if no more layer
        if self.first_mode:
            return None

        # calculates delta for input (the next layer)
        new_error = np.dot(delta, self.weights)

        # return new error if there is no layer before
        if self.layer_before is None:
            return new_error                                                                # next layer
        else:
            # if return input error is True return error for network-input
            if return_input_error:
                return self.layer_before.backward(new_error, return_input_error)
            # else start backprop for the layer before
            else:
                self.layer_before.backward(new_error)

    def update(self, batch_size):
        """
        updates the weights and biases at the end of a batch
        uses the optimizer
        :param batch_size: size of the batch
        :return: None
        """
        self.weights = self.weights_update.update_params(self.weights, self.change_weights / batch_size) # update weights

        if self.use_bias:
            self.bias = self.bias_update.update_params(self.bias, self.change_bias / batch_size)     # update biases

    def take_snapshot(self):
        self.snapshot.append([self.weights.copy(), self.bias.copy()])

    def load_snapshot(self, nr=-1):
        self.weights, self.bias = self.snapshot[nr]

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons, trainable-params, activation-function
        """
        trainable_params = len(self.weights) * len(self.weights[0]) + len(self.bias)        # count of trainable params
        neurons = len(self.weights)                                                         # count of neurons
        activation = self.activation_name if self.activation_name is not None else '-'

        return 'Dense', self.input_shape, self.output_shape, neurons, trainable_params, activation

