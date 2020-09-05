import numpy as np
from activationfunctions import *
from helperfunctions import *


class Conv2D:
    def __init__(self, input_shape, neurons, filter_size, optimizer, activation=None, stride=1, padding='valid',
                 use_bias=True, layer_before=None, next_layer=None, trainings_mode=True):
        """
        Convolutional Layer
        :param input_shape: shape of the input (3d)
        :param neurons: number of different filters
        :param filter_size: size of the filter (2d)
        :param activation: activation-function
        :param optimizer: optimizer for update the trainable params
        :param stride: number of steps for filter
        :param padding: define the overlapping operation
        """
        # Input-shape and dims
        self.input_shape = input_shape
        self.dims, self.width, self.height = input_shape

        # filter-size
        self.filter_size = filter_size
        try:
            self.filter_width, self.filter_height = filter_size
        except TypeError:
            self.filter_width, self.filter_height = filter_size, filter_size

        # stride
        self.stride = stride
        try:
            self.stride_width, self.stride_height = stride
        except TypeError:
            self.stride_width, self.stride_height = stride, stride

        # Output-shape and dims
        self.neurons = neurons

        # the output shape depends onf the padding method
        self.padding = padding.lower()
        if padding.lower() == 'valid':
            self.out_width = int((self.width - self.filter_width) / self.stride_width) + 1
            self.out_height = int((self.height - self.filter_height) / self.stride_height) + 1

        elif padding.lower() == 'same':
            # output with the same size of input
            self.out_width, self.out_height = self.width, self.height

            # calculate the added pad of zeros to get th same sized output
            pad_width = self.stride_width * self.out_width + self.filter_width - 1
            pad_height = self.stride_height * self.out_height + self.filter_width - 1
            w_add, h_add = (pad_width - self.out_width), (pad_height - self.out_height)
            self.pad_w_add, self.pad_h_add = (int(w_add / 2), w_add - int(w_add / 2)), (int(h_add / 2), h_add - int(h_add / 2))
            print(pad_width, pad_height)
            print(self.pad_w_add, self.pad_h_add)

        else:
            raise AttributeError(f"\'{padding}\' is an invalid Attribute for padding. Must be \'valid\' or \'same\'.")

        self.output_shape = self.neurons, self.out_width, self.out_height

        # init weights and bias
        self.filter = np.random.randn(self.neurons, self.dims, self.filter_width, self.filter_height)
        self.bias = np.random.randn(self.neurons)

        self.filter_change = np.zeros_like(self.filter)
        self.bias_change = np.zeros_like(self.bias)

        # optimizer
        self.optimizer = optimizer
        self.weights_update = self.optimizer.Update(self.optimizer, self.filter.shape)
        self.bias_update = self.optimizer.Update(self.optimizer, self.bias.shape)

        # activation
        self.activation_name = activation                                # safe the name of the used activation-function
        if activation is not None:
            activation_dict = give_activation_dict()                        # get dict of activation-functions
            self.function, self.derivative = activation_dict[activation.lower()]

        # modes
        self.trainings_mode = trainings_mode                            # does not have any effect on this layer
        self.first_mode = False                                         # backward does not return if True
        self.use_bias = use_bias

        # links to next and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

        # List of saved states of filter and bias values
        self.snapshot = []

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """

        :param params: dictionary with all params
        :param optimizer: optimizer for update the trainable params
        :return: class
        """
        stride = 1 if not 'stride' in params.keys() else params['stride']
        padding = 'valid' if not 'padding' in params.keys() else params['padding']
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        use_bias = True if not 'use_bias' in params.keys() else params['use_bias']
        activation = None if not 'activation' in params.keys() else params['activation'].lower()
        return cls(params['input_shape'], params['neurons'], params['filter_size'], optimizer=optimizer,
                   activation=activation, stride=stride, padding=padding, use_bias=use_bias, layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before=None):
        input_shape, neurons, filter_size, activation, stride, padding, use_bias = load
        return cls(retuple(input_shape), int(neurons), retuple(filter_size), optimizer=optimizer, activation=activation,
                   stride=retuple(stride), padding=padding, use_bias=use_bias, layer_before=layer_before)

    def set_weights_biases(self, params):
        """
        just load some values for filter and bias
        :param params:
        :return:
        """
        self.filter = params[0]             # set filter-params and
        self.bias = params[1]               # biases to the loaded values

    def set_trainingsmode(self, mode=True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def forward(self, input, target=None, give_return=False):
        """
        forward-propagation
        :param input: Input from data or layer before (3d)
        :return: output of the layer (3d)
        """
        self.Input = input
        self.Z = np.zeros((self.Input.shape[0], ) + self.output_shape)
        self.Output = np.zeros((self.Input.shape[0], ) + self.output_shape)

        if self.padding == 'same':
            self.pad = np.pad(input, ((0, 0), (0, 0), self.pad_w_add, self.pad_h_add), mode='constant', constant_values=0.0)
        else:
            self.pad = input

        b, d, w, h = self.pad.shape

        for x in range(0, (w + 1 - self.filter_width), self.stride_width):
            for y in range(0, (h + 1 - self.filter_height), self.stride_height):
                # print(x, y)
                self.Z[:, :, int(x / self.stride_width), int(y / self.stride_height)] = \
                    np.tensordot(self.pad[:, :, x:x+self.filter_width, y:y+self.filter_height],
                                 self.filter, axes=([1, 2, 3], [1, 2, 3]))

        if self.use_bias:
            self.Z += self.bias[None, :, None, None]

        if self.activation_name is not None:
            self.Output = self.function(self.Z)
        else:
            self.Output = self.Z

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error):
        """
        backpropagation for layer
        :param error: Error from the Layer after
        :return: error for the Layer before
        """
        # a very nice Explanation for the backpropagation in convolutional Layer i found
        # in this article https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509
        # on towards data science by Pierre JAUMIER

        # backprop the activation function
        if self.activation_name is not None:
            delta = self.derivative(self.Z, error)
        else:
            delta = error

        # create a array delta_w for the delta of the weights (filter)
        self.filter_change = np.zeros_like(self.filter)

        # create a pad with zeros between the values of delta
        b, n, w, h = delta.shape

        delta_pad = np.zeros((b, n, w + (w - 1) * (self.stride_width - 1), h + (h - 1) * (self.stride_height - 1)), dtype=delta.dtype)
        delta_pad[:, :, ::self.stride_width, ::self.stride_height] = delta

        b, n, d_width, d_height = delta_pad.shape

        for x in range(self.filter_width):
            for y in range(self.filter_height):
                #                          b  d
                input_selection = self.pad[:, :, x:x+d_width, y:y+d_height]
                dw = np.tensordot(delta_pad, self.pad[:, :, x:x+d_width, y:y+d_height],
                                                axes=([0, 2, 3], [0, 2, 3]))
                self.filter_change[:, :, x, y] = dw

        # calculate error if bias is used
        if self.use_bias:
            self.bias_change = np.sum(delta, axis=(0, 2, 3))

        # update trainable params with optimizer if trainings_mode is True
        if self.trainings_mode:
            self.update(error.shape[0])

        # error of the input
        if self.first_mode:
            return None

        new_error = np.zeros_like(self.pad)

        weights = np.flip(self.filter, axis=[2, 3])
        b, n, w, h = weights.shape
        _, _, weights_pad_w, weights_pad_h = weights.shape

        b, n, w, h = delta.shape
        delta_pad = np.zeros((b, n, w + (w - 1) * (self.stride_width - 1), h + (h - 1) * (self.stride_height - 1)),
                              dtype=delta.dtype)
        delta_pad[:, :, ::self.stride_width, ::self.stride_height] = delta
        delta_pad = np.pad(delta_pad, ((0, 0), (0, 0), (weights_pad_w-1, weights_pad_w-1), (weights_pad_h-1, weights_pad_h-1)), mode='constant', constant_values=0.0)

        # print(delta_pad[0,0])
        _, _, delta_pad_w, delta_pad_h = delta_pad.shape
        for x_out, x_in in enumerate(range(0, (delta_pad_w - weights_pad_w + 1))):
            for y_out, y_in in enumerate(range(0, (delta_pad_h - weights_pad_h + 1))):
                new_error[:, :, x_out, y_out] = np.tensordot(
                    delta_pad[:, :, x_in:x_in + weights_pad_w, y_in:y_in+weights_pad_h], weights,
                    axes=([1, 2, 3], [0, 2, 3]))

        if self.padding == 'same':
            new_error = new_error[:, :, self.pad_w_add[0]: -self.pad_w_add[1], self.pad_h_add[0]:-self.pad_h_add[1]]

        if self.layer_before is None:
            return new_error
        else:
            self.layer_before.backward(new_error)

    def update(self, batch_size):
        """
        updates the weights of the filters and biases at the end of a batch
        uses the optimizer
        :param batch_size: size of the batch
        :return: None
        """
        # there will be error because of the shape of the change!!!
        self.filter = self.weights_update.update_params(self.filter, self.filter_change / batch_size)    # update weights

        if self.use_bias:
            self.bias = self.bias_update.update_params(self.bias, self.bias_change / batch_size)           # update biases

    def save(self):
        # input_shape, neurons, filter_size, activation, optimizer, stride=1, padding='valid'
        hyperparameter = ['Conv2D', seperate_tuple(self.input_shape), self.neurons,
                          seperate_tuple(self.filter_size), self.activation_name, seperate_tuple(self.stride),
                          self.padding, self.use_bias]
        params = [self.filter, self.bias, np.array([]), np.array([])]
        return hyperparameter, params

    def take_snapshot(self):
        self.snapshot.append([self.filter, self.bias])

    def load_snapshot(self, nr=-1):
        self.filter, self.bias = self.snapshot[nr]

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons, trainable-params, activation-function
        """
        trainable_params = self.neurons * self.filter_width * self.filter_height       # count of trainable params
        neurons = self.neurons                                                         # count of neurons

        return 'Conv2D', self.input_shape, self.output_shape, neurons, trainable_params, self.activation_name

