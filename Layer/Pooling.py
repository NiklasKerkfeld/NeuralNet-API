import numpy as np
from typing import Union, Tuple
from helperfunctions import *


class Pooling:
    def __init__(self, input_shape: tuple, filter_size: Union[int, Tuple[int]], stride: Union[int, Tuple[int]] = (1, 1),
                 pooling: str = 'max', layer_before: Union[None, object] = None, next_layer: Union[None, object] = None,
                 trainings_mode: bool = True):
        """
        reduces the count of datapoints with pooling
        :param input_shape: shape of the input data
        :param filter_size: size of the pooling filter
        :param stride: distance between filter (x, y)
        :param pooling: method for pooling
        """
        # hyperparams
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.stride = stride

        # define filter height and width
        try:
            self.filter_width, self.filter_height = self.filter_size
        except TypeError:
            self.filter_width, self.filter_height = self.filter_size, self.filter_size

        # define stride height and width
        try:
            self.stride_width, self.stride_height = self.stride
        except TypeError:
            self.stride_width, self.stride_height = self.stride, self.stride

        # define pooling method (max, min, avarage)
        self.pooling = pooling
        if pooling.lower() == 'max':
            self.method = np.amax
            self.idx_method = np.argmax
        elif pooling.lower() == 'min':
            self.method = np.amin
            self.idx_method = np.argmin
        elif pooling.lower() == 'avg' or 'average':
            self.pooling = 'avg'
            self.method = np.mean
            self.multiplyer = 1 / (self.filter_width * self.filter_height)
        else:
            raise ImportError(f'pooling method \'{pooling}\' not understood must be \'max\', \'min\' or \'avg\'')

        # define input params
        self.in_neurons = input_shape[0]
        self.in_width = input_shape[1]
        self.in_height = input_shape[2]

        # define output params
        self.out_neurons = self.in_neurons
        self.out_width = int((self.in_width - self.filter_width) / self.stride_width + 1)
        self.out_height = int((self.in_height - self.filter_height) / self.stride_height + 1)
        self.output_shape = (self.out_neurons, self.out_width, self.out_height)

        # modes
        self.trainings_mode = trainings_mode
        self.first_mode = False                                     # backward function does not return if True

        # link to next layer and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the layer form dict
        :param params: dictionary with all params
        :param optimizer: just for standardization not needed in this layer
        :param layer_before: link to layer before
        :return: instance of layer
        """
        stride = 1 if not 'stride' in params.keys() else params['stride']
        pooling = 'max' if not 'pooling' in params.keys() else params['pooling']
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        return cls(params['input_shape'], params['filter_size'], stride=stride, pooling=pooling,
                   layer_before=layer_before, trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        """
        init the layer form save files
        :param load: List with hyperparams
        :param optimizer: just for standardization not needed in this layer
        :param layer_before: link to layer before
        :return: instance of layer
        """
        input_shape, filter_size, stride, pooling = load
        return cls(retuple(input_shape), retuple(filter_size), retuple(stride), pooling, layer_before=layer_before)

    def save(self):
        hyperparameter = ['Pooling', seperate_tuple(self.input_shape), seperate_tuple(self.filter_size), seperate_tuple(self.stride),
                          self.pooling]
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

    def set_firstmode(self, mode=False):
        self.first_mode = mode

    def set_trainingsmode(self, mode=True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def _reval_index(self, in_filter_index, array_index, filter_width):
        arr = np.zeros((in_filter_index.shape + (2,)))
        arr[:, :, 0] = in_filter_index % filter_width
        arr[:, :, 1] = in_filter_index // filter_width
        arr += array_index
        return arr

    def forward(self, input: np.ndarray, target: Union[None, np.ndarray] = None, give_return: bool = False):
        """
        forward propagation of the layer
        :param input: input-tensor
        :param target: target of the batch
        :param give_return: returns Output if True
        :return:
        """
        self.Output = np.zeros((input.shape[0],) + self.output_shape)

        # convolutes over the input tensor width and height
        # if the pooling method is min or max
        # save the position of min and max in Cache
        if self.pooling != 'avg':
            self.Cache = np.zeros(((input.shape[0],) + self.output_shape + (2,)), dtype=np.intp)

            for x_out, x_in in enumerate(range(0, self.in_width - (self.filter_width - 1), self.stride_width)):
                for y_out, y_in in enumerate(range(0, self.in_height - (self.filter_height - 1), self.stride_height)):
                    self.Output[:, :, x_out, y_out] = self.method(
                        input[:, :, x_in:x_in + self.filter_width, y_in:y_in + self.filter_height], axis=(2, 3))
                    self.Cache[:, :, x_out, y_out] = self._reval_index(
                        in_filter_index=self.idx_method(
                            input[:, :, x_in:x_in + self.filter_width, y_in:y_in + self.filter_height].reshape(
                            input.shape[0], self.output_shape[0], np.prod(self.filter_size)), axis=2),
                        array_index=np.array(input.shape[0] * [self.output_shape[0] * [[x_in, y_in]]]),
                                              filter_width=self.filter_width)
        else:
            # if the pooling method is avg
            for x_out, x_in in enumerate(range(0, self.in_width - (self.filter_width - 1), self.stride_width)):
                for y_out, y_in in enumerate(range(0, self.in_height - (self.filter_height - 1), self.stride_height)):
                    self.Output[:, :, x_out, y_out] = self.method(
                        input[:, :, x_in:x_in + self.filter_width, y_in:y_in + self.filter_height], axis=(2, 3))

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        backpropagation of the layer
        :param error: error of the layer output
        :param return_input_error: returns error of the network input if True
        :return:
        """
        if self.first_mode:
            return None

        if self.pooling == 'avg':
            delta = error * self.multiplyer
            new_error = np.zeros((delta.shape[0],) + self.input_shape)
            b, d, error_w, error_h = delta.shape

            for x_err, x_in in enumerate(range(0, self.in_width - self.filter_width + 1, self.stride_width)):
                for y_err, y_in in enumerate(range(0, self.in_height - self.filter_height + 1, self.stride_height)):
                    new_error[:, :, x_in:x_in + self.filter_width, y_in:y_in + self.filter_height] += \
                        np.broadcast_to(delta[:, :, x_err, y_err, None, None], (b, d, self.filter_width, self.filter_height))

        else:
            new_error = np.zeros((error.shape[0], ) + self.input_shape)
            b, d, width, heights = new_error.shape
            bd = np.flip(np.mgrid[0:d, 0:b], 0).T
            for x in range(self.out_width):
                for y in range(self.out_height):
                    indexes = np.append(bd, self.Cache[:,:,x,y], axis=2).reshape((b*d, 4))
                    for index in indexes:
                        new_error[tuple(index)] += error[index[0], index[1], x, y]

        if self.layer_before is None:
            return new_error  # next layer
        else:
            # if return input error is True return error for network-input
            if return_input_error:
                return self.layer_before.backward(new_error, return_input_error)
            # else start backprop for the layer before
            else:
                self.layer_before.backward(new_error)

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
        :return: Name, neurons, trainable-params, activation-function
        """
        trainable_params = 0                    # count of trainable params
        neurons = 0                             # count of neurons

        return 'Pooling', self.input_shape, self.output_shape, neurons, trainable_params, '-'