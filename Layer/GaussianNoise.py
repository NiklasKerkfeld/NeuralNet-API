import numpy as np
from typing import Union
from helperfunctions import *


class GaussianNoise:
    def __init__(self, input_shape: tuple, standard_diviation: float = .1, layer_before: Union[None, object] = None,
                 next_layer: Union[None, object] = None, trainings_mode: bool = True):
        """
        Adds a zero-centere Gaussian Noise to the input
        :param input_shape: shape of the input
        :param optimizer: just for standardization
        :param layer_before: layer before
        :param next_layer: next layer
        :param trainings_mode: No noise if False
        """
        # init-params
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.stadiv = standard_diviation

        self.Output = np.zeros(input_shape)        # saves the last output of the neurons
        self.Input = np.zeros(input_shape)         # save the last input to the layer
        self.Noise = np.zeros(input_shape)         # last output without the activation-function

        # modes
        self.trainings_mode = trainings_mode        # trainings-mode has no effect on this layer
        self.first_mode = False                     # if True backward-function does not return

        # links to next layer and layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the Layer from a dict
        :param params: dict with prams for init
        :param optimizer: just for standardization
        :return: instance of the layer
        """
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        standard_diviation = .1 if not 'standard_diviation' in params.keys() else params['standard_diviation']
        return cls(params['input_shape'], standard_diviation, layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        """
        init the Layer with the params form a save-file
        :param params: List with prams for init
        :param optimizer: just for standardization
        :return: instance of the layer
        """
        input_shape, standard_diviation = load
        return cls(retuple(input_shape), float(standard_diviation), layer_before=layer_before)

    def save(self):
        hyperparameter = ['GaussianNoise', seperate_tuple(self.input_shape), self.stadiv]
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
        forward_propagation [activation(input * weights + bias)]
        :param input: input-tensor to the Layer
        :param target: target(s) of the batch
        :param give_return: returns Output if True
        :return:
        """
        if self.trainings_mode:
            # make some the noise
            self.Noise = np.random.normal(0, self.stadiv, input.shape)

            # add the noise
            self.Output = input + self.Noise
        else:
            self.Output = input

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error: np.ndarray, return_input_error: bool = False):
        """
        Backpropagation
        just returns the error form layer before
        :param error: delta from next layer
        :param return_input_error: returns error of the network input if True
        :return: delta for layer before
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return error
        else:
            if return_input_error:
                return self.layer_before.backward(error, return_input_error)
            else:
                self.layer_before.backward(error)

    def take_snapshot(self):
        """
        no params to safe
        :return:
        """
        pass

    def load_snapshot(self, nr=-1):
        """
        no params to load
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
        neurons = 0                                                      # count of neurons

        return 'GaussianNoise', self.input_shape, self.output_shape, neurons, trainable_params, '-'


if __name__ == '__main__':
    import tensorflow.keras as keras
    from plots import *
    from Layer.Dropout import Dropout
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='mnist.npz')

    x_test = x_test / 255
    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    print(x_test.shape[1:])

    NoiseLayer = GaussianNoise(x_test.shape[1:], .1)
    DropoutLayer = Dropout(x_test.shape[1:], dropout=.2)

    out = NoiseLayer.forward(x_test, give_return=True)
    out = DropoutLayer.forward(out, give_return=True)
    show_images(out[:16])
