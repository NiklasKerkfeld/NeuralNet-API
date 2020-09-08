import numpy as np
from helperfunctions import *


class GaussianNoise:
    def __init__(self, input_shape, standard_diviation=.1, layer_before=None, next_layer=None, trainings_mode=True):
        """
        Adds a zero-centere Gaussian Noise to the input
        :param input_shape: shape of the input
        :param optimizer: just for standardization
        :param layer_before: layer before
        :param next_layer: next layer
        :param trainings_mode: No noise if False
        """

        self.input_shape = input_shape
        self.output_shape = input_shape
        self.stadiv = standard_diviation

        self.Output = None                                              # saves the last output of the neurons
        self.Input = None                                               # save the last input to the layer
        self.Noise = None                                               # last output without the activation-function

        self.trainings_mode = trainings_mode                            # trainings-mode has no effect on this layer
        self.first_mode = False                                         # if True backward-function does not return

        self.layer_before = layer_before
        self.next_layer = next_layer

        self.snapshot = []

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the Dense-Layer-class with params in a dictionary
        :param params: dict with prams for init
        :return: cls-object                                             # I'm not sure what exactly this is
        """
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        standard_diviation = .1 if not 'standard_diviation' in params.keys() else params['standard_diviation']
        return cls(params['input_shape'], standard_diviation, layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, neurons, activation, use_bias = load
        activation = None if activation is '' else activation
        return cls(retuple(input_shape), layer_before=layer_before)

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
        forward_propagation [activation(input * weights + bias)]
        :param input: input to the Layer
        :return: the output of the layer
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

    def backward(self, error):
        """
        Backpropagation
        just returns the error form layer before
        :param error: delta from next layer
        :return: delta for layer before
        """
        if self.first_mode:
            return None

        if self.layer_before is None:
            return error
        else:
            self.layer_before.backward(error)

    def save(self):
        # input_shape, neurons, activation
        hyperparameter = ['GaussianNoise', seperate_tuple(self.input_shape), seperate_tuple(self.stadiv)]
        params = [np.array([]), np.array([]), np.array([]), np.array([])]
        return hyperparameter, params

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
