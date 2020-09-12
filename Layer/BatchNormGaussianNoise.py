import numpy as np
from typing import Union
from helperfunctions import *


class BatchNormGaussianNoise:
    def __init__(self, input_shape: np.ndarray, scale: float = .1, layer_before: Union[None, object] = None,
                 next_layer: Union[None, object] = None, trainings_mode: bool = True):
        """
        This layer noises with a noise orientated on the variance of the batch
        Adds a zero-centere Gaussian Noise to the input
        :param input_shape: shape of the input
        :param optimizer: just for standardization
        :param layer_before: layer before
        :param next_layer: next layer
        :param trainings_mode: No noise if False
        """
        # define init params
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.scale = scale

        self.Output = None                                              # saves the last output of the neurons
        self.Input = None                                               # save the last input to the layer
        self.Noise = None                                               # last output without the activation-function

        # trainings- and first-mode
        self.trainings_mode = trainings_mode                            # trainings-mode has no effect on this layer
        self.first_mode = False                                         # if True backward-function does not return

        # links to the next layer and the layer before
        self.layer_before = layer_before
        self.next_layer = next_layer

        # snapshots of trainable prams
        self.snapshot = []

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init the Layer with params from a dictionary
        :param params: dict with prams for init
        :return: cls-object                                             # I'm not sure what exactly it is
        """
        trainings_mode = True if not 'traingingsmode' in params.keys() else params['trainingsmode']
        scale = .1 if not 'scale' in params.keys() else params['scale']
        return cls(params['input_shape'], scale, layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        input_shape, scale = load
        return cls(retuple(input_shape), float(scale), layer_before=layer_before)

    def save(self):
        hyperparameter = ['GaussianNoise', seperate_tuple(self.input_shape), self.scale]
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
        :param input: input to the Layer
        :param target: target of the forward propagation
        :param give_return: returns output if True
        :return: the output of the layer
        """
        if self.trainings_mode:
            # make some the noise
            mean = np.mean(input, axis=0)
            var = np.mean((input - mean) ** 2, axis=0)
            stadiv = np.sqrt(var)

            self.Noise = np.random.normal(0, self.scale, input.shape) * stadiv

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

    def load_snapshot(self, nr: int = -1):
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
    from Layer.GaussianNoise import GaussianNoise
    from activationfunctions import Relu_forward
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='mnist.npz')

    x_test = x_test / 255
    x_test = x_test.reshape(len(x_test), 1, 28, 28)
    print(x_test.shape[1:])

    NoiseLayer = GaussianNoise(x_test.shape[1:], .2)
    BatchNormGaussianNoiseLayer = BatchNormGaussianNoise(x_test.shape[1:], .5)
    DropoutLayer = Dropout(x_test.shape[1:], dropout=.2)

    # show_images(x_test[:16])

    out = NoiseLayer.forward(x_test, give_return=True)
    out = DropoutLayer.forward(out, give_return=True)
    show_images(out[:16])

    out = BatchNormGaussianNoiseLayer.forward(x_test, give_return=True)
    out = Relu_forward(out)
    out /= np.amax(out)
    print(np.amin(out), np.amax(out))

    print(x_test[0, 0, 0])
    print(out[0, 0, 0])
    show_images(out[:16])
