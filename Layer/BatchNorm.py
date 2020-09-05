import numpy as np
from helperfunctions import *
from optimizer import *


class BatchNormalization:
    def __init__(self, input_shape, optimizer, alpha=0.98, layer_before=None, next_layer=None, trainings_mode=True):
        """
        Batch-Norm layer normalize the input-data over batch
        :param input_shape: shape of the input data (without batch_size
        :param optimizer: optimizer for updating the trainable params (gamma, beta)
        :param alpha: factor for moving mean and variance
        :param layer_before: link to the layer before
        :param next_layer: link to the next layer
        :param trainings_mode: has no effect on this layer
        """
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.dims = input_shape  # this will be a problem if input is not flat
        self.alpha = alpha
        self.step = 1

        self.mean_history = []
        self.var_history = []

        self.moving_mean_history = []
        self.moving_var_history = []
        self.multiplier_history = []

        self.trainings_mode = trainings_mode
        self.first_mode = False  # backward function does not return if True

        self.layer_before = layer_before
        self.next_layer = next_layer

        self.epsilon = 0.000001  # avoid dividing by zero
        # print(self.dims)
        if isinstance(self.dims, tuple):
            self.gamma = np.random.randn(*self.dims)  # is this a vector or a number?
            self.beta = np.random.randn(*self.dims)  # same here number or vector?
        else:
            self.gamma = np.random.randn(self.dims)  # is this a vector or a number?
            self.beta = np.random.randn(self.dims)  # same here number or vector?

        self.delta_gamma = np.zeros(self.dims)
        self.delta_beta = np.zeros(self.dims)

        self.moving_mean = np.zeros(input_shape)
        self.moving_var = np.zeros(input_shape)

        # optimizer
        self.optimizer = optimizer  # safes the optimizer
        self.gamma_update = self.optimizer.Update(self.optimizer, self.gamma.shape)  # creates in instance of the optimizer (inner
        self.beta_update = self.optimizer.Update(self.optimizer, self.beta.shape)  # class) for optimization of weights and biases

        # list of saves of the trainable params
        self.snapshot = []

    @classmethod
    def from_dict(cls, params, optimizer, layer_before=None):
        """
        init layer from a dict
        :param params: dict with params
        :param optimizer: optimizer for updating the trainable params (gamma, beta)
        :param layer_before: link to the layer before
        :return: layer-object
        """
        trainings_mode = True if not 'trainings_mode' in params.keys() else params['trainingsmode']
        alpha = 0.1 if 'alpha' not in params.keys() else params['alpha']
        return cls(params['input_shape'], optimizer, alpha=alpha, layer_before=layer_before,
                   trainings_mode=trainings_mode)

    @classmethod
    def from_load(cls, load, optimizer, layer_before):
        """
        init the layer from saves
        :param load: pramas in the save-file (csv)
        :param optimizer: optimizer for updating the trainable params (gamma, beta)
        :param layer_before: link to the layer before
        :return: Layer-object
        """
        input_shape, alpha = load
        return cls(retuple(input_shape), optimizer=optimizer, alpha=float(alpha), layer_before=layer_before)

    def set_weights_biases(self, params):
        """
        set the values of gamma and beta to the values in params
        :param params: list with to np-arrays (gamma, beta
        :return:
        """
        self.gamma = params[0]  # set gamma-params and
        self.beta = params[1]  # beta-params to the loaded values
        self.moving_mean, self.moving_var = params[2], params[3]

    def set_trainingsmode(self, mode=True):
        self.trainings_mode = mode

    def set_next_layer(self, next_layer=None):
        self.next_layer = next_layer

    def set_layer_before(self, layer_before=None):
        self.layer_before = layer_before

    def forward(self, input, target=None, give_return=False):

        self.Input = input
        batch_size = input.shape[0]

        if self.trainings_mode:
            self.mean = np.mean(input, axis=0)
            self.var = np.mean((input - self.mean) ** 2, axis=0)

            self.mean_history.append(self.mean[0])
            self.var_history.append(self.var[0])

            self.x_hat = (self.Input - self.mean) / (np.sqrt(self.var + self.epsilon))

            self.Output = self.gamma * self.x_hat + self.beta

        else:
            mean = self.moving_mean / (1 - self.alpha ** self.step)
            var = self.moving_var / (1 - self.alpha ** self.step)
            self.x_hat = (self.Input - mean) / (np.sqrt(var + self.epsilon))

            self.Output = self.gamma * self.x_hat + self.beta

        if self.next_layer is None or give_return:
            return self.Output
        else:
            self.next_layer.forward(self.Output, target=target)

    def backward(self, error):
        batch_size = error.shape[0]

        # updating the moving mean and moving var with Bias Correction of Exponentially Weighted Averages
        self.moving_mean = self.alpha * self.moving_mean
        self.moving_mean += (1 - self.alpha) * self.mean

        self.moving_var = (self.alpha * self.moving_var)
        self.moving_var += (1 - self.alpha) * self.var

        self.step += 1

        # backprop
        # scale and shift
        self.delta_beta = np.sum(error, axis=0)

        self.delta_gamma = np.sum(error * self.x_hat, axis=0)

        delta_x_hat = error * self.gamma

        # normalize
        delta_mu_1 = 1 / (np.sqrt(self.var + self.epsilon)) * delta_x_hat

        delta_denom = -1 / (self.var + self.epsilon) * np.sum((self.Input - self.mean) * delta_x_hat,
                                                              axis=0)

        delta_var = 1 / (2 * np.sqrt(self.var + self.epsilon)) * delta_denom

        # mini-batch-variance
        delta_mu_2 = 2 * (self.Input - self.mean) * np.full_like(self.Input, 1 / batch_size) * delta_var

        delta_mu = -1 * np.sum(delta_mu_1 + delta_mu_2, axis=0)

        delta_x_2 = (1 / batch_size) * np.ones_like(self.Input) * delta_mu

        delta_x_1 = (delta_mu_1 + delta_mu_2)
        new_error = (delta_x_1 + delta_x_2)

        if self.trainings_mode:
            self.update(error.shape[0])

        if self.first_mode:
            return

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
        # update the gamma-values
        self.gamma = self.gamma_update.update_params(self.gamma, self.delta_gamma / batch_size)

        # update the beta-values
        self.beta = self.beta_update.update_params(self.beta, self.delta_beta / batch_size)

    def save(self):
        # input_shape, neurons, filter_size, activation, optimizer, stride=1, padding='valid'
        hyperparameter = ['BatchNormalization', seperate_tuple(self.input_shape), self.alpha]
        params = [self.gamma, self.beta, self.moving_mean, self.moving_var]
        return hyperparameter, params

    def take_snapshot(self):
        self.snapshot.append([self.gamma, self.beta, self.moving_mean, self.moving_var])

    def load_snapshot(self, nr=-1):
        self.gamma, self.beta, self.moving_mean, self.moving_var = self.snapshot[nr]

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons, trainable-params, activation-function
        """
        trainable_params = np.prod(self.gamma.shape) + np.prod(self.beta.shape)     # count of trainable params
        neurons = 0                                                                 # count of neurons

        return 'BatchNormalization', self.input_shape, self.output_shape, neurons, trainable_params, '-'

