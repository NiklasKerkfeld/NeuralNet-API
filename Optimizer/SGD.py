import numpy as np


class SGD:
    def __init__(self, lr: float = .4, lr_decay: float = .9999, momentum: float = .9, weight_decay: float = 1.0):
        """
        stochastic gradient decent
        outer_class to save params
        :param lr: learning-rate
        :param lr_decay: decay for learning-rate every batch (lr *= decay)
        :param momentum: momentum of change
        :param weight_decay: decay of weights every update
        """

        self.lr = lr
        self.decay = lr_decay
        self.use_momentum = True if momentum < 1 else False
        self.beta = momentum
        self.weight_decay = weight_decay

    class Update:
        """
        inner_class updates the params in layer
        """
        def __init__(self, outer_class, shape):
            # getting the params from outer_class
            self.lr = outer_class.lr
            self.decay = outer_class.decay

            self.use_momentum = outer_class.use_momentum
            self.beta = outer_class.beta

            self.weight_decay = outer_class.weight_decay

            # for saving the old changes
            self.change = np.zeros(shape)
            self.steps = 1

        def update_params(self, weights,  delta):
            """
            returns a change for subtract from the trainable param
            :param delta: delta for the trainable param from the backpropagation
            :return: change
            """
            # no momentum in the 1st update
            if self.use_momentum:
                self.change = self.beta * self.change + (1 - self.beta) * (self.lr * delta)

                # bias correction
                self.change *= (1 - self.beta ** self.steps)

            self.change = self.lr * delta

            # update the weights
            weights -= self.change
            weights *= self.weight_decay

            # reduce the learning-rate
            self.lr *= self.decay
            return weights
