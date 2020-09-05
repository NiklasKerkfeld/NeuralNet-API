import numpy as np


class RMSprop:
    def __init__(self, lr=.001, epsilon=10**-7, beta=.99, weight_decay=1):
        """
        adaptive gradient algorithm
        updates lr for every weight an bias individually
        outer_class to save the params
        inner_class used in the Layer to update trainable params
        :param lr: learning-rate (default: 1)
        :param epsilon: small number to avoid dividing zero (default: 1)
        :param momentum: momentum must be between 0 and 1; 0 means no momentum (default: 0)
        """
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta
        self.weight_decay = weight_decay

    class Update:
        """
        the inner_class to update the trainable params
        """
        def __init__(self, outer_class, shape):
            self.lr = outer_class.lr                        # init learning rate
            self.epsilon = outer_class.epsilon              # with epsilon 1 the used lr starts at self.lr
            self.beta = outer_class.beta
            self.Sd = np.zeros(shape)                               # saves squared change of every param
            self.step = 0
            self.weight_decay = outer_class.weight_decay
            # self.change_history = []

        def update_params(self, weights,  delta):
            """
            updates params
            :param delta: delta from backpropagation
            :param batch_size: size of the mini-batch
            :return: value to subtract from params
            """
            # exponential weighted average of Squared delta
            self.Sd = self.beta * self.Sd + (1 - self.beta) * (delta ** 2)
            self.step += 1

            # bias correction of Squared delta
            Sd = self.Sd / (1 - self.beta ** self.step)

            # compute the new weight values
            weights -= self.lr * (delta / (np.sqrt(Sd) + self.epsilon))
            weights *= self.weight_decay
            return weights