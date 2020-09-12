import numpy as np


class RMSprop:
    def __init__(self, lr: float = .001, epsilon: float = 10**-7, beta: float = .99, weight_decay: float = 1.0):
        """
        adaptive gradient algorithm
        updates lr for every weight an bias individually
        outer_class to save the params
        inner_class used in the Layer to update trainable params
        :param lr: learning-rate (default: 1)
        :param epsilon: small number to avoid dividing zero (default: 1)
        :param beta: beta for exponential weighted avarage of delta squared
        :param: weight_decay: decay of the weights every update
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

        def update_params(self, weights: np.ndarray,  delta: np.ndarray):
            """
            updates params
            :param weights: weights to update
            :param delta: delta from backpropagation
            :return: new weights
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
