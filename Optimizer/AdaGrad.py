import numpy as np


class AdaGrad:
    def __init__(self, lr: float = .1, epsilon: float = 10**-7, initial_accumulator_value: float = 0.1,
                 weight_decay: float = 1.0):
        """
        adaptive gradient algorithm
        updates lr for every weight an bias individually
        outer_class to save the params
        inner_class used in the Layer to update trainable params
        :param lr: learning-rate (default: 1)
        :param epsilon: small number to avoid dividing zero (default: 1)
        :param initial_accumulator_value: initial value of alpha
        :param weight_decay: to reduce weights every step (default 1 (no reduce))
        """
        self.lr = lr
        self.epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value
        self.weight_decay = weight_decay

    class Update:
        """
        the inner_class to update the trainable params
        """
        def __init__(self, outer_class, shape: tuple):
            self.lr = outer_class.lr                        # init learning rate
            self.epsilon = outer_class.epsilon              # with epsilon 1 the used lr starts at self.lr
            self.alpha = np.full(shape, outer_class.initial_accumulator_value)                               # saves squared change of every param
            self.weight_decay = outer_class.weight_decay

        def update_params(self, weights: np.ndarray, delta: np.ndarray):
            """
            updates params
            :param weights: weights to update
            :param delta: delta from backpropagation
            :return: value to subtract from params
            """
            # alpha increases every update of the params so the lr decreases
            self.alpha += delta ** 2

            # calculation of the new weights
            weights -= self.lr * (delta / (np.sqrt(self.alpha) + self.epsilon))

            # reduce weights with weight-decay if not 1
            weights *= self.weight_decay
            return weights