import numpy as np


class Default:
    def __init__(self, lr=.2, lr_decay=.9999):
        """
        optimizer with no fancy shit at all
        :param lr: learning-rate (default: 0.2)
        :param lr_decay: learning- rate-decay lr = lr * decay (default: 0.9999)
         """
        self.lr = lr
        self.decay = lr_decay

    class Update:
        def __init__(self, outer_class, shape):
            """this inner-class is the real Optimizer
            use update_params to update the params"""
            self.lr = outer_class.lr
            self.decay = outer_class.decay

        def update_params(self, weights, delta):
            """
            returns the actial update for the params at the end of the batch
            :param change: in backpropagation calculated sum of changes
            :param batch_size: number of examples in the batch
            :return: the actual change for the params
            """
            weights -= self.lr * delta
            self.lr *= self.decay
            return weights