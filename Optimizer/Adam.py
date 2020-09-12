import numpy as np


class Adam:
    def __init__(self, lr: float = 0.001, beta_1: float = .9, beta_2: float = .999, epsilon: float = 10**-7,
                 weight_decay: float = 1.0):
        """
         Adaptive Moment Estimation
        :param lr: learning-rate
        :param beta_1: beta for exponential weighted avarage of delta
        :param beta_2: beta for exponential weighted avarage of delta squared
        :param epsilon: small number for avoiding dividing by zero
        :param weight_decay: dacay of weights every time it's updated
        """
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    class Update:
        """
        the inner_class to update the trainable params
        """
        def __init__(self, outer_class, shape):
            self.lr = outer_class.lr
            self.beta_1 = outer_class.beta_1
            self.beta_2 = outer_class.beta_2
            self.epsilon = outer_class.epsilon
            self.weight_decay = outer_class.weight_decay

            self.Vd = np.zeros(shape)
            self.Sd = np.zeros(shape)
            self.step = 0

        def update_params(self, weights, delta):

            # exponential weighted avarage of delta and delta squared
            self.Vd = self.beta_1 * self.Vd + (1 - self.beta_1) * delta
            self.Sd = self.beta_2 * self.Sd + (1 - self.beta_2) * (delta ** 2)
            self.step += 1

            # Bias Correction
            Vd_correct = self.Vd / (1 - self.beta_1 ** self.step)
            Sd_correct = self.Sd / (1 - self.beta_2 ** self.step)

            # calculating change
            weights -= self.lr * (Vd_correct / (np.sqrt(Sd_correct) + self.epsilon))
            weights *= self.weight_decay
            return weights