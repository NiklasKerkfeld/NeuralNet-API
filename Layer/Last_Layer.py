from lossfunctions import *
from helperfunctions import *


class Last_Layer:
    def __init__(self, input_shape, loss_function, layer_before, trainings_mode=True):
        """
        this Layer must be the last in the Model. calculates loss and starts backprob
        :param loss_function: function to calculate loss
        """
        self.input_shape = input_shape

        self.function_name = loss_function
        loss_dict = give_loss_dict()
        self.Loss_func, self.Error_func, self.Acc_func = loss_dict[loss_function.lower()]
        # self.function, self.derivative = give_loss_dict(loss_function)
        self.layer_before = layer_before
        self.next_layer = None

        self.loss_history = []                                              # maybe i add some statistics here
        self.trainings_mode = trainings_mode
        self.evalutation_mode = False
        self.prediction_mode = False

        self.loss, self.acc = 0, False

    def set_trainingsmode(self, mode=False):
        self.trainings_mode = mode

    def set_evalutation_mode(self, mode=False):
        self.evalutation_mode = mode

    def set_prediction_mode(self, mode=False):
        self.prediction_mode = mode

    def forward(self, pred, target, give_return=False):
        """
        last point in the forward-propagation
        calculates error and starts backprop
        calculates Loss for statistics
        calculates acc for statistics
        or returns a prediction (not implemented yet)
        :param pred: prediction
        :param target: target or y_hat
        :param give_return: returns error if True else
        :return: None
        """

        if self.trainings_mode:
            error = self.Error_func(pred, target)
            if give_return:
                return error
            else:
                self.layer_before.backward(error)

        elif self.evalutation_mode:
            self.loss, self.acc = np.mean(self.Loss_func(pred, target)), self.Acc_func(pred, target)

        elif self.prediction_mode:
            self.pred = Softmax_forward(pred)

    def update(self, batch_size):
        """
        function just exist for standardization
        does nothing
        :param batch_size: size of a mini-batch
        :return:
        """
        pass

    def info(self):
        """
        retuns information of the layer for summary
        :return: Name, neurons (0), trainable-params (0), activation-function (-)
        """
        return self.function_name
