from typing import Union
from lossfunctions import *
from helperfunctions import *


class Last_Layer:
    def __init__(self, input_shape: tuple, loss_function: str, layer_before: object, trainings_mode: bool = True):
        """
        this Layer must be the last in the Model. calculates loss and starts backprob
        :param input_shape: shape of the input tensor
        :param loss_function: name of the loss function
        :param layer_before: link to the layer before
        :param trainings_mode: starts backprop if True
        """
        self.input_shape = input_shape
        self.layer_before = layer_before
        self.next_layer = None

        # defines functions to calculate the loss, the error and the accuracy
        self.function_name = loss_function
        loss_dict = give_loss_dict()
        self.Loss_func, self.Error_func, self.Acc_func = loss_dict[loss_function.lower()]

        self.loss_history = []                                              # maybe i add some statistics here

        self.trainings_mode = trainings_mode
        self.evalutation_mode = False
        self.prediction_mode = False

        self.give_input_error = False
        self.input_error = None

        self.loss, self.acc = 0, False

    def set_give_input_error(self, mode: bool = False):
        self.give_input_error = mode

    def set_trainingsmode(self, mode: bool = False):
        self.trainings_mode = mode

    def set_evalutation_mode(self, mode: bool = False):
        self.evalutation_mode = mode

    def set_prediction_mode(self, mode: bool = False):
        self.prediction_mode = mode

    def forward(self, pred: np.ndarray, target: Union[None, np.ndarray], give_return: bool = False):
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
                if self.give_input_error:
                    self.input_error = self.layer_before.backward(error, return_input_error=True)
                else:
                    self.layer_before.backward(error, return_input_error=False)

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
        :return: loss-function name
        """
        return self.function_name
