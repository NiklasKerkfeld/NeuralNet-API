import numpy as np
from activationfunctions import Softmax_forward, sigmoid_forward


def give_loss_dict():
    functions = {
        'squared_error': (SquaredError_Loss, SquaredError_Error, SquaredError_Acc),
        'categorical_crossentropy': (CrossEntropy_Loss, CrossEntropy_Error, CrossEntropy_Accuracy),
        'binary_crossentropy': (BinaryCrossEntopy_Loss, BinaryCrossEntopy_Error, BinaryCrossEntopy_Accuracy)
    }
    return functions


#####################################################################################################################
#        Squared_Error -- Squared_Error -- Squared_Error -- Squared_Error -- Squared_Error -- Squared_Error         #
#####################################################################################################################


def SquaredError_Loss(pred, target):
    return np.sum(.5 * (pred - target) ** 2, axis=1)


def SquaredError_Error(pred, target):
    return pred - target



def SquaredError_Acc(pred, target):
    guess = np.zeros_like(pred)
    guess[np.arange(len(pred)), pred.argmax(axis=1)] = 1
    result = (guess == target).all(axis=1)
    acc = np.sum(result)
    return acc / len(target)


#####################################################################################################################
#   CrossEntropy -- CrossEntropy -- CrossEntropy -- CrossEntropy -- CrossEntropy -- CrossEntropy -- CrossEntropy    #
#####################################################################################################################


def CrossEntropy_Loss(pred, target):
    prob = Softmax_forward(pred)
    return np.sum(-target * np.log(prob + 0.00000000000000000000000001), axis=1)


def CrossEntropy_Error(pred, target):
    prob = Softmax_forward(pred)
    return prob - target


def CrossEntropy_Accuracy(pred, target):
    guess = np.zeros_like(pred)
    guess[np.arange(len(pred)), pred.argmax(axis=1)] = 1
    result = (guess == target).all(axis=1)
    acc = np.sum(result)
    return acc / len(target)


#####################################################################################################################
#    BinaryCrossEntropy -- BinaryCrossEntropy -- BinaryCrossEntropy -- BinaryCrossEntropy -- BinaryCrossEntropy     #
#####################################################################################################################

def BinaryCrossEntopy_Loss(pred, target, minval=10**-10):
    prob = sigmoid_forward(pred)
    return -np.mean(target * np.log(prob.clip(min=minval)) +
                   (1 - target) * np.log(1 - prob.clip(max=1-minval)), axis=1)


def BinaryCrossEntopy_Error(pred, target):
    prob = sigmoid_forward(pred)
    return prob - target


def BinaryCrossEntopy_Accuracy(pred, target):
    prob = sigmoid_forward(pred)
    guess = np.zeros_like(pred)
    guess[np.where(prob >= 0.5)] = 1
    guess[np.where(prob <= 0.5)] = 0
    result = (guess == target).all(axis=1)
    acc = np.sum(result)
    return acc / len(target)


if __name__ == '__main__':
    from helperfunctions import one_hot
    batch = 32
    # pred = np.random.randint(0, 1000, (batch, 1))
    # pred = pred / 1000
    pred = np.random.randn(batch, 1)
    # target = np.random.randint(0, 2, (batch, 1))
    target = np.zeros_like(pred)
    target[np.where(pred >= 0)] = 1
    # pred[0:10] = 0
    # pred = target
    print(f'pred: {pred[:10]}')
    print(f'target: {target[:10]}')
    # print(np.where(pred != 0, np.log(pred), -99))
    # print(np.where(pred != 1, np.log(1 - pred), -99))

    # loss = CrossEntropy_Loss(pred, target)
    # error = CrossEntropy_Error(pred, target)
    #
    # print()
    # print(loss)
    # print()
    # print(error)

    loss = BinaryCrossEntopy_Loss(pred, target)
    error = BinaryCrossEntopy_Error(pred, target)
    acc = BinaryCrossEntopy_Accuracy(pred, target)

    print()
    print(loss)
    print()
    print(error)
    print()
    print(acc)


    # print(Error(pred, target))
    # print(squared_error_derivative(pred, target))