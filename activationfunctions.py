import numpy as np


def give_activation_dict():
    activation_dict = {
        'sigmoid': (sigmoid_forward, sigmoid_backward),
        'relu': (Relu_forward, Relu_backward),
        'leakyrelu': (Leakyrelu_forward, LeakyRelu_backward),
        'softplus': (Softmax_forward, Softmax_backward),
        'binarystep': (BinaryStep_forward, BinaryStep_backward),
        'softmax': (Softmax_forward, Softmax_backward),
        'tanh': (Tanh_forward, Tanh_backward)
    }
    return activation_dict


#####################################################################################################################
#     Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid    #
#####################################################################################################################


def sigmoid(x):
    if x <= -700:
        return 0
    elif x >= 700:
        return 1
    else:
        return 1 / (1 + np.exp(-x))


sigmoid_forward = np.vectorize(sigmoid, otypes=["float64"])


def sigmoid_derivate(x):
    return sigmoid_forward(x) * (1 - sigmoid_forward(x))


sigmoid_derivate = np.vectorize(sigmoid_derivate, otypes=["float64"])


def sigmoid_backward(x, error):
    return sigmoid_derivate(x) * error


#####################################################################################################################
#    Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu   #
#####################################################################################################################


def relu(x):
    if x < 0:
        return 0
    else:
        return x


Relu_forward = np.vectorize(relu, otypes=["float64"])


def relu_derivate(y):
    if y < 0:
        return 0
    else:
        return 1


relu_derivate = np.vectorize(relu_derivate, otypes=["float64"])


def Relu_backward(y, error):
    return relu_derivate(y) * error


#####################################################################################################################
# LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu #
#####################################################################################################################


def LeakyRelu(x):
    return x if x >= 0 else 0.01 * x


Leakyrelu_forward = np.vectorize(LeakyRelu, otypes=['float64'])


def LeakyRelu_derivate(x):
    return 1 if x >= 0 else 0.01


leakyrelu_derivate = np.vectorize(LeakyRelu_derivate, otypes=['float64'])


def LeakyRelu_backward(x, error):
    return leakyrelu_derivate(x) * error


#####################################################################################################################
#     Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus      #
#####################################################################################################################


def Softplus(x):
    return np.log(1 + np.exp(x))


softplus_forward = np.vectorize(Softplus, otypes=['float64'])


def softplus_derivate(x):
    return 1 / (1 + np.exp(-x))


softplus_derivate = np.vectorize(softplus_derivate, otypes=['float64'])


def Softplus_backward(x, error):
    return softplus_derivate(x) * error


#####################################################################################################################
#    BinaryStep -- BinaryStep -- BinaryStep -- BinaryStep -- BinaryStep -- BinaryStep -- BinaryStep -- BinaryStep   #
#####################################################################################################################


def binaryStep(x):
    return 1 if x >= 0 else 0


BinaryStep_forward = np.vectorize(binaryStep, otypes=['float64'])


def binaryStep_derivate(x):
    return 0


binaryStep_derivate = np.vectorize(binaryStep_derivate, otypes=['float64'])


def BinaryStep_backward(x, error):
    # at this positon the error becomes 0 and the backprop will not work anymore
    return binaryStep_derivate(x) * error


#####################################################################################################################
#     Softmax -- Softmax -- Softmax -- Softmax -- Softmax -- Softmax -- Softmax -- Softmax -- Softmax -- Softmax    #
#####################################################################################################################


def Softmax_forward(x):
    """softmax: e^xi / sum(e^x)"""
    exp = np.exp(x - np.max(x, axis=1)[:,None])
    exp_sum = np.sum(exp, axis=1)[:,None]
    return exp / exp_sum


def Softmax_backward(x, error):
    # derivate = y_hat * (delta - y_hat)
    batch, size = x.shape
    y_hat = Softmax_forward(x)                                  # calculate y_hat
    delta = np.array([np.eye(size, dtype=int)] * batch)         # create delta-matrix
    tensor1 = delta - y_hat[:, :, None]
    delta_x = tensor1 * y_hat[:, None]

    return np.einsum('ijk, ik->ij', delta_x, error)


#####################################################################################################################
#    Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh -- Tanh   #
#####################################################################################################################


def tanh(x):
    return np.tanh(x)


Tanh_forward = np.vectorize(tanh, otypes=['float64'])


def tanh_derivate(x):
    return 1 - tanh(x) ** 2


tanh_derivate = np.vectorize(tanh_derivate, otypes=['float64'])


def Tanh_backward(x, error):
    return tanh_derivate(x) * error


if __name__ == '__main__':
    data = np.random.randn(32,1)
    pred = Softmax_forward(data)
    print(pred)