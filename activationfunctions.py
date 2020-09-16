import numpy as np


def give_activation_dict():
    activation_dict = {
        'sigmoid': (sigmoid_forward, sigmoid_backward),
        'relu': (Relu_forward, Relu_backward),
        'leakyrelu': (Leakyrelu_forward, LeakyRelu_backward),
        'softplus': (Softmax_forward, Softmax_backward),
        'softmax': (Softmax_forward, Softmax_backward),
        'tanh': (Tanh_forward, Tanh_backward)
    }
    return activation_dict


#####################################################################################################################
#     Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid -- Sigmoid    #
#####################################################################################################################


def sigmoid_forward(x):
    a = np.exp(-x)
    return 1 / (1 + a)


def sigmoid_derivate(x):
    return sigmoid_forward(x) * (1 - sigmoid_forward(x))


def sigmoid_backward(x, error):
    return sigmoid_derivate(x) * error


#####################################################################################################################
#    Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu -- Relu   #
#####################################################################################################################


def Relu_forward(x):
    return np.where(x < 0, 0, x)


def relu_derivate(y):
    return np.where(y < 0, 0, 1)


def Relu_backward(y, error):
    return relu_derivate(y) * error


#####################################################################################################################
# LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu -- LeakyRelu #
#####################################################################################################################


def Leakyrelu_forward(x):
    return np.where(x >= 0, x, 0.001 * x)


def LeakyRelu_derivate(x):
    return np.where(x >= 0, 1, 0.001)


def LeakyRelu_backward(x, error):
    return LeakyRelu_derivate(x) * error


#####################################################################################################################
#     Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus -- Softplus      #
#####################################################################################################################


def softplus_forward(x):
    return np.log(1 + np.exp(x))


def softplus_derivate(x):
    return 1 / (1 + np.exp(-x))


def Softplus_backward(x, error):
    return softplus_derivate(x) * error


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


def Tanh_forward(x):
    return np.tanh(x)


def tanh_derivate(x):
    return 1 - Tanh_forward(x) ** 2


def Tanh_backward(x, error):
    return tanh_derivate(x) * error
