import csv
from tabulate import tabulate
from layer import give_layer_dict
from Layer.Last_Layer import Last_Layer
from helperfunctions import *
from lossfunctions import *
from optimizer import *
from plots import *
from loadingBar import *


class Model:
    """
    the main class for a neural net
    """
    def __init__(self, model_name=None):
        """
        create the network architecture
        :param architecture: list (of dicts of specifications)
        """
        # for the layer of the model
        self.architecture = []

        # for saving the model
        if model_name is None:
            self.model_name = f'model_{np.random.randn(1)}'
        else:
            self.model_name = model_name

    def compile(self, architecture, loss, optimizer):
        """
        compiles the model for training
        :param architecture: the architecture of the model
        :param loss: the loss-function
        :param optimizer: the optimizer to optimize the trainable params
        :return: None
        """

        # optimizer
        # if the optimizer is given as class
        if not isinstance(optimizer, str):
            self.optimizer = optimizer
        # if the optimizer is given with string
        else:
            optimizer_dict = give_optimizer_dict()
            self.optimizer = optimizer_dict[optimizer.lower()]()

        # network-architecture
        layer_dict = give_layer_dict()              # dict of layers and there names

        # as architecture we create a link-List where every layer is connected to the next layer and the layer before
        # so in propagation every layer starts the forward propagation of the next layer
        # the so called 'Loss_Layer' is the last part of this Link-List it calculates the Error and starts the backprob
        # this goes then back from layer to layer

        self.architecture = []
        first = True
        for layer in architecture:
            name = layer['layer_type']

            # set the output-shape of the layer before as input-shape of the current layer
            if 'input_shape' not in layer.keys():
                layer['input_shape'] = Layer.output_shape

            if first:
                # the first layer has no layer before
                Layer = layer_dict[name.lower()].from_dict(layer, optimizer=self.optimizer)

                # do not need to calculate a error for the layer before (first_mode)
                Layer.first_mode = True
                first = False

            else:
                Layer_before = Layer
                Layer = layer_dict[name.lower()].from_dict(layer, optimizer=self.optimizer, layer_before=Layer_before)
                Layer_before.set_next_layer(Layer)

            # add layer to architecture
            self.architecture.append(Layer)

        Layer_before = Layer
        # the Loss Layer is a special "Layer". It's not a real layer. It calculates the loss and the delta and starts
        # the backpropagation in training
        # when predicting a takes the results
        self.Loss_Layer = Last_Layer(input_shape=Layer_before.output_shape, loss_function=loss,
                                        layer_before=Layer_before)
        Layer_before.set_next_layer(self.Loss_Layer)
        self.architecture.append(self.Loss_Layer)
        self.First_Layer = self.architecture[0]

    def save(self):
        """
        saves the model in the Models folder
        :return: None
        """
        layer_conf = []             # saves the hyperparameter of every layer
        weights_bias = []           # saves the weights and biases of every layer

        for layer in self.architecture[:-1]:
            hyperparameter, params = layer.save()
            layer_conf.append(hyperparameter)
            weights_bias.append(params[0])
            weights_bias.append(params[1])
            weights_bias.append(params[2])
            weights_bias.append(params[3])

        # saving the hyperparameter as a csv-file in the Model folder
        with open(f'Models/{self.model_name}.csv', "w", newline='') as params_file:
            params_writer = csv.writer(params_file, delimiter=';')
            for params in layer_conf:
                params_writer.writerow(params)

        # saving the weights and biases as a npz-file in the Model folder
        np.savez(f'Models/{self.model_name}.npz', *weights_bias)

    def load(self, model_name, loss, optimizer):
        """
        load model hyperparameter and inits a network then load the weights and biases
        :param model_name: name of the two save files
        :return: None
        """

        # set the optimizer
        if not isinstance(optimizer, str):
            self.optimizer = optimizer
        else:
            optimizer = optimizer.lower()
            optimizer_dict = give_optimizer_dict()
            self.optimizer = optimizer_dict[optimizer]()

        self.architecture = []
        layer_dict = give_layer_dict()

        # load the weights and biases form the save.npz file
        container = np.load(f'Models/{model_name}.npz')
        params = [container[key] for key in container]

        # open the csv-file with the model settings
        with open(f'Models/{model_name}.csv', 'r') as params_file:
            params_reader = csv.reader(params_file, delimiter=';')

            # create the Layer with the hyperparameter form the csv-file and the weigths and biases from the npz-file
            first = True
            params_idx = 0
            for layer in params_reader:
                name = layer[0]

                # create a Layer with the from_load method
                if first:
                    Layer = layer_dict[name.lower()].from_load(layer[1:], optimizer=self.optimizer, layer_before=None)
                    Layer.first_mode = True
                    first = False
                else:
                    Layer_before = Layer
                    Layer = layer_dict[name.lower()].from_load(layer[1:], optimizer=self.optimizer,
                                                                layer_before=Layer_before)
                    Layer_before.set_next_layer(Layer)

                # set the weights to the saved values
                Layer.set_weights_biases(params[params_idx: params_idx+4])
                params_idx += 4
                self.architecture.append(Layer)

            Layer_before = Layer
            self.Loss_Layer = Last_Layer(input_shape=Layer_before.output_shape, loss_function=loss,
                                            layer_before=Layer_before)
            Layer_before.set_next_layer(self.Loss_Layer)
            self.architecture.append(self.Loss_Layer)
            self.First_Layer = self.architecture[0]

    def train_on_batch(self, input, target):
        """
        start forward and backward propagation
        :param input: a batch of training-data
        :param target: the target of this batch
        :return: None
        """
        self.First_Layer.forward(input, target)

    def _trainings_policy(self, policy, last_loss, loss, last_acc, acc):
        """
        checks the trainings-policy
        returns true if the model has to load a snapshot according to the policy
        acc: True if the acc of the last epoch was higher
        loss: True if the loss of the last epoch was lower
        both: True if the conditions of acc and loss are True
        one: True if the conditions of acc or loss are True
        none: never load a snapshot
        :param policy: string that define the policy
        :param last_loss: the loss of the last epoch
        :param loss: the lost of the actual epoch
        :param last_acc: the acc of the last epoch
        :param acc: the acc of the actual epoch
        :return: boolean for loading the last snapshot
        """
        # i think this need no explanation
        if policy is None:
            return False

        if policy.lower() == 'none':
            return False

        if policy.lower() == 'acc':
            if last_acc == 0:
                return False
            if last_acc > acc:
                return True
            else:
                return False

        if policy.lower() == 'loss':
            if last_loss == 0:
                return False
            if last_loss < loss:
                return True
            else:
                return False

        if policy.lower() == 'both':
            if last_loss == 0:
                return False
            if last_loss < loss and last_acc > acc:
                return True
            else:
                return False

        if policy.lower() == 'one':
            if last_loss == 0:
                return False
            if last_loss < loss or last_acc > acc:
                return True
            else:
                return False

        return False

    def train(self, x_train, y_train, epochs, batchsize= 1, shuffle=False, x_test=None, y_test=None, policy=None,
              create_plot=False):
        """
        the trainingsprocess of the network
        :param x_train: input trainings-data
        :param y_train: true output of the trainings-data
        :param epochs: number of epochs to train
        :param batchsize: size of the trainings-batches (stochastic gradient decent)
        :param shuffle: to shuffles or not to shuffle the trainings-data
        :param x_test: data to test the model (uses x_train if None)
        :param y_test: true output of the test-data (uses x_train if None)
        :param policy: policy for loading the last snapshot
        :param create_plot: creating a plot of the trainingspocess
        :return: progress of the metrics
        """
        # saving the trainingsdata
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()

        # test-data
        if x_test is None:
            self.x_test = x_train.copy()
            self.y_test = y_train.copy()
        else:
            self.x_test = x_test.copy()
            self.y_test = y_test.copy()

        # for statistics
        loss_history = []
        acc_history = []
        load_history = []

        # init for policy
        last_loss, last_acc = 0, 0

        # start training
        b_size = str(batchsize) if isinstance(batchsize, int) else f'{min(batchsize)}-{max(batchsize)}'
        print(f'start training [epochs: {epochs}, batchsize: {b_size}, shuffle: {shuffle}, policy: {policy}]')
        start_time = time.time()
        for e in range(1, epochs+1):                                                        # loop over epochs
            epoch_start_time = time.time()

            if shuffle:                                                                     # shuffles trainings-data
                x_train, y_train = self._shuffle(self.x_train, self.y_train)

            batch_number = 1                                                                # number of the actual batch
            if isinstance(batchsize, list):
                actual_batchsize = batchsize[e-1 if e < len(batchsize) else len(batchsize) - 1]
                batch_count = len(x_train) // actual_batchsize + 1
            else:
                actual_batchsize = batchsize
                batch_count = len(x_train) // batchsize + 1

            for b in range(0, len(y_train), actual_batchsize):                                     # in batch-steps
                print_loading_bar(batch_number, batch_count, e, epochs, epoch_start_time)   # print a loading bar
                batch_number += 1                                                   # increases the number of the batch

                self.train_on_batch(x_train[b:b+actual_batchsize], y_train[b:b+actual_batchsize])     # trains on batch

            loss, acc = self.evaluate(self.x_test, self.y_test)                             # evaluate on test-data
            print(' ')
            print(f'epoch {e}: loss: {loss} acc: {acc}')                                    # print result of evaluation
            print(' ')

            # check policy and load the last snapshot if true
            if self._trainings_policy(policy, last_loss, loss, last_acc, acc):
                load_history.append(True)
                for layer in self.architecture[:-1]:
                    layer.load_snapshot()
                print('snapshot loaded!')
            else:
                load_history.append(False)
                last_loss, last_acc = loss, acc
                # save a snapshot
                for layer in self.architecture[:-1]:
                    layer.take_snapshot()

            # for statistics
            loss_history.append(loss)
            acc_history.append(acc)

        # print results of the trainings-process
        print(f'training finished in {prettyTime(time.time() - start_time)}.')
        print(f'loss: {last_loss} accuracy: {last_acc}')

        # returns the history of loss and acc in the trainings-process
        plot_trainingsprocess(loss_history, acc_history, loads=load_history, name=self.model_name, save=False)
        return loss_history, acc_history

    def _shuffle(self, X, Y):
        """
        shuffles the data
        :param X: input-data
        :param Y: output-data
        :return: x, y (shuffled)
        """
        pos = np.random.randint(0, len(Y), len(Y))
        shuffled_x = np.zeros_like(X)
        shuffled_y = np.zeros_like(Y)
        for i in range(len(Y)):
            shuffled_x[i] = X[pos[i]]
            shuffled_y[i] = Y[pos[i]]

        return shuffled_x, shuffled_y

    def predict(self, input, return_probability=False):
        """
        predicts the input-data
        :param input: Data to predict
        :param return_probability: if true also returns the results for every class
        :return: list of predictions
        """
        # turn off trainings_mode
        for Layer in self.architecture:
            Layer.set_trainingsmode(False)

        # set the Loss_Layer to prediction mode
        self.Loss_Layer.set_trainingsmode(False)
        self.Loss_Layer.set_evalutation_mode(False)
        self.Loss_Layer.set_prediction_mode(True)

        # predict and get the results form the Loss-Layer
        self.First_Layer.forward(input)
        prob = self.Loss_Layer.pred

        # turn on the trainings mode again
        for Layer in self.architecture:
            Layer.set_trainingsmode(True)

        # set the Loss-Layer to trainings-mode again
        self.Loss_Layer.set_trainingsmode(True)
        self.Loss_Layer.set_evalutation_mode(False)
        self.Loss_Layer.set_prediction_mode(False)

        if return_probability:
            return prob.argmax(axis=1), prob
        else:
            return prob.argmax(axis=1)

    def evaluate(self, input, target):
        """
        evaluates the quality of the net in accuracy and loss
        :param input: input-data for the network
        :param target: true output ot the input-data
        :return: accuracy, loss
        """
        # turn off trainings_mode
        for Layer in self.architecture:
            Layer.set_trainingsmode(False)

        # set the last Layer to evaluation mode
        self.Loss_Layer.set_trainingsmode(False)
        self.Loss_Layer.set_evalutation_mode(True)
        self.Loss_Layer.set_prediction_mode(False)

                                                                        # of backprob
        self.First_Layer.forward(input, target)                         # computes test-data as a batch
        loss, acc = self.Loss_Layer.loss, self.Loss_Layer.acc           # gets results form Loss_layer

        for Layer in self.architecture:                                 # turn on trainings_mode
            Layer.set_trainingsmode(True)

        # set the last Layer to trainings-mode again
        self.Loss_Layer.set_trainingsmode(True)
        self.Loss_Layer.set_evalutation_mode(False)
        self.Loss_Layer.set_prediction_mode(False)

        return loss, acc

    def summary(self):
        """
        shows the summary of the model
        :return: None
        """
        neuron_sum = 0
        param_sum = 0

        print(f'{self.model_name} summary:')

        # iterates over the model and gets the params of every layer
        table = []
        for layer in self.architecture[:-1]:
            layer_type, input_shape, output_shape, neurons, trainable_params, activation = layer.info()
            table.append([layer_type, input_shape, output_shape, neurons, trainable_params, activation])
            neuron_sum += neurons
            param_sum += trainable_params
        loss_function = self.architecture[-1].info()

        # adding a activation function to the last layer comming from the crossentropy lossfunction
        if loss_function == 'categorical_crossentropy':
            table[-1][-1] = '(softmax)' if table[-1][-1] == '-' else table[-1][-1]

        if loss_function == 'binary_crossentropy':
            table[-1][-1] = '(sigmoid)' if table[-1][-1] == '-' else table[-1][-1]

        table.append(['------------------------', '-------------', '--------------', '---------', '------------------',
                      '---------------------'])
        table.append(['total:', table[0][1], table[-2][2], neuron_sum, param_sum, '-'])
        table.append([])

        # prints a table with this params
        print(tabulate(table, headers=['layer-type', 'input_shape', 'output_shape', 'neurons', 'trainable-params',
                                       'activation-function']))

        # prints out the loss-function and the optimizer
        print(f'lossfunction: {loss_function}       optimizer: {type(self.optimizer).__name__}')
        print('\n')


if __name__ == '__main__':
    import time
    import tensorflow.keras as keras  # just for downloading the dataset
    from helperfunctions import prettyTime
    from lossfunctions import CrossEntropy_Accuracy

    # load Data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='mnist.npz')

    # scale the data between 0 and 1 and reshape it
    x_train = np.array(x_train) / 255
    x_train = x_train.reshape((len(x_train), 1, 28, 28))
    print(x_train.shape)

    x_test = np.array(x_test) / 255
    x_test = x_test.reshape((len(x_test), 1,  28, 28))
    print(x_test.shape)

    # make Y-data hot-encoded
    y_train = one_hot(y_train)
    y_test = one_hot(y_test)
    print(y_train.shape)
    print(y_test.shape)

    # architecture
    conv_architecture = [
        # {'layer_type': 'Pooling', 'input_shape': (1, 28, 28), 'filter_size': (2, 2), 'stride': (2, 2)},
        {'layer_type': 'Conv2D', 'input_shape': (1, 14, 14), 'neurons': 64, 'filter_size': (6, 6), 'stride': (2, 2), 'use_bias': False, 'padding': 'valid'},
        {"layer_type": "BatchNorm"},
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},

        {'layer_type': 'Conv2D', 'neurons': 32, 'filter_size': (3, 3), 'use_bias': False, 'padding': 'valid'},      # 4
        {"layer_type": "BatchNorm"},
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},                                                 # 7

        {'layer_type': 'Flatten'},

        {"layer_type": "Dense", "neurons": 128, "use_bias": False},         # 9
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 10
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 64, "use_bias": False},          # 13
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 14
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense",  "neurons": 64, "use_bias": False},         # 17
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 18
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 32, 'use_bias': False},          # 21
        {"layer_type": "BatchNorm"},                                        # 22
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 10}                              # 25
    ]

    dense_architecture = [
        {"layer_type": "GaussianNoise", 'input_shape': (1, 28, 28), 'standard_diviation': .1},
        {"layer_type": "Dropout", "ratio": .2},
        {"layer_type": "Flatten"},

        {"layer_type": "Dense", "neurons": 256, "use_bias": False},         # 1
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 2
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 128, "use_bias": False},         # 5
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 6
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 64, "use_bias": False},          # 9
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 10
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense",  "neurons": 64, "use_bias": False},         # 13
        {"layer_type": "BatchNorm", 'alpha': 0.99},                         # 14
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 32, "use_bias": False},          # 17
        {"layer_type": "BatchNorm", 'alpha': 0.98},                         # 18
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 32, "use_bias": False},          # 21
        {"layer_type": "BatchNorm", 'alpha': 0.98},                         # 22
        {"layer_type": "Activation", "activation": 'relu'},
        {"layer_type": "BatchNormGaussianNoise", 'standard_diviation': .4},
        {"layer_type": "Dropout", "ratio": .2},

        {"layer_type": "Dense", "neurons": 10}      # , "activation": 'softmax'
    ]

    # create model
    optimizer = Adam(lr=0.1)
    nn = Model(model_name='model_01')
    nn.compile(dense_architecture, loss='categorical_crossentropy', optimizer=optimizer)
    # nn.load('model_28', loss='squared_error', optimizer=optimizer)

    nn.summary()

    # train model
    training = True
    batch_sizes = [128, 256, 256, 512]
    print(len(batch_sizes))
    if training:
        loss, acc = nn.train(x_train, y_train, batchsize=batch_sizes, epochs=5, shuffle=True,
                              x_test=x_train, y_test=y_train, policy='loss')

        nn.save()
        # plot_trainingsprocess(loss, acc, name=nn.model_name, save=True)
        print(loss)
        print(acc)

    pred, prob = nn.predict(x_test, return_probability=True)
    y_pred = one_hot(pred, 10)
    loss, acc = nn.evaluate(x_test, y_test)
    print(f'prediction loss: {loss}, accuracy: {acc}')

    y_test = y_test.argmax(axis=1)
    nr = np.random.randint(0, 10000-25)
    showImagesWithProbabilities(x_test[nr:nr+16], prob[nr:nr+16], y_test[nr:nr+16])

    show_images(x_test[nr:nr+25], y_pred=pred[nr:nr+25], y_true=y_test[nr:nr+25])

