import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


def plot_trainingsprocess(loss: list, acc: list, loads: Union[list, None] = None, name: str = 'model',
                          save: bool = False):
    """
    this function plots the trainings-process of the Model
    :param loss: list of loss values
    :param acc: list of acc values
    :param loads: bool list when snapshot was loaded
    :param name: name of the model (for the title)
    :param save: saves the plot as pdf if Ture
    :return: None
    """

    # define the size of the model
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    # define colors
    acc_color = 'tab:orange'
    loss_color = 'tab:blue'
    snap_color = 'dimgray'
    major = 'black'

    # arange the x-axis
    x = np.arange(1, len(acc) + 1, 1, dtype=np.int32)

    # create a plot
    fig, ax1 = plt.subplots()

    ax1.set_ylabel('accuracy', color=major)


    # new subplot on the same x-axis
    ax2 = ax1.twinx()

    # handle the loads
    if loads is not None:
        actual_acc = []
        actual_loss = []
        last_acc = 0
        last_loss = 0
        for idx in range(len(acc)):
            if not loads[idx]:
                last_acc = acc[idx]
                last_loss = loss[idx]
                actual_acc.append(acc[idx])
                actual_loss.append(loss[idx])
            else:
                actual_acc.append(last_acc)
                actual_loss.append(last_loss)

        # creating the line where the model get worse and a snapshot was loaded
        for idx, load in enumerate(loads):
            if load:
                acc_line = [[x[idx-1], x[idx]], [actual_acc[idx-1], acc[idx]]]
                loss_line = [[x[idx-1], x[idx]], [actual_loss[idx-1], loss[idx]]]
                ax1.plot(acc_line[0], acc_line[1], marker='x', linestyle='dashed', label='accuracy', color=snap_color)
                ax2.plot(loss_line[0], loss_line[1], marker='x', linestyle='dashed', label='loss', color=snap_color)
    else:
        actual_acc = acc
        actual_loss = loss

    # for showing whole numbers on x-axis ('epochs')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    line1, = ax1.plot(x, actual_acc, marker='x', linestyle='dashed', label='accuracy', color=acc_color)

    # plot loss
    ax2.set_ylim(0, np.amax(loss)*1.1)
    ax2.set_ylabel('loss', color=major)
    ax2.tick_params(axis='y', labelcolor=major)
    line2, = ax2.plot(x, actual_loss, marker='x', linestyle='dashed', label='loss', color=loss_color)

    # print the values of accuracy on the plot
    for i, j in zip(x, actual_acc):
        ax1.annotate(str(np.round(j, 4)), xy=(i,j))

    # make a legend and title
    plt.legend(handles=[line1, line2], loc="center left", fontsize=10)
    plt.title(f'trainings-process of {name}')

    # save the plot
    if save:
        plt.savefig(f'Plots/{name}.pdf')

    # use tight layout and show the plot
    fig.tight_layout()
    plt.show()


def show_images(images, y_pred=None, y_true=None):
    """
    shows some images with prediction and labels
    :param images: array of images
    :param y_pred: array of predictions
    :param y_true: array of labels
    :return: None
    """
    plt.figure(figsize=(10, 10))
    x = round(np.sqrt(len(images)))

    # show the images
    for i in range(len(images)):
        plt.subplot(x, x, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i][0], cmap=plt.cm.binary)

        # add prediction and label as label
        label = ''
        if y_pred is not None:
            label += f'prediction: {y_pred[i]}'
        if y_true is not None:
            label += f' label: {y_true[i]}'
        plt.xlabel(label)
    plt.show()


def showImagesWithProbabilities(images, probs, labels=None):
    """
    shows some images with a bar chart of there probabilities
    true labels is shown red. If an other label is predicted it's shown red
    :param images: array of the images
    :param probs: array of the probabilities
    :param labels: true label of the images
    :return:
    """
    fig = plt.figure(constrained_layout=True)

    gs0 = gridspec.GridSpec(1, 2, figure=fig)
    # gs0.update(wspace=0.025, hspace=0.05)

    gs1 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], wspace=0.1)
    idx = 0
    for n in range(8):
        # show the images on the left side
        if n % 2 == 0:
            ax = fig.add_subplot(gs1[n])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.imshow(images[idx, 0], cmap=plt.cm.binary)
        # show the probabilities on the right side
        else:
            ax = fig.add_subplot(gs1[n])
            colors = ['tab:blue' for x in range(len(probs[idx]))]
            colors[np.argmax(probs[idx])] = 'r'
            if labels is not None:
                colors[labels[idx]] = 'g'
            ax.set_xlim([0, 1])
            ax.set_xticks([])
            ax.set_yticks(range(10))
            ax.barh(range(10), probs[idx], color=colors)
            idx +=2

    gs2 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[1], wspace=0.1)
    for n in range(8):
        if n % 2 == 0:
            ax = fig.add_subplot(gs2[n])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.imshow(images[idx, 0], cmap=plt.cm.binary)
        else:
            ax = fig.add_subplot(gs2[n])
            colors = ['tab:blue' for x in range(len(probs[idx]))]
            colors[np.argmax(probs[idx])] = 'r'
            if labels is not None:
                colors[labels[idx]] = 'g'
            ax.set_xlim([0, 1])
            ax.set_xticks([])
            ax.set_yticks(range(10))
            ax.barh(range(10), probs[idx], color=colors)
            idx += 2

    plt.suptitle('probabilities')
    plt.show()


if __name__ == '__main__':
    acc = np.random.randint(0, 100, 10)
    acc = acc / 100
    loss = 1 - acc
    loads = [False, True] * 5
    plot_trainingsprocess(loss, acc, loads)

    """
    from activationfunctions import Softmax_forward
    x_train = np.load('mnist/x_train.npy')
    y_train = np.load('mnist/y_train.npy')
    prob = np.random.randn(16, 10)
    prob = Softmax_forward(prob)

    showImagesWithProbabilities(x_train, prob, y_train)

    # show_images(x_train[:9], y_pred=pred, y_true=y_train)

    """