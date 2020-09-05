import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


def plot_trainingsprocess(loss, acc, loads=None, name='model', save=False):

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100  # 200 e.g. is really fine, but slower
    # define colors
    acc_color = 'tab:orange'
    loss_color = 'tab:blue'
    snap_color = 'dimgray'
    major = 'black'

    x = np.arange(1, len(acc) + 1, 1, dtype=np.int32)

    snaps = []
    actual = acc.copy()
    if loads is not None:
        for idx, load in enumerate(loads):
            if load:
                actual[idx] = acc[idx-1]
                snaps.append([[x[idx-1], x[idx]], [actual[idx-1], acc[idx]]])
            else:
                actual[idx] = acc[idx]

    fig, ax1 = plt.subplots()

    if loads is not None:
        for lines in snaps:
            ax1.plot(lines[0], lines[1], marker='x', linestyle='dashed', label='loss', color=snap_color)

    # plot accuracy
    ax1.set_ylim(0, 1.01)
    ax1.set_xlabel('epochs', color=major)
    ax1.set_ylabel('accuracy', color=major)
    ax1.tick_params(axis='y', labelcolor=major)
    ax1.set_xticks(range(len(acc)), minor=True)

    # for showing whole numbers on x-axis ('epochs')
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    line1, = ax1.plot(x, actual, marker='x', linestyle='dashed', label='accuracy', color=acc_color)

    # new subplot on the same x-axis
    ax2 = ax1.twinx()

    # plot loss
    ax2.set_ylim(0, np.amax(loss)*1.1)
    ax2.set_ylabel('loss', color=major)
    ax2.tick_params(axis='y', labelcolor=major)
    line2, = ax2.plot(x, loss, marker='x', linestyle='dashed', label='loss', color=loss_color)

    # print the values of accuracy on the plot
    for i, j in zip(x, acc):
        ax1.annotate(str(np.round(j, 4)), xy=(i,j))

    # make a legend
    plt.legend(handles=[line1, line2], loc="center left", fontsize=10)

    plt.title(f'trainings-process of {name}')
    # save the plot
    if save:
        plt.savefig(f'Plots/{name}.pdf')

    # use tight layout and show the plot
    fig.tight_layout()
    plt.show()


def show_images(images, y_pred=None, y_true=None):
    plt.figure(figsize=(10, 10))
    x = round(np.sqrt(len(images)))

    for i in range(len(images)):
        plt.subplot(x, x, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i][0], cmap=plt.cm.binary)
        label = ''
        if y_pred is not None:
            label += f'prediction: {y_pred[i]}'
        if y_true is not None:
            label += f' label: {y_true[i]}'
        plt.xlabel(label)
    plt.show()


def showImagesWithProbabilities(images, probs, labels=None):
    fig = plt.figure(constrained_layout=True)

    gs0 = gridspec.GridSpec(1, 2, figure=fig)
    # gs0.update(wspace=0.025, hspace=0.05)

    gs1 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], wspace=0.1)
    idx = 0
    for n in range(8):
        if n % 2 == 0:
            ax = fig.add_subplot(gs1[n])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.imshow(images[idx, 0], cmap=plt.cm.binary)
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
    from activationfunctions import Softmax_forward
    x_train = np.load('mnist/x_train.npy')
    y_train = np.load('mnist/y_train.npy')
    prob = np.random.randn(16, 10)
    prob = Softmax_forward(prob)

    showImagesWithProbabilities(x_train, prob, y_train)

    # show_images(x_train[:9], y_pred=pred, y_true=y_train)

