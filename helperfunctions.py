import numpy as np


def one_hot(array, classes=None):
    """
    one-hot encodes an 1d-array
    :param array: 1d-np.array
    :param classes: count of classes (if none max(input))
    :return: 2d-array
    """
    if classes is None:
        classes = np.max(array) + 1

    one_hot = np.zeros((len(array), classes))
    for idx, row in enumerate(one_hot):
        row[array[idx]] = 1

    return one_hot


def seperate_tuple(t):
    if isinstance(t, int) or isinstance(t, float):
        return str(t)
    string = ''
    first = True
    for value in t:
        if first:
            string += f'{value}'
            first = False
        else:
            string += f'/{value}'

    return string


def retuple(t):
    lst = t.split("/")
    lst = [int(x) for x in lst]
    return tuple(lst) if len(lst) > 1 else lst[0]


def prettyTime(time):
    hours, left = np.divmod(time, 60*60)
    min, left = np.divmod(left, 60)
    sec = np.round(left)
    return f'{int(hours)}h {int(min)}min {int(sec)}sec'


def add_noise(images, value):
    images = images.copy()
    noise = np.random.normal(0, value, images.shape)
    images += noise
    images /= np.amax(images)
    return images


def shot_noise(images, value):
    images = images.copy()
    noise = np.random.binomial(1, (1 - value), images.shape)
    images *= noise
    return images


def impulse_noise(images, value):
    images = images.copy()
    noise = np.random.binomial(1, (1 - value), images.shape)
    images = noise * images + (1-noise) * np.where(images >= .5, 0, 1)
    return images


def motion_noise(images, value):
    images = images.copy()
    noise = images
    noise[:, :, 0:27, 0:27] += images[:, :, 1:28, 1:28] * value
    noise[:, :, 0:26, 0:26] += images[:, :, 2:28, 2:28] * (value**2)
    noise[:, :, 0:25, 0:25] += images[:, :, 3:28, 3:28] * (value**3)
    noise /= np.amax(noise)
    return noise


if __name__ == '__main__':
    shape = 128
    print(retuple(shape))

    from plots import show_images
    x_train = np.load('mnist/x_train.npy')
    x_train = x_train.reshape((len(x_train), 1, 28, 28))
    x_train = np.array(x_train) / 255
    x_train = impulse_noise(x_train, value=0.1)
    show_images(x_train[:16].reshape(16, 28, 28))

