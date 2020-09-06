import sys
import time
from helperfunctions import prettyTime


def print_loading_bar(actual, steps, epoch, total, starttime=0):
    """
    prints a loading bar
    predicts a process duration if starttime is given
    overwrites the current line
    :param actual: the actual step of the process
    :param steps: the count of all steps of the epoch
    :param epoch: the actual epoch
    :param total: the count of all epochs
    :param starttime: time when the process have started
    :return:
    """
    if actual == 0:
        procent = 0
    else:
        procent = round((actual / steps) * 50)

    bar = '>' * procent + ' ' * (50 - procent)
    b = f'epoch {epoch} / {total}: [{bar}] {actual} of {steps} batches trained '

    if starttime != 0:
        proc_time = time.time() - starttime
        batch_time = proc_time / actual
        need_time = (steps - actual) * batch_time
        b += f'({prettyTime(need_time)})'

    sys.stdout.write('\r' + b)


if __name__ == '__main__':
    start = time.time()
    for i in range(1, 1000):
        print_loading_bar(i, 1000, 1, 5, starttime=start)
        time.sleep(.1)
