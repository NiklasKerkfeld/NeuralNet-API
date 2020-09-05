import sys
import time
from helperfunctions import prettyTime


def print_loading_bar(actual, steps, epoch, total, starttime=0):
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
    for i in range(0, 101, 10):
        print('\r>> You have finished %d%%' % i,)
        sys.stdout.flush()
        time.sleep(1)

    for i in range(1000):
        print_loading_bar(i, 1000, 1)
        time.sleep(.1)
