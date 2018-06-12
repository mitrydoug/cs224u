import re
import sys

from matplotlib import pyplot as plt

if __name__ == '__main__':
    batch_num = [-1]
    batch_mse = []
    batch_mun = []
    train_mse = []
    train_acc = []
    p_norm = []
    val_num = []
    val_mse = []
    val_acc = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            m = re.match(('train.*'
                          'batch_num = ([0-9]+).*'
                          'batch_mse = ([.0-9]+).*'
                          'batch_mun = ([.0-9]+).*'
                          'train_mse = ([.0-9]+).*'
                          'train_acc = ([.0-9]+).*'
                          'p_norm = ([.0-9]+)'), line)
            if m is not None:
                batch_num.append(batch_num[-1] + 1)
                batch_mse.append(float(m.group(2)))
                batch_mun.append(float(m.group(3)))
                train_mse.append(float(m.group(4)))
                train_acc.append(float(m.group(5)))
                p_norm.append(float(m.group(6)))

            m = re.match(('test.*'
                          'val_loss = ([.0-9]+).*'
                          'val_acc = ([.0-9]+)'), line)

            if m is not None:
                val_num.append(batch_num[-1])
                val_mse.append(float(m.group(1)))
                val_acc.append(float(m.group(2)))

    batch_num = batch_num[1:]
    plt.plot(batch_num, batch_mse, label='batch mse')
    plt.plot(batch_num, batch_mun, label='batch mean user norm')
    plt.plot(batch_num, train_mse, label='train mse')
    plt.plot(batch_num, train_acc, label='train_acc')
    #plt.plot(batch_num, param_norm, label='param norm')
    plt.plot(val_num, val_mse, c='m', label='val mse')
    plt.plot(val_num, val_acc, c='c', label='val acc')
    plt.legend()
    plt.show()
