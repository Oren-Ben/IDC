import matplotlib.pyplot as plt
import numpy as np


def plot_func_value_vs_iter_num (func_val_list, methods, title): #,
    plt.plot(np.arange(len(func_val_list[0])),func_val_list[0])
    plt.plot(np.arange(len(func_val_list[1])),func_val_list[1])
    plt.legend(methods)
    plt.title(title)
    plt.show()
