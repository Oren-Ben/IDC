import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.constrained_min import *


def qp_plot_3d(x0, path_history, title):
    ax = plt.figure(figsize=(15, 9)).add_subplot(projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f"{title} path", fontsize=20, fontweight='bold')
    x_limit = np.array([1, 0, 0])
    y_limit = np.array([0, 1, 0])
    z_limit = np.array([0, 0, 1])
    limits = [x_limit, y_limit, z_limit]
    shape = Poly3DCollection([limits], alpha=.25, color='b')
    plt.gca().add_collection3d(shape)
    x0_len = len(x0)
    x_y_z_values = path_history[:, :x0_len]
    x = x_y_z_values[:, 0]
    y = x_y_z_values[:, 1]
    z = x_y_z_values[:, 2]
    plt.plot(x[:-1], y[:-1], z[:-1], '-o', label='path')
    plt.plot(x[-1], y[-1], z[-1], marker='^', markerfacecolor='yellow', markersize=12, label='final x')

    plt.show()


def function(x):
    return 1 - x


def lp_plot_2d(x0, path_history, title):
    x0_len = len(x0)
    x_y_values, func_val, path = path_history[:, :x0_len], path_history[:, x0_len], path_history[:, x0_len + 1]
    x = x_y_values[:, 0]
    y = x_y_values[:, 1]
    # y >= 0
    x1_lim = np.linspace(function(0), 2, 50)
    y1_lim = np.zeros(50)
    # y <=1
    x2_lim = np.linspace(function(1), 2, 50)
    y2_lim = np.ones(50)
    # x <=2
    x3_lim = np.ones(50) * 2
    y3_lim = np.linspace(0, 1, 50)
    # 0 >= 1-x-y
    x4_lim = function(y3_lim)
    y4_lim = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(x[:-1], y[:-1], '-o', label='path', color='r')
    ax.plot(x[-1], y[-1], marker='^', markerfacecolor='yellow', markersize=12, label='final x')
    ax.plot(x1_lim, y1_lim, color='y')
    ax.plot(x2_lim, y2_lim, color='y')
    ax.plot(x3_lim, y3_lim, color='y')
    ax.plot(x4_lim, y4_lim, color='y')
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title(f"{title} path", fontsize=20, fontweight='bold')
    # boundaries colored
    plt.fill_between(x4_lim, function(x4_lim), y2_lim, color='yellow', alpha=0.25)
    plt.fill_between(x1_lim, y1_lim, y2_lim, color='yellow', alpha=0.25)
    plt.show()


def plot_func_value_vs_iter_num(x0, path_history, title):
    x0_len = len(x0)
    if title == 'lp_func':
        path_history = path_history[:, x0_len] * (-1)
    else:
        path_history = path_history[:, x0_len]
    plt.plot(np.arange(len(path_history)), path_history)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("iterations")
    plt.ylabel("objective function value")
    plt.show()
