import matplotlib.pyplot as plt
import numpy as np


def plot_func_value_vs_iter_num(func_val_list, methods, title):  # ,
    plt.plot(np.arange(len(func_val_list[0])), func_val_list[0])
    plt.plot(np.arange(len(func_val_list[1])), func_val_list[1])
    plt.legend(methods)
    plt.title(title)
    plt.show()


def plot_contour_lines(f, methods,title, x1_list, x2_list,func_val_list,
                       x1_min=-2., x1_max=2., x2_min=-2., x2_max=2., n_points=100):
    x1 = np.linspace(x1_min, x1_max, n_points)
    x2 = np.linspace(x2_min, x2_max, n_points)
    x1_grid, x2_grid = np.meshgrid(x1,x2)
    z = np.zeros(x1_grid.shape)
    for i in range(x1_grid.shape[0]):
        for j in range(x1_grid.shape[1]):
            z[i,j]= f(np.array([[x1_grid[i,j]],[x2_grid[i,j]]]))[0]

    CS = plt.contour(x1_grid, x2_grid, z, 20)
    plt.clabel(CS, inline=True, fontsize=10)

    plt.plot(x1_list[0], x2_list[0], color='blue', linestyle='dashed', marker='o') #, label='gd_path')
    plt.plot(x1_list[1], x2_list[1], color='orange', linestyle='dashed', marker='o') #label='nt_path')

    plt.plot(x1_list[0][-1], x2_list[0][-1], marker='X' , markerfacecolor='blue', markeredgecolor='black',markersize=12) #, label='gd final x')
    plt.plot(x1_list[1][-1], x2_list[1][-1] , marker='X' , markerfacecolor='orange', markeredgecolor='black', markersize=12) #, label='nt final x') ,


    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(methods)
    plt.colorbar()
    plt.title(title)

    plt.show()

