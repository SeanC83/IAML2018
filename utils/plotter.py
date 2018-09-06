## =============================================================================== ##
## 																				   ##
##	This file contains a number of plotting functions for use within the labs/	   ##
##	assignments.																   ##
## 																				   ##
## =============================================================================== ##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_grid(x, shape=None, **heatmap_params):
    """Function for reshaping and plotting vector data.
    If shape not given, assumed square.
    """
    if shape is None:
        width = int(np.sqrt(len(x)))
        if width == np.sqrt(len(x)):
            shape = (width, width)
        else:
            print('Data not square, supply shape argument')
    sns.heatmap(x.reshape(shape), annot=True, **heatmap_params)


def plot_hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    Source: https://matplotlib.org/examples/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
