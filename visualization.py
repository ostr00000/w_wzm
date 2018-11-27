from itertools import tee

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.path import Path


def exec_plot(func, *args, save=False, save_filename='', sleep=False, **kwargs):
    plt.cla()
    plt.tight_layout()
    func(*args, **kwargs)
    if save:
        plt.savefig(save_filename + '.jpg', dpi=200)
    elif sleep:
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.001)
        # plt.pause(0.5)
    else:
        plt.ioff()
        plt.show()


def board_plot(board: np.ndarray):
    plt.imshow(board)


def sequence_plot(data):
    plt.plot(data)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def path_plot(board, vertices):
    board_plot(board)
    ax = plt.gca()

    max_width = 20
    size = vertices.shape[0]
    widths = np.linspace(0, max_width, size)
    colors = cm.rainbow(np.linspace(0, 1, size))
    for color, width, (v1, v2) in zip(colors, reversed(widths), pairwise(vertices)):
        path = Path([v1[::-1], v2[::-1]])
        ax.add_patch(patches.PathPatch(path, fill=False, edgecolor=color, linewidth=width))
