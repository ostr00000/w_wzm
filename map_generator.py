# coding=utf-8
import random

import numpy as np


class Field(object):
    WITCHER = 0
    MOUNTAIN = 1
    EMPTY = 2
    DZIABERŁAK = 3
    MAYOR = 4


def generate_map(size=10, mountain_ratio=0.2, dziaberłak_choices=None):
    assert size > 3
    board = np.full((size, size), Field.EMPTY, dtype='b')

    mountain = np.random.randint(0, size, size=(int(board.size * mountain_ratio), 2))
    board[mountain[:, 0], mountain[:, 1]] = Field.MOUNTAIN

    witcher = 1, 1
    board[witcher] = Field.WITCHER

    dziaberłak = random.choice(dziaberłak_choices) if dziaberłak_choices else (size - 2, size - 2)
    board[dziaberłak] = Field.DZIABERŁAK

    mayor = 1, size - 2
    board[mayor] = Field.MAYOR

    return board

