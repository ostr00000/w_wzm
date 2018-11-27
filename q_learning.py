from typing import Union, Tuple, NewType

import numpy as np

from map_generator import Field

QTyp = NewType('QTyp', np.ndarray)
StateTyp = NewType('StateTyp', Tuple[int, int])
ActionTyp = NewType('ActionTyp', Tuple[int, int])


class Actions:
    """
    ------> axis X
    |
    |
    V
    axis Y

    point(y, x)
    """
    MOVE_UP: ActionTyp = (1, 0)
    MOVE_DOWN: ActionTyp = (-1, 0)
    MOVE_RIGHT: ActionTyp = (0, 1)
    MOVE_LEFT: ActionTyp = (0, -1)
    ALL = {
        MOVE_UP: 0, MOVE_DOWN: 1, MOVE_RIGHT: 2, MOVE_LEFT: 3
    }


def is_factor(factor):
    assert isinstance(factor, float) and 0 <= factor <= 1


class QLearning:
    def __init__(self, board: np.ndarray, learning_rate: float = 0.5, discount_factor=0.5, max_actions=100,
                 measure_rewards=False, record_path=False, start_pos: StateTyp = (1, 1), end_pos=None):
        # to copy value
        self._start_position = start_pos
        self._original_board = board

        # renewable
        self.state = None
        self.board = None
        self.reward = 0
        self.move_number = 0
        self.target_reward = 0

        # coefficient
        self.max_actions = max_actions
        is_factor(learning_rate)
        self.learning_rate = learning_rate
        is_factor(discount_factor)
        self.discount_factor = discount_factor

        # Q
        self.Q: QTyp = np.zeros(board.shape + (4,), dtype=float)

        # log structures
        self.reward_log = np.empty(max_actions, dtype=float) if measure_rewards else None
        self._path = np.empty((max_actions + 1, 2), dtype=int) if record_path else None
        self.path = self._path

        self.reset()

    def reset(self):
        self.board = self._original_board.copy()
        self.state = self._start_position
        self.board[self._start_position] = Field.EMPTY

        if self._path is not None:
            self._path.fill(0)
            self._path[0] = self._start_position

        self.reward = 0
        self.move_number = 0
        self.target_reward = 0

    def move(self) -> bool:
        action = self.choose_action()
        old_state = self.state
        self.state = StateTyp((old_state[0] + action[0], old_state[1] + action[1]))

        reward = -1
        if self.move_number + 1 == self.max_actions:
            reward = -100

        filed = self.board[self.state]
        if filed == Field.MAYOR:
            self.target_reward = 1000
        elif filed == Field.DZIABERŁAK:
            reward = self.target_reward
        self.reward += reward

        self.move_number += 1

        if self._path is not None:
            self._path[self.move_number] = self.state

        self.update_q(old_state, action, reward, self.state, None)
        return not (filed == Field.DZIABERŁAK or self.move_number == self.max_actions)

    def run_epoch(self, visualize=False):
        self.reset()
        epoch_exist = True
        while epoch_exist:
            epoch_exist = self.move()

            # DEBUG
            print(f"Move number: {self.move_number} to pos: {self.state}")
            if visualize:
                from visualization import board_plot, exec_plot
                old = self.board[self.state]
                self.board[self.state] = Field.WITCHER
                exec_plot(board_plot, self.board, sleep=True)
                self.board[self.state] = old

        if self._path is not None:
            self.path = self._path[:self.move_number + 1]

        if self.reward_log is not None:
            self.reward_log[self.move_number] = self.reward

    def choose_action(self) -> ActionTyp:
        action_value = sorted(Actions.ALL.keys(), key=lambda k: self.Q[self.state + (Actions.ALL[k],)])
        d = [(k, self.Q[self.state + (v,)]) for k, v in Actions.ALL.items()]
        d.sort(key=lambda x: x[1])

        while True:
            best_action = action_value.pop()
            if self.is_allowed_action(best_action):
                return best_action

    def is_allowed_action(self, action: ActionTyp):
        x = self.state[0] + action[0]
        y = self.state[1] + action[1]
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.board[x, y] != Field.MOUNTAIN:
                return True
        return False

    def set_learning_mode(self, val: bool):
        pass

    def update_q(self, s: StateTyp, a: ActionTyp, r: float, ss: StateTyp, aa: ActionTyp):
        index = s + (Actions.ALL[a],)
        old_val = self.Q[index]
        best_future_value = max(self.Q[ss + (action_index,)] for action, action_index in Actions.ALL.items()
                                if self.is_allowed_action(action))
        self.Q[index] += self.learning_rate * (r + self.discount_factor * best_future_value - old_val)
