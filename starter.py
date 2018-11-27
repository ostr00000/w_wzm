from map_generator import generate_map, Field
from visualization import board_plot, exec_plot, path_plot
from q_learning import QLearning

import numpy

numpy.random.seed(5)

def main():
    board = generate_map(mountain_ratio=0.4)
    # exec_plot(board_plot, board)

    q = QLearning(board, record_path=True, max_actions=100)
    for _ in range(50):
        q.run_epoch(visualize=False)
        q.board[q.state] = Field.WITCHER
        exec_plot(path_plot, q.board, q.path)



if __name__ == '__main__':
    main()
