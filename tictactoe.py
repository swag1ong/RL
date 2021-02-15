import numpy as np


class Board:

    def __init__(self):

        self._board = np.zeros((3, 3), dtype=int)
        self._finished = False
        self._win = False
        self.action_map = {
            'q': [0, 0],
            'w': [0, 1],
            'e': [0, 2],
            'a': [1, 0],
            's': [1, 1],
            'd': [1, 2],
            'z': [2, 0],
            'x': [2, 1],
            'c': [2, 2]
        }

    def place(self, pos, marker):

        if not self._finished:
            x, y = self.action_map[pos]

            if self._board[x][y] == 0:
                self._board[x][y] = marker
                self._is_finished()

            else:
                print('location not available!')

        print(self._board)

        return self._next_state()

    def _is_finished(self):

        diag_sum = self._board[0][0] + self._board[1][1] + self._board[2][2]
        neg_diag_sum = self._board[0][2] + self._board[1][1] + self._board[2][0]

        if any(self._board.sum(axis=1) == -3) or any(self._board.sum(axis=0) == -3) or diag_sum == -3 or neg_diag_sum == -3:
            self._finished = True
            self._win = True

        elif any(self._board.sum(axis=1) == 3) or any(self._board.sum(axis=0) == 3) or diag_sum == 3 or neg_diag_sum == 3:
            self._finished = True

    def _next_state(self):

        next_state = tuple(self._board.reshape(-1))
        reward = 1 if self._win else 0

        if not self._finished:
            actions = np.where(self._board == 0)
            action_space = [(x, y) for x, y in zip(actions[0], actions[1])]

        else:
            action_space = []

        return next_state, reward, self._finished, action_space

    def reset(self):

        self._board = np.zeros((3, 3), dtype=int)
        self._finished = False
        self._win = False
        self.action_map = {
            'q': [0, 0],
            'w': [0, 1],
            'e': [0, 2],
            'a': [1, 0],
            's': [1, 1],
            'd': [1, 2],
            'z': [2, 0],
            'x': [2, 1],
            'c': [2, 2]
        }

        return self._next_state()

class AiPlayer:

    def __init__(self, num_iter=5000, method='off-policy-WIS', gamma=0.5):

        self.num_iter = num_iter
        self.method = method
        self.gamma = gamma
        self.board = Board()

    def _train(self):

        while

    def _generate_episode(self):

