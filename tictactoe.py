import numpy as np


class Board:

    def __init__(self):

        self._board = np.zeros((3, 3), dtype=int)
        self._finished = False
        self._win = False
        self._draw = False
        self.action_map = {
            'q': (0, 0),
            'w': (0, 1),
            'e': (0, 2),
            'a': (1, 0),
            's': (1, 1),
            'd': (1, 2),
            'z': (2, 0),
            'x': (2, 1),
            'c': (2, 2)
        }

    def place(self, pos, marker):

        if not self._finished:

            if isinstance(pos, str):
                x, y = self.action_map[pos]

            else:
                x, y = pos

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

        elif 0 not in self._board:
            self._finished = True
            self._draw = True

    def _next_state(self):

        next_state = tuple(self._board.reshape(-1))

        if self._win:
            reward = 2

        elif self._draw:
            reward = 1

        else:
            reward = 0

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

    def __init__(self, num_iter=5000, gamma=0.5, init_q=0.5):

        self.num_iter = num_iter
        self.gamma = gamma
        self.init_q = init_q
        self.board = Board()
        self.b = {}
        self.pi = {}
        self.q = {}
        self.C = {}

    def _train(self):

        pass

    def _add_states(self, s, actions):

        init_prob = len(actions) * [1 / len(actions)]

        if s not in self.b.keys():
            self.b[s] = (actions, init_prob)
            action_index = np.random.choice(range(len(actions)), p=init_prob)
            self.pi[s] = actions[action_index]

            for a in actions:
                self.q[s] = {a: self.init_q}
                self.C[s] = {a: 0}

    def _generate_episode(self):

        curr_state, reward, stop, actions = self.board.reset()
        episode = []

        while not stop:

            # random pick one action for marker 1
            action_index = np.random.choice(range(len(actions)))
            a_1 = actions[action_index]
            curr_state, reward, stop, actions = self.board.place(a_1, 1)

            if not stop:
                self._add_states(curr_state, actions)
                episode.append(curr_state)

                action_index = np.random.choice(range(len(actions)), p=self.b[curr_state][1])
                a_2 = actions[action_index]
                episode.append(a_2)

                curr_state, reward, stop, actions = self.board.place(a_2, -1)
                episode.append(reward)

        return episode


