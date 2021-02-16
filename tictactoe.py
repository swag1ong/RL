import numpy as np


class Board:

    def __init__(self):

        self._board = np.zeros((3, 3), dtype=int)
        self._finished = False
        self.win = False
        self.draw = False
        self.lose = False
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

    def place(self, pos, marker, if_display=False):

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

            if if_display:
                print(self._board)

        return self._next_state()

    def _is_finished(self):

        diag_sum = self._board[0][0] + self._board[1][1] + self._board[2][2]
        neg_diag_sum = self._board[0][2] + self._board[1][1] + self._board[2][0]

        if any(self._board.sum(axis=1) == -3) or any(self._board.sum(axis=0) == -3) or diag_sum == -3 or neg_diag_sum == -3:
            self._finished = True
            self.win = True

        elif any(self._board.sum(axis=1) == 3) or any(self._board.sum(axis=0) == 3) or diag_sum == 3 or neg_diag_sum == 3:
            self._finished = True
            self.lose = True

        elif 0 not in self._board:
            self._finished = True
            self.draw = True

    def _next_state(self):

        next_state = tuple(self._board.reshape(-1))

        if self.win:
            reward = 1

        elif self.lose:
            reward = 0

        else:
            reward = 0.5

        if not self._finished:
            actions = np.where(self._board == 0)
            action_space = [(x, y) for x, y in zip(actions[0], actions[1])]

        else:
            action_space = []

        return next_state, reward, self._finished, action_space

    def reset(self):

        self._board = np.zeros((3, 3), dtype=int)
        self._finished = False
        self.win = False
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

    def train(self):

        count = 0

        while count < self.num_iter:
            W = 1
            G = 0
            s_epi, a_epi, r_epi = self._generate_episode()

            for i in range(len(s_epi) - 1, -1, -1):
                s = s_epi[i]
                a = a_epi[i]
                r = r_epi[i]

                G = self.gamma * G + r
                self.C[s][a] = self.C[s][a] + W
                self.q[s][a] = self.q[s][a] + W * (G - self.q[s][a]) / self.C[s][a]
                self.pi[s] = self._get_arg_max(s)

                if self.pi[s] == a:
                    action_index = self.b[s][0].index(a)
                    W = W / self.b[s][1][action_index]

            count += 1

            print(f'iteration {count} finished')

    def _add_states(self, s, actions):

        init_prob = len(actions) * [1 / len(actions)]

        if s not in self.b.keys():
            self.b[s] = (actions, init_prob)
            action_index = np.random.choice(range(len(actions)), p=init_prob)
            self.pi[s] = actions[action_index]
            self.q[s] = {a: self.init_q for a in actions}
            self.C[s] = {a: 0 for a in actions}

    def _generate_episode(self):

        curr_state, reward, stop, actions = self.board.reset()
        a_lst = []
        r_lst = []
        s_lst = []

        while not stop:

            # random pick one action for marker 1
            a_1 = self._dumb_ai(actions)
            curr_state, reward, stop, actions = self.board.place(a_1, 1)

            if not stop:
                self._add_states(curr_state, actions)
                s_lst.append(curr_state)

                action_index = np.random.choice(range(len(actions)), p=self.b[curr_state][1])
                a_2 = actions[action_index]
                a_lst.append(a_2)

                curr_state, reward, stop, actions = self.board.place(a_2, -1)
                r_lst.append(reward)

        return s_lst, a_lst, r_lst

    def _get_arg_max(self, s):

        curr_largest = -1
        arg_max = 0
        a_dict = self.q[s]

        for k, v in a_dict.items():

            if v > curr_largest:
                arg_max = k
                curr_largest = v

        return arg_max

    def eval_performance(self, num_ep=100):

        win = 0
        draw = 0
        loss = 0
        count = 0

        while count < num_ep:
            curr_state, reward, stop, actions = self.board.reset()

            try:

                while not stop:
                    action_index = np.random.choice(range(len(actions)))
                    a_1 = actions[action_index]
                    curr_state, reward, stop, actions = self.board.place(a_1, 1)

                    if not stop:
                        a_2 = self.pi[curr_state]
                        curr_state, reward, stop, actions = self.board.place(a_2, -1)

                if self.board.win:
                    win += 1

                elif self.board.draw:
                    draw += 1

                else:
                    loss += 1

            except KeyError:

                continue

            count += 1

        print(f'number of wins: {win}')
        print(f'number of draws: {draw}')
        print(f'number of losses: {loss}')

    def _dumb_ai(self, actions):

        curr_board = self.board._board
        row_sum = curr_board.sum(axis=1)
        col_sum = curr_board.sum(axis=0)
        pos_dia_sum = curr_board[0][0] + curr_board[1][1] + curr_board[2][2]
        neg_dia_sum = curr_board[0][2] + curr_board[1][1] + curr_board[2][0]

        if any(row_sum == 2):
            x = np.argmax(row_sum == 2)
            y = np.argmax(curr_board[x] == 0)

            return x, y

        elif any(col_sum == 2):
            y = np.argmax(col_sum == 2)
            x = np.argmax(curr_board[:, y] == 0)

            return x, y

        elif pos_dia_sum == 2:

            if curr_board[0][0] == 0:

                return 0, 0

            elif curr_board[1][1] == 0:

                return 1, 1

            else:

                return 2, 2

        elif neg_dia_sum == 2:

            if curr_board[0][2] == 0:

                return 0, 2

            elif curr_board[1][1] == 0:

                return 1, 1

            else:

                return 2, 0

        else:

            a = np.random.choice(range(len(actions)))
            action = actions[a]

            return action


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_iter', type=int, default=10000)
    parser.add_argument('-gamma', type=float, default=0.8)
    parser.add_argument('-init_q', type=float, default=0.5)

    args = parser.parse_args()

    ai_player = AiPlayer(num_iter=args.num_iter, gamma=args.gamma, init_q=args.init_q)
    ai_player.train()

    while True:

        board = Board()
        stop = False

        while not stop:
            player = input('please place, you are 1:')
            curr_state, _, stop, _ = board.place(player, 1, if_display=True)

            if not stop:

                ai_a = ai_player.pi[curr_state]
                _, _, stop, _ = board.place(ai_a, -1, if_display=True)

            if stop:

                if board.win:
                    print('computer win!')

                elif board.draw:
                    print('draw !')

                else:
                    print('you win!')

        again = input('play again? Yes or No')

        if again == 'No':
            break


