import numpy as np
import typing


class Connect4:
    def __init__(self, height=4, width=5, on_move=1, field=None):
        self.height = height
        self.width = width
        self.on_move = on_move
        self.winner = None  # None: no winner yet, 0: draw, +/-1 player
        self.lastMove = None
        if field is None:
            self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        else:
            self.field = np.copy(field)
        # highest row is 0

    def play(self, move: int) -> typing.Union[int, None]:
        assert move is not None
        self.lastMove = move
        for h in range(self.height - 1, -1, -1):
            if self.field[h, move] == 0:
                self.field[h, move] = self.on_move
                break
        self.winner = self._checkWinner((h, move))
        self.on_move *= -1

        return self.winner

    def _checkWinner(self, position):
        """
        :param position: where the last piece was placed
        :return: -1, 1 for a possible winner, 0 for a draw, False for nothing yet
        """
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        for d in directions:
            counter = 1
            cur_pos = Connect4._mapping(position, d, self.height, self.width)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = Connect4._mapping(cur_pos, d, self.height, self.width)
                counter += 1

            d_ = tuple((i * -1 for i in d))
            cur_pos = Connect4._mapping(position, d_, self.height, self.width)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = Connect4._mapping(cur_pos, d_, self.height, self.width)
                counter += 1

            if counter >= 4:
                return self.field[position]

        if len(self.possible_moves()) == 0:
            return 0
        return None

    @staticmethod
    def _mapping(position, move, height, width):
        new_pos = position[0] + move[0], position[1] + move[1]
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= height or new_pos[1] >= width:
            return None
        else:
            return new_pos

    def possible_moves(self):
        return tuple((i for i in range(len(self.field[0])) if self.field[0,i] == 0))

    def best_next_move(self, qnet):
        field_player_view = np.copy(self.field) * self.on_move
        q_max = -1 * float("inf")
        best_move = None
        for possible_move in self.possible_moves():
            net_input = Connect4.create_net_input(field_player_view, possible_move)
            q_cur = qnet.eval(net_input)
            if q_cur > q_max:
                q_max = q_cur
                best_move = possible_move
        return best_move, q_max

    @staticmethod
    def create_net_input(field, move):
        assert field is not None and move is not None
        state = np.copy(field).reshape(1,-1)
        action = np.zeros((1, len(field[0])))
        action[0, move] = 1
        return np.concatenate((state, action), axis=1)
