import numpy as np
import typing


class Connect4:
    def __init__(self, height=4, width=5, onMove=1, field=None):
        self.height = height
        self.width = width
        self.onMove = onMove
        self.winner = None  # None: no winner yet, 0: draw, +/-1 player
        if field is None:
            self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        else:
            self.field = np.copy(field)
        # highest row is 0

    def play(self, col: int) -> typing.Union[int, None]:
        assert col is not None
        for h in range(self.height - 1, -1, -1):
            if self.field[h, col] == 0:
                self.field[h, col] = self.onMove
                break
        self.winner = self._checkWinner((h, col))
        self.onMove *= -1

        return self.winner

    def _checkWinner(self, position):
        """
        :param position: where the last piece was placed
        :param field_param: a state, if None use self.field
        :return: -1, 1 for a possible winner, 0 for a draw, False for nothing yet
        """
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        for d in directions:
            counter = 1
            cur_pos = game._mapping(position, d, self.height, self.width)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = game._mapping(cur_pos, d, self.height, self.width)
                counter += 1

            d_ = tuple((i * -1 for i in d))
            cur_pos = game._mapping(position, d_, self.height, self.width)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = game._mapping(cur_pos, d_, self.height, self.width)
                counter += 1

            if counter >= 4:
                return self.field[position]

        if len(self.possibleMoves()) == 0:
            return 0
        return None

    @staticmethod
    def _mapping(position, move, height, width):
        new_pos = position[0] + move[0], position[1] + move[1]
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= height or new_pos[1] >= width:
            return None
        else:
            return new_pos

    def possibleMoves(self):
        return tuple((i for i in range(len(self.field[0])) if self.field[0,i] == 0))

    def bestNextMove(self, qnet):
        field_player_view = np.copy(self.field) * self.onMove
        q_max = -1 * float("inf")
        best_move = None
        for possibleMove in self.possibleMoves():
            netInput = game.createNetInput(field_player_view, possibleMove)
            q_cur = qnet.eval(netInput)
            if q_cur > q_max:
                q_max = q_cur
                best_move = possibleMove
        return best_move, q_max

    @staticmethod
    def createNetInput(field, move):
        assert field is not None and move is not None
        state = np.copy(field).reshape(1,-1)
        action = np.zeros((1, len(field[0])))
        action[0, move] = 1
        return np.concatenate((state, action), axis=1)

    def showNetInput(self, netInput, reward):
        print("State")
        print(netInput[0, :-self.width].reshape((self.height, self.width)))
        print("Action --> Reward")
        print(netInput[0, -self.width:], reward)
