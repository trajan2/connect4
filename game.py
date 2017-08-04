import numpy as np


class Connect4:
    def __init__(self, height, width, onMove=1):
        self.width = width
        self.height = height
        self.onMove = onMove
        self.winner = None  # None: no winner yet, 0: draw, +/-1 player
        self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        # heighest row is 0

    def play(self, col):
        self.field, self.winner = Connect4.play_static(col, self.field, self.onMove)
        self.onMove *= -1
        return self.winner

    @staticmethod
    def play_static(col: int, field: np.array, color: int):
        """
        if field/color are None, the regular game values are used,
        otherwised the passed parameters are used
        returns the field after the move was made, and a flag saying if the game is over thereafter
        """
        #assert color is None or field_param is not None
        # if field_param is not None:
        #     field = np.copy(field_param)
        # else:
        #     field = self.field
        field = np.copy(field)
        height = len(field)
        assert col is not None and field is not None and color is not None
        assert field[0, col] == 0, "col " + str(col) + " already full"
        for h in range(height - 1, -1, -1):
            if field[h, col] == 0:
                field[h, col] = color
                break
        winner = Connect4._checkWinner((h, col), field)
        return field, winner

    @staticmethod
    def _checkWinner(position, field):
        """
        :param position: where the last piece was placed
        :param field_param: a state, if None use self.field
        :return: -1, 1 for a possible winner, 0 for a draw, False for nothing yet
        """
        height = len(field)
        width = len(field[0])
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        for d in directions:
            counter = 1
            cur_pos = Connect4._mapping(position, d, height, width)
            while cur_pos is not None and field[cur_pos] == field[position]:
                cur_pos = Connect4._mapping(cur_pos, d, height, width)
                counter += 1

            d_ = tuple((i * -1 for i in d))
            cur_pos = Connect4._mapping(position, d_, height, width)
            while cur_pos is not None and field[cur_pos] == field[position]:
                cur_pos = Connect4._mapping(cur_pos, d_, height, width)
                counter += 1

            if counter >= 4:
                return field[position]

        if len(Connect4.possibleMoves_static(field=field)) == 0:
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
        return Connect4.possibleMoves_static(self.field)

    @staticmethod
    def possibleMoves_static(field):
        return tuple((i for i in range(len(field[0])) if field[0,i] == 0))

    @staticmethod
    def createNetInput(state, move):
        width = len(state[0])
        state_ = np.copy(state).reshape(1,-1)
        action = np.zeros((1, width))
        action[0, move] = 1
        return np.concatenate((state_, action), axis=1)

    def showNetInput(self, netInput, reward):
        print("State")
        print(netInput[0, :-self.width].reshape((self.height, self.width)))
        print("Action --> Reward")
        print(netInput[0, -self.width:], reward)

    def reset(self):
        self.field = np.zeros((self.height, self.width), dtype=np.int32)