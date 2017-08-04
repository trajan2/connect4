import numpy as np


class Connect4:
    def __init__(self, height, width, onMove=1):
        self.width = width
        self.height = height
        self.onMove = onMove
        self.winner = 0
        self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        # heighest row is 0

    def play(self, col, field_param=None, color=None):
        """
        if field/color are None, the regular game values are used,
        otherwised the passed parameters are used
        returns the field after the move was made, and a flag saying if the game is over thereafter
        """
        if field_param is not None:
            field = np.copy(field_param)
        else:
            field = self.field
        assert field[0, col] == 0, "col " + str(col) + " already full"
        for h in range(self.height - 1, -1, -1):
            if field[h, col] == 0:
                field[h, col] = self.onMove if color is None else color
                break
        if color is None:
            self.onMove *= -1
        return field, self.checkWinner((h, col), field_param=None if field_param is None else field)

    def checkWinner(self, position, field_param=None):
        """
        :param position: where the last piece was placed
        :param field_param: a state, if None use self.field
        :return: -1, 1 for a possible winner, 0 for a draw, False for nothing yet
        """
        field = self.field if field_param is None else field_param
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        for d in directions:
            counter = 1
            cur_pos = self.mapping(position, d)
            while cur_pos is not None and field[cur_pos] == field[position]:
                cur_pos = self.mapping(cur_pos, d)
                counter += 1

            d_ = tuple((i * -1 for i in d))
            cur_pos = self.mapping(position, d_)
            while cur_pos is not None and self.field[cur_pos] == field[position]:
                cur_pos = self.mapping(cur_pos, d_)
                counter += 1

            if counter >= 4:
                if field_param is not None:
                    self.winner = field[position]
                return field[position]
        if len(self.possibleMoves(field=field_param)) == 0:
            return 0
        return None

    def mapping(self, position, move):
        new_pos = position[0] + move[0], position[1] + move[1]
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.height or new_pos[1] >= self.width:
            return None
        else:
            return new_pos

    def possibleMoves(self, field=None):
        field = self.field if field is None else field
        return tuple((i for i in range(self.width) if field[0,i] == 0))

    def state_action(self, move: int):
        field = np.copy(self.field)
        one_hot = self.moveToOneHot(move)
        reshaped = np.reshape(field, (1, -1))
        #formatted = np.concatenate((reshaped, one_hot), axis=1)
        return reshaped, one_hot

    def oneHotToMove(self, action):
        return np.argmax(action)

    def moveToOneHot(self, move):
        one_hot = np.zeros((1, self.width))
        one_hot[0, move] = 1
        return one_hot

    def createNetInput(self, state, move):
        state_ = np.copy(state).reshape(1,-1)
        action = self.moveToOneHot(move)
        return np.concatenate((state_, action), axis=1)

    def splitNetInput(self, netInput):
        state = np.copy(netInput[0, :-self.width])
        action = np.copy(netInput[0, -self.width:])
        return state, action

    def showNetInput(self, netInput, reward):
        print("State")
        print(netInput[0, :-self.width].reshape((self.height, self.width)))
        print("Action --> Reward")
        print(netInput[0, -self.width:], reward)


    # def processAction(self, state, action):
    #     "returns s+1"
    #     col = np.argmax(action)
    #     state = state.reshape((self.height, self.width))
    #     for h in range(self.height - 1, -1, -1):
    #         if state[h, col] == 0:
    #             state[h, col] = 1  # we are always player 1
    #             break
    #
    # def
    #     possibleOpponentActions = self.possibleMoves(state)
    #     for possibleOpponentAction in possibleOpponentActions:
    #
    #
    #
    #     return


    def show(self):
        return self.field

    def reset(self):
        self.field = np.zeros((self.height, self.width), dtype=np.int32)