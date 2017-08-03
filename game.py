import numpy as np


class Connect4:
    def __init__(self, height, width, onMove=1):
        self.width = width
        self.height = height
        self.onMove = onMove
        self.winner = 0
        self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        # heighest row is 0

    def play(self, col):
        assert self.field[0, col] == 0, "col " + str(col) + " already full"
        for h in range(self.height - 1, -1, -1):
            if self.field[h, col] == 0:
                self.field[h, col] = self.onMove
                break
        self.onMove *= -1
        return self.checkWinner((h, col))


    def show(self):
        return self.field

    def reset(self):
        self.field = np.zeros((self.height, self.width), dtype=np.int32)

    def checkWinner(self, position):
        directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
        for d in directions:
            counter = 1
            cur_pos = self.mapping(position, d)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = self.mapping(cur_pos, d)
                counter += 1

            d_ = tuple((i * -1 for i in d))
            cur_pos = self.mapping(position, d_)
            while cur_pos is not None and self.field[cur_pos] == self.field[position]:
                cur_pos = self.mapping(cur_pos, d_)
                counter += 1

            if counter >= 4:
                self.winner = self.field[position]
                return True
        return False

    def mapping(self, position, move):
        new_pos = position[0] + move[0], position[1] + move[1]
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.height or new_pos[1] >= self.width:
            return None
        else:
            return new_pos

    def possibleMoves(self):
        return tuple((i for i in range(self.width) if self.field[0,i] == 0))

    def state_action(self, move: int):
        field = np.copy(self.field)
        one_hot = np.zeros((1, self.width))
        one_hot[0, move] = 1
        reshaped = np.reshape(field, (1, -1))
        #formatted = np.concatenate((reshaped, one_hot), axis=1)
        return reshaped, one_hot

    def netInput(self, state, action):
        return np.concatenate((state,action), axis=1)

    def netInputToState(self, netInput):
        return (netInput[:-self.width], netInput[-self.width:])

    def showNetInput(self, netInput, reward):
        print("Action --> Reward")
        print(netInput[0, -self.width:], reward)
        print("Field")
        print(netInput[0, :-self.width].reshape((self.height, self.width)))

    def processAction(self, state, action):
        "returns s+1"
        col = np.argmax(action)
        state = state.reshape((self.height, self.width))
        for h in range(self.height - 1, -1, -1):
            if state[h, col] == 0:
                state[h, col] = 1  # we are always player 1
                break
        next_state = state * -1
        return next_state

