import numpy as np
import copy

winning_bonus = 100
draw_malus = -50


class State:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.winner = None  # 1 -> beginner has won, -1 -> second player has won, 0 -> draw and None if not finished
        self.round = 0  # counter for number of rounds played yet
        self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1

    def possible_actions(self):
        moves = tuple((Action(i) for i in range(self.width) if self.field[0, i] == 0))

        if self.winner is None and len(moves) > 0:
            return moves
        else:
            return tuple([Action(-1)])  # do not play at all (net_input for this action is a zero vector)

    def show(self):
        print("State. Winner is ", self.winner, ". Round is ", self.round)
        print(self.field)


class Action:
    def __init__(self, move):
        self.move = move

    def show(self):
        print("Action: ", self.move)


def play(state: State, action: Action):
    """
    Performs given action on the given state. Assumes player "1" is playing.
    Returns the resulting state. This state is shown from perspective of the other player (inverted).
    """

    new_state = copy.deepcopy(state)

    if new_state.winner is not None:
        assert action.move == -1, "Illegal action. Trying to play a token despite of game finished!"
        new_state.field *= -1
        new_state.round += 1
        return new_state

    h = None

    for h in range(new_state.height - 1, -1, -1):
        if new_state.field[h, action.move] == 0:
            new_state.field[h, action.move] = 1  # one is the color of the current player
            break

    assert h is not None, "Illegal Action performed! Column " + str(action.move) + " is already full!"

    # order is important!
    new_state.winner = check_winner((h, action.move), new_state)
    new_state.field *= -1
    new_state.round += 1

    return new_state


def check_winner(position, state: State):
    """
    :param position: where the last token was placed
    :param state: a state on which the check is done
    :return: -1, 1 for a possible winner, 0 for a draw, None for nothing yet
    """

    directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    for d in directions:
        counter = 1
        cur_pos = next_position(position, d, state)
        while cur_pos is not None and state.field[cur_pos] == state.field[position]:
            cur_pos = next_position(cur_pos, d, state)
            counter += 1

        d_ = tuple((i * -1 for i in d))
        cur_pos = next_position(position, d_, state)
        while cur_pos is not None and state.field[cur_pos] == state.field[position]:
            cur_pos = next_position(cur_pos, d_, state)
            counter += 1

        if counter >= 4:
            return (state.round % 2) * -2 + 1

    if state.possible_actions()[0].move == -1:  # draw is given if -1 = 'do nothing' is the only option
        return 0
    return None


def next_position(position, move, state: State):
    new_pos = position[0] + move[0], position[1] + move[1]
    if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= state.height or new_pos[1] >= state.width:
        return None
    else:
        return new_pos


def create_net_input(state: State, action: Action):
    flat_state = np.copy(state.field).reshape(1, -1)  # TODO is this copy necessary??? depends on keras
    one_hot = np.zeros((1, state.width))
    if action.move != -1:
        one_hot[0, action.move] = 1

    return np.concatenate((flat_state, one_hot), axis=1)


def get_net_input_dim(height, width):
    return height * width + width


def get_reward(state: State, action: Action):
    next_state = play(state, action)  # TODO write get_winner for state, action (next_state is not necessary)

    mapping = {
        None: 0,  # game is not finished yet
        0: draw_malus,  # game was a draw
        1: ((state.round % 2) * -2 + 1) * winning_bonus,  # 1 means the beginning player has won
        -1: ((state.round % 2) * 2 - 1) * winning_bonus,  # -1 means the second player has won
    }

    return mapping[next_state.winner]

    # def state_action(self, move: int):
    #    field = np.copy(self.field)
    #    one_hot = self.moveToOneHot(move)
    #    reshaped = np.reshape(field, (1, -1))
    #    #formatted = np.concatenate((reshaped, one_hot), axis=1)
    #    return reshaped, one_hot

    # def oneHotToMove(self, action):
    #    return np.argmax(action)

    # def moveToOneHot(self, move):
    #    one_hot = np.zeros((1, self.width))
    #    one_hot[0, move] = 1
    #    return one_hot

    # def createNetInput(self, state, move):
    #    state_ = np.copy(state).reshape(1,-1)
    #    action = self.moveToOneHot(move)
    #    return np.concatenate((state_, action), axis=1)

    # def splitNetInput(self, netInput):
    #    state = np.copy(netInput[0, :-self.width])
    #    action = np.copy(netInput[0, -self.width:])
    #   return state, action

    # def showNetInput(self, netInput, reward):
    #    print("State")
    #    print(netInput[0, :-self.width].reshape((self.height, self.width)))
    #    print("Action --> Reward")
    #    print(netInput[0, -self.width:], reward)

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

    # def show(self):
    #    return self.field

    # def reset(self):
    #    self.field = np.zeros((self.height, self.width), dtype=np.int32)