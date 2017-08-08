import numpy as np
import copy
import uuid

winning_bonus = 100
draw_malus = -50

class State:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.winner = None  # 1: current player (1) has won, -1: opponent player (-1) has won, 0 draw, None not finished
        self.round = 0  # counter for number of rounds played yet
        self.field = np.zeros((height, width), dtype=np.int32)  # 0: empty, 1,-1: player 1,-1
        self.id = str(uuid.uuid4())

    def possible_actions(self):
        moves = tuple((Action(i) for i in range(self.width) if self.field[0, i] == 0))

        if self.winner is None and len(moves) > 0:
            return moves
        else:
            return tuple([Action(-1)])  # do not play at all (net_input for this action is a zero vector)

    def random_action(self):
        possible_actions = self.possible_actions()
        assert len(possible_actions) >= 1
        return possible_actions[np.random.randint(0, len(possible_actions))]

    def show(self):
        print("State. Winner is ", self.winner, ". Round is ", self.round)
        print(self.field)

    def to_graphviz(self):
        mapping = {
            0: "white",
            1: "yellow",
            -1: "red"
        }

        result = "<<table bgcolor=\"blue\" cellspacing=\"5\">"
        for y in range(self.height):
            result += "<tr>"
            for x in range(self.width):
                result += "<td width=\"15\" height=\"15\" bgcolor=\"" + mapping[self.field[y, x]] + "\">" + "</td>"
            result += "</tr>"
        result += "</table>>"
        return result


class Action:
    def __init__(self, move, rating=None):
        self.move = move
        self.rating = rating

    def show(self):
        print("Action: ", self.move)

    def to_graphviz(self):
        return "Column: " + str(self.move)


def play(state: State, action: Action):
    """
    Performs given action on the given state. Assumes player "1" is playing.
    Returns the resulting state. This state is shown from perspective of the other player (inverted).
    """

    new_state = copy.deepcopy(state)
    new_state.id = str(uuid.uuid4())

    if new_state.winner is not None:
        assert action.move == -1, "Illegal action. Trying to play a token despite of game finished!"
        new_state.field *= -1
        new_state.round += 1
        new_state.winner *= -1
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
            return 1 #(state.round % 2) * -2 + 1

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
        1: winning_bonus, # 1 means the player (1) has won
        -1: -1 * winning_bonus # -1 means the opponent (-1) has won
    }

    return mapping[next_state.winner]
