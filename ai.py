import numpy as np
import network
import game
from abc import ABCMeta, abstractmethod


class AI:
    __metaclass__ = ABCMeta
    name = None

    def next_exploring_move(self, state: game.State, epsilon: float = 0.2):
        if np.random.rand() < epsilon:
            next_action = state.random_action()
        else:
            next_action = self.next_move(state)
        return next_action

    @abstractmethod
    def next_move(self, state: game.State):
        pass

class RandomAI(AI):
    def __init__(self):
        self.name = "RandomAI"

    def next_move(self, state: game.State):
        return state.random_action()


class NetAI(AI):
    def __init__(self, in_dim: int, load_file=None, hiddens=(100, 100)):
        self.qnet = network.Network(in_dim, load_file, hiddens=hiddens)
        self.name = "NetAI/"+str(load_file)

    def next_move(self, state: game.State):
        max_q = -1 * float("inf")
        action = None
        for possible_action in state.possible_actions():
            cur_q = self.qnet.eval(game.create_net_input(state, possible_action))
            if max_q < cur_q:
                action = possible_action
                max_q = cur_q
        action.rating = max_q
        return action

    def store(self, name):
        self.qnet.store(name)


class FillupAI(AI):
    def __init__(self):
        self.column = None
        self.name = "FillupAI"

    def next_move(self, state: game.State):
        if self.column is None:
            self.column = state.random_action()

        for i in range(state.height):
            if state.field[i, self.column.move] != 0:
                break

        if state.field[i, self.column.move] == -1:
            self.column = state.random_action()
        return self.column