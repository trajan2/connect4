import numpy as np
import network
from typing import Union
from connect4 import Connect4 as Game

class AI:
    def __init__(self, color: int):
        self.color = color

    def exploreMove(self, c4: Game, epsilon=0.2):
        assert c4.onMove == self.color
        explore = np.random.rand() < epsilon
        if explore:
            move = np.random.choice(c4.possibleMoves())
        else:
            move = self.nextMove(c4)
        return move, {"explore": explore}

    def nextMove(self, c4: Game):
        return np.random.choice(c4.possibleMoves())


class NetAI(AI):
    def __init__(self, qnet: Union[network.Network, None],
                 color=-1):
        """
        mimics an opponent playing the game
        :param qnet: neural net evaluating the next moves
        :param color: this AIs "color"
        :param epsilon: 0.0: always use the net's suggestion,
                        1.0: always pick a random move from the possible ones
        """
        super().__init__(color)
        self.qnet = qnet

    def nextMove(self, c4: Game):
        move, _ = c4.bestNextMove(self.qnet)
        return move

class RepeatAI(AI):
    def __init__(self, color=-1):
        super().__init__(color)

    def nextMove(self, c4: Game):
        if c4.lastMove is not None and c4.lastMove in c4.possibleMoves():
            return c4.lastMove
        else:
            return np.random.choice(c4.possibleMoves())

class RepeatVicinityAI(AI):
    def __init(self, color=-1):
        super().__init__(color)

    def nextMove(self, c4: Game):
        if c4.lastMove is not None:
            vicinty = list(set(c4.possibleMoves()).intersection(range(c4.lastMove-1,c4.lastMove+2)))
            if len(vicinty) > 0:
                return np.random.choice(vicinty)
        return np.random.choice(c4.possibleMoves())