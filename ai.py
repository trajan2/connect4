import numpy as np
import network
from connect4 import Connect4 as Game

class AI:
    def __init__(self, qnet: network.Network, color: int):
        self.color = color
        self.qnet = qnet

    def nextMove(self, c4: Game, epsilon=0.2):
        assert c4.onMove == self.color
        possibleMoves = c4.possibleMoves()
        explore = np.random.rand() < epsilon
        if explore:
            move = np.random.choice(possibleMoves)
        else:
            move, _ = c4.bestNextMove(self.qnet)
        state = np.copy(c4.field)
        end = c4.play(move)
        return state, move, end, explore
