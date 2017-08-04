import numpy as np
import network
import game

class AI:
    def __init__(self, qnet: network.Network, color: int):
        self.color = color
        self.qnet = qnet

    def nextMove(self, c4: game.Connect4, epsilon=0):
        assert c4.onMove == self.color

        possibleMoves = c4.possibleMoves()
        if np.random.rand() < epsilon:
            move = possibleMoves[np.random.randint(len(possibleMoves))]
            #print("chose Random move:", move)
        else:
            max_q = -1 * float("inf")
            move = None
            for possibleMove in possibleMoves:
                netInput = c4.createNetInput(c4.field, possibleMove)
                cur_q = self.qnet.eval(netInput)
                if max_q < cur_q:
                    move = possibleMove
                    max_q = cur_q
                #print("curq:", cur_q, "is move:", possibleMove)
            #print ("will give max:", max_q, "result move:", move)
        # order important!
        state = np.copy(c4.field)
        _, end = c4.play(move)
        return state, move, end