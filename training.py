import game
import network
import ai
import numpy as np

class Training:
    def __init__(self, winning_bonus=100, draw_malus=-50, gamma=0.96):

        self.winning_bonus = winning_bonus
        self.draw_malus = draw_malus
        self.gamma = gamma
        self.qnet = network.Network(4, 5)

        self.ais = {
            -1: ai.AI(self.qnet,-1),
             1: ai.AI(self.qnet, 1)
        }

    def playGame(self):
        # TODO: remove
        self.c4 = game.Connect4(4, 5)

        netInput_list = []
        rewards = []

        while self.c4.winner == 0 and len(self.c4.possibleMoves())> 0 :
            color = self.c4.onMove
            state, action = self.ais[color].nextMove(self.c4, 0.2)
            state *= color
            netInput_list.append(self.c4.netInput(state,action))

            reward = 0
            if self.c4.winner == color:
                reward = self.winning_bonus
            elif self.c4.winner == -1 * color:
                reward = self.winning_bonus * -1
            elif len(self.c4.possibleMoves()) == 0:
                reward = self.draw_malus
            rewards.append(reward)

        isEnd_list = [False for _ in rewards]
        isEnd_list[-1] = True

        for netInput, reward in zip(netInput_list, rewards):
            self.c4.showNetInput(netInput, reward)

        # for i in range(len(netInput_list)-1, -1, -1):
        #     netInput = netInput_list[i]
        #     reward = rewards[i]
        #     isEnd = isEnd_list[i]
        #     Q_ = 0  # minimal future reward, in case of end of game
        #     if not isEnd:
        #         state, action = self.c4.netInputToState(netInput)
        #         # next state but already inverted, we are player 1
        #         next_state = self.c4.processAction(state, action)
        #         possibleMoves = [i for i in range(self.c4.width) if netInput[0, i] == 0]
        #
        #         for possibleMove in possibleMoves:
        #             Q_cur = self.qnet.eval(netInput)
        #             if Q_cur > Q_:
        #                 Q_ = Q_cur
        #
        #
        #     netInput_list = netInput_list[::-1]
        #     rewards = rewards[::-1]
        #
        #     netTargets = rewards + self.gamma * Q_