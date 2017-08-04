from game import Connect4
import network
import ai
import numpy as np


class Training:
    def __init__(self, winning_bonus=1, draw_malus=-0.5, gamma=0.96, height=4, width=5, model_path=None):

        self.winning_bonus = winning_bonus
        self.draw_malus = draw_malus
        self.gamma = gamma
        self.qnet = network.Network(height, width, model_path="4x5_20-20-1.h5") #TODO: make generic
        self.height = height
        self.width = width

        self.ais = {
            -1: ai.AI(self.qnet,-1),
             1: ai.AI(self.qnet, 1)
        }

    def trainGame(self):
        c4 = Connect4(self.height, self.width)

        state_list = []
        move_list = []
        #reward_list = []

        while c4.winner == None and len(c4.possibleMoves()) > 0:
            color = c4.onMove
            state, move, end, explore = self.ais[color].nextMove(c4, 0.2)
            # print("state")
            # print(state)
            # print("move:", move, " (explore=", explore, ") ends=", end)

            state *= color
            state_list.append(state)
            move_list.append(move)
            #reward_list.append(self.endReward(end, color))

            if end in (-1, 0, 1):
                break

        netInput_list = []
        netTarget_list = []
        netLoss_list = []


        for i in range(len(state_list)-1, -1, -1):
            state = state_list[i]
            move = move_list[i]
            #reward = reward_list[i]
            netInput = Connect4.createNetInput(state, move)
            netOutput = self.qnet.eval(netInput)

            print("we are player 1 in state:")
            print(state)
            # next state but already inverted, we are player 1
            next_state, end = Connect4.play_static(col=move, field=state, color=1)
            print("we play", move, "which resulted end=", end)
            # print(next_state)
            Q_ = 0
            reward = self.endReward(end, 1)
            if end is None:
                next_state, best_q, best_next_move, end = self.bestNextMove(next_state, -1)

                # next_state_opponent_view = -1 * next_state
                # possibleOpponentMoves = Connect4.possibleMoves_static(field=next_state_opponent_view)
                # Q_opp_max = -1*float("inf")
                # opp_best_move = None
                # for possibleMove in possibleOpponentMoves:
                #     netInput_opponent = Connect4.createNetInput(next_state_opponent_view, possibleMove)
                #     Q_opp_cur = self.qnet.eval(netInput_opponent)
                #     if Q_opp_cur > Q_opp_max:
                #         Q_opp_max = Q_opp_cur
                #         opp_best_move = possibleMove
                #
                # next_next_state, end = Connect4.play_static(col=opp_best_move, field=next_state, color=-1)
                # print("then the opponent plays", opp_best_move, "for him we can expect", Q_opp_max, "and this ends=", end)
                # print(next_next_state)
                if end is not None:
                    #Q_ = self.endReward(end, 1)
                    r = self.endReward(end, 1)  # treat a direct loss next round also as reward
                else:
                    # this is the next move, here we will just use the highest q value for us.
                    next_state, best_q, best_move, end = self.bestNextMove(next_state, 1)
                    Q_ = best_q
                    #
                    # best_next_next_move = None
                    # possibleMoves = Connect4.possibleMoves_static(field=next_next_state)
                    # for possibleMove in possibleMoves:
                    #     next_next_netInput = Connect4.createNetInput(next_next_state, possibleMove)
                    #     Q_ = -1 * float("inf")
                    #     Q_cur = self.qnet.eval(next_next_netInput)
                    #     if Q_cur > Q_:
                    #         Q_ = Q_cur
                    #         best_next_next_move = possibleMove

            #         print("then we should play", best_next_next_move, "for which we expect", Q_)
            netTarget = np.array([reward + self.gamma * Q_]).reshape(1,1)
            print("==> Q(s,a)=", netOutput)
            print("==> r=", reward, " Q_=", Q_)
            netTarget_list.append(netTarget)

            netLoss = self.qnet.train(netInput, netTarget)
            netLoss_list.append(netLoss)
            print("---------------")

        return netLoss_list

    def testGame(self):
        c4 = Connect4(self.height, self.width)

    def bestNextMove(self, state, color):
        state_player_view = np.copy(state) * color
        possibleMoves = Connect4.possibleMoves_static(field=state_player_view)
        Q_max = -1 * float("inf")
        best_move = None
        for possibleMove in possibleMoves:
            netInput = Connect4.createNetInput(state_player_view, possibleMove)
            Q_opp_cur = self.qnet.eval(netInput)
            if Q_opp_cur > Q_max:
                Q_max = Q_opp_cur
                best_move = possibleMove
        next_state, end = Connect4.play_static(col=best_move, field=state, color=color)
        print(state)
        print(color, "plays", best_move, "(with q=", Q_max, ")and this ends=", end)
        return next_state, Q_max, best_move, end

    def endReward(self, end, color):
        if end == color:
            return self.winning_bonus
        elif end == -1 * color:
            return self.winning_bonus * -1
        elif end == 0:
            return self.draw_malus
        elif end is None:
            return 0