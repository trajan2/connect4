import game
import network
import ai
import numpy as np

class Training:
    def __init__(self, winning_bonus=100, draw_malus=-50, gamma=0.96, height=4, width=5):

        self.winning_bonus = winning_bonus
        self.draw_malus = draw_malus
        self.gamma = gamma
        self.qnet = network.Network(height, width)
        self.height = height
        self.width = width

        self.ais = {
            -1: ai.AI(self.qnet,-1),
             1: ai.AI(self.qnet, 1)
        }

    def playGame(self):
        c4 = game.Connect4(self.height, self.width)


        state_list = []
        move_list = []
        reward_list = []

        while c4.winner == 0 and len(c4.possibleMoves())> 0 :
            color = c4.onMove
            state, move, end = self.ais[color].nextMove(c4, 0.2)
            # print("state")
            # print(state.reshape(self.height, self.width))
            # print("action:", action)

            state *= color
            state_list.append(state)
            move_list.append(move)

            reward = 0
            if end is not None:
                reward = self.endReward(end, color)
            reward_list.append(reward)

        # for netInput, reward in zip(netInput_list, reward_list):
        #     c4.showNetInput(netInput, reward)

        netInput_list = []
        netTarget_list = []


        for i in range(len(state_list)-1, -1, -1):
            state = state_list[i]
            move = move_list[i]
            reward = reward_list[i]
            netInput = c4.createNetInput(state, move)
            netOutput = self.qnet.eval(netInput)

            print("we are player 1 in state:")
            print(state)
            # next state but already inverted, we are player 1
            next_state, end = c4.play(col=move, field_param=state, color=1)
            print("we play", move, "which resulted end=", end)
            print(next_state)
            Q_ = 0
            if end is not None:
                #Q_ = self.endReward(end, 1)
                pass
            else:
                next_state_oppent_view = -1 * next_state
                possibleOpponentMoves = c4.possibleMoves(field=next_state_oppent_view)
                Q_opp_max = -1*float("inf")
                opp_best_move = None
                for possibleMove in possibleOpponentMoves:
                    netInput_opponent = c4.createNetInput(next_state_oppent_view, possibleMove)
                    Q_opp_cur = self.qnet.eval(netInput_opponent)
                    if Q_opp_cur > Q_opp_max:
                        Q_opp_max = Q_opp_cur
                        opp_best_move = possibleMove

                next_next_state, end = c4.play(col=opp_best_move, field_param=next_state, color=-1)
                print("then the opponent plays", opp_best_move, "for him we can expect", Q_opp_max, "and this ends=", end)
                print(next_next_state)
                if end is not None:
                    # Q_ = self.endReward(end, 1)
                    pass
                else:
                    best_next_next_move = None
                    possibleMoves = c4.possibleMoves(field=next_next_state)
                    for possibleMove in possibleMoves:
                        next_next_netInput = c4.createNetInput(next_next_state, possibleMove)
                        Q_cur =  self.qnet.eval(next_next_netInput)
                        if Q_cur > Q_:
                            Q_ = Q_cur
                            best_next_next_move = possibleMove

                    print("then we should play", best_next_next_move, "for which we expect", Q_)
            print("==> Q(s,a)=", netOutput)
            netTarget = np.array([reward + self.gamma * Q_]).reshape(1,1)
            print("==> r=", reward, " Q_=", Q_)
            netTarget_list.append(netTarget)
            self.qnet.train(netInput, netTarget)
            print("---------------")

        if False:  # Train on Batch
            netInput_np  = np.array(netInput_list[::-1])
            netTarget_np = np.array(netTarget_list)  # already inverted, bc upper for loop runs inverted

            self.qnet.train(netInput_np, netTarget_np)


    def endReward(self, end, color):
        if end == color:
            return self.winning_bonus
        elif end == -1 * color:
            return self.winning_bonus * -1
        elif end == 0:
            return  self.draw_malus