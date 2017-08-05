from connect4 import Connect4 as Game
import network
import ai
import numpy as np


class RLFramework:
    def __init__(self, winning_bonus=1, draw_malus=-0.5, gamma=0.96, height=4, width=5,
                 model_path=None, opponent_ai=None, epsilon=0.2):

        self.qnet = network.Network(model_path=model_path, in_dim=width * height + width)
        self.height = height
        self.width = width
        self.rewards = {None: 0, -1: -1 * winning_bonus, 0: draw_malus, 1: winning_bonus}
        self.gamma = gamma
        self.epsilon = epsilon

        # AIs for exploring the field, use the epsilon parameter below to get more randomness
        self.ais = {
            -1: ai.NetAI(self.qnet, 1) if opponent_ai is None else opponent_ai,
             1: ai.NetAI(self.qnet, 1)
        }
        #
        # if opponent_mode == "rand":
        #     self.opponent_ai = ai.AI(self.qnet, -1, epsilon=1.0)
        # elif opponent_mode == "frozen":
        #     self.opponent_ai = ai.AI(self.frozen_qnet, -1, epsilon=0.0)
        # elif opponent_mode == "qnet":
        #     self.opponent_ai = ai.AI(self.qnet, -1, epsilon=0.0)

    def playGame(self, verbose=False):
        """
        explore another round of the game
            verbose: bool
                show the process of this game
            returns: list[np.array], list[int]
                the state of the playing field before the move and the move
        """
        c4 = Game(self.height, self.width)

        state_list = []
        move_list = []
        # generate training data through two AIs playing against each other
        while c4.winner is None:  # and len(c4.possibleMoves()) > 0:
            color = c4.onMove
            move, meta = self.ais[color].exploreMove(c4, epsilon=self.epsilon)
            state_list.append(np.copy(c4.field)*color)
            if verbose:
                print(c4.field)
            c4.play(move)
            if verbose:
                print("move:", move, " (explore=", meta["explore"], ") ends=", c4.winner)
            move_list.append(move)

            if c4.winner in (-1, 0, 1):
                break
        return state_list, move_list

    def trainGame(self, state_list, move_list, verbose=False):
        netLoss_list = []
        for i in range(len(state_list)-1, -1, -1):
            move = move_list[i]
            c4 = Game(height = self.height, width=self.width, field=state_list[i], onMove=1)
            netInput = Game.createNetInput(c4.field, move)
            netOutput = self.qnet.eval(netInput)  # calculate now before moves are played Q(s,a)

            if verbose:
                print("we are player 1 in state:\n", c4.field)
            # next state but already inverted, we are player 1
            c4.play(move)
            if verbose:
                print("and do action ", move, "which resulted end=", c4.winner, "\n", c4.field)

            q_future = 0
            reward = self.rewards[c4.winner]
            if c4.winner is None:
                # opponent plays random move, result also counts directly to reward
                assert c4.onMove == -1
                move = self.ais[-1].nextMove(c4)
                c4.play(move)
                reward = self.rewards[c4.winner]

                if c4.winner is None:
                    assert c4.onMove == 1
                    move, q_max = c4.bestNextMove(self.qnet)
                    q_future = q_max

            netTarget = np.array([reward + self.gamma * q_future]).reshape(1,1)
            if verbose:
                print("==> Q(s,a)=", netOutput)
                print("==> r=", reward, " Q_=", q_future)

            netLoss = self.qnet.train(netInput, netTarget)
            netLoss_list.append(netLoss)
        return netLoss_list

    def testGame(self, enemy="opp", verbose=False, qnet_starts=True):
        c4 = Game(self.height, self.width, onMove=1 if qnet_starts else -1)
        while c4.winner is None:
            if qnet_starts:
                move, q_max = c4.bestNextMove(self.qnet)
                c4.play(move)
                if verbose:
                    print(c4.field, "\nplayer 1 (qnet) played", move)
            qnet_starts = True

            if c4.winner is None:
                if enemy == "qnet":
                    move, q_max = c4.bestNextMove(self.qnet)
                elif enemy == "opp":
                    move = self.ais[-1].nextMove(c4)
                elif enemy == "rand":
                    move = np.random.choice(c4.possibleMoves())
                elif enemy == "human":
                    print("Current Field state:\n", c4.field)
                    move = int(input("Your move: "+str(c4.possibleMoves())))

                if verbose:
                    print("player -1", enemy, "played", move)
                c4.play(move)
        return c4.winner

    def saveModel(self, model_path):
        self.qnet.save(model_path)