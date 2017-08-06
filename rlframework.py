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

    def play_game(self, on_move=None, verbose=False):
        """
        explore another round of the game
            verbose: bool
                show the process of this game
            returns: list[np.array], list[int]
                the state of the playing field before the move and the move
        """
        c4 = Game(self.height, self.width, on_move=np.random.choice([-1,1]) if on_move is None else on_move)

        state_list = []
        move_list = []
        # generate training data through two AIs playing against each other
        while c4.winner is None:  # and len(c4.possibleMoves()) > 0:
            color = c4.on_move
            move, meta = self.ais[color].next_exploring_move(c4, epsilon=self.epsilon)
            state_list.append(np.copy(c4.field)*color)
            print_cond(c4.field, cond=verbose)
            c4.play(move)
            print_cond("move:", move, " (explore=", meta["explore"], ") ends=", c4.winner, cond=verbose)
            move_list.append(move)

            if c4.winner in (-1, 0, 1):
                break
        return state_list, move_list

    def trainGame(self, state_list, move_list, verbose=False):
        net_losses = []
        for i in range(len(state_list)-1, -1, -1):
            move = move_list[i]
            c4 = Game(height=self.height, width=self.width, field=state_list[i], on_move=1)
            net_input = Game.create_net_input(c4.field, move)
            net_output = self.qnet.eval(net_input)  # calculate now before moves are played Q(s,a)

            print("we are player 1 in state:\n", c4.field, cond=verbose)
            # next state but already inverted, we are player 1
            c4.play(move)
            print_cond("and do action ", move, "which resulted end=", c4.winner, "\n", c4.field, cond=verbose)

            q_future = 0
            reward = self.rewards[c4.winner]
            if c4.winner is None:
                # opponent plays random move, result also counts directly to reward
                assert c4.on_move == -1
                move = self.ais[-1].next_move(c4)
                c4.play(move)
                reward = self.rewards[c4.winner]

                if c4.winner is None:
                    assert c4.on_move == 1
                    move, q_max = c4.best_next_move(self.qnet)
                    q_future = q_max

            net_target = np.array([reward + self.gamma * q_future]).reshape(1,1)
            print_cond("==> Q(s,a)=", net_output, cond=verbose)
            print_cond("==> r=", reward, " Q_=", q_future, cond=verbose)

            net_loss = self.qnet.train(net_input, net_target)
            net_losses.append(net_loss)
        return net_losses

    def test_game(self, enemy="opp", verbose=False, player1_starts=True):
        c4 = Game(self.height, self.width, on_move=1 if player1_starts else -1)
        while c4.winner is None:
            if player1_starts:
                move, q_max = c4.best_next_move(self.qnet)
                c4.play(move)
                print_cond(c4.field, "\nplayer 1 (qnet) played", move, cond=verbose)
            player1_starts = True

            if c4.winner is None:
                if enemy == "qnet":
                    move, q_max = c4.best_next_move(self.qnet)
                elif enemy == "opp":
                    move = self.ais[-1].next_move(c4)
                elif enemy == "rand":
                    move = np.random.choice(c4.possible_moves())
                elif enemy == "human":
                    print("Current Field state:\n", c4.field)
                    move = int(input("Your move: " + str(c4.possible_moves())))

                print_cond("player -1", enemy, "played", move, cond=verbose)
                c4.play(move)
        return c4.winner

    def save_model(self, model_path):
        self.qnet.save(model_path)


def print_cond(*args, cond):
    if cond:
        print(*args)