import numpy as np
import network
import game


class AI:
    def __init__(self, in_dim : int):
        self.qnet = network.Network(in_dim)

    def perform_best_move(self, state : game.State, epsilon=0):
        """
        Chooses the next action and performs this
        :param state: State on which an action should be chosen
        :param epsilon: Exploration parameter. Used to decide random in some cases.
        :return: Returns the new state after performing the chosen action and the action itself
        """


        if np.random.rand() < epsilon:
            possible_actions = state.possible_actions()
            move = possible_actions[np.random.randint(len(possible_actions))]
            #print("chose Random move:", move)
        else:
            move, _ = self.calc_best_move(state)

        return game.play(state, move), move

    def calc_best_move(self, state : game.State):
        max_q = -1 * float("inf")
        move = None
        for possible_action in state.possible_actions():
            cur_q = self.qnet.eval(game.create_net_input(state, possible_action))
            if max_q < cur_q:
                move = possible_action
                max_q = cur_q

        return move, max_q
