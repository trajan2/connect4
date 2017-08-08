import game
import ai
import numpy as np
import graphviz as gv
from typing import List

class Training:
    def __init__(self, height=4, width=5, load_file=None, gamma=0.96, exploration=0.2):
        self.gamma = gamma
        self.exploration = exploration
        self.height = height
        self.width = width
        self.train_ai = ai.NetAI(game.get_net_input_dim(height, width), load_file)

    def play_game(self, opponent: ai.AI = None):
        ais = {
            0: self.train_ai,
            1: opponent if opponent is not None else self.train_ai
        }
        state_action_list = []

        cur_state = game.State(self.height, self.width)
        cur_ai_id = np.random.randint(0,2)
        last = False
        while cur_state.winner is None or last:
            action = ais[cur_ai_id].next_exploring_move(cur_state, self.exploration)
            state_action_list.append((cur_state, action))
            cur_state = game.play(cur_state, action)
            last = cur_state.winner is None and not last
            cur_ai_id = 1 - cur_ai_id

        # invariant: each state has infinite following states (even if game is already over)
        # invariant: each state is seen by the current player (every player sets a '1' token)

        return state_action_list

    def train_game_deprecated(self, state_action_list):

        # # first try: did not work very well on 75.000 games
        # for state, action in reversed(state_action_list):
        #     reward = game.get_reward(state, action)
        #     next_state = game.play(state, action)  # perform the stored action
        #     next_state, _ = self.ai.perform_best_move(next_state)  # perform the best action for the opponent
        #
        #     _, q_value = self.ai.calc_best_move(next_state)  # calc q value of the best possible move
        #
        #     # state.show()
        #     # for possible_action in state.possible_actions():
        #     #    print("Options: ", possible_action.move, " with q_value ",
        # self.ai.qnet.eval(game.create_net_input(next_state, possible_action)))
        #     # print("q_value of best move: ", q_value)
        #
        #     net_input = game.create_net_input(state, action)
        #     target = reward + self.gamma * q_value
        #     # print("Reward: ", reward, " Target: ", target)
        #
        #     self.ai.qnet.train(net_input, target)  # actual training
        pass

    def calc_targets(self, state_action_list):
        # second try: rewards are directly known because of total sequence
        target_one = 0.0
        target_two = 0.0
        target_list = []
        for state, action in reversed(state_action_list):
            if state.round % 2 == 0:
                target_one = game.get_reward(state, action) + self.gamma * target_one
                target = target_one
            else:
                target_two = game.get_reward(state, action) + self.gamma * target_two
                target = target_two
            target_list.append(target)

    # def train_batch(self, state_action_list, target_list):
    #
    #     net_input = game.create_net_input(state, action)
    #     self.train_ai.qnet.train(net_input, np.array([target]))  # actual training

    def clear_test_results(self, result_file="results"):
        f = open(result_file + ".csv", 'w')
        f.write("Number of games, Number of wins, Number of defeats, Number of ties\n")
        f.close()

    def test(self, opponent: ai.AI, num_test_games: int = 2000, result_file="results", verbose=False):

        results = {
            -1: 0,  # player 0 wins
            0: 0,  # draw
            1: 0  # player 1 wins
        }

        for x in range(num_test_games):
            print("Test number", x)
            abs_winner = self.test_game(opponent, verbose)
            results[abs_winner] += 1

        print("Results", results)

        if result_file is not None:
            text = str(num_test_games) + ", " + str(results[1]) + ", " + str(results[-1]) + ", "+str(results[0]) + "\n"
            f = open(result_file + ".csv", 'a')
            f.write(text)
            f.close()

    def test_game(self, opponent: ai.AI, verbose=False, draw_graph=False):
        assert opponent is not None
        ais = {
            "t": self.train_ai,
            "o": opponent
        }
        cur_state = game.State(self.height, self.width)
        beginner = np.random.choice(("t", "o"))
        cur_player = beginner

        if draw_graph:
            g_all = gv.Digraph(format='svg')
            g_all.graph_attr.update(rankdir="TB")
            draw_graph_layer(g_all, cur_state, is_top_node=True)

        while cur_state.winner is None:
            # cur_state.show()
            print_cond(ais[cur_player].name, "played.", cond=verbose)
            action = ais[cur_player].next_move(cur_state)
            new_state = game.play(cur_state, action)

            if draw_graph:
                draw_graph_layer(g_all, cur_state, action, new_state, ais[cur_player].last_ratings)
            cur_state = new_state
            cur_player = "t" if cur_player == "o" else "o"
        abs_winner = cur_state.winner if cur_player == "o" else cur_state.winner * -1
        if draw_graph:
            g_all.render(filename="connect4")

        return abs_winner

    def store(self, name):
        self.train_ai.store(name)


def draw_graph_layer(g_all: gv.Digraph, cur_state: game.State, action: game.Action = None,
                     new_state: game.State = None, last_ratings = None, is_top_node=False, ):

    with g_all.subgraph(name='cluster_' + str(cur_state.round + (0 if is_top_node else 1))) as c:
        c.attr(style='filled')
        c.attr(color='lightgrey')
        c.attr(margin='10px')
        c.attr(label='Round ' + str(cur_state.round + (0 if is_top_node else 1)))
        if is_top_node:
            c.node(cur_state.id, label=cur_state.to_graphviz(), style="filled", fillcolor="limegreen",
                   shape="rectangle", margin="0.5")
        else:
            for (possible_action, q_value) in last_ratings: # ais[cur_player].last_ratings:
                action_label = possible_action.to_graphviz() + "\n" + "Q-Value: " + str(round(float(q_value), 3))
                if action.move == possible_action.move:
                    c.node(new_state.id, label=new_state.to_graphviz(), style="filled",
                           fillcolor="limegreen",
                           shape="rectangle", margin="0.5")
                    g_all.edge(cur_state.id, new_state.id, label=action_label, penwidth="6", weight="0.9")
                else:
                    tmp_state = game.play(cur_state, possible_action)
                    c.node(tmp_state.id, label=tmp_state.to_graphviz(), shape="rectangle", margin="0.5")
                    g_all.edge(cur_state.id, tmp_state.id, label=action_label)


def print_cond(*args, cond=False):
    if cond:
        print(*args)