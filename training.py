import game
import ai


class Training:
    def __init__(self, height=4, width=5, load_file=None, gamma=0.96, exploration=0.2):
        self.gamma = gamma
        self.exploration = exploration
        self.height = height
        self.width = width
        self.ai = ai.AI(game.get_net_input_dim(height, width), load_file)

    def play_game(self):
        state_action_list = []

        cur_state = game.State(self.height, self.width)

        while cur_state.winner is None:
            new_state, action = self.ai.perform_best_move(cur_state, self.exploration)
            state_action_list.append((cur_state, action))
            cur_state = new_state

        # invariant: each state has infinite following states (even if game is already over)
        # invariant: each state is seen by the current player (every player sets a '1' token)

        # add last state to list for negative rewards
        new_state, action = self.ai.perform_best_move(cur_state, self.exploration)
        state_action_list.append((cur_state, action))

        # first try: did not work very well on 75.000 games
        for state, action in reversed(state_action_list):
            reward = game.get_reward(state, action)
            next_state = game.play(state, action)  # perform the stored action
            next_state, _ = self.ai.perform_best_move(next_state)  # perform the best action for the opponent

            _, q_value = self.ai.calc_best_move(next_state)  # calc q value of the best possible move

            # state.show()
            # for possible_action in state.possible_actions():
            #    print("Options: ", possible_action.move, " with q_value ", self.ai.qnet.eval(game.create_net_input(next_state, possible_action)))
            # print("q_value of best move: ", q_value)

            net_input = game.create_net_input(state, action)
            target = reward + self.gamma * q_value
            # print("Reward: ", reward, " Target: ", target)

            self.ai.qnet.train(net_input, target)  # actual training


            # second try: rewards are directly known because of total sequence
            # target_one = 0.0
            # target_two = 0.0
            # for state, action in reversed(state_action_list):
            #     if state.round % 2 == 0:
            #         target_one = game.get_reward(state, action) + self.gamma * target_one
            #         target = target_one
            #     else:
            #         target_two = game.get_reward(state, action) + self.gamma * target_two
            #         target = target_two
            #
            #     for possible_action in state.possible_actions():
            #         print("Options: ", possible_action.move, " with q_value ", self.ai.qnet.eval(game.create_net_input(state, possible_action)))
            #     print("--------------")
            #
            #     net_input = game.create_net_input(state, action)
            #     self.ai.qnet.train(net_input, np.array([target]))  # actual training

    def store(self, name):
        self.ai.store(name)
