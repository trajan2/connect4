import training
from keras.models import save_model

learn = training.Training(5, 7)

for x in range(75000):
    print(x)
    learn.play_game()


learn.ai.qnet.model.save("connect4.h5")

# state = game.State(5, 7)
#
# while state.winner is None:
#     state.show()
#
#     actions = state.possible_actions()
#     state = game.play(state, actions[0])
#
# state.show()
