import training
import select
import sys
import ai
import itertools

BATCH_SIZE = 100

learn = training.Training(5, 7, "connect4")
learn.clear_test_results()

opp_ai = ai.RandomAI()

for x in itertools.count():

    if x % 10 == 0:
        learn.store()
        learn.test(opp_ai, x*BATCH_SIZE, 100)

    print("Training batch number ", x)

    games_list = []
    for i in range(BATCH_SIZE):
        state_action_list = learn.play_game()
        games_list.append(state_action_list)

    learn.train_batch(games_list, BATCH_SIZE)

    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        break

learn.store()
learn.test_game(opp_ai, draw_graph=True)


# while True:
#     gui.GUI(5, 7, "connect4")
