import training
import select
import sys
import ai

#BATCH_SIZE = 100

learn = training.Training(5, 7, "connect4")

# learn.create_graph()
#
# while True:
#     gui.GUI(5, 7, "connect4")

opp_ai = ai.RandomAI()
learn.clear_test_results()

for x in range(1):
    print("Training number ", x)

    if x % 10 == 0:
        learn.store("connect4")
        #learn.test(opp_ai, 10)
        #learn.create_graph()
        #learn.test(learn.train_ai, 100)

    state_action_list = learn.play_game()
    target_list = learn.calc_targets(state_action_list)


    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        break
learn.test(opp_ai, 1, draw_graph=True)
