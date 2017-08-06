# import training
# import sys
# import select
# import user
import gui

while True:
    gui.GUI(5, 7, "connect4")

    # learn = training.Training(5, 7)
    # learn = training.Training(5, 7, "connect4")

    # learn.test(1, None)

    # learn.clear_test_results()
    #
    # for x in range(1000000):
    #     print("Training number ", x)
    #
    #     if x % 1000 == 0:
    #         learn.store("connect4")
    #         learn.test(1000)
    #
    #     learn.play_game()
    #
    #     if select.select([sys.stdin, ], [], [], 0.0)[0]:
    #         break
