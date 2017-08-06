import training
import sys
import select

learn = training.Training(5, 7)
# learn = training.Training(5, 7, "connect4")


for x in range(1000000):
    print("Training number ", x)
    learn.play_game()

    if x % 1000 == 0:
        learn.store("connect4")

    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        break

learn.store("connect4")
learn.test()
