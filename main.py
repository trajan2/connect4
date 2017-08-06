import training
import sys
import select

learn = training.Training(5, 7)
# learn = training.Training(5, 7, "connect4")

for x in range(1000000):
    print(x)
    learn.play_game()

    if select.select([sys.stdin, ], [], [], 0.0)[0]:
        break

learn.store("connect4")
