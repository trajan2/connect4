import training
from game import Connect4
import numpy as np

# state = np.zeros((4,5))
# res = Connect4.play_static(4, field=state, color=1)


learn = training.Training()
for i in range(1000):
    losses = learn.trainGame()
    print("game:", i, "losses:", losses)
print("Done")