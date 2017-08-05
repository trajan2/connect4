import rlframework
from collections import Counter

learn = rlframework.RLFramework(model_path="model.h5")
for i in range(10):
    state_list, move_list = learn.playGame()
    losses = learn.trainGame(state_list, move_list)
    #print("game:", i, "losses:", losses)
learn.saveModel("model.h5")
ends = []
for i in range(100):
    ends.append(learn.testGame())
    #print("test:", i, "end:", ends[-1])
print(Counter(ends))
#learn.testGame(verbose=True)

