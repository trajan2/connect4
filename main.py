import rlframework
import network
import ai
from collections import Counter
import numpy as np


TRAINGAMES_NO = 10
TESTGAMES_NO = 100

ai_opp = ai.RepeatAI(-1)
ai_opp = ai.NetAI(network.Network(model_path="model.h5"),-1)
ai_opp = ai.RepeatVicinityAI(-1)

learn = rlframework.RLFramework(
    model_path="model.h5",
    opponent_ai=ai_opp
)
print("Training on", TRAINGAMES_NO, "games...")

for i in range(TRAINGAMES_NO):
    if i % 100 == 0:
        print("Training on Game #", i, "-", i+100)
    state_list, move_list = learn.playGame()
    losses = learn.trainGame(state_list, move_list)
    #print("game:", i, "losses:", losses)
learn.saveModel("model.h5")

ends = []
print("Testing  on", TESTGAMES_NO, "games...")
for _ in range(TESTGAMES_NO):
    ends.append(learn.testGame(
        qnet_starts=np.random.randint(0,2)))
print(Counter(ends))
learn.testGame(verbose=True, enemy="human", qnet_starts=False)
