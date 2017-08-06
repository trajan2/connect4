import rlframework
import network
import ai
from collections import Counter
import numpy as np


TRAINGAMES_NO = 10
TESTGAMES_NO = 10

ai_opp = ai.RepeatAI(-1)
ai_opp = ai.NetAI(network.Network(model_path="model.h5"),-1)
#ai_opp = ai.RepeatVicinityAI(-1)

learn = rlframework.RLFramework(
    model_path="model.h5",
    opponent_ai=ai_opp
)
print("Training on", TRAINGAMES_NO, "games...")

for i in range(TRAINGAMES_NO):
    if i % 100 == 0:
        print("Training on Game #", i, "-", i+100)
    state_list, move_list = learn.play_game()
    losses = learn.trainGame(state_list, move_list)
learn.save_model("model.h5")

ends = []
print("Testing on", TESTGAMES_NO, "games...")
for _ in range(TESTGAMES_NO):
    player1_starts = np.random.choice([True, False])
    print("player1 starts", player1_starts)
    ends.append(learn.test_game(
        player1_starts=player1_starts,
        enemy="opp",
        verbose=True
    ))
print(Counter(ends))
learn.test_game(verbose=True, enemy="human", player1_starts=False)
