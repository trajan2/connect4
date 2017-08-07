# import game
# import graphviz as gv
# import random
#
# cur_state = game.State(5, 7)
#
# g1 = gv.Digraph(format='png')
# g1.node(cur_state.id, label=cur_state.to_graphviz(), style="filled", fillcolor="gray")
#
# while cur_state.winner is None:
#     actions = cur_state.possible_actions()
#     random_action = actions[random.randint(0, len(actions) - 1)]
#
#     for action in actions:
#         tmp_state = game.play(cur_state, action)
#
#         if random_action.move == action.move:
#             g1.node(tmp_state.id, label=tmp_state.to_graphviz(), style="filled", fillcolor="gray")
#             g1.edge(cur_state.id, tmp_state.id, label=action.to_graphviz(), penwidth="6")
#             new_state = tmp_state
#         else:
#             g1.node(tmp_state.id, label=tmp_state.to_graphviz())
#             g1.edge(cur_state.id, tmp_state.id, label=action.to_graphviz())
#
#
#     # new_state = game.play(cur_state, random_action)
#
#     #g1.node(new_state.id, label=new_state.to_graphviz())
#     #g1.edge(cur_state.id, new_state.id, label=random_action.to_graphviz(), penwidth="3")
#
#     cur_state = new_state
#
#
# g1.render(filename="connect4")  # import training


# import sys
# import select
# import user
# import gui
import training

# learn = training.Training(5, 7)
learn = training.Training(5, 7, "connect4")
learn.create_graph()

# while True:
#     gui.GUI(5, 7, "connect4")

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
