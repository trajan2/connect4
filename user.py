import game
import sys
import copy


class User:
    def __init__(self):
        pass

    def perform_action(self, state: game.State):
        print("Your turn. Choose wisely.")
        state.show()

        for line in sys.stdin:
            try:
                number = int(line)

                if len(list(filter((lambda x: x == number), map((lambda x: x.move), state.possible_actions())))) != 1:
                    raise ValueError("Invalid move!")

                action = game.Action(number)

                return game.play(state, action)

            except ValueError:
                print("I am afraid ", line, " is not a valid option. Please enter a valid column number.")
