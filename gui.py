from tkinter import *
import random
import game
import ai


class GUI:
    def __init__(self, height=4, width=5, load_file: str = None, max_height_px=600, max_width_px=800, border_px=50):
        self.master = Tk()
        self.master.wm_title("Vier gewinnt")
        self.master.resizable(False, False)

        self.height = height
        self.width = width
        self.border_px = border_px

        self.cell_size_px = min(
            [int((max_height_px - 2 * border_px) / (height + 1)), int((max_width_px - 2 * border_px) / width)])
        self.padding_px = self.cell_size_px / 8
        self.outline_px = self.cell_size_px / 10
        self.board_width_px = self.cell_size_px * width
        self.board_height_px = self.cell_size_px * (height + 1)
        self.width_px = self.board_width_px + 2 * self.border_px
        self.height_px = self.board_height_px + 2 * self.border_px

        self.w = Canvas(self.master, width=self.width_px, height=self.height_px)
        self.w.pack()
        # self.w.configure(background="floral white")

        x1 = self.border_px
        y1 = self.border_px + self.cell_size_px
        x2 = self.border_px + self.board_width_px
        y2 = self.border_px + self.board_height_px
        self.w.create_rectangle(x1, y1, x2, y2, fill="royal blue")

        self.buttons = []
        for x in range(self.width):
            b = Button(self.master, text=str(x), command=lambda col=x: self.play_column(col))
            b.place(x=self.border_px + x * self.cell_size_px, y=self.border_px, width=self.cell_size_px,
                    height=self.cell_size_px)
            self.buttons.append(b)

        # create a new game
        self.state = game.State(self.height, self.width)
        self.ai = ai.AI(game.get_net_input_dim(height, width), load_file)

        if random.randint(0, 1) == 0:
            self.state, _ = self.ai.perform_best_move(self.state)

        self.draw_state()

        mainloop()

    def play_column(self, col):
        self.state = game.play(self.state, game.Action(col))
        self.state, _ = self.ai.perform_best_move(self.state)

        possible_columns = list(map(lambda a: a.move, self.state.possible_actions()))
        for i, button in enumerate(self.buttons):
            if i not in possible_columns:
                button['state'] = DISABLED

        self.draw_state()

    def draw_state(self):
        for pos_y in range(self.state.height):
            for pos_x in range(self.state.width):

                mapping = {
                    0: ("gainsboro", None),
                    1: ("gold", "gold2"),
                    -1: ("red", "red2"),
                }

                color, ring_color = mapping[self.state.field[pos_y, pos_x]]

                y1 = self.border_px + (pos_y + 1) * self.cell_size_px + self.padding_px
                x1 = self.border_px + pos_x * self.cell_size_px + self.padding_px
                y2 = self.border_px + (pos_y + 2) * self.cell_size_px - self.padding_px
                x2 = self.border_px + (pos_x + 1) * self.cell_size_px - self.padding_px
                self.w.create_oval(x1, y1, x2, y2, fill=color, width=0)

                if ring_color is not None:
                    out = self.outline_px / 2
                    self.w.create_oval(x1 + out, y1 + out, x2 - out, y2 - out, outline=ring_color,
                                       width=self.outline_px)
