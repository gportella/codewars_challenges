#! /usr/bin/env python
""" Unfinished 
Thought of adding some some graphical representation to the 
knight puzzle but lost interest. Here in case I pick it up later.
"""

import tkinter as tk


BOARD_SIZE = 8
TILE_SIZE = 60
LIGHT_COLOR = "#F0D9B5"  # light brown
DARK_COLOR = "#B58863"  # dark brown
HIGHLIGHT_COLOR = "#FFD54F"


def square_name(row, col):
    # Chess notation: files a–h (columns), ranks 1–8 (rows from white's perspective)
    file_letter = chr(ord("a") + col)
    rank_number = BOARD_SIZE - row
    return f"{file_letter}{rank_number}"


def draw_connection(
    canvas, from_sq, to_sq, tile_size=60, line_color="red", line_width=3, dot_radius=4
):
    """
    Draw a line between the centers of two squares and a dot at the midpoint.

    Parameters
    ----------
    canvas : tk.Canvas
        The canvas to draw on.
    from_sq : tuple[int, int]
        (row, col) of the starting square.
    to_sq : tuple[int, int]
        (row, col) of the ending square.
    tile_size : int
        Size of a tile in pixels.
    line_color : str
        Color for the line and dot.
    line_width : int
        Width of the line.
    dot_radius : int
        Radius of the midpoint dot in pixels.

    Returns
    -------
    dict
        {"line": line_id, "dot": dot_id}
    """
    r1, c1 = from_sq
    r2, c2 = to_sq
    x1, y1 = center_of_square(r1, c1, tile_size)
    x2, y2 = center_of_square(r2, c2, tile_size)

    line_id = canvas.create_line(
        x1, y1, x2, y2, fill=line_color, width=line_width, capstyle="round"
    )

    # Midpoint
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dot_id = canvas.create_oval(
        mx - dot_radius,
        my - dot_radius,
        mx + dot_radius,
        my + dot_radius,
        fill=line_color,
        outline="",
    )

    return {"line": line_id, "dot": dot_id}


class ChessBoard(tk.Frame):
    def __init__(self, master=None, on_click=None):
        super().__init__(master)
        self.canvas = tk.Canvas(
            self, width=BOARD_SIZE * TILE_SIZE, height=BOARD_SIZE * TILE_SIZE
        )
        self.canvas.pack()
        self.tiles = {}  # (row, col) -> rect_id
        self.on_click = on_click
        self.highlighted = None
        self.draw_board()
        self.canvas.bind("<Button-1>", self.handle_click)

    def draw_board(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=color, outline=""
                )
                self.tiles[(row, col)] = rect

                # Optional: draw square labels in corner
                name = square_name(row, col)
                self.canvas.create_text(
                    x1 + 6,
                    y1 + 6,
                    text=name,
                    anchor="nw",
                    font=("Arial", 8),
                    fill="#333" if color == LIGHT_COLOR else "#EEE",
                )

    def handle_click(self, event):
        col = event.x // TILE_SIZE
        row = event.y // TILE_SIZE
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            # Toggle highlight
            self.clear_highlight()
            self.highlight_square(row, col)

            # Compute value(s) to report
            name = square_name(row, col)  # e.g., "e4"
            value = {"name": name, "row": row, "col": col}  # you can adapt this

            print(f"Clicked: {value}")  # Console output

            # Callback hook if you want to consume the value elsewhere
            if self.on_click:
                self.on_click(value)

    def highlight_square(self, row, col):
        rect_id = self.tiles[(row, col)]
        self.canvas.itemconfig(rect_id, fill=HIGHLIGHT_COLOR)
        self.highlighted = (row, col)

    def clear_highlight(self):
        if self.highlighted:
            r, c = self.highlighted
            color = LIGHT_COLOR if (r + c) % 2 == 0 else DARK_COLOR
            rect_id = self.tiles[(r, c)]
            self.canvas.itemconfig(rect_id, fill=color)
            self.highlighted = None

    def draw_connection(
        self, from_sq, to_sq, line_color="red", line_width=3, dot_radius=4
    ):
        return draw_connection(
            self.canvas, from_sq, to_sq, TILE_SIZE, line_color, line_width, dot_radius
        )


def center_of_square(row, col, tile_size):
    cx = col * tile_size + tile_size / 2
    cy = row * tile_size + tile_size / 2
    return cx, cy


def main():
    root = tk.Tk()
    root.title("Chess Board")

    def on_square_click(value):
        pass

    board = ChessBoard(root, on_click=on_square_click)
    board.pack()
    root.mainloop()


if __name__ == "__main__":
    main()
