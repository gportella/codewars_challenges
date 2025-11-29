#! /usr/bin/env python

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.containers import Grid


class ChessBoardApp(App[None]):
    CSS = """
    Grid {
        grid-size: 8 8;
        & .cell {
            width: 1fr;
            height: 1fr;
            content-align: center middle;
            text-style: bold; /* Make pieces stand out */
        }
        & .white-bg { background: #f0d9b5; color: black; }
        & .black-bg { background: #b58863; color: white; }
    }
    """

    # Define a starting board state (using FEN-like notation for simplicity)
    # Rooks, Knights, Bishops, Queens, Kings, Bishops, Knights, Rooks
    # Black pieces are lowercase, white pieces are uppercase
    STARTING_POS = [
        "♜♞♝♛♚♝♞♜",
        "♟♟♟♟♟♟♟♟",
        "        ",
        "        ",
        "        ",
        "        ",
        "♙♙♙♙♙♙♙♙",
        "♖♘♗♕♔♗♘♖",
    ]

    def __init__(self):
        super().__init__()
        self.grid_widgets = {}

    def compose(self) -> ComposeResult:
        with Grid():
            for i in range(8):  # Rows (0 to 7)
                for j in range(8):  # Columns (0 to 7)
                    color_class = "white-bg" if (i + j) % 2 == 0 else "black-bg"
                    # Get the initial piece character from our starting position data
                    piece_char = self.STARTING_POS[i][j]

                    widget = Static(piece_char, classes=f"cell {color_class}")
                    self.grid_widgets[(i, j)] = widget
                    yield widget

    def on_mount(self) -> None:
        # Example: Move a white pawn forward
        self.move_piece(6, 0, 5, 0)

    def move_piece(self, start_i, start_j, end_i, end_j):
        """Helper method to simulate a move."""
        start_widget = self.grid_widgets[(start_i, start_j)]
        print(f"Haha {start_widget.content}")
        piece_char = start_widget.content.strip()
        if piece_char:  # Check if there's a piece
            self.update_cell_content(start_i, start_j, "")  # Clear start cell
            self.update_cell_content(
                end_i, end_j, piece_char
            )  # Place piece in end cell

    def update_cell_content(self, i: int, j: int, content: str) -> None:
        """Helper method to update a cell's content by index."""
        if (i, j) in self.grid_widgets:
            widget = self.grid_widgets[(i, j)]
            widget.update(content)


if __name__ == "__main__":
    app = ChessBoardApp()
    app.run()
