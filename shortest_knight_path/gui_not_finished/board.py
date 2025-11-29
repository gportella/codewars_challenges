#! /usr/bin/env python 

from textual.app import App, ComposeResult
from textual.widgets import Static


from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.containers import Grid
from textual.css.query import NoMatches

class ChessBoardApp(App[None]):
    CSS = """
    Grid {
        grid-size: 8 8;
        /* Optional: style the grid cells */
        & .cell {
            width: 1fr;
            height: 1fr;
            content-align: center middle;
        }
        /* Chessboard colors */
        & .white { background: #f0d9b5; color: black; }
        & .black { background: #b58863; color: white; }
    }
    """

    def __init__(self):
        super().__init__()
        # This dictionary stores a reference to every widget by its (i, j) index
        self.grid_widgets = {}

    def compose(self) -> ComposeResult:
        with Grid():
            for i in range(8):  # Rows
                for j in range(8):  # Columns
                    # Determine cell color
                    color_class = "white" if (i + j) % 2 == 0 else "black"
                    # Create the widget
                    widget = Static(f"{i},{j}", classes=f"cell {color_class}")
                    # Store a reference in our Python data structure
                    self.grid_widgets[(i, j)] = widget
                    # Yield the widget to the Textual DOM
                    yield widget

    def on_mount(self) -> None:
        # Example: Accessing a specific cell after the app has mounted
        self.update_cell_content(1, 2, "Knight")
        self.highlight_cell(7, 7)

    def update_cell_content(self, i: int, j: int, content: str) -> None:
        """Helper method to update a cell's content by index."""
        if (i, j) in self.grid_widgets:
            widget = self.grid_widgets[(i, j)]
            widget.update(content)

    def highlight_cell(self, i: int, j: int) -> None:
        """Helper method to add a 'highlight' class to a cell by index."""
        if (i, j) in self.grid_widgets:
            widget = self.grid_widgets[(i, j)]
            widget.add_class("highlight")

if __name__ == "__main__":
    app = ChessBoardApp()
    app.run()

