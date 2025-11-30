import pytest

import chess_engine as ce


@pytest.fixture
def midboard_king_board():
    """Board with the white king placed on e4 for move-generation tests."""
    fen = "8/8/8/8/4K3/8/8/8 w - - 0 1"
    parsed = ce.parse_fen(fen)
    board = ce.Board(pieces=parsed.pieces)
    # zero any other state fields we care about in tests
    board.side = ce.Color.white
    board.en_passant = -1
    board.pawns = {
        ce.Color.white: ce.U64(0),
        ce.Color.black: ce.U64(0),
        ce.Color.both: ce.U64(0),
    }
    return board
