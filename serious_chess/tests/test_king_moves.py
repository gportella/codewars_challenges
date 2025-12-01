import pytest
import chess_engine as ce


@pytest.fixture
def empty_board():
    board = ce.Board()
    board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
    return board


class TestKingMoveGeneration:
    """Reference expectations for eventual king move generation."""

    @pytest.mark.xfail(reason="King move generation not implemented yet", strict=False)
    def test_king_moves_from_center_square(self, empty_board):
        center_square = ce.Sqr["e4"]
        expected_targets = {
            ce.Sqr["d5"],
            ce.Sqr["e5"],
            ce.Sqr["f5"],
            ce.Sqr["d4"],
            ce.Sqr["f4"],
            ce.Sqr["d3"],
            ce.Sqr["e3"],
            ce.Sqr["f3"],
        }
        king_moves = getattr(ce, "generate_king_moves")
        moves = king_moves(empty_board, center_square)
        assert set(moves) == expected_targets

    @pytest.mark.xfail(reason="King move generation not implemented yet", strict=False)
    def test_king_moves_from_corner_square(self, empty_board):
        corner_square = ce.Sqr["a1"]
        expected_targets = {ce.Sqr["a2"], ce.Sqr["b1"], ce.Sqr["b2"]}
        king_moves = getattr(ce, "generate_king_moves")
        moves = king_moves(empty_board, corner_square)
        assert set(moves) == expected_targets
