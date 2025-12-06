import chess_engine as ce

from types_and_masks import generate_k_attack_bm, generate_rook_attack_bm, iter_bits


class TestKingMoveGeneration:
    """Reference expectations for eventual king move generation."""

    def test_king_moves_from_center_square(self):
        board = ce.Board()
        board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
        center_square = ce.Sqr["e4"]
        board.pieces[center_square] = ce.Pcs.K
        ce.init_bitboards(board)

        mask = generate_k_attack_bm(board, center_square, ce.Color.white)
        moves = {ce.Sqr(sq64) for sq64 in iter_bits(int(mask))}

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
        assert moves == expected_targets

    def test_king_moves_from_corner_square(self):
        board = ce.Board()
        board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
        corner_square = ce.Sqr["a1"]
        board.pieces[corner_square] = ce.Pcs.K
        ce.init_bitboards(board)

        mask = generate_k_attack_bm(board, corner_square, ce.Color.white)
        moves = {ce.Sqr(sq64) for sq64 in iter_bits(int(mask))}

        expected_targets = {ce.Sqr["a2"], ce.Sqr["b1"], ce.Sqr["b2"]}
        assert moves == expected_targets


class TestRookMoveGeneration:
    def test_rook_moves_on_open_file(self):
        board = ce.Board()
        board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
        rook_square = ce.Sqr["a1"]
        board.pieces[rook_square] = ce.Pcs.R
        ce.init_bitboards(board)

        mask = generate_rook_attack_bm(board, rook_square, ce.Color.white)
        moves = {ce.Sqr(sq64) for sq64 in iter_bits(int(mask))}

        expected_targets = {
            ce.Sqr["a2"],
            ce.Sqr["a3"],
            ce.Sqr["a4"],
            ce.Sqr["a5"],
            ce.Sqr["a6"],
            ce.Sqr["a7"],
            ce.Sqr["a8"],
            ce.Sqr["b1"],
            ce.Sqr["c1"],
            ce.Sqr["d1"],
            ce.Sqr["e1"],
            ce.Sqr["f1"],
            ce.Sqr["g1"],
            ce.Sqr["h1"],
        }
        assert moves == expected_targets


class TestCheckDetection:
    def test_white_rook_checking_black_king(self):
        board = ce.Board()
        board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
        board.pieces[ce.Sqr["a1"]] = ce.Pcs.R
        board.pieces[ce.Sqr["a8"]] = ce.Pcs.k
        ce.init_bitboards(board)

        assert ce.is_in_check(board, ce.Color.black)

    def test_blocked_rook_not_checking(self):
        board = ce.Board()
        board.pieces = [ce.Pcs.empty] * ce.BRD_SQ_NUM
        board.pieces[ce.Sqr["a1"]] = ce.Pcs.R
        board.pieces[ce.Sqr["a4"]] = ce.Pcs.P
        board.pieces[ce.Sqr["a8"]] = ce.Pcs.k
        ce.init_bitboards(board)

        assert not ce.is_in_check(board, ce.Color.black)
