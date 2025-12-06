import time
from pathlib import Path

import chess_engine as ce


def _play_game(fen_str: str, depth: int = 3, max_plies: int = 512):
    fen = ce.parse_fen(fen_str)
    starting_side = ce.Color.white if fen.player == "w" else ce.Color.black
    board = ce.Board(pieces=fen.pieces, side=starting_side)
    ce.init_bitboards(board)

    current_side = board.side
    start = time.perf_counter()

    for ply in range(max_plies):
        board.side = current_side

        if board.is_threefold_repetition():
            elapsed = time.perf_counter() - start
            return "Draw by threefold repetition.", "1/2-1/2", elapsed, board.his_ply

        if board.king_captured is not None:
            winner = (
                ce.Color.white
                if board.king_captured == ce.Color.black
                else ce.Color.black
            )
            result_message = (
                f"{board.king_captured.name.capitalize()} king captured. Game over."
            )
            result = "1-0" if winner == ce.Color.white else "0-1"
            elapsed = time.perf_counter() - start
            return result_message, result, elapsed, board.his_ply

        if not board.has_legal_move(current_side):
            if ce.is_in_check(board, current_side):
                winner = (
                    ce.Color.white if current_side == ce.Color.black else ce.Color.black
                )
                result_message = f"Checkmate! {winner.name.capitalize()} wins."
                result = "1-0" if winner == ce.Color.white else "0-1"
            else:
                result_message = "Stalemate."
                result = "1/2-1/2"
            elapsed = time.perf_counter() - start
            return result_message, result, elapsed, board.his_ply

        move = ce.generate_next_move(board, current_side, depth=depth)
        assert move is not None, "Engine failed to produce a move"

        board.make_move(*move)
        current_side = (
            ce.Color.black if current_side == ce.Color.white else ce.Color.white
        )

    raise AssertionError(
        f"Exceeded {max_plies} plies without reaching a terminal state for FEN: {fen_str}"
    )


def test_trouble_positions_finish_with_checkmate_quickly():
    trouble_path = Path(__file__).resolve().parent.parent / "trouble_positions.txt"
    fen_positions = [
        line.strip()
        for line in trouble_path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]

    for fen in fen_positions:
        outcome = None
        for depth in (3, 4):
            message, result, elapsed, plies = _play_game(fen, depth=depth)
            full_moves_played = max(1, (plies + 1) // 2)
            if (
                message == "Checkmate! White wins."
                and result == "1-0"
                and full_moves_played <= 16
            ):
                outcome = (depth, message, result, elapsed, plies, full_moves_played)
                break

        assert outcome is not None, (
            f"Engine failed to deliver a fast checkmate (depths 3-4) for {fen}"
        )

        depth_used, message, result, elapsed, plies, full_moves_played = outcome
        assert elapsed < 4.0, (
            f"Engine took too long ({elapsed:.2f}s) at depth {depth_used} for {fen}"
        )
