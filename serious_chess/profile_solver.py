#!/usr/bin/env python
"""Profile the KRK solver without the UCI Stockfish overhead."""

from __future__ import annotations

import math
import cProfile
from typing import Iterable, Sequence

from fastq_krk_final_submission  import Color, WhitePlayer, generate_next_move

# Same sample suite we use in play_against_stock.py, with their identifiers for reference.
EXAMPLE_CASES: tuple[tuple[str, str], ...] = (
    ("game_1", "Ka4,Rb5 - Ka8"),
    ("game_2", "Ke3,Rf4 - Kg5"),
    ("game_3", "Kh3,Rf7 - Kh8"),
    ("game_4", "Kf3,Re4 - Kf5"),
    ("game_5", "Kd3,Ra5 - Kf4"),
    ("game_6", "Kc1,Rg7 - Kh5"),
    ("game_7", "Kh1,Ra3 - Kd4"),
    ("game_8", "Kd3,Ra5 - Kf4"),
    ("game_9", "Kf3,Re4 - Kf5"),
    ("game_10", "Ka1,Rb1 - Ka6"),
)

POSITIONS: tuple[str, ...] = tuple(pos for _, pos in EXAMPLE_CASES)


def play_out(position: str, max_plies: int = 32) -> None:
    """Play out a KRK position in self-play until termination or max plies."""
    board = WhitePlayer(position).board
    side = Color.white
    for _ in range(max_plies):
        move = generate_next_move(board, side)
        if move is None:
            break
        board.make_move(*move)
        if board.is_terminal():
            break
        side = Color.black if side == Color.white else Color.white


def run_batch(
    positions: Iterable[str], total_games: int = 500, max_plies: int = 32
) -> None:
    pool: Sequence[str] = tuple(positions)
    if not pool:
        return
    per_position = math.ceil(total_games / len(pool))
    for _ in range(per_position):
        for pos in pool:
            play_out(pos, max_plies=max_plies)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    run_batch(POSITIONS, total_games=500, max_plies=32)
    profiler.disable()
    profiler.print_stats("cumtime")


if __name__ == "__main__":
    main()
