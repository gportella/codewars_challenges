#! /usr/bin/env python
from codewars_submissin.trimmed import play_white_vs_stockfish

positions = [
    ("game_failed_1", "Ka4,Rb5 - Ka8"),
    ("game_failed_2", "Ke3,Rf4 - Kg5"),
    ("game_failed_3", "Kh3,Rf7 - Kh8"),
    ("game_failed_4", "Kf3,Re4 - Kf5"),
    ("game_failed_5", "Kd3,Ra5 - Kf4"),
    ("game_failed_6", "Kc1,Rg7 - Kh5"),
    ("game_failed_7", "Kh1,Ra3 - Kd4"),
    ("game_failed_8", "Kd3,Ra5 - Kf4"),
    ("game_failed_9", "Kf3,Re4 - Kf5"),
    ("game_failed_10", "Ka1,Rb1 - Ka6"),
]

lines = []
for name, pos in positions:
    try:
        outcome = play_white_vs_stockfish(
            pos,
            stockfish_depth=19,
            stockfish_movetime=None,
            max_plies=32,
            verbose=False,
        )
        plies = outcome["plies"]
        moves = (plies + 1) // 2
        lines.append(f"{name}: plies={plies} moves={moves} king_captured={outcome['black_in_check']}")
    except Exception as exc:
        lines.append(f"{name}: ERROR {exc}")

print("\n".join(lines))
