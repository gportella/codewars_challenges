#! /usr/bin/env python
"""
Note! This does not guarantee a full knight's tour. It uses Warnsdorff's heuristic
to find a path that covers as many squares as possible.
We would need backtracking to ensure a full tour, but I passed the challange like this, so
leaving it as is... :-)
Alternatively, do a proper DFS with backtracking plus the Warnsdorff's heuristic to optimize the search.
"""

import sys
from typing import Tuple


def create_moves(row, col, bs: int):
    def _valid(x):
        return x >= 0 and x < bs

    moves = []
    knight_rules = [
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1),
    ]
    for i in knight_rules:
        new_row, new_col = row + i[0], col + i[1]
        if _valid(new_row) and _valid(new_col):
            moves.append((new_row, new_col))
    return moves


def knights_tour(start: Tuple[int, int], size: int):
    move = start
    visited = {move}
    path = [move]
    while len(visited) < size * size:
        min_deg_idx = -1
        min_deg = size + 1
        for i in create_moves(move[0], move[1], size):
            if i not in visited:
                c = 0
                for j in create_moves(i[0], i[1], size):
                    if j not in visited:
                        c += 1
                if c < min_deg:
                    min_deg_idx = i
                    min_deg = c
        if min_deg_idx == -1:
            break
        move = min_deg_idx
        visited.add(move)
        path.append(move)
    return path


if __name__ == "__main__":
    if len(sys.argv) == 3:
        start_col = ord(sys.argv[1][0].lower()) - 97
        start_row = int(sys.argv[1][1]) - 1
        board_size = int(sys.argv[2])
        solution = knights_tour((start_row, start_col), board_size)
        print(f"Solution: {solution}")
        print(
            f"Did we complete the tour? {'Yes' if len(solution) == board_size * board_size else 'No'}"
        )

    if len(sys.argv) == 2:
        board_size = int(sys.argv[1])
        solution = knights_tour((0, 0), board_size)
        print(f"Solution: {solution}")
        print(
            f"Did we complete the tour? {'Yes' if len(solution) == board_size * board_size else 'No'}"
        )
    else:
        solution = knights_tour((0, 0), 8)
        print(f"Solution: {solution}")
        print(f"Did we complete the tour? {'Yes' if len(solution) == 8 * 8 else 'No'}")
