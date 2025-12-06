#! /usr/bin/env python
"""Quick converter from custom KRK transcript to PGN."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple
from datetime import date

FILES = "abcdefgh"


def position_to_fen(position: str, side_to_move: str) -> str:
    board = ["1"] * 64
    white_part, black_part = [part.strip() for part in position.split(" - ", 1)]

    for piece_pos in filter(None, white_part.split(",")):
        piece_pos = piece_pos.strip()
        if not piece_pos:
            continue
        piece, file_c, rank_c = piece_pos[0], piece_pos[1], piece_pos[2]
        file_idx = FILES.index(file_c)
        rank_idx = int(rank_c) - 1
        square = rank_idx * 8 + file_idx
        board[square] = piece

    for piece_pos in filter(None, black_part.split(",")):
        piece_pos = piece_pos.strip()
        if not piece_pos:
            continue
        piece, file_c, rank_c = piece_pos[0], piece_pos[1], piece_pos[2]
        file_idx = FILES.index(file_c)
        rank_idx = int(rank_c) - 1
        square = rank_idx * 8 + file_idx
        board[square] = piece.lower()

    rows: List[str] = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file_idx in range(8):
            square = rank * 8 + file_idx
            piece = board[square]
            if piece == "1":
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += piece
        if empty:
            row += str(empty)
        rows.append(row)

    side_char = "w" if side_to_move == "white" else "b"
    return f"{'/'.join(rows)} {side_char} - - 0 1"


def parse_move_part(part: str) -> Optional[str]:
    stripped = part.strip()
    if not stripped or stripped == "...":
        return None

    tokens = stripped.split()
    move = tokens[0]
    suffix = ""
    for token in tokens[1:]:
        lowered = token.lower()
        if lowered.startswith("mate"):
            suffix = "#"
        elif lowered.startswith("check"):
            suffix = "+"
    return move + suffix


def _parse_unnumbered_moves(
    lines: List[str],
) -> List[Tuple[int, Optional[str], Optional[str]]]:
    if not lines:
        return []

    side_to_move = "white"
    if lines[0].startswith("..."):
        side_to_move = "black"

    moves: List[Tuple[int, Optional[str], Optional[str]]] = []
    move_no = 1

    for line in lines:
        if side_to_move == "black":
            if " - " in line:
                _, black_part = [part.strip() for part in line.split(" - ", 1)]
            else:
                black_part = line.strip()
            black_move = parse_move_part(black_part)
            moves.append((move_no, None, black_move))
            move_no += 1
            side_to_move = "white"
        else:
            if " - " in line:
                white_part, black_part = [part.strip() for part in line.split(" - ", 1)]
            else:
                white_part, black_part = line.strip(), None

            white_move = parse_move_part(white_part)
            black_move = parse_move_part(black_part) if black_part is not None else None
            moves.append((move_no, white_move, black_move))
            move_no += 1
            if black_move is None:
                side_to_move = "black"

    return moves


def parse_transcript(
    text: str,
) -> Tuple[str, List[Tuple[int, Optional[str], Optional[str]]]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty transcript")

    header = lines[0]
    if header.endswith(":"):
        header = header[:-1].strip()

    move_lines = lines[1:]
    if not move_lines:
        return header, []

    move_re = re.compile(r"^(\d+)\.\s*(.*)")
    numbered = all(move_re.match(line) for line in move_lines)

    if numbered:
        moves: List[Tuple[int, Optional[str], Optional[str]]] = []
        for line in move_lines:
            match = move_re.match(line)
            assert match is not None
            move_no = int(match.group(1))
            rest = match.group(2).strip()

            white_part = rest
            black_part = ""
            if " - " in rest:
                white_part, black_part = [part.strip() for part in rest.split(" - ", 1)]

            white_move = parse_move_part(white_part)
            black_move = parse_move_part(black_part)
            moves.append((move_no, white_move, black_move))

        return header, moves

    moves = _parse_unnumbered_moves(move_lines)
    if not moves:
        raise ValueError("No moves parsed")
    return header, moves


def detect_starting_side(moves: List[Tuple[int, Optional[str], Optional[str]]]) -> str:
    if moves and moves[0][1] is None and moves[0][2] is not None:
        return "black"
    return "white"


def detect_result(moves: List[Tuple[int, Optional[str], Optional[str]]]) -> str:
    if not moves:
        return "*"
    last_no, last_white, last_black = moves[-1]
    if last_black and last_black.endswith("#"):
        return "0-1"
    if last_white and last_white.endswith("#") and last_black is None:
        return "1-0"
    if last_black and last_black.endswith("#"):
        return "0-1"
    if last_white and last_white.endswith("#"):
        return "1-0"
    return "*"


def moves_to_pgn(moves: List[Tuple[int, Optional[str], Optional[str]]]) -> str:
    parts: List[str] = []
    for move_no, white_move, black_move in moves:
        if white_move is not None and black_move is not None:
            parts.append(f"{move_no}. {white_move} {black_move}")
        elif white_move is not None:
            parts.append(f"{move_no}. {white_move}")
        elif black_move is not None:
            parts.append(f"{move_no}... {black_move}")
    return " ".join(parts)


def convert_transcript_to_pgn(text: str) -> str:
    header, moves = parse_transcript(text)
    starting_side = detect_starting_side(moves)
    fen = position_to_fen(header, starting_side)
    result = detect_result(moves)
    move_text = moves_to_pgn(moves)
    current_date = date.today().strftime("%Y.%m.%d")

    headers = [
        '[Event "?"]',
        '[Site "?"]',
        f'[Date "{current_date}"]',
        '[Round "?"]',
        '[White "?"]',
        '[Black "?"]',
        f'[Result "{result}"]',
        '[SetUp "1"]',
        f'[FEN "{fen}"]',
    ]

    if starting_side == "black":
        headers.append('[PlyCount "0"]')

    header_str = "\n".join(headers)
    return f"{header_str}\n\n{move_text} {result}".strip()


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pgn_from_text.py <transcript-file>", file=sys.stderr)
        sys.exit(1)

    transcript_path = sys.argv[1]

    try:
        with open(transcript_path, "r", encoding="utf-8") as handle:
            input_text = handle.read()
    except OSError as error:
        print(f"Error reading '{transcript_path}': {error}", file=sys.stderr)
        sys.exit(1)

    if not input_text.strip():
        print("Transcript file is empty", file=sys.stderr)
        sys.exit(1)

    try:
        pgn_output = convert_transcript_to_pgn(input_text)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    print(pgn_output)


if __name__ == "__main__":
    main()
