from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Dict, List, NewType, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from chess_engine import Board

U64 = NewType("U64", int)
U64_MASK = (1 << 64) - 1  # 0xFFFFFFFFFFFFFFFF
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
NOT_FILE_A = U64(~FILE_A & U64_MASK)
NOT_FILE_H = U64(~FILE_H & U64_MASK)

FILES = "abcdefgh"


class Pcs(Enum):
    empty = 0
    P = 1
    N = 2
    B = 3
    R = 4
    Q = 5
    K = 6
    p = 7
    n = 8
    b = 9
    r = 10
    q = 11
    k = 12


PIECES_REP: Dict[Pcs, str] = {
    Pcs.r: "♜",
    Pcs.n: "♞",
    Pcs.b: "♝",
    Pcs.q: "♛",
    Pcs.k: "♚",
    Pcs.p: "♟",
    Pcs.P: "♙",
    Pcs.R: "♖",
    Pcs.N: "♘",
    Pcs.B: "♗",
    Pcs.Q: "♕",
    Pcs.K: "♔",
}


class Color(Enum):
    white = 0
    black = 1
    both = 2


class CastlingRights(Enum):
    No = 0
    WKCA = 1
    WQCA = 2
    BKCA = 4
    BQCA = 8


WHITE_PIECES = frozenset({Pcs.P, Pcs.N, Pcs.B, Pcs.R, Pcs.Q, Pcs.K})
BLACK_PIECES = frozenset({Pcs.p, Pcs.n, Pcs.b, Pcs.r, Pcs.q, Pcs.k})


def to_u64(value: int) -> U64:
    return U64(value & U64_MASK)


def format_binary_bytes(n: int) -> str:
    """Format an integer as 8 byte-sized binary blocks."""
    binary_str = format(n, "064b")
    chunks = [binary_str[i : i + 8] for i in range(0, 64, 8)]
    return "_".join(chunks)


def print_attack_mask(
    mask: U64,
    pieces: Sequence | None = None,
    piece_symbols: Mapping | None = None,
) -> None:
    bb = int(mask)
    symbols = piece_symbols or PIECES_REP
    for rank in range(8, 0, -1):
        rank_idx = rank - 1
        line = f"{rank} |"
        for file_idx, _ in enumerate(FILES):
            sq64 = rank_idx * 8 + file_idx
            marker = " X " if (bb >> sq64) & 1 else " . "
            if pieces is not None:
                sq120 = 21 + file_idx + 10 * rank_idx
                piece = pieces[sq120]
                if piece != Pcs.empty:
                    symbol = symbols.get(piece)
                    if symbol is not None:
                        marker = f" {symbol} "
            line += marker
        print(line)
    print("--------------------------")
    print("    a  b  c  d  e  f  g  h")


def build_king_attack_patterns() -> List[U64]:
    attacks: List[U64] = []
    for sq in range(64):
        bit = 1 << sq
        mask = 0
        mask |= (bit << 8) & U64_MASK
        mask |= (bit >> 8) & U64_MASK
        if bit & NOT_FILE_H:
            mask |= (bit << 9) & U64_MASK
            mask |= (bit >> 7) & U64_MASK
            mask |= (bit << 1) & U64_MASK
        if bit & NOT_FILE_A:
            mask |= (bit << 7) & U64_MASK
            mask |= (bit >> 9) & U64_MASK
            mask |= (bit >> 1) & U64_MASK
        attacks.append(U64(mask & U64_MASK))
    return attacks


def generate_rook_rays() -> Dict[int, Dict[str, U64]]:
    rays: Dict[int, Dict[str, U64]] = {}
    for sq in range(64):
        north = south = east = west = 0
        for target in range(sq + 8, 64, 8):
            north |= 1 << target
        for target in range(sq - 8, -1, -8):
            south |= 1 << target
        file = sq % 8
        for step in range(file + 1, 8):
            target = sq + (step - file)
            east |= 1 << target
        for step in range(file - 1, -1, -1):
            target = sq - (file - step)
            west |= 1 << target
        rays[sq] = {
            "north": to_u64(north),
            "south": to_u64(south),
            "east": to_u64(east),
            "west": to_u64(west),
        }
    return rays


def generate_king_attack_bm(
    board: "Board",
    attack_patterns: Sequence[U64],
    king_sq64: int,
    side: Color,
    *,
    debug: bool = False,
) -> U64:
    """Return king attack bitboard pruned by friendly occupancy."""

    attacks = attack_patterns[king_sq64]
    friendly_occ = board.occupied[side]
    legal_attacks = to_u64(attacks & ~friendly_occ)

    if debug:
        print_attack_mask(legal_attacks, pieces=board.pieces)

    return legal_attacks


if __name__ == "__main__":
    king_masks = build_king_attack_patterns()
    for sq, attack_mask in enumerate(king_masks):
        print(f"King attacks for {sq} -> {format_binary_bytes(int(attack_mask))}")
        print_attack_mask(attack_mask)
        print()
