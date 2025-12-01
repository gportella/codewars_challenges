#! /usr/bin/env python
"""
Learning. I'm going to base it on Programming a chess engine in C by Bluefever software
Doing it in python to pass some Katas in python, as they don't have a C/C++ version
Over the top, but seemed fun to do.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List

from types_and_masks import (
    BLACK_PIECES,
    PIECES_REP,
    Pcs,
    WHITE_PIECES,
    CastlingRights,
    Color,
    U64,
    chebyshev_dist,
    generate_k_attack_bm,
    generate_king_moves,
    generate_rook_attack_bm,
    generate_rook_rays,
    is_in_check,
    iter_bits,
    print_attack_mask,
    to_u64,
)


BRD_SQ_NUM = 120
MAXMOVES_GAME = 2048
FILES = "abcdefgh"
RANKS = range(1, 9)


CANONICAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
wE4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
RUY_LOPEZ_FEN = "r1bqkb1r/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
ROOK_ENDGAME = "8/8/8/8/5R2/3k4/5K2/8 b - - 0 1"
ROOK_ENDGAME_CHECK = "8/8/8/8/3k1R2/8/5K2/8 b - - 0 1"


def fr2sq(f, r):
    return 21 + f + 10 * (r - 1)


attack_120_directions = {
    Pcs.P: [11, 9],
    Pcs.p: [-11, -9],
    Pcs.N: [21, 19, 12, 8, -21, -19, -12, -8],
    Pcs.n: [21, 19, 12, 8, -21, -19, -12, -8],
    Pcs.B: [11, 9, -11, -9],
    Pcs.b: [11, 9, -11, -9],
    Pcs.R: [10, -10, 1, -1],
    Pcs.r: [10, -10, 1, -1],
    Pcs.K: [11, 10, 9, 1, -1, -9, -10, -11],
    Pcs.k: [11, 10, 9, 1, -1, -9, -10, -11],
    Pcs.Q: [11, 10, 9, 1, -1, -9, -10, -11],
    Pcs.q: [11, 10, 9, 1, -1, -9, -10, -11],
}


_file_vals = {f"file_{file}": idx for idx, file in enumerate(FILES)}
_file_vals["file_none"] = len(FILES)
File = IntEnum("File", _file_vals)

_rank_vals = {f"rank_{idx + 1}": idx for idx in range(len(RANKS))}
_rank_vals["rank_none"] = len(RANKS)
Rank = IntEnum("Rank", _rank_vals)

_square_vals = {
    f"{file}{rank}": fr2sq(file_idx, rank)
    for rank in RANKS
    for file_idx, file in enumerate(FILES)
}
Sqr = IntEnum("Square", _square_vals)


SQ120_TO_SQ64 = [-1] * BRD_SQ_NUM
SQ64_TO_SQ120 = [-1] * 64
for rank in RANKS:
    for file_idx, file in enumerate(FILES):
        sq120 = fr2sq(file_idx, rank)
        sq64 = (rank - 1) * 8 + file_idx
        SQ120_TO_SQ64[sq120] = sq64
        SQ64_TO_SQ120[sq64] = sq120


def _empty_piece_bitboards() -> Dict[Color, Dict[Pcs, U64]]:
    return {
        Color.white: {piece: U64(0) for piece in WHITE_PIECES},
        Color.black: {piece: U64(0) for piece in BLACK_PIECES},
    }


@dataclass
class Undo:
    move: int
    calstlng_rights: CastlingRights = CastlingRights.No
    en_passant: int = -1
    position_key: U64 = U64(0)


@dataclass
class Board:
    pieces: List[Pcs] = field(default_factory=lambda: [Pcs.empty] * BRD_SQ_NUM)
    side: Color = Color.white
    en_passant: int = -1
    fifty_move: int = 0
    ply: int = 0
    his_ply: int = 0
    castling_rights: CastlingRights = CastlingRights.No
    position_key: U64 = U64(0)
    pieces_bb: Dict[Color, Dict[Pcs, U64]] = field(
        default_factory=_empty_piece_bitboards
    )
    kings_pos: List[int] = field(default_factory=lambda: [-1, -1])
    rooks_pos: list[int] = field(
        default_factory=lambda: [-1] * 4
    )  # assume no promotions
    occupied: dict[Color, U64] = field(
        default_factory=lambda: {
            Color.white: U64(0),
            Color.black: U64(0),
            Color.both: U64(0),
        }
    )  # also kept as a masked 64-bit
    history: List[Undo] = field(default_factory=list)


def set_bit(bb: U64, sq64: int) -> U64:
    return to_u64(int(bb) | (1 << sq64))


def clear_bit(bb: U64, sq64: int) -> U64:
    return to_u64(int(bb) & ~(1 << sq64))


@dataclass
class Fen:
    pieces: list[Pcs]
    player: str
    castle_rights: str
    en_passant: str
    hlf_clk: int
    fll_mv: int


def parse_fen(fen_str) -> Fen:
    """Read FEN"""
    fen_chnks = [board_pos, player, castle_rights, en_passant, hlf_clk, fll_mv] = (
        fen_str.split()
    )
    list_of_pieces: List[Pcs] = [Pcs.empty] * BRD_SQ_NUM
    if len(fen_chnks) != 6:
        raise ValueError(f"Wrong format FEN {fen_str}")
    for rnk_n, rnk in enumerate(board_pos.split("/")):
        ct = 0
        for p in rnk:
            if p.isdigit():
                ct += int(p)
            else:
                loc = fr2sq(ct, 8 - rnk_n)
                list_of_pieces[loc] = Pcs[p]
                ct += 1

    fen = Fen(
        pieces=list_of_pieces,
        player=player,
        castle_rights=castle_rights,
        en_passant=en_passant,
        hlf_clk=hlf_clk,
        fll_mv=fll_mv,
    )
    return fen


def print_board(board: Board):
    for rank in range(8, 0, -1):
        line = f"{rank} |"
        for file in FILES:
            sq = fr2sq(FILES.index(file), rank)
            piece = board.pieces[sq]
            if piece != Pcs.empty:
                line += f" {PIECES_REP[piece]} "
            else:
                line += " . "
        print(line)
    print("--------------------------")
    print("    a  b  c  d  e  f  g  h")


def move(board: Board, from_sq: int, to_sq: int):
    piece = board.pieces[from_sq]
    board.pieces[to_sq] = piece
    board.pieces[from_sq] = Pcs.empty


def init_bitboards(board: Board):
    """Initialize bitboards for pawns and occupied squares"""
    board.pieces_bb = _empty_piece_bitboards()
    board.occupied = {
        Color.white: U64(0),
        Color.black: U64(0),
        Color.both: U64(0),
    }
    board.kings_pos = [-1, -1]
    board.rooks_pos = [-1] * 4

    for sq in range(BRD_SQ_NUM):
        piece = board.pieces[sq]
        sq64 = SQ120_TO_SQ64[sq]
        if sq64 == -1:
            continue
        if piece == Pcs.empty:
            continue

        if piece in WHITE_PIECES:
            color = Color.white
        elif piece in BLACK_PIECES:
            color = Color.black
        else:
            continue

        board.pieces_bb[color][piece] = set_bit(board.pieces_bb[color][piece], sq64)
        board.occupied[Color.both] = set_bit(board.occupied[Color.both], sq64)
        board.occupied[color] = set_bit(board.occupied[color], sq64)

        if piece == Pcs.K:
            board.kings_pos[Color.white.value] = sq64
        elif piece == Pcs.k:
            board.kings_pos[Color.black.value] = sq64
        elif piece == Pcs.R:
            if board.rooks_pos[0] == -1:
                board.rooks_pos[0] = sq64
            else:
                board.rooks_pos[1] = sq64
        elif piece == Pcs.r:
            if board.rooks_pos[2] == -1:
                board.rooks_pos[2] = sq64
            else:
                board.rooks_pos[3] = sq64
    pass


if __name__ == "__main__":
    # fen = parse_fen("8/8/2k5/8/4KP2/8/8/8 w - - 0 1")
    # fen = parse_fen(ROOK_ENDGAME)
    fen = parse_fen(ROOK_ENDGAME_CHECK)
    # fen = parse_fen(RUY_LOPEZ_FEN)

    board = Board(pieces=fen.pieces)
    print("The board\n")
    print_board(board)
    init_bitboards(board)
    white_king_attack = generate_k_attack_bm(
        board,
        board.kings_pos[Color.white.value],
        Color.white,
    )
    black_king_attack = generate_k_attack_bm(
        board,
        board.kings_pos[Color.black.value],
        Color.black,
    )
    print("\nThe white king attack! \n")

    white_king_attack &= ~black_king_attack

    print_attack_mask(to_u64(white_king_attack), pieces=board.pieces)
    rook_attack_rays = generate_rook_rays()

    white_rook_attack = generate_rook_attack_bm(
        board=board,
        sq64=board.rooks_pos[0],
        side=Color.white,
    )
    print("\nThe white rook attack rays \n")
    print_attack_mask(white_rook_attack, pieces=board.pieces)

    is_in_check(board, Color.white)

    print("\nThe white rook and king attack rays \n")
    both_ray = to_u64(white_rook_attack | white_king_attack)
    print_attack_mask(both_ray, pieces=board.pieces)

    print("\nIs white in check?", is_in_check(board, Color.white))
    print("Is black in check?", is_in_check(board, Color.black))

    print(f"\nPotential black king moves from {board.kings_pos[Color.black.value]}")
    black_king_moves = generate_king_moves(board, Color.black)
    print_attack_mask(black_king_moves, pieces=board.pieces)

    black_sq = board.kings_pos[Color.black.value]
    for piece_type in WHITE_PIECES:
        for attacker_sq in iter_bits(int(board.pieces_bb[Color.white][piece_type])):
            distance = chebyshev_dist(black_sq, attacker_sq)
            print(
                f"Distance from black king at {black_sq} to {piece_type} at {attacker_sq} is {distance}"
            )
