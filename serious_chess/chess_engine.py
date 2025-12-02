#! /usr/bin/env python
"""
Learning. I'm going to base it on Programming a chess engine in C by Bluefever software
Doing it in python to pass some Katas in python, as they don't have a C/C++ version
Over the top, but seemed fun to do.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from render_board import show_fancy_board
from types_and_masks import (
    BLACK_PIECES,
    PIECES_REP,
    Pcs,
    WHITE_PIECES,
    CastlingRights,
    Color,
    U64,
    chebyshev_dist,
    compute_position_key,
    generate_k_attack_bm,
    generate_king_moves,
    generate_next_move,
    generate_rook_attack_bm,
    is_in_check,
    iter_bits,
    minimax,
    print_attack_mask,
    to_u64,
    BRD_SQ_NUM,
)


MAXMOVES_GAME = 2048
FILES = "abcdefgh"
RANKS = range(1, 9)


def fr2sq(file_idx: int, rank: int) -> int:
    """Convert file (0-7) and rank (1-8, with 1 = rank1) to 0..63 index."""

    return (rank - 1) * 8 + file_idx


def sq2coord(square: int) -> str:
    """Convert 0..63 index to algebraic coordinate like 'e4'."""

    file_idx = square % 8
    rank = square // 8 + 1
    return f"{FILES[file_idx]}{rank}"


_square_vals = {
    f"{file}{rank}": fr2sq(file_idx, rank)
    for rank in RANKS
    for file_idx, file in enumerate(FILES)
}
Sqr = IntEnum("Square", _square_vals)


CANONICAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
wE4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
RUY_LOPEZ_FEN = "r1bqkb1r/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
ROOK_ENDGAME = "8/8/8/8/5R2/3k4/5K2/8 b - - 0 1"
ROOK_ENDGAME_CHECK = "8/8/8/8/3k1R2/8/5K2/8 b - - 0 1"


def _empty_piece_bitboards() -> Dict[Color, Dict[Pcs, U64]]:
    return {
        Color.white: {piece: U64(0) for piece in WHITE_PIECES},
        Color.black: {piece: U64(0) for piece in BLACK_PIECES},
    }


@dataclass
class Undo:
    move: Tuple[int, int] = (-1, -1)
    calstlng_rights: CastlingRights = CastlingRights.No
    en_passant: int = -1
    position_key: U64 = U64(0)
    moved_piece: Pcs = Pcs.empty
    captured_piece: Pcs = Pcs.empty
    mover_bb: U64 = U64(0)
    captured_bb: Optional[U64] = None
    side: Color = Color.white
    fifty_move: int = 0
    ply: int = 0
    his_ply: int = 0
    occupied_white: U64 = U64(0)
    occupied_black: U64 = U64(0)
    occupied_both: U64 = U64(0)
    kings_pos: Tuple[int, int] = (-1, -1)
    rooks_pos: Tuple[int, int, int, int] = (-1, -1, -1, -1)
    king_captured: Optional[Color] = None


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
    )  # assume no promotions for now
    occupied: dict[Color, U64] = field(
        default_factory=lambda: {
            Color.white: U64(0),
            Color.black: U64(0),
            Color.both: U64(0),
        }
    )  # also kept as a masked 64-bit
    history: Deque[U64] = field(default_factory=deque)
    king_captured: Optional[Color] = None

    def __post_init__(self):
        self.position_key = compute_position_key(self)
        self.history.append(self.position_key)

    def to_fen(self):
        fen_parts = []
        for rank in range(8, 0, -1):
            line = ""
            empty_count = 0
            for file in FILES:
                sq = fr2sq(FILES.index(file), rank)
                piece = self.pieces[sq]
                if piece == Pcs.empty:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        line += str(empty_count)
                        empty_count = 0
                    line += piece.name
            if empty_count > 0:
                line += str(empty_count)
            if rank > 1:
                line += "/"
            fen_parts.append(line)
        fen_parts = ["".join(fen_parts)]
        fen_parts.append("w" if self.side == Color.white else "b")
        fen_parts.append(self.castling_rights.to_fen_str())
        fen_parts.append(sq2coord(self.en_passant) if self.en_passant != -1 else "-")
        fen_parts.append(str(self.fifty_move))
        fen_parts.append(str(self.his_ply // 2 + 1))

        return " ".join(fen_parts)

    def is_terminal(self) -> bool:
        if self.king_captured is not None:
            return True

        moves = generate_king_moves(self, self.side)
        if int(moves) == 0:
            return True

        return False

    def make_move(self, from_sq: int, to_sq: int) -> Undo:
        moved_piece = self.pieces[from_sq]
        captured_piece = self.pieces[to_sq]
        mover_color = Color.white if moved_piece in WHITE_PIECES else Color.black
        opponent = Color.white if mover_color == Color.black else Color.black
        captured_color: Optional[Color] = None
        if captured_piece != Pcs.empty:
            captured_color = (
                Color.white if captured_piece in WHITE_PIECES else Color.black
            )

        undo = Undo(
            move=(from_sq, to_sq),
            calstlng_rights=self.castling_rights,
            en_passant=self.en_passant,
            position_key=self.position_key,
            moved_piece=moved_piece,
            captured_piece=captured_piece,
            mover_bb=self.pieces_bb[mover_color][moved_piece],
            captured_bb=(
                self.pieces_bb[captured_color][captured_piece]
                if captured_color is not None
                else None
            ),
            side=mover_color,
            fifty_move=self.fifty_move,
            ply=self.ply,
            his_ply=self.his_ply,
            occupied_white=self.occupied[Color.white],
            occupied_black=self.occupied[Color.black],
            occupied_both=self.occupied[Color.both],
            kings_pos=(self.kings_pos[0], self.kings_pos[1]),
            rooks_pos=(
                self.rooks_pos[0],
                self.rooks_pos[1],
                self.rooks_pos[2],
                self.rooks_pos[3],
            ),
            king_captured=self.king_captured,
        )

        self.pieces[to_sq] = moved_piece
        self.pieces[from_sq] = Pcs.empty

        mover_bb = clear_bit(self.pieces_bb[mover_color][moved_piece], from_sq)
        mover_bb = set_bit(mover_bb, to_sq)
        self.pieces_bb[mover_color][moved_piece] = mover_bb

        if captured_color is not None:
            captured_bb = clear_bit(
                self.pieces_bb[captured_color][captured_piece], to_sq
            )
            self.pieces_bb[captured_color][captured_piece] = captured_bb

        self.occupied[mover_color] = set_bit(
            clear_bit(self.occupied[mover_color], from_sq),
            to_sq,
        )
        self.occupied[Color.both] = set_bit(
            clear_bit(self.occupied[Color.both], from_sq),
            to_sq,
        )
        if captured_color is not None:
            self.occupied[captured_color] = clear_bit(
                self.occupied[captured_color], to_sq
            )

        if moved_piece == Pcs.K:
            self.kings_pos[Color.white.value] = to_sq
        elif moved_piece == Pcs.k:
            self.kings_pos[Color.black.value] = to_sq

        if captured_piece == Pcs.K:
            self.king_captured = Color.white
        elif captured_piece == Pcs.k:
            self.king_captured = Color.black

        if moved_piece == Pcs.R:
            for idx in range(0, 2):
                if self.rooks_pos[idx] == from_sq:
                    self.rooks_pos[idx] = to_sq
                    break
        elif moved_piece == Pcs.r:
            for idx in range(2, 4):
                if self.rooks_pos[idx] == from_sq:
                    self.rooks_pos[idx] = to_sq
                    break

        if captured_piece == Pcs.R:
            for idx in range(0, 2):
                if self.rooks_pos[idx] == to_sq:
                    self.rooks_pos[idx] = -1
                    break
        elif captured_piece == Pcs.r:
            for idx in range(2, 4):
                if self.rooks_pos[idx] == to_sq:
                    self.rooks_pos[idx] = -1
                    break

        if moved_piece in (Pcs.P, Pcs.p) or captured_piece != Pcs.empty:
            self.fifty_move = 0
        else:
            self.fifty_move += 1

        self.ply += 1
        self.his_ply += 1

        self.en_passant = -1

        self.side = opponent

        self.position_key = compute_position_key(self)
        self.history.append(self.position_key)

        return undo

    def undo_move(self, undo: Undo):
        from_sq, to_sq = undo.move
        opponent = Color.white if undo.side == Color.black else Color.black

        if self.history:
            self.history.pop()

        self.side = undo.side
        self.castling_rights = undo.calstlng_rights
        self.en_passant = undo.en_passant
        self.position_key = undo.position_key
        self.fifty_move = undo.fifty_move
        self.ply = undo.ply
        self.his_ply = undo.his_ply

        self.pieces[from_sq] = undo.moved_piece
        self.pieces[to_sq] = undo.captured_piece

        self.pieces_bb[self.side][undo.moved_piece] = undo.mover_bb
        if undo.captured_piece != Pcs.empty and undo.captured_bb is not None:
            self.pieces_bb[opponent][undo.captured_piece] = undo.captured_bb

        self.occupied[Color.white] = undo.occupied_white
        self.occupied[Color.black] = undo.occupied_black
        self.occupied[Color.both] = undo.occupied_both

        self.kings_pos = [undo.kings_pos[0], undo.kings_pos[1]]
        self.rooks_pos = list(undo.rooks_pos)
        self.king_captured = undo.king_captured


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


# def move(board: Board, from_sq: int, to_sq: int):
#     piece = board.pieces[from_sq]
#     board.pieces[to_sq] = piece
#     board.pieces[from_sq] = Pcs.empty


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
        if piece == Pcs.empty:
            continue

        if piece in WHITE_PIECES:
            color = Color.white
        elif piece in BLACK_PIECES:
            color = Color.black
        else:
            continue

        board.pieces_bb[color][piece] = set_bit(board.pieces_bb[color][piece], sq)
        board.occupied[Color.both] = set_bit(board.occupied[Color.both], sq)
        board.occupied[color] = set_bit(board.occupied[color], sq)

        if piece == Pcs.K:
            board.kings_pos[Color.white.value] = sq
        elif piece == Pcs.k:
            board.kings_pos[Color.black.value] = sq
        elif piece == Pcs.R:
            if board.rooks_pos[0] == -1:
                board.rooks_pos[0] = sq
            else:
                board.rooks_pos[1] = sq
        elif piece == Pcs.r:
            if board.rooks_pos[2] == -1:
                board.rooks_pos[2] = sq
            else:
                board.rooks_pos[3] = sq
    board.position_key = compute_position_key(board)
    board.history = deque([board.position_key])


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

    white_king_mask = int(white_king_attack) & ~int(black_king_attack)
    white_king_attack = to_u64(white_king_mask)

    print_attack_mask(white_king_attack, pieces=board.pieces)

    white_rook_attack = generate_rook_attack_bm(
        board=board,
        sq64=board.rooks_pos[0],
        side=Color.white,
    )
    print("\nThe white rook attack rays \n")
    print_attack_mask(white_rook_attack, pieces=board.pieces)

    current_side = Color.black
    game_moves = 0
    while board.king_captured is None:
        game_moves += 1
        print(f"\nMove {game_moves}, {current_side.name} to move")
        move = generate_next_move(board, current_side)
        if move is None:
            print("No move found, game over.")
            break
        board.make_move(*move)
        show_fancy_board(board.to_fen(), size=400)
        current_side = Color.black if current_side == Color.white else Color.white
    # generate_next_move(board, Color.white)

    # minimax_score, best_move = minimax(
    #     board, depth=3, side_to_move=Color.white, alpha=-99999, beta=99999
    # )
    # print(f"\nMinimax evaluation score for white: {minimax_score} for move {best_move}")

    # print("\nThe white rook and king attack rays \n")
    # both_ray = to_u64(int(white_rook_attack) | int(white_king_attack))
    # print_attack_mask(both_ray, pieces=board.pieces)

    # print("\nIs white in check?", is_in_check(board, Color.white))
    # print("Is black in check?", is_in_check(board, Color.black))

    # print(f"\nPotential black king moves from {board.kings_pos[Color.black.value]}")
    # black_king_moves = generate_king_moves(board, Color.black)
    # print_attack_mask(black_king_moves, pieces=board.pieces)

    # # show_fancy_board(our_fen, attacks=black_king_moves, size=400)
    # black_sq = board.kings_pos[Color.black.value]
    # for piece_type in WHITE_PIECES:
    #     for attacker_sq in iter_bits(int(board.pieces_bb[Color.white][piece_type])):
    #         distance = chebyshev_dist(black_sq, attacker_sq)
    #         print(
    #             f"Distance from black king at {black_sq} to {piece_type} at {attacker_sq} is {distance}"
    #         )
