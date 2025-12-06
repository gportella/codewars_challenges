from collections.abc import Mapping, Sequence
from enum import Enum
import math
import random
from typing import Dict, List, NewType, Optional, TYPE_CHECKING, Tuple
from functools import lru_cache

if TYPE_CHECKING:
    from chess_engine import Board

U64 = NewType("U64", int)
U64_MASK = (1 << 64) - 1  # 0xFFFFFFFFFFFFFFFF
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
NOT_FILE_A = U64(~FILE_A & U64_MASK)
NOT_FILE_H = U64(~FILE_H & U64_MASK)

FILES = "abcdefgh"
BRD_SQ_NUM = 64

TT_EXACT = "exact"
TT_LOWER = "lower"
TT_UPPER = "upper"

# key -> (depth searched, score, flag, best_move)
TRANSPOSITION_TABLE: Dict[
    Tuple[int, int, int], Tuple[int, float, str, Optional[Tuple[int, int]]]
] = {}


def lsb(number: int) -> int:
    """Return the least significant blocker index of the given number."""
    if number == 0:
        return -1  # Handle zero input (no set bits)
    return int(math.log2(number & -number))


def msb(number: int) -> int:
    """Return the most significant blocker index of the given number."""
    assert number != 0
    return number.bit_length() - 1


def chebyshev_dist(sq1_index, sq2_index):
    f1 = sq1_index % 8
    r1 = sq1_index // 8
    f2 = sq2_index % 8
    r2 = sq2_index // 8

    file_dist = abs(f1 - f2)
    rank_dist = abs(r1 - r2)

    distance = max(file_dist, rank_dist)

    return distance


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

PIECES_VALUES = {
    Pcs.P: 100,
    Pcs.N: 320,
    Pcs.B: 330,
    Pcs.R: 500,
    Pcs.Q: 900,
    Pcs.K: 20000,
    Pcs.p: -100,
    Pcs.n: -320,
    Pcs.b: -330,
    Pcs.r: -500,
    Pcs.q: -900,
    Pcs.k: -20000,
}


MATE_SCORE = 100_000


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

    def to_fen_str(self) -> str:
        rights = []
        if self.value & CastlingRights.WKCA.value:
            rights.append("K")
        if self.value & CastlingRights.WQCA.value:
            rights.append("Q")
        if self.value & CastlingRights.BKCA.value:
            rights.append("k")
        if self.value & CastlingRights.BQCA.value:
            rights.append("q")
        return "".join(rights) if rights else "-"


WHITE_PIECES = frozenset({Pcs.P, Pcs.N, Pcs.B, Pcs.R, Pcs.Q, Pcs.K})
BLACK_PIECES = frozenset({Pcs.p, Pcs.n, Pcs.b, Pcs.r, Pcs.q, Pcs.k})


def __init_zobrist_keys() -> Dict[object, int]:
    keys: Dict[object, int] = {}
    for sq in range(BRD_SQ_NUM):
        for piece in Pcs:
            if piece == Pcs.empty:
                continue
            keys[(sq, piece)] = random.getrandbits(64)

    keys["side"] = random.getrandbits(64)

    for rights_value in range(16):
        keys[("castling", rights_value)] = random.getrandbits(64)

    for file_idx in range(8):
        keys[("en_passant", file_idx)] = random.getrandbits(64)
    return keys


ZOBRIST_KEYS = __init_zobrist_keys()


def compute_position_key(board: "Board") -> U64:
    key = 0
    for sq in range(BRD_SQ_NUM):
        piece = board.pieces[sq]
        if piece != Pcs.empty:
            key ^= ZOBRIST_KEYS[(sq, piece)]
    if board.side == Color.black:
        key ^= ZOBRIST_KEYS["side"]
    key ^= ZOBRIST_KEYS[("castling", board.castling_rights.value)]
    if board.en_passant != -1:
        key ^= ZOBRIST_KEYS[("en_passant", board.en_passant % 8)]
    return to_u64(key)


def to_u64(value: int) -> U64:
    return U64(value & U64_MASK)


def print_attack_mask(
    mask: U64,
    pieces: Sequence | None = None,
    piece_symbols: Mapping | None = None,
) -> None:
    bb = int(mask)
    symbols = piece_symbols or PIECES_REP
    for rank in range(7, -1, -1):
        line = f"{rank + 1} |"
        for file_idx, _ in enumerate(FILES):
            sq64 = rank * 8 + file_idx
            marker = " X " if (bb >> sq64) & 1 else " . "
            if pieces is not None:
                piece = pieces[sq64]
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


KING_ATTACK_PATTERNS = build_king_attack_patterns()
ROOK_ATTACK_RAYS = generate_rook_rays()


@lru_cache(maxsize=4096)
def rook_attacks_from(sq: int, occupancy: int) -> int:
    """Return rook attacks for a square given board occupancy."""

    rays = ROOK_ATTACK_RAYS[sq]
    attacks = 0
    occ = occupancy & U64_MASK

    for direction in ("north", "south", "east", "west"):
        ray = int(rays[direction])
        blockers = ray & occ
        if blockers:
            if direction in ("north", "east"):
                blocker_sq = lsb(blockers)
                attacks |= ray & ((1 << blocker_sq) - 1)
            else:
                blocker_sq = msb(blockers)
                attacks |= ray & ~((1 << (blocker_sq + 1)) - 1)
            attacks |= 1 << blocker_sq
        else:
            attacks |= ray

    return attacks & U64_MASK


def generate_rook_attack_bm(
    board: "Board",
    sq64: int,
    side: Color,
) -> U64:
    """Return rook attack bitboard pruned by friendly occupancy."""

    occupancy = int(board.occupied[Color.both])
    friendly_occ = int(board.occupied[side])
    attacks = rook_attacks_from(sq64, occupancy)
    legal_attacks = attacks & ~friendly_occ

    return to_u64(legal_attacks)


def iter_bits(mask: int):
    while mask:
        lsb = mask & -mask  # isolate lowest set bit
        idx = lsb.bit_length() - 1
        yield idx
        mask ^= lsb


def generate_k_attack_bm(
    board: "Board",
    sq64: int,
    side: Color,
    *,
    debug: bool = False,
) -> U64:
    """Return king attack bitboard pruned by friendly occupancy."""

    attacks = KING_ATTACK_PATTERNS[sq64]
    friendly_occ = board.occupied[side]
    legal_attacks = to_u64(int(attacks) & ~int(friendly_occ))

    return legal_attacks


attack_vectors = {
    Pcs.k: generate_k_attack_bm,
    Pcs.K: generate_k_attack_bm,
    Pcs.R: generate_rook_attack_bm,
    Pcs.r: generate_rook_attack_bm,
}


def is_check(board: "Board", side: Color, sq64: int) -> bool:
    """Determine if the given side is in check."""
    other_side = Color.white if side == Color.black else Color.black
    own_bb = 1 << sq64

    for piece_type in WHITE_PIECES if side == Color.black else BLACK_PIECES:
        for attacker_sq in iter_bits(int(board.pieces_bb[other_side][piece_type])):
            if (attack_vector := attack_vectors.get(piece_type)) is not None:
                if (attack_vector(board, attacker_sq, other_side) & own_bb) != 0:
                    return True
    return False


def is_in_check(board: "Board", side: Color) -> bool:
    """Determine if the given side is in check."""
    return is_check(board, side, board.kings_pos[side.value])


def evaluate_board_material(board: "Board") -> int:
    """Evaluate the board position and return a score."""
    score = 0
    for sq in range(64):
        piece = board.pieces[sq]
        score += PIECES_VALUES.get(piece, 0)
    return score


def evaluate_board(board: "Board") -> int:
    """Evaluate a move and return a score.

    Absolutely not:
        - can not move into check
    Positives:
        - check // checkmate
    """

    score_move = 0
    if is_in_check(board, Color.white):
        score_move -= 10000
    if is_in_check(board, Color.black):
        score_move += 10000
    score_move += evaluate_board_material(board)
    rook_mobility_score = mobility(board, Color.white, Pcs.R)
    king_mobility_score = mobility(board, Color.white, Pcs.K) - mobility(
        board, Color.black, Pcs.K
    )
    score_move += rook_mobility_score * 3
    score_move += king_mobility_score * 30
    score_move += king_progress_score(board)
    score_move -= enemy_king_mobility_penalty(board)
    score_move += rook_barrier_bonus(board)
    score_move += checkmate_pattern_bonus(board)
    return score_move


def terminal_score(board: "Board", depth_remaining: int) -> Optional[int]:
    """Return a decisive score for terminal positions."""

    if board.king_captured is not None:
        mate_bonus = MATE_SCORE + depth_remaining
        return mate_bonus if board.king_captured == Color.black else -mate_bonus

    if board.repetition_count() >= 3:
        return 0

    if depth_remaining == 0 and not board.has_legal_move(board.side):
        if is_in_check(board, board.side):
            mate_bonus = MATE_SCORE + depth_remaining
            return mate_bonus if board.side == Color.black else -mate_bonus
        return 0

    return None


def king_progress_score(board: "Board") -> int:
    white_king_sq = board.kings_pos[Color.white.value]
    black_king_sq = board.kings_pos[Color.black.value]

    distance_between_kings = chebyshev_dist(white_king_sq, black_king_sq)
    score = (7 - distance_between_kings) * 420

    corners = [0, 7, 56, 63]
    enemy_corner_dist = min(chebyshev_dist(black_king_sq, corner) for corner in corners)
    own_corner_dist = min(chebyshev_dist(white_king_sq, corner) for corner in corners)
    score += (7 - enemy_corner_dist) * 200
    score -= own_corner_dist * 5

    enemy_file = black_king_sq % 8
    enemy_rank = black_king_sq // 8
    own_file = white_king_sq % 8
    own_rank = white_king_sq // 8
    file_span = abs(enemy_file - own_file)
    rank_span = abs(enemy_rank - own_rank)
    board_area = (file_span + 1) * (rank_span + 1)
    score += (64 - board_area) * 24

    file_edge_dist = min(enemy_file, 7 - enemy_file)
    rank_edge_dist = min(enemy_rank, 7 - enemy_rank)
    edge_distance = file_edge_dist + rank_edge_dist
    score += (6 - edge_distance) * 260

    return score


def enemy_king_mobility_penalty(board: "Board") -> int:
    mobility_count = mobility(board, Color.black, Pcs.k)
    return mobility_count * 260


def rook_barrier_bonus(board: "Board") -> int:
    black_king_sq = board.kings_pos[Color.black.value]
    white_king_sq = board.kings_pos[Color.white.value]

    if black_king_sq == -1 or white_king_sq == -1:
        return 0

    rook_positions = [sq for sq in board.rooks_pos[:2] if sq != -1]
    if not rook_positions:
        return 0

    bonus = 0
    bk_file = black_king_sq % 8
    bk_rank = black_king_sq // 8
    wk_distance = chebyshev_dist(white_king_sq, black_king_sq)

    for rook_sq in rook_positions:
        r_file = rook_sq % 8
        r_rank = rook_sq // 8

        if r_file == bk_file:
            distance = abs(r_rank - bk_rank)
            closeness = max(0, 5 - distance)
            if closeness:
                bonus += closeness * 250
            if wk_distance <= 3:
                bonus += 150

        if r_rank == bk_rank:
            distance = abs(r_file - bk_file)
            closeness = max(0, 5 - distance)
            if closeness:
                bonus += closeness * 250
            if wk_distance <= 3:
                bonus += 150

    return bonus


def checkmate_pattern_bonus(board: "Board") -> int:
    black_king_sq = board.kings_pos[Color.black.value]
    if black_king_sq == -1:
        return 0

    corners = {0, 7, 56, 63}
    if black_king_sq not in corners:
        return 0

    white_king_sq = board.kings_pos[Color.white.value]
    if white_king_sq == -1:
        return 0

    if chebyshev_dist(white_king_sq, black_king_sq) > 2:
        return 0

    rook_positions = [sq for sq in board.rooks_pos[:2] if sq != -1]
    if not rook_positions:
        return 0

    bk_file = black_king_sq % 8
    bk_rank = black_king_sq // 8

    for rook_sq in rook_positions:
        r_file = rook_sq % 8
        r_rank = rook_sq // 8
        if r_file == bk_file or r_rank == bk_rank:
            return 6000

    return 0


def mobility(board: "Board", side: Color, piece_type: Pcs) -> int:
    actual_piece = piece_type
    if side == Color.white and piece_type in BLACK_PIECES:
        actual_piece = Pcs(piece_type.value - 6)
    elif side == Color.black and piece_type in WHITE_PIECES:
        actual_piece = Pcs(piece_type.value + 6)

    attack_fn = attack_vectors.get(actual_piece)
    if attack_fn is None:
        return 0

    moves = 0
    for from_sq in iter_bits(board.pieces_bb[side][actual_piece]):
        attack_bm = attack_fn(board, from_sq, side)
        moves += attack_bm.bit_count()
    return moves


def generate_legal_moves(board: "Board", side: Color):
    for piece_type in WHITE_PIECES if side == Color.white else BLACK_PIECES:
        for from_sq in iter_bits(board.pieces_bb[side][piece_type]):
            attack_bm = attack_vectors[piece_type](board, from_sq, side)
            for to_sq in iter_bits(int(attack_bm)):
                yield (from_sq, to_sq)


def minimax(board, depth, side_to_move, alpha, beta):
    repetition = board.repetition_count()
    key = (int(board.position_key), repetition, depth)
    entry = TRANSPOSITION_TABLE.get(key)
    stored_move: Optional[Tuple[int, int]] = None
    if entry is not None:
        stored_depth, stored_score, flag, stored_move = entry
        if stored_depth >= depth:
            if flag == TT_EXACT:
                return stored_score, stored_move
            if flag == TT_LOWER:
                alpha = max(alpha, stored_score)
            elif flag == TT_UPPER:
                beta = min(beta, stored_score)
            if alpha >= beta:
                return stored_score, stored_move

    terminal = terminal_score(board, depth)
    if terminal is not None:
        return terminal, None

    if depth == 0:
        return evaluate_board(board), None

    alpha_init = alpha
    beta_init = beta

    best_score = -math.inf if side_to_move == Color.white else math.inf
    best_move: Optional[Tuple[int, int]] = None
    legal_move_found = False

    moves = list(generate_legal_moves(board, side_to_move))
    captures: List[Tuple[int, int]] = []
    quiet: List[Tuple[int, int]] = []
    for move in moves:
        captured_piece = board.pieces[move[1]]
        if captured_piece != Pcs.empty:
            captures.append(move)
        else:
            quiet.append(move)

    if stored_move in captures:
        captures.remove(stored_move)
        captures.insert(0, stored_move)
    elif stored_move in quiet:
        quiet.remove(stored_move)
        quiet.insert(0, stored_move)

    ordered_moves = captures + quiet

    for move in ordered_moves:
        opposite_side = Color.white if side_to_move == Color.black else Color.black
        undo = board.make_move(*move)  # mutate board, remember undo info
        if is_in_check(board, side_to_move):
            board.undo_move(undo)
            continue

        repetition_count = board.repetition_count()
        if repetition_count >= 3:
            score = 0  # draw by repetition
        else:
            score, _ = minimax(board, depth - 1, opposite_side, alpha, beta)

        board.undo_move(undo)

        legal_move_found = True

        if side_to_move == Color.white:  # maximizing node
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        else:  # minimizing node
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)

        if beta <= alpha:  # alpha-beta cutoff
            break

    if not legal_move_found:
        mate_bonus = MATE_SCORE + depth
        if is_in_check(board, side_to_move):
            best_score = mate_bonus if side_to_move == Color.black else -mate_bonus
        else:
            best_score = 0
        best_move = None

    if len(TRANSPOSITION_TABLE) > 200_000:
        TRANSPOSITION_TABLE.clear()

    if best_score <= alpha_init:
        flag = TT_UPPER
    elif best_score >= beta_init:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    TRANSPOSITION_TABLE[key] = (depth, best_score, flag, best_move)

    return best_score, best_move


def generate_next_move(
    board: "Board", side: Color, depth: int = 3
) -> Optional[Tuple[int, int]]:
    """Generate the next best move for the given side using minimax."""
    _, best_move = minimax(board, depth, side, -math.inf, math.inf)
    return best_move


if __name__ == "__main__":
    king_masks = build_king_attack_patterns()
    for sq, attack_mask in enumerate(king_masks):
        print(f"King attacks for {sq}: {int(attack_mask):064b}")
        print_attack_mask(attack_mask)
        print()
