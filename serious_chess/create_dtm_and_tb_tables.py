#!/usr/bin/env python
"""
Build a KRK (King+Rook vs King) tablebase via retrograde analysis.

Outputs:
- krk_tb.pkl: dict mapping raw index -> (wdl, dtm, mover, to_sq)
- krk_tb_base64.txt: base64+zlib compressed packed array of length 524,288 bytes
  where each byte encodes [wdl(2 bits) | mover(2 bits) | to_sq(6 bits)].

WDL relative to side-to-move: 0=draw, 1=win, 2=loss.
mover: 0=none, 1=king, 2=rook (only meaningful for wins; black wins don’t exist in KRK).
to_sq: 0..63 (only meaningful for wins).
DTM: distance to mate in plies; -1 for draws.
"""

from array import array
from typing import List, Tuple
from collections import deque
import base64
import zlib
import pickle


BRD_SQ_NUM = 64
WHITE, BLACK = 0, 1


def bit(sq: int) -> int:
    return 1 << sq


def sq_file(sq: int) -> int:
    return sq % 8


def sq_rank(sq: int) -> int:
    return sq // 8


# Chebyshev distance (kings metric)
CHEBYSHEV = [[0] * BRD_SQ_NUM for _ in range(BRD_SQ_NUM)]
for a in range(BRD_SQ_NUM):
    fa, ra = sq_file(a), sq_rank(a)
    for b in range(BRD_SQ_NUM):
        CHEBYSHEV[a][b] = max(abs(fa - sq_file(b)), abs(ra - sq_rank(b)))


def kings_adjacent(a: int, b: int) -> bool:
    return CHEBYSHEV[a][b] <= 1


U64_MASK = (1 << 64) - 1
FILE_A = 0x0101010101010101
FILE_H = 0x8080808080808080
NOT_FILE_A = ~FILE_A & U64_MASK
NOT_FILE_H = ~FILE_H & U64_MASK


def build_king_attack_patterns() -> List[int]:
    attacks = []
    for sq in range(64):
        bit_sq = 1 << sq
        mask = 0
        mask |= (bit_sq << 8) & U64_MASK
        mask |= (bit_sq >> 8) & U64_MASK
        if bit_sq & NOT_FILE_H:
            mask |= (bit_sq << 9) & U64_MASK
            mask |= (bit_sq >> 7) & U64_MASK
            mask |= (bit_sq << 1) & U64_MASK
        if bit_sq & NOT_FILE_A:
            mask |= (bit_sq << 7) & U64_MASK
            mask |= (bit_sq >> 9) & U64_MASK
            mask |= (bit_sq >> 1) & U64_MASK
        attacks.append(mask & U64_MASK)
    return attacks


KING_ATTACK_MASKS = build_king_attack_patterns()


def rook_attacks_on_the_fly(square: int, occupancy: int) -> int:
    attacks = 0
    r = sq_rank(square)
    f = sq_file(square)
    for rr in range(r + 1, 8):
        sq = rr * 8 + f
        attacks |= 1 << sq
        if occupancy & (1 << sq):
            break
    for rr in range(r - 1, -1, -1):
        sq = rr * 8 + f
        attacks |= 1 << sq
        if occupancy & (1 << sq):
            break
    for ff in range(f + 1, 8):
        sq = r * 8 + ff
        attacks |= 1 << sq
        if occupancy & (1 << sq):
            break
    for ff in range(f - 1, -1, -1):
        sq = r * 8 + ff
        attacks |= 1 << sq
        if occupancy & (1 << sq):
            break
    return attacks & U64_MASK


def rook_attacks_from(sq: int, occupancy: int) -> int:
    return rook_attacks_on_the_fly(sq, occupancy)



N = 2 * 64 * 64 * 64  # 524,288 raw states


def krk_index(side: int, wk: int, wr: int, bk: int) -> int:
    # side (1 bit), wk (6), wr (6), bk (6) => index in 0..524287
    return (side << 18) | (wk << 12) | (wr << 6) | bk


def krk_decode(idx: int) -> Tuple[int, int, int, int]:
    side = (idx >> 18) & 1
    wk = (idx >> 12) & 63
    wr = (idx >> 6) & 63
    bk = idx & 63
    return side, wk, wr, bk


def pack_entry(wdl: int, mover: int, to_sq: int) -> int:
    # 1 byte: [wdl(2)|mover(2)|to_sq(6)]
    return ((wdl & 0b11) << 6) | ((mover & 0b11) << 4) | (to_sq & 0b111111)




def in_check(side: int, wk: int, wr: int, bk: int) -> bool:
    occ = bit(wk) | bit(wr) | bit(bk)
    if side == BLACK:
        # black in check if attacked by white king or rook
        if KING_ATTACK_MASKS[wk] & (1 << bk):
            return True
        rook_mask = rook_attacks_from(wr, occ)
        return (rook_mask & (1 << bk)) != 0
    else:
        # white in check only by adjacency (illegal states), but keep generic
        return (KING_ATTACK_MASKS[bk] & (1 << wk)) != 0


def is_valid_state(wk: int, wr: int, bk: int) -> bool:
    if wk == wr or wk == bk or wr == bk:
        return False
    if kings_adjacent(wk, bk):
        return False
    return True




def children_with_moves(
    side: int, wk: int, wr: int, bk: int
) -> Tuple[List[Tuple[int, int, int]], bool]:
    """
    Returns: (list of (child_idx, mover_code, to_sq), has_terminal_draw)
    - mover_code: 1=king, 2=rook (White side); Black side only 1=king
    - has_terminal_draw: True if Black can capture the rook (K vs K); terminal draw edge
    """
    occ = bit(wk) | bit(wr) | bit(bk)
    moves: List[Tuple[int, int, int]] = []
    has_draw = False

    if side == WHITE:
        # White king moves
        kmask = KING_ATTACK_MASKS[wk] & ~occ
        for dest in range(64):
            if not (kmask & (1 << dest)):
                continue
            if kings_adjacent(dest, bk):
                continue
            child_idx = krk_index(BLACK, dest, wr, bk)
            moves.append((child_idx, 1, dest))
        # White rook moves (cannot capture king)
        rmask = rook_attacks_from(wr, occ)
        rmask &= ~(1 << wk)
        rmask &= ~(1 << bk)
        for dest in range(64):
            if not (rmask & (1 << dest)):
                continue
            child_idx = krk_index(BLACK, wk, dest, bk)
            moves.append((child_idx, 2, dest))
        return moves, False

    else:
        # Black king moves
        kmask = KING_ATTACK_MASKS[bk]
        for dest in range(64):
            if not (kmask & (1 << dest)):
                continue
            if dest == wk:
                continue
            if dest == wr:
                # Attempt to capture rook -> K vs K terminal draw if not adjacent to white king
                if kings_adjacent(dest, wk):
                    continue
                has_draw = True
                continue  # terminal draw edge; not in KRK domain
            if kings_adjacent(dest, wk):
                continue
            occ2 = (occ & ~(1 << bk)) | (1 << dest)
            # Cannot move into rook check
            rmask = rook_attacks_from(wr, occ2)
            if (rmask & (1 << dest)) != 0:
                continue
            child_idx = krk_index(WHITE, wk, wr, dest)
            moves.append((child_idx, 1, dest))
        return moves, has_draw




def build_krk_tablebase():
    # Arrays (length N = 524_288)
    # wdl: -1 unknown, 0 draw, 1 win, 2 loss (relative to side-to-move)
    wdl = array("b", [-1] * N)
    # dtm: plies to mate; -1 for draws/unknown
    dtm = array("h", [-1] * N)
    # best move for wins
    mover_best = array("b", [0] * N)  # 0 none, 1 king, 2 rook (white only)
    to_sq_best = array("B", [0] * N)  # 0..63

    valid_count = 0

    for side in (WHITE, BLACK):
        for wk in range(64):
            for wr in range(64):
                if wr == wk:
                    continue
                for bk in range(64):
                    if not is_valid_state(wk, wr, bk):
                        continue
                    idx = krk_index(side, wk, wr, bk)
                    valid_count += 1

                    child_moves, has_draw = children_with_moves(side, wk, wr, bk)
                    inchk = in_check(side, wk, wr, bk)

                    if len(child_moves) == 0:
                        if inchk:
                            if side == BLACK and has_draw:
                                # Only escape is capturing the rook -> terminal draw
                                wdl[idx] = 0
                                dtm[idx] = -1
                            else:
                                # Checkmated
                                wdl[idx] = 2
                                dtm[idx] = 0
                        else:
                            # Stalemate
                            wdl[idx] = 0
                            dtm[idx] = -1

    print(f"Valid states: {valid_count}")

    # Phase 1: Assign new Wins and Losses until no more unknowns can be classified.
    while True:
        changed_win = 0
        # Assign Wins where any child is a Loss (choose minimal DTM child)
        for idx in range(N):
            if wdl[idx] != -1:
                continue
            side, wk, wr, bk = krk_decode(idx)
            if not is_valid_state(wk, wr, bk):
                wdl[idx] = 0
                dtm[idx] = -1
                continue
            child_moves, _ = children_with_moves(side, wk, wr, bk)
            best_d = None
            best_mv = (0, 0)
            for child_idx, mover_code, to_sq in child_moves:
                if wdl[child_idx] == 2:
                    d = dtm[child_idx] + 1
                    if best_d is None or d < best_d:
                        best_d = d
                        best_mv = (mover_code, to_sq)
            if best_d is not None:
                wdl[idx] = 1
                dtm[idx] = best_d
                mover_best[idx] = best_mv[0]
                to_sq_best[idx] = best_mv[1]
                changed_win += 1

        changed_loss = 0
        # Assign Losses where all children are Wins and Black has no terminal draw capture
        for idx in range(N):
            if wdl[idx] != -1:
                continue
            side, wk, wr, bk = krk_decode(idx)
            if not is_valid_state(wk, wr, bk):
                wdl[idx] = 0
                dtm[idx] = -1
                continue
            child_moves, has_draw = children_with_moves(side, wk, wr, bk)
            if not child_moves:
                # Already seeded (mate/stalemate)
                continue
            if side == BLACK and has_draw:
                # Black can draw by capturing the rook; not a Loss
                continue
            all_win = True
            max_child_d = -1
            for child_idx, _, _ in child_moves:
                if wdl[child_idx] != 1:
                    all_win = False
                    break
                if dtm[child_idx] > max_child_d:
                    max_child_d = dtm[child_idx]
            if all_win:
                wdl[idx] = 2
                dtm[idx] = max_child_d + 1
                changed_loss += 1

        print(f"Assign: wins={changed_win}, losses={changed_loss}")
        if changed_win == 0 and changed_loss == 0:
            break

    # Phase 2: Converge DTM for already assigned Wins/Losses
    # Keep improving (lowering) win DTM and (raising) loss DTM until stable
    while True:
        improved = 0

        # Improve Wins: pick minimal child Loss DTM + 1
        for idx in range(N):
            if wdl[idx] != 1:
                continue
            side, wk, wr, bk = krk_decode(idx)
            child_moves, _ = children_with_moves(side, wk, wr, bk)
            best_d = None
            best_mv = (mover_best[idx], to_sq_best[idx])
            for child_idx, mover_code, to_sq in child_moves:
                if wdl[child_idx] == 2:
                    d = dtm[child_idx] + 1
                    if best_d is None or d < best_d:
                        best_d = d
                        best_mv = (mover_code, to_sq)
            if best_d is not None and best_d < dtm[idx]:
                dtm[idx] = best_d
                mover_best[idx] = best_mv[0]
                to_sq_best[idx] = best_mv[1]
                improved += 1

        # Improve Losses: pick maximal child Win DTM + 1
        for idx in range(N):
            if wdl[idx] != 2:
                continue
            side, wk, wr, bk = krk_decode(idx)
            child_moves, has_draw = children_with_moves(side, wk, wr, bk)
            if side == BLACK and has_draw:
                # Loss can't be improved when Black has draw edge; but such losses shouldn't exist
                continue
            max_child_d = -1
            all_win = True
            for child_idx, _, _ in child_moves:
                if wdl[child_idx] != 1:
                    all_win = False
                    break
                if dtm[child_idx] > max_child_d:
                    max_child_d = dtm[child_idx]
            if all_win and max_child_d >= 0 and max_child_d + 1 > dtm[idx]:
                dtm[idx] = max_child_d + 1
                improved += 1

        print(f"Refine dtm: improved entries={improved}")
        if improved == 0:
            break

    # Mark remaining unknowns as Draw
    undecided = 0
    for idx in range(N):
        if wdl[idx] == -1:
            wdl[idx] = 0
            dtm[idx] = -1
            mover_best[idx] = 0
            to_sq_best[idx] = 0
            undecided += 1
    print(f"Unknown -> draws: {undecided}")

    # Pack 1 byte per entry
    packed = bytearray(N)
    for idx in range(N):
        packed[idx] = pack_entry(wdl[idx], mover_best[idx], to_sq_best[idx])

    return wdl, dtm, mover_best, to_sq_best, packed


def pack_dtm_blob(wdl, dtm) -> bytearray:
    # 1 byte per entry: 0..63 for plies to mate; 255 for draws
    N = 2 * 64 * 64 * 64
    out = bytearray(N)
    for idx in range(N):
        if int(wdl[idx]) == 0 or int(dtm[idx]) < 0:
            out[idx] = 255
        else:
            d = int(dtm[idx])
            if d < 0:  # safety
                out[idx] = 255
            else:
                out[idx] = min(d, 63)
    return out


def save_base64_blob_bytes(data: bytearray, path: str, level: int = 9):
    compressed = zlib.compress(bytes(data), level=level)
    b64 = base64.b64encode(compressed).decode("ascii")
    with open(path, "w") as f:
        f.write(b64)
    print(f"Saved {path}: base64 chars={len(b64)}, compressed bytes={len(compressed)}")



def save_pickle(wdl, dtm, mover_best, to_sq_best, path: str = "krk_tb.pkl"):
    tb = {}
    for idx in range(N):
        tb[idx] = (
            int(wdl[idx]),
            int(dtm[idx]),
            int(mover_best[idx]),
            int(to_sq_best[idx]),
        )
    with open(path, "wb") as f:
        pickle.dump(tb, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved pickle: {path} entries={len(tb)}")


def save_base64_blob(
    packed: bytearray, path: str = "krk_tb_base64.txt", level: int = 9
):
    compressed = zlib.compress(bytes(packed), level=level)
    b64 = base64.b64encode(compressed).decode("ascii")
    with open(path, "w") as f:
        f.write(b64)
    print(
        f"Saved base64+zlib blob: {path} size={len(b64)} chars (compressed={len(compressed)} bytes)"
    )




def main():
    wdl, dtm, mover_best, to_sq_best, packed = build_krk_tablebase()
    # Save both formats
    save_pickle(wdl, dtm, mover_best, to_sq_best, "krk_tb.pkl")
    save_base64_blob(packed, "krk_tb_base64.txt", level=9)
    dtm_blob = pack_dtm_blob(wdl, dtm)
    save_base64_blob_bytes(dtm_blob, "krk_dtm_base64.txt", level=9)
    print("Done.")


if __name__ == "__main__":
    main()
