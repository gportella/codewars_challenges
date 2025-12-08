from typing import List

NOT_FILE_A = 0xFEFEFEFEFEFEFEFE
NOT_FILE_H = 0x7F7F7F7F7F7F7F7F


U64_MASK = 0xFFFFFFFFFFFFFFFF


def build_king_attack_patterns() -> List[int]:
    attacks: List[int] = []
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
        attacks.append(mask & U64_MASK)
    return attacks


KING_ATTACKS = build_king_attack_patterns()
print("KING_ATTACKS =", KING_ATTACKS)
