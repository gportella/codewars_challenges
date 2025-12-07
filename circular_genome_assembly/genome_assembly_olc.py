#!/usr/bin/env python
"""
Genome assembly using Overlap-Layout-Consensus (OLC) approach.
Handles circular genomes with error tolerance through approximate matching.
"""

from typing import List, Optional, Tuple
from collections import defaultdict
import heapq
from test_solutions import test_solutions


def rc(s: str) -> str:
    """Return reverse complement of DNA sequence."""
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return s.translate(comp)[::-1]


def canonicalize_circular(sequence: str) -> str:
    """Find lexicographically smallest rotation of circular sequence."""
    if not sequence:
        return sequence

    s = sequence + sequence
    f = [-1] * len(s)
    k = 0

    for j in range(1, len(s)):
        i = f[j - k - 1]
        while i != -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j - i - 1
            i = f[i]

        if i == -1 and s[j] != s[k + i + 1]:
            if s[j] < s[k + i + 1]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1

    return sequence[k:] + sequence[:k]


def hamming_distance(s1: str, s2: str, max_dist: int = None) -> int:
    """Calculate Hamming distance between two strings."""
    if len(s1) != len(s2):
        return float("inf")

    dist = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            dist += 1
            if max_dist is not None and dist > max_dist:
                return dist
    return dist


def find_overlap(
    a: str, b: str, min_overlap: int, max_mismatches: int = 0
) -> Tuple[int, int]:
    """
    Find overlap between suffix of a and prefix of b.
    Returns (overlap_length, num_mismatches).
    """
    best_overlap = 0
    best_mismatches = max_mismatches + 1

    # Try all possible overlap lengths
    for olen in range(min(len(a), len(b)), min_overlap - 1, -1):
        suffix = a[-olen:]
        prefix = b[:olen]

        mismatches = hamming_distance(suffix, prefix, max_mismatches)

        if mismatches <= max_mismatches:
            if olen > best_overlap or (
                olen == best_overlap and mismatches < best_mismatches
            ):
                best_overlap = olen
                best_mismatches = mismatches
                # If we found exact match, can stop
                if mismatches == 0:
                    break

    return best_overlap, best_mismatches


def find_circular_overlap(
    a: str, b: str, min_overlap: int, max_mismatches: int = 0
) -> Tuple[int, int]:
    """
    Find overlap considering circular nature of b.
    Returns (overlap_length, num_mismatches).
    """
    # First try regular linear overlap
    olen, mism = find_overlap(a, b, min_overlap, max_mismatches)

    # If we got good overlap or reads are short, return
    if olen >= len(b) * 0.6 or len(b) < 20:
        return olen, mism

    # Try circular: extend b by concatenating with itself
    bb = b + b
    best_olen = olen
    best_mism = mism

    for o in range(min(len(a), len(b)), max(min_overlap - 1, best_olen), -1):
        suffix = a[-o:]
        prefix = bb[:o]

        mism_count = hamming_distance(suffix, prefix, max_mismatches)

        if mism_count <= max_mismatches:
            # Actual overlap is capped at length of b
            actual_olen = min(o, len(b))
            if actual_olen > best_olen or (
                actual_olen == best_olen and mism_count < best_mism
            ):
                best_olen = actual_olen
                best_mism = mism_count
                if mism_count == 0:
                    break

    return best_olen, best_mism


def merge_reads(a: str, b: str, overlap: int) -> str:
    """Merge two reads given overlap length."""
    return a + b[overlap:]


class OverlapGraph:
    """Graph representing overlaps between reads."""

    def __init__(self, reads: List[str], min_overlap: int, max_mismatches: int = 0):
        self.reads = [r.strip().upper() for r in reads]
        self.min_overlap = min_overlap
        self.max_mismatches = max_mismatches
        self.overlaps = defaultdict(
            list
        )  # read_idx -> [(next_idx, overlap_len, mismatches)]
        self._build_graph()

    def _build_graph(self):
        """Build overlap graph."""
        n = len(self.reads)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                olen, mism = find_circular_overlap(
                    self.reads[i], self.reads[j], self.min_overlap, self.max_mismatches
                )

                if olen >= self.min_overlap:
                    # Store as (overlap_len, mismatches, target_idx)
                    # Higher overlap and lower mismatches are better
                    self.overlaps[i].append((olen, mism, j))

        # Sort overlaps by quality (longer overlap, fewer mismatches)
        for i in self.overlaps:
            self.overlaps[i].sort(key=lambda x: (-x[0], x[1]))

    def greedy_assembly(self) -> str:
        """Greedy assembly: always take best overlap."""
        if not self.reads:
            return ""

        if len(self.reads) == 1:
            return self.reads[0]

        contigs = list(range(len(self.reads)))
        contig_seqs = {i: self.reads[i] for i in range(len(self.reads))}

        # Track merges to avoid getting stuck
        iterations_without_merge = 0
        max_iterations = len(self.reads) * 2

        while len(contigs) > 1 and iterations_without_merge < max_iterations:
            best_merge = None
            best_quality = (-1, float("inf"), 0, 0)  # (overlap, mismatches, -from, -to)

            # Find best overlap between any two contigs
            for i in contigs:
                if i not in self.overlaps:
                    continue

                for olen, mism, j in self.overlaps[i]:
                    if j not in contigs or j == i:
                        continue

                    quality = (
                        olen,
                        -mism,
                        -i,
                        -j,
                    )  # Maximize overlap, minimize mismatches
                    if quality > best_quality:
                        best_quality = quality
                        best_merge = (i, j, olen)

            # If no valid merge found, try with lower overlap threshold
            if best_merge is None:
                # Try to find any overlap at all
                reduced_min = max(1, self.min_overlap // 2)
                for i in contigs:
                    for j in contigs:
                        if i == j:
                            continue

                        olen, mism = find_circular_overlap(
                            contig_seqs[i],
                            contig_seqs[j],
                            reduced_min,
                            self.max_mismatches + 1,  # Allow slightly more errors
                        )

                        if olen >= reduced_min:
                            quality = (olen, -mism, -i, -j)
                            if quality > best_quality:
                                best_quality = quality
                                best_merge = (i, j, olen)

                if best_merge is None:
                    break

            i, j, olen = best_merge

            # Merge contigs
            merged_seq = merge_reads(contig_seqs[i], contig_seqs[j], olen)

            # Remove old contigs and add merged one
            contigs.remove(i)
            contigs.remove(j)
            new_idx = len(contig_seqs)
            contigs.append(new_idx)
            contig_seqs[new_idx] = merged_seq

            # Update overlaps for new contig
            self.overlaps[new_idx] = []
            for k in contigs:
                if k == new_idx:
                    continue

                # Check overlap from new contig to k
                olen, mism = find_circular_overlap(
                    merged_seq, contig_seqs[k], self.min_overlap, self.max_mismatches
                )
                if olen >= self.min_overlap:
                    self.overlaps[new_idx].append((olen, mism, k))

                # Check overlap from k to new contig
                olen, mism = find_circular_overlap(
                    contig_seqs[k], merged_seq, self.min_overlap, self.max_mismatches
                )
                if olen >= self.min_overlap:
                    if k not in self.overlaps:
                        self.overlaps[k] = []
                    self.overlaps[k].append((olen, mism, new_idx))

            # Sort new overlaps
            self.overlaps[new_idx].sort(key=lambda x: (-x[0], x[1]))

            iterations_without_merge = 0

        # Return the final contig (or join remaining if multiple)
        if contigs:
            if len(contigs) == 1:
                final_seq = contig_seqs[contigs[0]]
            else:
                # Join remaining contigs (shouldn't happen often)
                final_seq = "".join(contig_seqs[c] for c in contigs)

            # For circular genomes, remove potential duplication
            return self._trim_circular(final_seq)

        return ""

    def _trim_circular(self, sequence: str) -> str:
        """Remove circular duplication from sequence."""
        n = len(sequence)
        if n == 0:
            return sequence

        # Use KMP failure function to find border
        pi = [0] * n
        j = 0
        for i in range(1, n):
            while j > 0 and sequence[i] != sequence[j]:
                j = pi[j - 1]
            if sequence[i] == sequence[j]:
                j += 1
            pi[i] = j

        border = pi[-1]
        if 0 < border < n:
            return sequence[:-border]
        return sequence


def assemble_olc(
    reads: List[str], min_overlap: int = None, max_mismatches: int = 0
) -> str:
    """
    Assemble genome using Overlap-Layout-Consensus approach.

    Args:
        reads: List of sequencing reads
        min_overlap: Minimum overlap required (auto-selected if None)
        max_mismatches: Maximum mismatches allowed in overlap

    Returns:
        Assembled circular genome sequence
    """
    if not reads:
        return ""

    if len(reads) == 1:
        return reads[0].strip().upper()

    # Auto-select minimum overlap
    if min_overlap is None:
        read_len = len(reads[0].strip())
        if read_len <= 5:
            min_overlap = max(1, read_len - 2)
        elif read_len < 10:
            min_overlap = max(1, read_len - 3)
        else:
            min_overlap = max(1, int(read_len * 0.6))

    # Build overlap graph and assemble
    graph = OverlapGraph(reads, min_overlap, max_mismatches)
    assembled = graph.greedy_assembly()

    # Canonicalize circular sequence
    return canonicalize_circular(assembled)


def reconstruct_genome(reads: List[str], has_errors: bool = False) -> str:
    """
    Reconstruct circular genome from reads using OLC.

    Args:
        reads: List of sequencing reads
        has_errors: Whether reads may contain errors

    Returns:
        Assembled genome sequence
    """
    if not reads:
        return ""

    read_len = len(reads[0].strip())

    # Adjust parameters based on error status and read length
    if has_errors:
        # Allow more mismatches and require less overlap
        if read_len < 10:
            max_mismatches = 1
            min_overlap = max(1, read_len - 3)
        else:
            max_mismatches = min(2, max(1, read_len // 25))
            min_overlap = max(1, int(read_len * 0.4))
    else:
        # No errors expected
        if read_len <= 5:
            max_mismatches = 0
            min_overlap = max(1, read_len - 2)
        else:
            max_mismatches = 0
            min_overlap = max(1, int(read_len * 0.6))

    result = assemble_olc(reads, min_overlap=min_overlap, max_mismatches=max_mismatches)

    return result


if __name__ == "__main__":

    print("Testing Overlap-Layout-Consensus Assembly\n")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_solutions):
        solution = test_case["solution"]
        reads = test_case["reads"]
        has_errors = test_case.get("has_errors", False)

        my_solution = reconstruct_genome(reads, has_errors=has_errors)
        canonical_reference = canonicalize_circular(solution)

        if canonical_reference == my_solution:
            print(f"Test {i}: PASS")
            passed += 1
        else:
            print(f"Test {i}: FAIL")
            print(f"  Expected length: {len(canonical_reference)}")
            print(f"  Got length:      {len(my_solution)}")
            print(f"  Has errors:      {has_errors}")
            print(f"  Read length:     {len(reads[0])}")
            print(f"  Num reads:       {len(reads)}")
            if len(my_solution) < 200:
                print(f"  Expected: {canonical_reference}")
                print(f"  Got:      {my_solution}")
            failed += 1

    print("=" * 80)
    print(
        f"\nResults: {passed} passed, {failed} failed out of {len(test_solutions)} tests"
    )
