#!/usr/bin/env python
"""
Genome assembly using de Bruijn graph approach.
Handles circular genomes with error tolerance.
"""

from typing import List, Optional
from collections import defaultdict, Counter


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


def trim_circular(sequence: str) -> str:
    """Remove circular duplication from sequence using KMP."""
    n = len(sequence)
    if n == 0:
        return sequence

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


class DeBruijnGraph:
    """de Bruijn graph for genome assembly."""

    def __init__(self, k: int):
        self.k = k
        self.graph = defaultdict(list)  # node -> [next_nodes]
        self.edge_usage = defaultdict(int)  # (u, v) -> times_used

    def add_kmer(self, kmer: str):
        """Add a k-mer to the graph."""
        prefix = kmer[:-1]
        suffix = kmer[1:]
        self.graph[prefix].append(suffix)

    def build_from_reads(self, reads: List[str], min_count: int = 1):
        """Build graph from reads with k-mer filtering."""
        kmer_counts = Counter()

        # Count all k-mers
        for read in reads:
            read = read.strip().upper()
            if len(read) < self.k:
                continue

            for i in range(len(read) - self.k + 1):
                kmer = read[i : i + self.k]
                if set(kmer) <= set("ACGT"):
                    kmer_counts[kmer] += 1

        # Add k-mers that pass threshold
        for kmer, count in kmer_counts.items():
            if count >= min_count:
                self.add_kmer(kmer)

    def find_eulerian_path(self) -> List[str]:
        """Find Eulerian path using Hierholzer's algorithm."""
        if not self.graph:
            return []

        # Count degrees
        in_deg = Counter()
        out_deg = Counter()

        for u in self.graph:
            out_deg[u] = len(self.graph[u])
            for v in self.graph[u]:
                in_deg[v] += 1

        # Find start node (prefer unbalanced source, else any)
        start = None
        for node in set(self.graph.keys()) | set(in_deg.keys()):
            if out_deg[node] > in_deg[node]:
                start = node
                break

        if start is None:
            # All balanced - pick node with highest out-degree
            start = max(self.graph.keys(), key=lambda x: len(self.graph[x]))

        # Hierholzer's algorithm
        curr_path = [start]
        path = []
        graph_copy = {u: list(vs) for u, vs in self.graph.items()}

        while curr_path:
            curr = curr_path[-1]
            if curr in graph_copy and graph_copy[curr]:
                next_node = graph_copy[curr].pop()
                curr_path.append(next_node)
            else:
                path.append(curr_path.pop())

        return path[::-1]

    def path_to_sequence(self, path: List[str]) -> str:
        """Convert node path to DNA sequence."""
        if not path:
            return ""
        if len(path) == 1:
            return path[0]

        seq = path[0]
        for node in path[1:]:
            if node:
                seq += node[-1]
        return seq


def assemble_with_k(reads: List[str], k: int, min_count: int = 1) -> str:
    """Assemble genome with specific k value."""
    if k < 2:
        return ""

    graph = DeBruijnGraph(k)
    graph.build_from_reads(reads, min_count=min_count)

    if not graph.graph:
        return ""

    path = graph.find_eulerian_path()
    sequence = graph.path_to_sequence(path)

    return sequence


def select_best_k(read_len: int, num_reads: int, has_errors: bool) -> List[int]:
    """Select candidate k values to try."""
    if read_len <= 3:
        return [2]
    elif read_len <= 5:
        return [read_len - 1, read_len - 2, 2]
    elif read_len <= 7:
        # For 6-7 bp reads, need careful k selection
        return [4, 3, 5, 2]
    elif read_len <= 10:
        candidates = [read_len - 3, read_len - 2, read_len - 4, read_len - 5]
    elif read_len <= 20:
        candidates = [read_len - 5, read_len - 7, read_len - 9, read_len - 3]
    elif read_len <= 30:
        if has_errors:
            candidates = [15, 17, 13, 19, 11]
        else:
            candidates = [read_len - 10, read_len - 12, read_len - 8, read_len - 14]
    else:
        if has_errors:
            candidates = [21, 19, 23, 17, 25]
        else:
            candidates = [25, 27, 23, 29, 21]

    return [k for k in candidates if k >= 2]


def assemble_debruijn(
    reads: List[str], k: Optional[int] = None, min_count: int = 1
) -> str:
    """
    Assemble genome using de Bruijn graph.

    Args:
        reads: List of sequencing reads
        k: k-mer size (auto-selected if None)
        min_count: Minimum k-mer count to include

    Returns:
        Assembled circular genome sequence
    """
    if not reads:
        return ""

    read_len = len(reads[0].strip())
    num_reads = len(reads)

    # Try multiple k values
    if k is not None:
        k_values = [k]
    else:
        k_values = select_best_k(read_len, num_reads, False)

    results = []
    target_len = read_len  # Approximate expected genome length

    for k_val in k_values:
        result = assemble_with_k(reads, k_val, min_count)

        if not result:
            continue

        # Trim potential circular duplication
        trimmed = trim_circular(result)

        # Store with score
        score = len(trimmed)  # Prefer longer assemblies initially
        results.append((score, trimmed, k_val))

    if not results:
        return ""

    # Sort by score and return best
    results.sort(key=lambda x: (abs(x[0] - target_len), -x[0]))
    return results[0][1]


def estimate_genome_length(reads: List[str]) -> int:
    """Estimate genome length from reads."""
    if not reads:
        return 0

    read_len = len(reads[0].strip())
    num_reads = len(reads)

    # For small genomes with good coverage, estimate from unique content
    if read_len <= 10:
        # Collect all unique substrings
        unique_content = set()
        for read in reads:
            r = read.strip().upper()
            for i in range(len(r)):
                for j in range(i + 1, len(r) + 1):
                    unique_content.add(r[i:j])

        # Rough estimate: genome is likely 1-3x read length
        return read_len * 3

    # For larger reads, assume decent coverage
    # Total bases / coverage ~ genome length
    total_bases = read_len * num_reads
    estimated_coverage = max(10, num_reads // 5)  # Conservative estimate
    return max(read_len, total_bases // estimated_coverage)


def reconstruct_genome(reads: List[str], has_errors: bool = False) -> str:
    """
    Reconstruct circular genome from reads using de Bruijn graph.

    Args:
        reads: List of sequencing reads
        has_errors: Whether reads may contain errors

    Returns:
        Assembled genome sequence
    """
    if not reads:
        return ""

    read_len = len(reads[0].strip())
    num_reads = len(reads)
    estimated_len = estimate_genome_length(reads)

    # Adjust parameters based on read length and error status
    if has_errors:
        # With errors, use lower k and lower threshold
        min_counts = [1]
        k_values = select_best_k(read_len, num_reads, has_errors=True)
    else:
        # Without errors, can try different thresholds
        min_counts = [1, 2] if num_reads > 5 else [1]
        k_values = select_best_k(read_len, num_reads, has_errors=False)

    results = []

    for min_count in min_counts:
        for k in k_values:
            result = assemble_with_k(reads, k, min_count)

            if not result:
                continue

            # Trim circular duplication
            trimmed = trim_circular(result)
            canonical = canonicalize_circular(trimmed)

            # Score based on length proximity to estimate
            len_diff = abs(len(canonical) - estimated_len)
            score = -len_diff  # Higher score is better

            # Bonus for reasonable length
            if estimated_len * 0.5 <= len(canonical) <= estimated_len * 2:
                score += 100

            results.append((score, canonical))

    if not results:
        return ""

    # Return best scoring result
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1]


if __name__ == "__main__":
    from test_solutions import test_solutions

    print("Testing de Bruijn Graph Assembly\n")
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
            print(f"Test {i}: ✓ PASS")
            passed += 1
        else:
            print(f"Test {i}: ✗ FAIL")
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
