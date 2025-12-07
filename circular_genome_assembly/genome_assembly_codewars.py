#! /usr/bin/env python
from typing import List, Tuple, Optional
from test_solutions import test_solutions
from collections import Counter, defaultdict


def hamming_mismatches(s1: str, s2: str, max_allowed: int = float("inf")) -> int:
    mismatches = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            mismatches += 1
            if mismatches > max_allowed:
                return mismatches
    return mismatches


def linear_overlap_exact(a: str, b: str, min_length: int) -> int:
    for olen in range(min(len(a), len(b)), min_length - 1, -1):
        if a[-olen:] == b[:olen]:
            return olen
    return 0


def linear_overlap_approx(
    a: str, b: str, min_length: int, max_mismatches: int
) -> Tuple[int, Optional[int]]:
    if max_mismatches == 0:
        return linear_overlap_exact(a, b, min_length), 0

    best = (0, None)
    for olen in range(min(len(a), len(b)), min_length - 1, -1):
        mism = hamming_mismatches(a[-olen:], b[:olen], max_mismatches)
        if mism <= max_mismatches:
            if olen > best[0] or (
                olen == best[0] and (best[1] is None or mism < best[1])
            ):
                best = (olen, mism)
                break
    return best


def circularoverlap(
    a: str, b: str, min_length: int, max_mismatches: int
) -> Tuple[int, Optional[int]]:
    olen, mism = linear_overlap_approx(a, b, min_length, max_mismatches)
    best = (olen, mism)

    if olen < min(len(a), len(b)) * 0.6:
        bb = b + b
        n = len(b)
        for olen in range(min(len(a), n), max(min_length - 1, best[0]), -1):
            mism = hamming_mismatches(a[-olen:], bb[:olen], max_mismatches)
            if mism <= max_mismatches:
                candidate_olen = min(olen, n)
                if candidate_olen > best[0] or (
                    candidate_olen == best[0] and (best[1] is None or mism < best[1])
                ):
                    best = (candidate_olen, mism)
                    break

    return best


def best_pair(
    a: str, b: str, k: int, e: int
) -> Tuple[Optional[str], Optional[str], int, Optional[int]]:
    oa_len, oa_mism = circularoverlap(a, b, min_length=k, max_mismatches=e)
    ob_len, ob_mism = circularoverlap(b, a, min_length=k, max_mismatches=e)

    if oa_len == 0 and ob_len == 0:
        return None, None, 0, None

    if (
        oa_len > ob_len
        or (
            oa_len == ob_len
            and oa_mism is not None
            and ob_mism is not None
            and oa_mism < ob_mism
        )
        or (oa_len == ob_len and oa_mism == ob_mism and a < b)
    ):
        return a, b, oa_len, oa_mism
    else:
        return b, a, ob_len, ob_mism


def is_better_overlap(
    olen: int,
    mism: Optional[int],
    merged: str,
    best_olen: int,
    best_mism: Optional[int],
    best_merged: Optional[str],
) -> bool:
    """Determine if current overlap is better than the best so far"""
    return (
        olen > best_olen
        or (
            olen == best_olen
            and best_mism is not None
            and mism is not None
            and mism < best_mism
        )
        or (
            olen == best_olen
            and mism == best_mism
            and best_merged is not None
            and len(merged) < len(best_merged)
        )
        or (
            olen == best_olen
            and mism == best_mism
            and best_merged is not None
            and len(merged) == len(best_merged)
            and merged < best_merged
        )
    )


def scs_optimized(reads: List[str], k: int = 1, e: int = 1) -> str:
    if not reads:
        return ""

    reads = list(reads)
    if len(reads) == 1:
        return reads[0]

    min_useful_overlap = max(k, min(len(read) for read in reads) // 3)

    while len(reads) > 1:
        best_left = best_right = None
        best_olen = 0
        best_mism = None
        best_merged = None
        found_excellent_overlap = False

        for i in range(len(reads)):
            if found_excellent_overlap:
                break
            for j in range(len(reads)):
                if i == j:
                    continue

                left, right, olen, mism = best_pair(reads[i], reads[j], k, e)
                if olen < min_useful_overlap:
                    continue

                merged = left + right[olen:]

                min_read_len = min(len(reads[i]), len(reads[j]))
                if olen >= min_read_len * 0.8:
                    best_left, best_right = left, right
                    best_olen, best_mism, best_merged = olen, mism, merged
                    found_excellent_overlap = True
                    break

                if is_better_overlap(
                    olen, mism, merged, best_olen, best_mism, best_merged
                ):
                    best_left, best_right, best_olen, best_mism, best_merged = (
                        left,
                        right,
                        olen,
                        mism,
                        merged,
                    )

        if best_olen == 0 or best_left is None or best_right is None:
            return "".join(reads)

        reads.remove(best_left)
        reads.remove(best_right)
        reads.append(best_merged)

        if len(reads) <= 10:
            min_useful_overlap = max(1, min_useful_overlap - 1)

    return reads[0] if reads else ""


def greedy_scs(reads: List[str], k: int = 1, e: int = 1) -> str:
    """Faster greedy approach for large datasets"""
    if not reads:
        return ""
    reads = list(reads)
    if len(reads) == 1:
        return reads[0]
    while len(reads) > 1:
        best_merge = None
        best_quality = (0, float("inf"))

        for i in range(len(reads)):
            for j in range(i + 1, len(reads)):
                for read_i, read_j in [(reads[i], reads[j]), (reads[j], reads[i])]:
                    left, right, olen, mism = best_pair(read_i, read_j, k, e)
                    if olen > 0:
                        quality = (olen, mism or 0)
                        if quality[0] > best_quality[0] or (
                            quality[0] == best_quality[0]
                            and quality[1] < best_quality[1]
                        ):
                            best_quality = quality
                            best_merge = (i, j, left + right[olen:])

        if best_merge is None:
            return "".join(reads)

        i, j, merged = best_merge
        reads.pop(max(i, j))
        reads.pop(min(i, j))
        reads.append(merged)

    return reads[0] if reads else ""


def trim_circular(S: str) -> str:
    n = len(S)
    if n == 0:
        return S
    pi = [0] * n
    j = 0
    for i in range(1, n):
        while j > 0 and S[i] != S[j]:
            j = pi[j - 1]
        if S[i] == S[j]:
            j += 1
        pi[i] = j

    border = pi[-1]
    if 0 < border < n:
        return S[:-border]
    return S


def reconstruct_genome(reads: List[str], has_errors: bool = False) -> str:
    if not reads:
        return ""

    e = 2 if has_errors else 0
    read_len = len(next(iter(reads)))
    if read_len < 10:
        the_k = 1
    else:
        the_k = int(0.9 * len(next(iter(reads))))

    if len(reads) > 100:
        long_genome = greedy_scs(reads, k=the_k, e=e)
    else:
        long_genome = scs_optimized(reads, k=the_k, e=e)

    trimmed = trim_circular(long_genome)
    return trimmed


def align_read_hamming(reads: List[str], genome: str, max_allowed: int = float("inf")):
    R = len(genome)
    ref2 = genome + genome

    good_reads = []
    for read in reads:
        L = len(read)

        best_start = None
        best_mismatches = float("inf")
        # best_mismatches = 2
        best_window = None
        # Only consider starts within the original genome length
        for s in range(R):
            window = ref2[s : s + L]
            if len(window) < L:
                break  # should not happen with ref2 length, but safety
            mm = hamming_mismatches(
                read,
                window,
                max_allowed=best_mismatches
                if max_allowed == float("inf")
                else min(best_mismatches, max_allowed),
            )
            if mm < best_mismatches:
                best_mismatches = mm
                best_start = s % R
                best_window = window
                if best_mismatches == 0:
                    break
            if best_mismatches <= max_allowed:
                pass

        if best_mismatches < 2:
            good_reads.append(read)
        window = best_window.upper()
        markers = "".join("|" if a == b else "." for a, b in zip(window, read))
        assert len(read) == len(window)

        # Optional prefix showing circular position
        prefix = f"pos {best_start % R}: "

        print(prefix + best_window)
        print(" " * len(prefix) + markers)
        print(" " * len(prefix) + read)

    return good_reads
    # return best_start, best_mismatches, best_window


def canonicalize_circular(sequence: str) -> str:
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


def align_best_start_le1(read: str, contig: str) -> Optional[int]:
    contig = contig.strip().upper()
    read = read.strip().upper()
    R = len(contig)
    ref2 = contig + contig
    L = len(read)

    best_s = None
    best_mm = 2
    for s in range(R):
        w = ref2[s : s + L]
        if len(w) < L:
            break
        mm = hamming_mismatches(read, w, max_allowed=1)
        if mm <= 1 and mm < best_mm:
            best_mm = mm
            best_s = s
            if mm == 0:
                break
    return best_s


def correct_read_against_contig(read: str, contig: str) -> Optional[str]:
    """Return read with its single mismatch corrected to the contig base; otherwise unchanged."""
    r = read.strip().upper()
    c = contig.strip().upper()
    if not r or not c:
        return None

    s = align_best_start_le1(r, c)
    if s is None:
        return None

    ref2 = c + c
    L = len(r)
    window = ref2[s : s + L]

    mism_idxs = [i for i, (rb, wb) in enumerate(zip(r, window)) if rb != wb]

    if len(mism_idxs) == 0:
        return read
    if len(mism_idxs) > 2:
        return None

    # Correct the single mismatch
    for i in mism_idxs:
        corrected_base = window[i]
        corrected = list(r)
        corrected[i] = corrected_base
    return "".join(corrected)


def correct_reads(reads: List[str], contig: str) -> List[str]:
    """Polish reads by correcting at most one mismatch per read based on contig."""
    corrected = []
    for read in reads:
        read_fixed = correct_read_against_contig(read, contig)
        if read_fixed:
            corrected.append(read_fixed)
    return corrected


def rc(s: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return s.translate(comp)[::-1]


def kmers(seq: str, k: int):
    for i in range(len(seq) - k + 1):
        yield seq[i : i + k]


def build_kmer_counts(reads, k: int):
    counts = Counter()
    for r in reads:
        r = r.strip().upper()
        if len(r) < k:
            continue
        for s in (r, rc(r)):
            for kmer in kmers(s, k):
                if set(kmer) <= set("ACGT"):
                    counts[kmer] += 1
    return counts


def build_graph(counts, min_count: int):
    graph = defaultdict(Counter)  # node -> Counter(next_node -> weight)
    for kmer, c in counts.items():
        if c < min_count:
            continue
        u = kmer[:-1]
        v = kmer[1:]
        graph[u][v] += c
    return graph


def simplify_graph(graph):
    return {u: vs for u, vs in graph.items() if vs}


def best_cycle(graph):
    if not graph:
        return ""
    start = max(graph.keys(), key=lambda u: sum(graph[u].values()))
    path_nodes = [start]
    visited_edges = Counter()
    u = start
    for _ in range(10000):
        if not graph.get(u):
            break
        v, w = max(graph[u].items(), key=lambda kv: kv[1])
        edge = (u, v)
        visited_edges[edge] += 1
        path_nodes.append(v)
        u = v
        if v == start and len(path_nodes) > 1:
            break
    if len(path_nodes) < 2:
        return ""
    seq = path_nodes[0]
    for node in path_nodes[1:]:
        seq += node[-1]
    return seq


def assemble_debruijn(reads, k: int = 20, min_count: int = 2):
    counts = build_kmer_counts(reads, k)
    graph = build_graph(counts, min_count=min_count)
    graph = simplify_graph(graph)
    contig = best_cycle(graph)
    return canonicalize_circular(contig)


if __name__ == "__main__":
    for i, test_case in enumerate(test_solutions):
        print("@@@@@@@@@@ Start @@@@@@@@@@@")
        solution = test_case["solution"]
        reads = test_case["reads"]
        has_errors = test_case.get("has_errors", False)
        my_solution = reconstruct_genome(reads, has_errors=has_errors)
        canonical_reference = canonicalize_circular(solution)
        canonical_mine = canonicalize_circular(my_solution)
        reads_fixed = correct_reads(reads, my_solution)
        len_read = len(next(iter(reads)))

        if len_read > 20 and len(reads_fixed) != len(reads):
            print("De Bruijn")
            my_solution = assemble_debruijn(reads, k=21, min_count=2)
            canonical_mine = canonicalize_circular(my_solution)
            reads_fixed = correct_reads(reads, canonical_mine)
            print(f"len fixed is {len(reads_fixed)}")
            my_solution = reconstruct_genome(reads_fixed, has_errors=has_errors)
            canonical_mine = canonicalize_circular(my_solution)

        if canonical_reference != canonical_mine:
            print("\n#### Wrong! #####\n")
        else:
            print("\n================> That's same sequence")
        print("My trimmed solution")
        print(canonical_mine)
        print("Solution canonical")
        print(canonical_reference)
        print()
        if len_read > 20:
            align_read_hamming([canonical_reference], canonical_mine, max_allowed=1)
        print("\n********* Done ***************")
