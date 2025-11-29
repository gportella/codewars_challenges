#! /usr/bin/env python

import itertools
from collections import Counter
import sys


def overlap(a, b, min_length=3):
    """Return length of longest suffix of 'a' matching
    a prefix of 'b' that is at least 'min_length'
    characters long.  If no such overlap exists,
    return 0."""
    start = 0  # start all the way at the left
    while True:
        start = a.find(b[:min_length], start)  # look for b's suffx in a
        if start == -1:  # no more occurrences to right
            return 0
        # found occurrence; check for full suffix/prefix match
        if b.startswith(a[start:]):
            return len(a) - start
        start += 1  # move just past previous match


def scs(ss):
    """Returns shortest common superstring of given
    strings, which must be the same length"""
    shortest_sup = None
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss) - 1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i + 1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i + 1][olen:]
        if shortest_sup is None or len(sup) < len(shortest_sup):
            shortest_sup = sup  # found shorter superstring
    return shortest_sup  # return shortest


def scs_list(ss):
    """Returns shortest common superstring of given
    strings, which must be the same length"""
    shortest_sup = []
    shortest = None
    shortest_list = []
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss) - 1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i + 1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i + 1][olen:]
        if len(shortest_sup) == 0 or len(sup) <= len(shortest_sup[0]):
            shortest_sup.append(sup)  # found shorter superstring
    # find the shortest
    for dd in shortest_sup:
        if shortest is None or len(dd) < shortest:
            shortest = len(dd)
    for dd in shortest_sup:
        if len(dd) == shortest:
            shortest_list.append(dd)
    return shortest_list  # return shortest


def smart_pick_maximal_overlap(reads, kmermap, k):
    """Return a pair of reads from the list with a
    maximal suffix/prefix overlap >= k.  Returns
    overlap length 0 if there are no such overlaps."""
    reada, readb = None, None
    best_olen = 0
    for a, b in itertools.permutations(reads, 2):
        if b in kmermap[a[-k:]]:
            olen = overlap(a, b, min_length=k)
            if olen > best_olen:
                reada, readb = a, b
                best_olen = olen
    return reada, readb, best_olen


def find_kmers(read, k):
    kmers = []
    for i in range(0, len(read) - k + 1):
        kmers.append(read[i : i + k])
    return sorted(kmers)


def buildKmerDict(reads, k):
    kmermap = {}
    for read in reads:
        kmers = find_kmers(read, k)
        for km in kmers:
            kmermap.setdefault(km, set()).add(read)
    return kmermap


def greedy_scs(reads, k):
    """Greedy shortest-common-superstring merge.
    Repeat until no edges (overlaps of length >= k)
    remain."""
    kmap = buildKmerDict(reads, k)
    read_a, read_b, olen = smart_pick_maximal_overlap(reads, kmap, k)
    while olen > 0:
        print(f" {olen}", end="\r")
        sys.stdout.flush()
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        kmap = buildKmerDict(reads, k)
        read_a, read_b, olen = smart_pick_maximal_overlap(reads, kmap, k)
    return "".join(reads)


def readFastq(filename):
    sequences = []
    qualities = []
    with open(filename) as fh:
        while True:
            fh.readline()  # skip name line
            seq = fh.readline().rstrip()  # read base sequence
            fh.readline()  # skip placeholder line
            qual = fh.readline().rstrip()  # base quality line
            if len(seq) == 0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities


reads, quals = readFastq("ads1_week4_reads.fq")
print("There are", len(reads), "reads")
genome = greedy_scs(reads, 10)
print(genome)
print("Length of genome", len(genome))
cnt = Counter(genome)
print("Counts", cnt)
