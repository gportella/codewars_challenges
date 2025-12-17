#! /usr/bin/env python

from graph import Graph
import sys


def buildGraph(wordFile):
    bucket_d = {}
    g = Graph()

    with open(wordFile) as fi:
        for line in fi:
            word = line.rstrip()
            for i in range(len(word)):
                w_b = word[:i] + "_" + word[i + 1 :]
                if w_b in bucket_d:
                    bucket_d[w_b].append(word)
                else:
                    bucket_d[w_b] = [word]

    for bucket in bucket_d:
        for word_1 in bucket_d[bucket]:
            for word_2 in bucket_d[bucket]:
                if word_1 != word_2:
                    g.addEdge(word_1, word_2)

    return g


my_g = buildGraph("4-letter-words-processed.txt")

my_g.bfs("ABED")
my_g.traverse("ZERO", show_path=True)
