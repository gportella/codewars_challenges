#! /usr/bin/env python

from graph import Graph
import math


class KnightFinder:
    def __init__(self, board_size: int = 8):
        self.bs = board_size
        self.graph = Graph()
        self._knightGraph()

    def _knightGraph(self):
        for row in range(self.bs):
            for col in range(self.bs):
                nodeId = self.pos_to_id(row, col)
                newPositions = self.create_moves(row, col, self.bs)
                for e in newPositions:
                    nid = self.pos_to_id(e[0], e[1])
                    self.graph.addEdge(nodeId, nid)

    def pos_to_id(self, row, column):
        return (row * self.bs) + column

    def id_to_pos(self, node_id):
        return (math.floor(node_id / self.bs), node_id % self.bs)

    def create_moves(self, x, y, bdSize):
        newMoves = []
        moveOffsets = [
            (-1, -2),
            (-1, 2),
            (-2, -1),
            (-2, 1),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ]
        for i in moveOffsets:
            newX = x + i[0]
            newY = y + i[1]
            if self._legal(newX) and self._legal(newY):
                newMoves.append((newX, newY))
        return newMoves

    def _legal(self, x):
        return x >= 0 and x < self.bs

    def algebraic_to_node(self, p1: str):
        """very minimal validation"""
        if len(p1) != 2:
            raise ValueError(f"Input {p1} incorrect format")
        col, row = ord(p1[0]) - 97, int(p1[1]) - 1
        return self.pos_to_id(row, col)

    def node_to_algebraic(self, node_id):
        """very minimal validation"""
        row, col = self.id_to_pos(node_id)
        return chr(row + 97) + str(col + 1)

    def min_path(self, p1, p2):
        start_node = self.algebraic_to_node(p1)
        end_node = self.algebraic_to_node(p2)
        if self.graph.bfs(start_node):
            end_v = self.graph.getVertex(end_node)
            if end_v:
                print(f"Min movements: {end_v.distance}")


def knight(p1, p2):
    kt = KnightFinder()
    kt.min_path(p1, p2)
    steps_follow = " -> ".join(
        [kt.node_to_algebraic(s) for s in kt.graph.traverse(kt.algebraic_to_node(p2))]
    )
    print(f"Path taken: {steps_follow}")


print("A1 to H8")
knight("a1", "h8")
print("C1 to H8")
knight("c2", "h8")
print("A3 to B5")
knight("a3", "b5")

