#! /usr/bin/env python

from typing import Dict, Optional, Any
import sys


class Vertex:
    def __init__(self, key) -> None:
        self.key = key
        self.connections: Dict["Vertex", Optional[int]] = {}
        self.color = "white"
        self.distance = sys.maxsize
        self.previous: Optional["Vertex"] = None

    def __eq__(self, other: "Vertex") -> bool:
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


class Graph:
    def __init__(self) -> None:
        self.vertex_list: Dict[Any, Vertex] = {}

    def add_vertex(self, key):
        new_v = Vertex(key)
        self.vertex_list[key] = new_v

    def get_vertex(self, n: Any):
        if n in self.vertex_list:
            return self.vertex_list[n]
        else:
            return None

    def add_edge(self, f, t, weight=0):
        if f not in self.vertex_list:
            self.add_vertex(f)
        if t not in self.vertex_list:
            self.add_vertex(t)
        self.vertex_list[f].connections[self.vertex_list[t]] = weight

    def bfs(self, start: Any) -> bool:
        if (start := self.get_vertex(start)) is None:
            return False
        start.distance = 0
        start.previous = None
        queue = [start]
        while queue:
            next_vert = queue.pop(0)
            for vertex in next_vert.connections:
                if vertex.color == "white":
                    vertex.color = "gray"
                    vertex.distance = next_vert.distance + 1
                    vertex.previous = next_vert
                    queue.append(vertex)
            next_vert.color = "black"
        return True


class KnightFinder:
    def __init__(self, board_size: int = 8):
        self.bs = board_size
        self.graph = Graph()
        self._build_graph()

    def _build_graph(self):
        for row in range(self.bs):
            for col in range(self.bs):
                node_id = self.pos_to_id(row, col)
                pos = self.create_moves(row, col)
                for e in pos:
                    nid = self.pos_to_id(e[0], e[1])
                    self.graph.add_edge(node_id, nid)

    def pos_to_id(self, row, column):
        return (self.bs * row) + column

    def create_moves(self, row, col):
        moves = []
        knight_rules = [
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
            (-1, -2),
            (-1, 2),
            (-2, -1),
            (-2, 1),
        ]
        for i in knight_rules:
            new_row, new_col = row + i[0], col + i[1]
            if self._valid(new_col) and self._valid(new_col):
                moves.append((new_row, new_col))
        return moves

    def _valid(self, x):
        return x >= 0 and x < self.bs

    def __algebraic_to_node(self, p1: str):
        if len(p1) != 2:
            raise ValueError(f"Input {p1} incorrect format")
        col, row = ord(p1[0]) - 97, int(p1[1]) - 1
        return self.pos_to_id(row, col)

    def min_path(self, p1, p2) -> int:
        start_node = self.__algebraic_to_node(p1)
        end_node = self.__algebraic_to_node(p2)
        if self.graph.bfs(start_node):
            end_v = self.graph.get_vertex(end_node)
            if end_v:
                return end_v.distance
        return -1


def knight(p1, p2):
    kt = KnightFinder()
    return kt.min_path(p1, p2)


out = knight("a1", "b2")
print(out)
out = knight("a3", "b5")
print(out)
