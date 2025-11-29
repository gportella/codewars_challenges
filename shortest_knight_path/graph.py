#! /usr/bin/env python
from typing import List, Tuple, Dict, Optional, Any
import sys


class Vertex:
    def __init__(self, key) -> None:
        self.key = key
        self.connectedTo: Dict["Vertex", Optional[int]] = {}
        self.color = "white"
        self.distance = sys.maxsize
        self.previous: Optional["Vertex"] = None
        self.discovery_time = 0
        self.closing_time = 0

    def addNeighbour(self, nbr: "Vertex", weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def __str__(self):
        return f"{self.key} is connected to {[x.key for x in self.connectedTo]}"

    def __eq__(self, other: "Vertex") -> bool:
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: "Vertex") -> bool:
        """Less than operator required for heapify"""
        return self.key < other.key


class Graph:
    def __init__(self) -> None:
        self.vertList: Dict[Any, Vertex] = {}
        self.numVertices = 0

    def addVertex(self, key) -> Vertex:
        self.numVertices += 1
        new_v = Vertex(key)
        self.vertList[key] = new_v
        return new_v

    def getVertex(self, n: Any):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            _ = self.addVertex(f)
        if t not in self.vertList:
            _ = self.addVertex(t)
        self.vertList[f].addNeighbour(self.vertList[t], weight=weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

    def __repr__(self) -> str:
        return f"G with {self.numVertices} vertices"

    def bfs(self, start: Any) -> bool:
        if (start := self.getVertex(start)) is None:
            return False
        start.distance = 0
        start.previous = None
        queue = [start]
        while queue:
            next_vert = queue.pop(0)
            for vertex in next_vert.getConnections():
                if vertex.color == "white":
                    vertex.color = "gray"
                    vertex.distance = next_vert.distance + 1
                    vertex.previous = next_vert
                    queue.append(vertex)
            next_vert.color = "black"
        return True

    def traverse(self, end: Any, show_path=False) -> List[Any]:
        end_v = self.getVertex(end)
        path = []
        if not end_v:
            print(f"Vertex {end} does not exist")
            return []
        current = end_v
        while current.previous:
            path.append(current.key)
            current = current.previous
        path.append(current.key)
        if show_path:
            print("HAHA", path)
            print(f"{' -> '.join([str(x) for x in path])}")
        return path


# g = Graph()

# for i in range(3):
#     g.addVertex(i)


# g.addEdge(0, 1, 5)
# g.addEdge(0, 6, 5)
# g.addEdge(1, 2, 5)
# g.addEdge(1, 3, 5)
# g.addEdge(1, 4, 5)
# g.addEdge(2, 5, 5)
# g.addEdge(5, 6, 5)

# xx = g.getVertex(1)
# print(f"X is {xx}")
# if xx:
#     g.bfs(xx)
# fafa = g.getVertex(6)
# if fafa:
#     print(f"Distance from 0 to 6 is {fafa.distance}")

# g.traverse(1, 6)

# for gg in g:
#     for nb in gg.getConnections():
#         print(f"{gg.id} --> {nb.id} w: {gg.getWeight(nb)}")
